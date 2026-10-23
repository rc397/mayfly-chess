"""The Kaggle notebook that trains MayFly's main run on a TPU session, and its CPU test.

kaggle_train_ctl.py pushes it with private datasets attached, holding the code (train_jax.py, train_xla.py and
train_av.py's modules), the run's resume checkpoint (ckpt_<TAG>_last.pt with model and AdamW moments) and a slice of
the training shards. It copies the checkpoint somewhere writable, makes sure PyTorch/XLA is there, trains on every
chip for the budget, and leaves the new checkpoint (train_av's format) and xla_summary.json in /kaggle/working for
the controller to resume from.

MODE "tpu" is the real run. "cpu" is a free test on XLA's CPU backend with a tiny network, which runs every path but
the chip.
"""
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

TIME_BUDGET = 8 * 3600 + 15 * 60          # replaced by the pusher
MODE = "__MODE__"                           # "tpu" or "cpu"
RUN = "__RUN__"
T0 = time.time()
WORK = "/kaggle/working"
CODE = "/kaggle/tmp/mayfly_train"
# The ways to train on the TPU, best first. JAX with one process driving every chip (train_jax.py), then
# train_xla.py with a process per chip, one process with the batch sharded over every chip, one chip at eight times
# the local batch, and one chip at the local batch.
MODES = [("jax", {}), ("procs", {}), ("spmd", {}), ("single", {"XLA_BATCH_X": "8"}),
         ("single", {"XLA_BATCH_X": "1"})]
START_MIN = float(os.environ.get("START_MIN", 10))            # until xla_started, when the chips answer and sum
FIRST_STEP_MIN = float(os.environ.get("FIRST_STEP_MIN", 45))  # until the first heartbeat, which covers the compile
STALL_MIN = float(os.environ.get("STALL_MIN", 15))            # between heartbeats once training
GIVE_UP_MIN = float(os.environ.get("GIVE_UP_MIN", 100))       # if no mode trains by then, stop before Kaggle's 2 h rule
RESTARTS = 2


def say(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def sh(cmd, check=True, timeout=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        say(f"FAILED ({r.returncode}): {cmd}\n{(r.stdout or '')[-1500:]}\n{(r.stderr or '')[-2500:]}")
        raise SystemExit(1)
    return r


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None


def ensure_torch_xla(tpu):
    """Make sure PyTorch/XLA imports.

    torch_xla only works with the torch it was built against, so a working pair is left alone. Otherwise the newest
    torch_xla releases on PyPI are tried in turn, each with the CPU build of the same torch version, until one imports."""
    check = f"{sys.executable} -c \"import torch, torch_xla; print(torch.__version__, torch_xla.__version__)\""
    r = sh(check, check=False)
    if r.returncode == 0:
        say(f"torch and torch_xla: {r.stdout.strip()}")
        return
    r = sh(f"{sys.executable} -m pip index versions torch_xla", check=False)
    import re
    m = re.search(r"Available versions:\s*(.+)", r.stdout or "")
    versions = [v.strip() for v in m.group(1).split(",")] if m else []
    versions = [v for v in versions if re.fullmatch(r"\d+\.\d+(\.\d+)?", v)][:4]
    say(f"torch_xla does not import here; trying releases {versions}")
    for v in versions:
        base = ".".join(v.split(".")[:2])
        t = sh(f"{sys.executable} -m pip install -q 'torch=={base}.*' --index-url https://download.pytorch.org/whl/cpu",
               check=False, timeout=1800)
        x = sh(f"{sys.executable} -m pip install -q '{'torch_xla[tpu]' if tpu else 'torch_xla'}=={v}'", check=False,
               timeout=1800)
        r = sh(check, check=False)
        if r.returncode == 0:
            say(f"torch and torch_xla: {r.stdout.strip()} (installed for {v})")
            return
        say(f"  {v}: does not import ({(t.stderr or '')[-200:]} | {(x.stderr or '')[-200:]} | {(r.stderr or '')[-300:]})")
    say("no torch_xla release imports here")
    raise SystemExit(1)


def start_labelling(seconds):
    """Label a batch from the Kaggle queue on the spare host cores while the TPU trains.

    Runs kaggle_kernel.py with every core but `spare` (kept for the trainer's processes and loaders), at the lowest
    priority, writing into /kaggle/working/label and finishing a few minutes before training. Only with a batch in the
    code (label.json)."""
    cfg_p = os.path.join(CODE, "label.json")
    if not os.path.exists(cfg_p):
        return None
    cfg = json.load(open(cfg_p))
    # Not beside the trainer by default, since labelling engines starved the trainer processes of the host CPUs they
    # need for tracing and loaders. "with_training": true in label.json turns it back on.
    if MODE == "tpu" and not cfg.get("with_training"):
        say(f"not labelling batch {cfg.get('batch')}: the trainer has the host's CPUs to itself")
        return None
    src = open(os.path.join(CODE, "kaggle_kernel.py"), encoding="utf-8").read()
    spare = int(cfg.get("spare", 24))
    subs = [('BATCH = "__BATCH__"', f'BATCH = "{cfg["batch"]}"'),
            ('LINES_FILE = "__LINES_FILE__"', f'LINES_FILE = "{cfg["file"]}"'),
            ("WORKERS = cpus()", f"WORKERS = max(8, cpus() - {spare})"),
            ("TIME_BUDGET = 8 * 3600 + 30 * 60", f"TIME_BUDGET = {max(600, seconds - 300)}"),
            ('OUT = "/kaggle/working"', 'OUT = "/kaggle/working/label"')]
    for a, b in subs:
        if a not in src:
            say(f"kaggle_kernel.py has no {a!r}; no labelling this session")
            return None
        src = src.replace(a, b)
    os.makedirs(os.path.join(WORK, "label"), exist_ok=True)
    path = os.path.join(CODE, "label_kernel.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    log = open(os.path.join(WORK, "label", "label.log"), "w")
    say(f"labelling batch {cfg['batch']} ({cfg.get('positions', '?')} positions) on the spare cores, nice 19")
    return subprocess.Popen(["nice", "-n", "19", sys.executable, path], cwd=CODE, stdout=log, stderr=subprocess.STDOUT)


def kill_tree(p):
    """Kill the trainer and every process it made, since a child left alive would hold its chip and the next mode
    could not start. Each trainer is started in its own process group, so the group goes at once."""
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    p.wait()
    for _ in range(30):
        try:
            os.killpg(p.pid, 0)
        except (ProcessLookupError, PermissionError):
            break
        time.sleep(2)
    time.sleep(10)                                # the TPU runtime lets go of the chips


def attempt(mode, extra, env, ck_dir, end):
    """Start the trainer in one mode (train_jax.py for jax, train_xla.py for the rest) and see it to its first step.

    Returns (process, None) once it trains, or (None, why) after stopping it, when it did not answer within START_MIN,
    answered but made no step within FIRST_STEP_MIN, or exited first."""
    started, beat = os.path.join(ck_dir, "xla_started"), os.path.join(ck_dir, "xla_heartbeat")
    for f in (started, beat):
        try:
            os.remove(f)
        except OSError:
            pass
    e = dict(env, XLA_MODE=mode, XLA_SECONDS=str(int(end - time.time())), **extra)
    script = "train_jax.py" if mode == "jax" else "train_xla.py"
    say(f"starting {script}: {mode} {extra or ''} for {(end - time.time()) / 3600:.2f}h")
    t0 = time.time()
    p = subprocess.Popen([sys.executable, os.path.join(CODE, script)], cwd=CODE, env=e, start_new_session=True)
    p.t0 = t0
    while True:
        if os.path.exists(beat):
            say(f"{mode}: training, first step {time.time() - t0:.0f}s after the start")
            return p, None
        rc, up = p.poll(), os.path.exists(started)
        if rc is not None:
            kill_tree(p)
            return None, f"exited {rc} before its first step" + ("" if up else ", the chips never answered")
        if not up and time.time() - t0 > START_MIN * 60:
            kill_tree(p)
            return None, f"the chips did not answer within {START_MIN:.0f} minutes"
        if time.time() - t0 > FIRST_STEP_MIN * 60:
            kill_tree(p)
            return None, f"no first step within {FIRST_STEP_MIN:.0f} minutes (the compile)"
        time.sleep(10)


def watch(p, ck_dir):
    """Watch a training trainer to its end. Returns (exit code, why the watchdog stopped it or None).

    A trainer that wrote its summary in this attempt has finished whatever its exit code, since the runtime's teardown
    can abort after everything is saved, and it must never be taken for a crash and started again."""
    beat, summ = os.path.join(ck_dir, "xla_heartbeat"), os.path.join(ck_dir, "xla_summary.json")
    finished = lambda: os.path.exists(summ) and os.path.getmtime(summ) >= p.t0  # noqa: E731
    while p.poll() is None:
        time.sleep(30)
        if p.poll() is not None:                  # it ended during the sleep, so not a stall
            break
        if os.path.exists(beat) and time.time() - os.path.getmtime(beat) > STALL_MIN * 60:
            kill_tree(p)
            if finished():                        # all written, then hung in its teardown
                say(f"the trainer hung after writing its summary; stopped, finished")
                return 0, None
            say(f"watchdog: no step for {STALL_MIN:.0f} minutes; trainer stopped")
            return p.returncode if p.returncode is not None else -9, f"no step for {STALL_MIN:.0f} minutes"
    rc = p.wait()
    if rc != 0:
        kill_tree(p)                              # whatever of it is left
        if finished():
            say(f"the trainer exited {rc} after writing its summary: finished")
            rc = 0
    return rc, None


def main():
    code = find("train_xla.py")
    ckpt = find("ckpt_mayfly_last.pt")
    if not code or not ckpt:
        say(f"missing from /kaggle/input: code {code}, checkpoint {ckpt}")
        raise SystemExit(1)
    src = os.path.dirname(code)
    shutil.rmtree(CODE, ignore_errors=True)
    shutil.copytree(src, CODE, ignore=shutil.ignore_patterns("*.bin", "*.pt", "*.npy"))
    shards = os.path.dirname(find("av_*.bin") or code)
    ck_dir = os.path.join(WORK, "ckpt")
    os.makedirs(ck_dir, exist_ok=True)
    shutil.copy(ckpt, os.path.join(ck_dir, "ckpt_mayfly_last.pt"))
    for extra in ("prefix_selfplay_keys.npy",):
        p = find(extra)
        if p:
            shutil.copy(p, ck_dir)
    say(f"checkpoint {os.path.getsize(ckpt) / 1e9:.2f} GB, shards from {shards}: "
        + ", ".join(sorted({os.path.basename(p).split('_')[0] for p in glob.glob(os.path.join(shards, '*.bin'))})))
    ensure_torch_xla(MODE == "tpu")
    sh(f"{sys.executable} -m pip install -q chess", check=False)
    # the main run's settings (autostart.json, passed in by the pusher), then this session's
    env = dict(os.environ)
    run_env = json.load(open(os.path.join(CODE, "train_env.json")))
    env.update({k: str(v) for k, v in run_env.items()})
    env.update(CKPT_DIR=ck_dir, SHARDS=shards, VERSION_DIR=os.path.join(WORK, "versions_unused"), COMPILE="0",
               PYTHONUNBUFFERED="1", DP_DOWN_KEYS=os.path.join(ck_dir, "prefix_selfplay_keys.npy")
               if os.path.exists(os.path.join(ck_dir, "prefix_selfplay_keys.npy")) else "",
               XLA_SECONDS=str(int(TIME_BUDGET - (time.time() - T0) - min(1200, 0.15 * TIME_BUDGET))))
    sess = os.path.join(CODE, "session.json")   # this session's extras from the controller (a cooled copy)
    if os.path.exists(sess):
        extra = json.load(open(sess))
        if extra.get("cool_min"):
            env["XLA_COOL_MIN"] = str(extra["cool_min"])
            say(f"the last {extra['cool_min']} minutes cool a copy for the next candidate")
    say("TPU settings: " + json.dumps({k: v for k, v in sorted(os.environ.items())
                                        if k.startswith(("TPU_", "PJRT_", "LIBTPU", "CLOUD_TPU"))}))
    r = sh(f"{sys.executable} -m pip show libtpu", check=False)
    say("libtpu " + (" ".join(l for l in (r.stdout or "").splitlines() if l.startswith("Version")) or "not installed"))
    env.update(XLA_CACHE_DIR="/kaggle/tmp/xla_cache")
    if MODE == "tpu":
        env.update(PJRT_DEVICE="TPU")
        chain = MODES
    else:
        # One replica, since XLA's CPU backend runs replicas as threads of one process, which share train_av's module
        # state and deadlock at the first gradient average. This tests everything but the averaging itself.
        env.update(PJRT_DEVICE="CPU", CPU_NUM_DEVICES="1", XLA_LOG_EVERY="5", XLA_SAVE_MIN="3", XLA_MIN_SPS="0")
        chain = [("single", {})]
    end = T0 + TIME_BUDGET - min(1200, 0.15 * TIME_BUDGET)    # the trainer is done by then, the rest packs and saves
    say(f"training for {(end - time.time()) / 3600:.2f}h")
    lab = start_labelling(int(end - time.time()))
    # Try every way to train, best first, until one trains. A mode that cannot start, or starts and never steps, is
    # stopped and the next is tried, so only a session where none trains comes home without a checkpoint.
    attempts = []

    def first_that_trains(i, t_from, note=""):
        """The ways from chain[i] on, in turn, until one trains. Returns (process, its index) or (None, None)."""
        for j in range(i, len(chain)):
            mode, extra = chain[j]
            if time.time() - t_from > GIVE_UP_MIN * 60 or end - time.time() < 1800:
                attempts.append({"mode": mode, **extra, "result": "not tried: out of time" + note})
                return None, None
            p, why = attempt(mode, extra, env, ck_dir, end)
            attempts.append({"mode": mode, **extra, "result": (why or "trained") + note})
            if why is None:
                return p, j
            say(f"{mode} {extra or ''}: {why}; the next way")
        return None, None

    p, i = first_that_trains(0, time.time())
    # The watchdog stops a trainer with no step for STALL_MIN rather than leave it holding an idle TPU. A trainer that
    # crashes or stalls with half an hour left is restarted from its own last save, up to RESTARTS times, in the same
    # mode if it had saved since it started, otherwise in the next mode.
    ck = os.path.join(ck_dir, "ckpt_mayfly_last.pt")
    why, rc, restarts = None, (None if p else 1), 0
    while p is not None:
        rc, why = watch(p, ck_dir)
        if rc == 0 or restarts >= RESTARTS or end - time.time() < 1800:
            break
        restarts += 1
        saved = os.path.getmtime(ck) >= p.t0
        mode, extra = chain[i]
        say(f"the trainer stopped mid-session (exit {rc}{', ' + why if why else ''}); "
            + (f"it had saved, so {mode} again" if saved else f"it never saved, so the next way after {mode}")
            + f" from its last save, {(end - time.time()) / 3600:.2f}h left (restart {restarts})")
        p, i = first_that_trains(i if saved else i + 1, time.time(), " (restart)")
        if p is None:
            rc, why = 1, why or "no way trained after the stop"
            break
    if lab is not None:
        try:
            lab.wait(timeout=600 if rc == 0 else 5)
        except subprocess.TimeoutExpired:
            # never Stockfish alone on a TPU session, so a trainer that stopped early stops the labelling with it

            lab.kill()
            say("labelling stopped with the trainer")
    summ = os.path.join(ck_dir, "xla_summary.json")
    s = json.load(open(summ)) if os.path.exists(summ) else {}
    s.update(run=RUN, machine=MODE, exit=rc, session_s=round(time.time() - T0), watchdog=why, attempts=attempts,
             restarts=restarts, labelled=os.path.exists(os.path.join(WORK, "label", "summary.json")))
    with open(os.path.join(WORK, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=1)
    for f in glob.glob(os.path.join(ck_dir, "*.npy")):
        os.remove(f)
    say(f"done: {json.dumps(s)}")


if __name__ == "__main__":
    main()
