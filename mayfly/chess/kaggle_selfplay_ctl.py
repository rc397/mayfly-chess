"""Keep Kaggle TPU and GPU sessions busy with MayFly self-play and download the games, one pass per run.

Each slot named in selfplay_slots.txt ("tpu", "gpu", or "cpu" for a free dry run) keeps one notebook running
kaggle_selfplay.py (<KAGGLE_USER>/mayfly-selfplay-<slot>) with the private self-play dataset attached. A free slot
gets a new push while its weekly quota lasts, with its budget cut to the quota left, and a finished notebook's
output (positions.txt.gz, targets.jsonl.gz, summary.json) is downloaded into SELFPLAY_OUT/<run>. The GPU slot waits
while kaggle_label.py's own GPU batch is out, since the weekly GPU hours are shared. The CPU dry run borrows one of
the labeller's CPU sessions, and with "restore_slots" in the state the labeller's slots.txt gets that number back
when it ends. The games are left in SELFPLAY_OUT.

    python chess/kaggle_selfplay_ctl.py          one pass, for example every ten minutes from a scheduler
    env KAGGLE_DIR, SELFPLAY_OUT, SP_SESSION_H=8.25, SP_CPU_TEST_H=0.4, KAGGLE_USER
"""
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kaggle_label as kl                     # its kaggle() wrapper (the token) and quota()
import paths

E = os.environ.get
BASE = E("KAGGLE_DIR", paths.under("kaggle"))
SLOTS_FILE = os.path.join(BASE, "selfplay_slots.txt")
HOURS_FILE = os.path.join(BASE, "selfplay_hours.json")      # optional per-slot session cap in hours
STATE = os.path.join(BASE, "selfplay_state.json")
LOG = os.path.join(BASE, "kaggle_selfplay.log")
PUSH = os.path.join(BASE, "sp_push")
OUT = E("SELFPLAY_OUT", os.path.join(paths.SELFPLAY, "kaggle"))
DATASET = f"{kl.USER}/mayfly-selfplay"
SESSION_H = float(E("SP_SESSION_H", 8.25))
CPU_TEST_H = float(E("SP_CPU_TEST_H", 0.4))
MIN_QUOTA_H = 1.0
# accelerator id and quota name for each slot, "cpu" is the dry run, which uses no quota
SHAPES = {"tpu": ("TpuV5E8", "TPU"), "gpu": ("NvidiaTeslaT4", "GPU"), "cpu": (None, None)}
KERNEL_BUDGET = "TIME_BUDGET = 8 * 3600 + 15 * 60"


def say(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def slug(slot):
    return f"{kl.USER}/mayfly-selfplay-{slot}"


def load_state():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except (OSError, ValueError):
        return {"slots": {}, "runs": 0, "wait": {}}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)
    os.replace(tmp, STATE)


def status(slot):
    rc, out = kl.kaggle("kernels", "status", slug(slot), timeout=120)
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?([A-Za-z_]+)', out)
    if m:
        s = m.group(1).lower()
        return {"cancel_requested": "running", "cancel_acknowledged": "cancelled"}.get(s, s)
    return "missing" if ("404" in out or "not found" in out.lower()) else "unknown"


def fetch(slot, job):
    """Download a finished notebook's output into OUT/<run>. True when summary.json arrived."""
    d = os.path.join(OUT, job["run"])
    os.makedirs(d, exist_ok=True)
    rc, out = kl.kaggle("kernels", "output", slug(slot), "-p", d, timeout=3600)
    summ = os.path.join(d, "summary.json")
    if not os.path.exists(summ):
        say(f"slot {slot}: run {job['run']} output has no summary.json yet ({out.strip()[-200:]})")
        return False
    s = json.load(open(summ, encoding="utf-8"))
    job["summary"] = s
    say(f"slot {slot}: run {job['run']} home: {s.get('positions_written', 0):,} positions, "
        f"{s.get('games', 0):,} games, {s.get('evals_per_s', 0):,} evals a second, "
        f"{s.get('positions_per_hour', 0):,} positions an hour over {s.get('seconds', 0) / 3600:.2f}h")
    return True


def tend(slot, st):
    """True while the slot's notebook is busy. A finished one is fetched, a failed one logged, and the slot freed."""
    job = st["slots"].get(slot)
    if not job or job.get("done"):
        return False
    s = status(slot)
    if s in ("running", "queued"):
        if s != job.get("seen"):
            job["seen"] = s
            say(f"slot {slot}: run {job['run']} {s}")
            save_state(st)
        return True
    if s == "complete":
        if not fetch(slot, job):
            job["tries"] = job.get("tries", 0) + 1
            if job["tries"] < 6:
                save_state(st)
                return True
    elif s in ("error", "cancelled"):
        d = os.path.join(OUT, job["run"])
        os.makedirs(d, exist_ok=True)
        kl.kaggle("kernels", "output", slug(slot), "-p", d, timeout=600)   # the log, when there is one
        say(f"slot {slot}: run {job['run']} ended {s}; its log (if any) is in {d}")
    else:
        say(f"slot {slot}: status {s}, leaving it")
        return True
    job["done"] = True
    job["ended"] = time.strftime("%Y-%m-%d %H:%M")
    job["status"] = s
    if slot == "cpu" and st.get("restore_slots"):
        # the dry run borrowed one of the labeller's CPU sessions (its slots.txt lowered by one), so give it back
        with open(os.path.join(BASE, "slots.txt"), "w", encoding="utf-8") as fh:
            fh.write(str(st.pop("restore_slots")))
        say("slot cpu: the labeller's CPU session given back (slots.txt restored)")
    save_state(st)
    return False


def labeller_on(slot):
    """True while kaggle_label.py's own slot of this name (its GPU or TPU labelling) has a batch out, so only one user
    of an accelerator's quota runs at a time."""
    try:
        job = json.load(open(os.path.join(BASE, "state.json"), encoding="utf-8"))["slots"].get(slot)
    except (OSError, ValueError, KeyError):
        return False
    return bool(job) and not job.get("done")


def push(slot, st, hours):
    acc, _ = SHAPES[slot]
    st["runs"] = st.get("runs", 0) + 1
    run = time.strftime("%Y%m%d_%H%M") + f"_{slot}_{st['runs']:04d}"
    d = os.path.join(PUSH, slot)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    src = open(os.path.join(HERE, "kaggle_selfplay.py"), encoding="utf-8").read()
    if KERNEL_BUDGET not in src:
        say(f"slot {slot}: kaggle_selfplay.py has no '{KERNEL_BUDGET}'; not pushed")
        return False
    src = (src.replace(KERNEL_BUDGET, f"TIME_BUDGET = {int(hours * 3600)}")
           .replace('"__ACCEL__"', f'"{slot}"').replace('"__RUN__"', f'"{run}"'))
    # not "selfplay.py", since the workers import MayFly's selfplay module and multiprocessing's spawn runs this file
    # in every child as well
    with open(os.path.join(d, "mayfly_sp_kernel.py"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    meta = {"id": slug(slot), "title": f"mayfly selfplay {slot}", "code_file": "mayfly_sp_kernel.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": slot == "gpu", "enable_tpu": slot == "tpu",
            "enable_internet": True, "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": []}
    if acc:
        meta["machine_shape"] = acc
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    rc, out = kl.kaggle("kernels", "push", "-p", d, *(("--accelerator", acc) if acc else ()))
    ok = rc == 0 and "successfully pushed" in out
    say(f"slot {slot}: push of run {run} ({hours:.2f}h" + (f" on {acc}" if acc else ", CPU dry run") + ")"
        + (" ok" if ok else f" FAILED: {out.strip()[-300:]}"))
    if ok:
        st["slots"][slot] = {"run": run, "pushed": time.strftime("%Y-%m-%d %H:%M"), "hours": round(hours, 2)}
    save_state(st)
    return ok


LOCK = os.path.join(BASE, ".selfplay_ctl.lock")


def main():
    """One pass, never two at once, or two passes could each push a session. A lock older than two hours was left
    by a pass that died."""
    os.makedirs(BASE, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        if time.time() - os.path.getmtime(LOCK) < 7200:
            print("another pass is running", flush=True)
            return
        os.utime(LOCK, None)
    try:
        one_pass()
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


def one_pass():
    os.makedirs(BASE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    try:
        names = open(SLOTS_FILE, encoding="utf-8").read().replace(",", " ").split()
    except OSError:
        names = []
    st = load_state()
    waits = st.setdefault("wait", {})
    # a slot taken out of selfplay_slots.txt is still tended until its run is home
    for slot in [s for s, j in st["slots"].items() if s not in names and not j.get("done")]:
        tend(slot, st)
    for slot in names:
        if slot not in SHAPES:
            continue
        if tend(slot, st) or time.time() < waits.get(slot, 0):
            continue
        acc, q = SHAPES[slot]
        if slot == "gpu" and labeller_on(slot):
            continue                          # the labeller's GPU batch finishes first, the weekly GPU hours are shared
        if q is None:
            if st["slots"].get(slot, {}).get("done"):
                continue                      # the dry run runs once, delete its entry in the state to run it again
            hours = CPU_TEST_H
        else:
            left = kl.quota(q)
            if left is None or left < MIN_QUOTA_H:
                waits[slot] = time.time() + 3600   # look again in an hour, the quota resets weekly

                save_state(st)
                continue
            hours = min(SESSION_H, left - 0.25)
        try:                                  # a shorter session for one slot, e.g. {"tpu": 1.25} for a test
            cap = json.load(open(HOURS_FILE, encoding="utf-8")).get(slot)
            if cap:
                hours = min(hours, float(cap))
        except (OSError, ValueError, AttributeError):
            pass
        if not push(slot, st, hours):
            waits[slot] = time.time() + 1800
            save_state(st)


if __name__ == "__main__":
    main()
