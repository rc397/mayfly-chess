"""MayFly on Kaggle: self-play on a TPU or GPU, Stockfish labelling on a CPU.

Volunteers: Copy & Edit this notebook, pick an accelerator if you have one (Settings > Accelerator > TPU v5e-8),
then Save Version. With a TPU or GPU it plays MayFly against itself; with neither it labels hard positions with
Stockfish at depth 35+. Everything it makes ends up in /kaggle/working.

kaggle_selfplay_ctl.py also pushes this file for our own runs, filling in TIME_BUDGET, ACCEL and RUN below.
"""
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import time

TIME_BUDGET = 8 * 3600 + 15 * 60          # the controller replaces this line with the session's budget
ACCEL = "__ACCEL__"                         # "tpu" or "gpu", filled in by the controller
RUN = "__RUN__"                             # the run's name, filled in by the controller
T0 = time.time()
WORK = "/kaggle/working"
CODE = "/kaggle/tmp/mayfly"

LABEL_LINES = 600                     # more than one session gets through
LABEL_BUDGET = 11 * 3600 + 30 * 60    # Kaggle stops CPU sessions at 12 hours


def say(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode:
        say(f"FAILED ({r.returncode}): {cmd}\n{(r.stdout or '')[-1500:]}\n{(r.stderr or '')[-1500:]}")
        raise SystemExit(1)
    return r


def find_dataset():
    hits = glob.glob("/kaggle/input/**/selfplay_remote.py", recursive=True)
    if not hits:
        say("no selfplay_remote.py under /kaggle/input: the dataset is not attached")
        raise SystemExit(1)
    return os.path.dirname(hits[0])


def build_native():
    """Compile the small C helpers (movegen, walk, pick). mcts falls back to Python if any of them fail."""
    setup = os.path.join(CODE, "setup_linux.py")
    with open(setup, "w") as fh:
        fh.write(
            "import numpy\n"
            "from setuptools import setup, Extension\n"
            "a = ['-O2', '-fno-fast-math', '-ffp-contract=off']\n"
            "setup(name='mayfly_native', ext_modules=[\n"
            "    Extension('mayfly_pick', ['pick.c'], extra_compile_args=a),\n"
            "    Extension('mayfly_movegen', ['movegen.c'], include_dirs=[numpy.get_include()], extra_compile_args=a),\n"
            "    Extension('mayfly_walk', ['walk.c'], include_dirs=[numpy.get_include()], extra_compile_args=a)])\n")
    r = sh(f"cd {CODE} && {sys.executable} setup_linux.py build_ext --inplace", check=False)
    built = glob.glob(os.path.join(CODE, "mayfly_*.so"))
    say(f"compiled helpers: {len(built)} of 3 built" + ("" if len(built) == 3 else f"\n{r.stderr[-1500:]}"))


def need(module, pip_name):
    try:
        __import__(module)
    except ImportError:
        say(f"installing {pip_name}")
        sh(f"{sys.executable} -m pip install -q {pip_name}")


def detect():
    """Work out what this session has when nobody filled in ACCEL (a volunteer's copy)."""
    global ACCEL, RUN
    if RUN == "__" + "RUN__":
        RUN = "volunteer_" + time.strftime("%Y%m%d_%H%M")
    if ACCEL != "__" + "ACCEL__":
        return
    # ask in a separate process: whoever starts JAX on a TPU keeps it until they exit
    r = sh(f"{sys.executable} -c \"import jax; print(jax.devices()[0].platform)\"", check=False)
    if (r.stdout or "").strip().endswith("tpu"):
        ACCEL = "tpu"
        return
    r = sh(f"{sys.executable} -c \"import torch; print(torch.cuda.is_available())\"", check=False)
    if (r.stdout or "").strip().endswith("True"):
        ACCEL = "gpu"
        return
    ACCEL = "label"
    say("no TPU or GPU here, so this session labels positions with Stockfish instead. "
        "For self-play, pick Settings > Accelerator > TPU v5e-8 (or a GPU) and run it again.")


def label():
    """Label a random chunk of the dataset's hard positions with Stockfish (kaggle_kernel.py does the work)."""
    import random
    import socket
    src = find_dataset()
    pool = [p for p in glob.glob(os.path.join(src, "label_pool.txt*")) if os.path.isfile(p)]
    if not pool:
        say("no label_pool.txt in the dataset")
        raise SystemExit(1)
    raw = open(pool[0], "rb").read()
    lines = [l for l in (gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw).decode("utf-8").splitlines()
             if l.count("|") >= 3]
    rng = random.Random(f"{time.time_ns()}/{os.getpid()}/{socket.gethostname()}")
    at = rng.randrange(max(1, len(lines) - LABEL_LINES))
    chunk = lines[at:at + LABEL_LINES]
    # saved so the results can be matched back to their positions
    with open(os.path.join(WORK, "chunk.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(chunk) + "\n")

    code = open(os.path.join(src, "kaggle_kernel.py"), encoding="utf-8").read()
    fills = [('r"""__LINES__"""', 'r"""' + "\n".join(chunk) + '"""'),
             ("TIME_BUDGET = 8 * 3600 + 30 * 60", f"TIME_BUDGET = {LABEL_BUDGET}")]
    # use the dataset's Stockfish so sessions without internet work too. The archive is stored as .bin
    # because Kaggle unpacks uploaded .gz files; an unpacked folder works as well
    sf = sorted(glob.glob(os.path.join(src, "stockfish*.bin")) + glob.glob(os.path.join(src, "stockfish*.tar*")))
    sf_dir = [p for p in glob.glob(os.path.join(src, "**", "stockfish"), recursive=True) if os.path.isdir(p)]
    if sf:
        fills.append(('urllib.request.urlretrieve(URL, "/tmp/sf.tgz")', f'__import__("shutil").copy({sf[0]!r}, "/tmp/sf.tgz")'))
    elif sf_dir:
        fills.append(('    urllib.request.urlretrieve(URL, "/tmp/sf.tgz")\n    with tarfile.open("/tmp/sf.tgz") as tf:\n'
                      '        tf.extractall("/tmp/sf")\n',
                      f'    __import__("shutil").copytree({sf_dir[0]!r}, "/tmp/sf/stockfish", dirs_exist_ok=True)\n'))
    for a, b in fills:
        if code.count(a) != 1:
            say(f"kaggle_kernel.py has changed ({a[:40]!r} not found once), can't label")
            raise SystemExit(1)
        code = code.replace(a, b)
    os.makedirs(CODE, exist_ok=True)
    run = os.path.join(CODE, "label_run.py")
    with open(run, "w", encoding="utf-8") as fh:
        fh.write(code)

    say(f"labelling {len(chunk)} of {len(lines):,} positions (from line {at:,}) for up to "
        f"{LABEL_BUDGET / 3600:.1f}h, Stockfish {'from the dataset' if sf or sf_dir else 'downloaded'}")
    r = subprocess.run([sys.executable, run], cwd=WORK)
    try:
        summ = json.load(open(os.path.join(WORK, "summary.json")))
    except (OSError, ValueError):
        summ = {}
    summ.update(kind="label", run=RUN, chunk_at=at, chunk_lines=len(chunk), exit=r.returncode)
    with open(os.path.join(WORK, "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    say(f"labelling done (exit {r.returncode}): {summ.get('labelled', summ.get('done', '?'))} positions")


def pack(src_glob, dest, keep):
    """Concatenate the workers' files into one gzip, keeping lines that look complete."""
    n = 0
    with gzip.open(dest, "wt", encoding="utf-8") as out:
        for f in sorted(glob.glob(src_glob)):
            for line in open(f, encoding="utf-8", errors="replace"):
                if keep(line):
                    out.write(line)
                    n += 1
    return n


def main():
    detect()
    if ACCEL == "label":
        return label()

    src = find_dataset()
    shutil.rmtree(CODE, ignore_errors=True)
    shutil.copytree(src, CODE, ignore=shutil.ignore_patterns("*.npz", "corpus_starts.txt"))
    sys.path.insert(0, CODE)
    os.chdir(CODE)
    need("chess", "chess")
    need("scipy", "scipy")
    cores = os.cpu_count() or 4

    spv = os.path.join(src, "sp_version.json")
    ver = json.load(open(spv)) if os.path.exists(spv) else {"version": "mayfly_v5.pt", "npz": "mayfly_v5.npz"}
    say(f"playing {ver['version']}")
    env = {"SP_NPZ": os.path.join(src, ver["npz"]), "VERSION": ver["version"], "VERSION_DIR": CODE,
           "SP_STARTS": os.path.join(src, "corpus_starts.txt"), "SP_OUT": os.path.join(WORK, "sp_out"),
           "VISITS": "1600", "SP_REPORT": "600", "SP_SEED": str(int(time.time())),
           # a quarter of moves get the full search and become training targets, the rest are quick
           "FULL_FRAC": "0.25", "FAST_VISITS": "200"}

    if ACCEL == "tpu":
        need("jax", "'jax[tpu]'")
        r = sh(f"{sys.executable} -c \"import jax; d = jax.devices(); print(d[0].platform, len(d), d[0].device_kind)\"",
               check=False)
        got = (r.stdout or "").strip().splitlines()[-1:] or [""]
        if not got[0].startswith("tpu"):
            # Kaggle stops TPU sessions that don't use the TPU, so don't fall back to the CPU
            say(f"no TPU here ({got[0]!r}; {(r.stderr or '')[-500:]}), stopping")
            raise SystemExit(1)
        say(f"TPU: {got[0]}, {cores} cores")
        env.update(SP_BACKEND="jax", SP_WORKERS=str(min(240, 2 * cores)))
    elif ACCEL == "cpu":
        # dry run of the whole pipeline on a free CPU session, at 32 visits so games finish
        say(f"dry run on the CPU, {cores} cores")
        env.update(SP_BACKEND="cpu", SP_WORKERS="3", VISITS="32", SP_BUCKETS="16,32,64,128", SP_REPORT="120")
    else:
        import torch
        if not torch.cuda.is_available():
            say("no CUDA card here, stopping")
            raise SystemExit(1)
        say(f"GPU: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}, {cores} cores")
        env.update(SP_BACKEND="torch", SP_WORKERS=str(max(8, 4 * cores)))
    need("torch", "torch --index-url https://download.pytorch.org/whl/cpu")   # every worker imports mcts, which needs torch

    if ACCEL == "tpu" and not RUN.startswith("volunteer_"):
        # our own runs benchmark the TPU first; volunteers skip straight to playing
        for mode in ("jax", "torch"):
            out = os.path.join(WORK, f"tpu_bench_{mode}.json")
            r = sh(f"timeout 1200 {sys.executable} {os.path.join(CODE, 'tpu_bench.py')} {mode} {env['SP_NPZ']} {out}",
                   check=False)
            got = [l for l in (r.stdout or "").splitlines() if l.startswith("BENCH ")]
            say(f"tpu bench {mode}: " + (got[-1][6:][:1500] if got else
                                         f"no result (exit {r.returncode}): {(r.stderr or '')[-800:]}"))

    build_native()
    os.environ.update(env)
    import selfplay_remote
    left = TIME_BUDGET - (time.time() - T0) - min(15 * 60, 0.1 * TIME_BUDGET)   # leave time to pack and save
    say(f"playing for {left / 3600:.2f}h with {env['SP_WORKERS']} workers")
    res = selfplay_remote.run(seconds=left, out=env["SP_OUT"])

    out = env["SP_OUT"]
    is_row = lambda line: line.startswith("{") and line.endswith("}\n")   # noqa: E731
    n_pos = pack(os.path.join(out, "positions_w*.txt"), os.path.join(WORK, "positions.txt.gz"),
                 lambda line: line.count("/") == 7 and line.endswith("\n"))
    n_tgt = pack(os.path.join(out, "targets_w*.jsonl"), os.path.join(WORK, "targets.jsonl.gz"), is_row)
    n_fast = pack(os.path.join(out, "fast_w*.jsonl"), os.path.join(WORK, "fast.jsonl.gz"), is_row)
    shutil.rmtree(out, ignore_errors=True)
    res.update(run=RUN, accel=ACCEL, version=ver["version"], positions_written=n_pos, targets_written=n_tgt,
               fast_written=n_fast, session_s=round(time.time() - T0))
    with open(os.path.join(WORK, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    say(f"saved {n_pos:,} positions and {n_tgt:,} targets: {json.dumps(res)}")


# selfplay_remote uses multiprocessing spawn, which re-runs this file in each child
if __name__ == "__main__":
    main()
