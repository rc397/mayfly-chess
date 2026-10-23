"""The Kaggle notebook that runs MayFly self-play on a TPU or GPU session.

kaggle_selfplay_ctl.py pushes it with its private self-play dataset attached, holding the self-play code, v5's
weights as chessnet_jax's npz, versions.json (v5's sealed search) and a file of corpus positions for mixed starts.
It builds the compiled tree helpers for Linux, checks the accelerator is really there, runs selfplay_remote.run for
TIME_BUDGET, and leaves positions.txt.gz, targets.jsonl.gz and summary.json in /kaggle/working for the controller.

Kaggle stops a TPU session that does not use the TPU, so with no TPU (or no CUDA device on a GPU session) this
stops with an error instead of playing on the CPU.
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
    """Build mayfly_movegen, mayfly_walk and mayfly_pick for this Linux, beside the code.

    Each checks itself against numpy and python-chess when mcts loads it, and is left unused wherever it differs."""
    nat = CODE                                 # the dataset keeps movegen.c, walk.c and pick.c beside the code
    setup = os.path.join(nat, "setup_linux.py")
    with open(setup, "w") as fh:
        fh.write(
            "import numpy\n"
            "from setuptools import setup, Extension\n"
            "a = ['-O2', '-fno-fast-math', '-ffp-contract=off']\n"
            "setup(name='mayfly_native', ext_modules=[\n"
            "    Extension('mayfly_pick', ['pick.c'], extra_compile_args=a),\n"
            "    Extension('mayfly_movegen', ['movegen.c'], include_dirs=[numpy.get_include()], extra_compile_args=a),\n"
            "    Extension('mayfly_walk', ['walk.c'], include_dirs=[numpy.get_include()], extra_compile_args=a)])\n")
    r = sh(f"cd {nat} && {sys.executable} setup_linux.py build_ext --inplace", check=False)
    built = glob.glob(os.path.join(nat, "mayfly_*.so"))
    say(f"compiled helpers: {len(built)} of 3 built" + ("" if len(built) == 3 else f"\n{r.stderr[-1500:]}"))


def need(module, pip_name):
    try:
        __import__(module)
    except ImportError:
        say(f"installing {pip_name}")
        sh(f"{sys.executable} -m pip install -q {pip_name}")


def main():
    src = find_dataset()
    shutil.rmtree(CODE, ignore_errors=True)
    shutil.copytree(src, CODE, ignore=shutil.ignore_patterns("*.npz", "corpus_starts.txt"))
    sys.path.insert(0, CODE)
    os.chdir(CODE)
    need("chess", "chess")
    need("scipy", "scipy")
    cores = os.cpu_count() or 4
    env = {"SP_NPZ": os.path.join(src, "mayfly_v5.npz"), "VERSION": "mayfly_v5.pt", "VERSION_DIR": CODE,
           "SP_STARTS": os.path.join(src, "corpus_starts.txt"), "SP_OUT": os.path.join(WORK, "sp_out"),
           "VISITS": "1600", "SP_REPORT": "600", "SP_SEED": str(int(time.time()))}
    if ACCEL == "tpu":
        need("jax", "'jax[tpu]'")
        # ask in a throwaway process, since a process that starts JAX on the TPU holds the chips until it exits and
        # the server process must be the one that has them
        r = sh(f"{sys.executable} -c \"import jax; d = jax.devices(); print(d[0].platform, len(d), d[0].device_kind)\"",
               check=False)
        got = (r.stdout or "").strip().splitlines()[-1:] or [""]
        if not got[0].startswith("tpu"):
            say(f"no TPU here ({got[0]!r}; {(r.stderr or '')[-500:]}): stopping rather than playing on the CPU")
            raise SystemExit(1)
        say(f"TPU: {got[0]}, {cores} cores")
        env.update(SP_BACKEND="jax", SP_WORKERS=str(min(240, 2 * cores)))
    elif ACCEL == "cpu":
        # A dry run of everything but the accelerator on a free CPU session, at 32 visits so a game ends inside it.
        # Never pushed with an accelerator.
        say(f"dry run on the CPU, {cores} cores")
        env.update(SP_BACKEND="cpu", SP_WORKERS="3", VISITS="32", SP_BUCKETS="16,32,64,128", SP_REPORT="120")
    else:
        import torch
        if not torch.cuda.is_available():
            say("no CUDA card here: stopping rather than playing on the CPU")
            raise SystemExit(1)
        say(f"GPU: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}, {cores} cores")
        env.update(SP_BACKEND="torch", SP_WORKERS=str(max(8, 4 * cores)))
    need("torch", "torch --index-url https://download.pytorch.org/whl/cpu")   # mcts.py imports it in every worker
    if ACCEL == "tpu":
        # Measure the TPU first with tpu_bench.py (training across every chip in JAX, the forward pass at self-play's
        # bucket, and whether PyTorch/XLA is there). Each runs in its own process, which hands the TPU back on exit.
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
    left = TIME_BUDGET - (time.time() - T0) - min(15 * 60, 0.1 * TIME_BUDGET)   # time to pack and save
    say(f"playing for {left / 3600:.2f}h with {env['SP_WORKERS']} workers")
    res = selfplay_remote.run(seconds=left, out=env["SP_OUT"])
    n_pos = n_tgt = 0
    with gzip.open(os.path.join(WORK, "positions.txt.gz"), "wt", encoding="utf-8") as out:
        for f in sorted(glob.glob(os.path.join(env["SP_OUT"], "positions_w*.txt"))):
            for line in open(f, encoding="utf-8", errors="replace"):
                if line.count("/") == 7 and line.endswith("\n"):
                    out.write(line)
                    n_pos += 1
    with gzip.open(os.path.join(WORK, "targets.jsonl.gz"), "wt", encoding="utf-8") as out:
        for f in sorted(glob.glob(os.path.join(env["SP_OUT"], "targets_w*.jsonl"))):
            for line in open(f, encoding="utf-8", errors="replace"):
                if line.startswith("{") and line.endswith("}\n"):
                    out.write(line)
                    n_tgt += 1
    shutil.rmtree(env["SP_OUT"], ignore_errors=True)
    res.update(run=RUN, accel=ACCEL, positions_written=n_pos, targets_written=n_tgt,
               session_s=round(time.time() - T0))
    with open(os.path.join(WORK, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    say(f"saved {n_pos:,} positions and {n_tgt:,} targets: {json.dumps(res)}")


# guarded, since selfplay_remote starts its servers and workers with multiprocessing's spawn, which runs this file
# again in every child process

if __name__ == "__main__":
    main()
