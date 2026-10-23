"""Helpers for the cuBLASLt autotuner (MAYFLY_CUDANET_TUNE, gemm_tune.cpp).

  python tune_cudanet.py fill --cnw X.cnw [--sizes 1-48] [--level 1]
      runs every batch size once with tuning on, so the cache file is complete before a search needs it
  python tune_cudanet.py ab --cnw X.cnw [--sizes 8,16,32,48] [--reps 20] [--rounds 5] [--level 1]
      the whole pass as a CUDA graph and the products (eager), heuristic against tuned, as two nets in one process
      timed in alternating rounds, medians printed

MAYFLY_CUDANET_ACC16 and MAYFLY_CUDANET_TUNE_FILE are read as usual. Tuning times the card, so run both with nothing
else on it (with another process on the card the tuner keeps its choices out of the file).
"""
import argparse
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cudanet_py import CudaNet  # noqa: E402

STEPS = ("qkv", "out", "ff1", "ff2", "heads")


def sizes_arg(s):
    """'1-48' or '8,16,32,48'."""
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def dummy(B):
    """An empty board in every row (ids, hm, ep and cast zero), one legal move."""
    return (np.zeros((B, 64), np.int64), np.zeros((B, 4), np.float32), np.zeros(B, np.int64),
            np.zeros(B, np.int64), np.zeros((B, 1), np.int64), np.ones((B, 1), np.uint8))


def make_net(args, level):
    if level:
        os.environ["MAYFLY_CUDANET_TUNE"] = str(level)
    else:
        os.environ.pop("MAYFLY_CUDANET_TUNE", None)
    return CudaNet(args.cnw, args.device, args.dll)   # the switch is read when the net is built


def fill(args):
    net = make_net(args, args.level)
    for B in sizes_arg(args.sizes):
        net.evaluate(*dummy(B))
        print(f"  B={B} done", flush=True)
    net.close()


def ab(args):
    sizes = sizes_arg(args.sizes)
    base = make_net(args, 0)
    tuned = make_net(args, args.level)
    nets = {"heuristic": base, "tuned": tuned}
    res = {name: {B: {k: [] for k in STEPS + ("graph_pass",)} for B in sizes} for name in nets}
    for B in sizes:
        tuned.profile(B, 1)          # tunes this size first, outside the timed rounds
        for r in range(args.rounds):
            for name in (("heuristic", "tuned") if r % 2 == 0 else ("tuned", "heuristic")):
                ms = nets[name].profile(B, args.reps)
                for k in res[name][B]:
                    res[name][B][k].append(ms[k])
    base.close()
    tuned.close()
    acc = os.environ.get("MAYFLY_CUDANET_ACC16", "0")
    print(f"ab: ACC16={acc}, TUNE={args.level}, medians of {args.rounds} rounds of {args.reps} passes; products "
          f"eager (an event after each step), graph_pass the whole pass as a CUDA graph; ms")
    print(f"  {'B':>3s} {'step':10s} {'heuristic':>10s} {'tuned':>10s} {'speed-up':>9s}")
    for B in sizes:
        for k in STEPS + ("graph_pass",):
            h = statistics.median(res["heuristic"][B][k])
            t = statistics.median(res["tuned"][B][k])
            print(f"  {B:>3d} {k:10s} {h:>10.3f} {t:>10.3f} {h / t:>8.3f}x")


def main(argv=None):
    ap = argparse.ArgumentParser(description="cudanet cuBLASLt autotuning: fill the cache, or A/B it")
    ap.add_argument("mode", choices=("fill", "ab"))
    ap.add_argument("--cnw", required=True)
    ap.add_argument("--dll")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--sizes", default=None)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args(argv)
    if args.sizes is None:
        args.sizes = "1-48" if args.mode == "fill" else "8,16,32,48"
    fill(args) if args.mode == "fill" else ab(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
