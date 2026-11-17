r"""The same C++ search (mayfly_cppsearch, v5's sealed settings) on the LibTorch trace and on the hand-tuned CUDA net,
one network per process. A .cnw path in the trace's place makes the tree on the CUDA net (cpp/pyext.cpp, a build with
MAYFLY_CUDANET). Prints evaluations, visits and time a search and the chosen moves, and appends a JSON line to --out.

  python bench_search.py NET(.cuda.ts|.cnw) --ckpt CKPT [--visits 1600] [--positions 24] [--batch 48] [--out X.jsonl]
      [--warm 1] [--prof] [--pipeline 0|1|2] [--dump X.json]
  env CPPSEARCH_DIR = the folder of the mayfly_cppsearch build to use

  --warm N      warm-up searches before the timed ones (the first N positions). --warm 0 with --positions P warms on
                all P, so every batch size's CUDA graph is captured before the clock starts
  --prof        the search's own timers (settings "_prof") and the CUDA net's, printed as milliseconds a batch
  --pipeline N  MCTS's pipeline key in the C++ tree, 0 is the v1 to v6 search
  --dump X      every search's root statistics, chosen move and principal variation, for a bit-identity check

A C++ module only loads beside the torch it was linked against, so build against the torch you run.
  venv     python -m venv, then pip install torch==2.10.0+cu128 (download.pytorch.org/whl/cu128), numpy and chess
  kit      a folder whose include and cuda\include point at the venv's torch\include and the CUDA headers, and whose
           lib holds the venv's torch, torch_cpu, torch_cuda, c10 and c10_cuda .lib files and cudart.lib
  build    set TORCH_KIT=<kit> and OUT=<a folder outside the repo>, then cmd /c chess\cpp\build_rig.bat
  run      CPPSEARCH_DIR=<OUT> <venv>\Scripts\python.exe bench_search.py ...
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cpp"))
import numpy as np  # noqa: E402
import torch  # noqa: E402,F401  (first, because the LibTorch build links this torch)

import export_ts  # noqa: E402
import mcts  # noqa: E402
import mcts_cpp  # noqa: E402
import test_cudanet  # noqa: E402

# the tree's own timers (seconds) and the CUDA net's (milliseconds, counts), as t.prof() names them
TREE_PARTS = ("descend", "leaf", "encode", "net", "submit", "collect", "expand")
NET_PARTS = ("net_h_stage", "net_h_queue_upload", "net_h_queue_pass", "net_h_queue_tail", "net_h_wait", "net_h_out",
             "net_g_upload", "net_g_pass", "net_g_post", "net_g_download")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--visits", type=int, default=1600)
    ap.add_argument("--positions", type=int, default=24)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--prof", action="store_true")
    ap.add_argument("--pipeline", type=int, default=0)
    ap.add_argument("--dump")
    a = ap.parse_args()
    if mcts_cpp.mayfly_cppsearch is None:
        raise SystemExit(f"no mayfly_cppsearch: {mcts_cpp.IMPORT_ERROR}")
    fens = test_cudanet.positions(max(a.positions, 48), a.seed)[:a.positions]
    # v5's settings for the C++ tree's cfg, read off an MCTS on the CPU (the card's memory is for the net)
    tree = mcts.MCTS(export_ts.load_v5(a.ckpt, "cpu"), device="cpu")
    tree.batch = a.batch
    cfg = mcts_cpp._settings(tree)
    cfg["use_tb"] = 0.0
    cfg["_ts"] = a.net
    if a.prof:
        cfg["_prof"] = 1.0
    if a.pipeline:
        cfg["pipeline"] = float(a.pipeline)
    t = mcts_cpp.mayfly_cppsearch.Tree(cfg, None, lambda fen: None, a.net, "cuda", True)
    piped = hasattr(t, "finish")

    def search(fen):
        t.reset()
        t.begin(fen, [], False)
        if t.needs_eval():
            t.expand_root()
        while t.step(float(a.visits), -1.0) == 0:
            pass
        if piped:
            t.finish()
        return t.stats()

    for fen in (fens[:a.warm] if a.warm > 0 else fens):    # warm-up for kernels and graphs
        search(fen)
    rows, dump, t0 = [], [], time.perf_counter()
    tot = {k: 0.0 for k in TREE_PARTS + ("step", "steps")}
    net0 = t.prof() if a.prof else {}
    for fen in fens:
        ts = time.perf_counter()
        s = search(fen)
        sec = time.perf_counter() - ts
        r = t.root()
        best = t.choose()
        move = r["moves"][best] if best >= 0 else ""
        rows.append({"fen": fen, "seconds": sec, "evals": s.get("evals"), "batches": s.get("batches"),
                     "visits": r["sum_n"], "best": move})
        if a.prof:
            p = t.prof()
            for k in tot:
                tot[k] += p[k]
        if a.dump:
            arr = {k: np.frombuffer(r[k], np.float32).tolist() for k in ("N", "W", "V2", "F", "X", "Q0", "P")}
            dump.append({"fen": fen, "best": move, "pv": t.pv(best, 12), "sum_n": r["sum_n"], "value": r["value"],
                         "kids_n": r["kids_n"], "stats": s, "nodes": t.tree_nodes(), **arr})
    el = time.perf_counter() - t0
    evals = sum(r["evals"] or 0 for r in rows)
    batches = sum(r["batches"] or 0 for r in rows)
    res = {"net": os.path.basename(a.net), "visits": a.visits, "positions": len(fens), "batch": a.batch,
           "pipeline": a.pipeline, "warm": a.warm, "seconds": el, "s_per_search": el / len(fens),
           "evals_s": evals / el, "mean_batch": evals / max(batches, 1), "batches": batches,
           "stats_keys": sorted(s.keys()), "rows": rows, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "gpu": torch.cuda.get_device_name(0)}
    print(f"{res['net']}: {len(fens)} searches of {a.visits} visits at batch {a.batch} (pipeline {a.pipeline}): "
          f"{res['s_per_search'] * 1000:.1f} ms a search, {res['evals_s']:,.0f} evaluations a second, mean batch "
          f"{res['mean_batch']:.1f}, {batches / len(fens):.1f} batches a search")
    print("moves: " + " ".join(r["best"] for r in rows))
    if a.prof:
        net1 = t.prof()
        nb = max(batches, 1)
        per = {k: 1000.0 * tot[k] / nb for k in TREE_PARTS + ("step",)}
        per["outside_steps"] = 1000.0 * (el - tot["step"]) / nb      # Python, begin, root pass, finish, stats
        for k in NET_PARTS:
            if k in net1:
                per[k] = (net1[k] - net0.get(k, 0.0)) / nb
        for k in ("net_batches", "net_eager"):
            if k in net1:
                res[k] = net1[k] - net0.get(k, 0.0)
        res["ms_per_batch"] = per
        print(f"ms a batch over {batches} batches (wall {1000.0 * el / nb:.3f}): " +
              ", ".join(f"{k} {v:.3f}" for k, v in per.items()))
        if "net_eager" in res:
            print(f"net batches {res['net_batches']:.0f}, eager (first use of a batch size) {res['net_eager']:.0f}")
    if a.dump:
        with open(a.dump, "w", encoding="utf-8") as fh:
            json.dump(dump, fh)
        h = hashlib.sha256(json.dumps(dump, sort_keys=True).encode()).hexdigest()
        print(f"dump {a.dump}: sha256 {h}")
    if a.out:
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
