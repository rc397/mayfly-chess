"""Benchmark FAST_EVAL against autocast and check it does not change the move.

Two searches on the same weights, position and visit count, one with FAST_EVAL off (autocast, ragged
batches) and one with it on (the trunk cast to bfloat16 once, batches padded to a fixed shape). Prints
visits per second for each and whether the move, root visit counts and evaluation match, since a
frozen version must stay the same opponent.

    python chess/fast_eval_bench.py [VISITS]     env VERSION=mayfly_v4.pt, FEN
"""
import os, sys, time, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

E = os.environ.get
VISITS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
VERSION = E("VERSION", "mayfly_v4.pt")
FEN = E("FEN", "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N1P/PP1P1PP1/RNBQR1K1 b - - 0 9")

with contextlib.redirect_stdout(sys.stderr):
    import torch, chess
    from play_web import load_version


def run(fast):
    os.environ["FAST_EVAL"] = "1" if fast else "0"
    import importlib
    import mcts
    importlib.reload(mcts)                      # FAST_EVAL is read when the tree is built
    with contextlib.redirect_stdout(sys.stderr):
        model, label, _ = load_version(VERSION)
    dev = str(next(model.parameters()).device)
    t = mcts.MCTS(model, dev, batch=48, cache_size=0).apply(mcts.pinned_search(VERSION))
    b = chess.Board(FEN)
    t.search(b.copy(), visits=200, reuse=False)  # warm the kernels
    t.root = None
    torch.cuda.synchronize() if dev != "cpu" else None
    t0 = time.time()
    mv, info = t.search(b.copy(), visits=VISITS, reuse=False)
    torch.cuda.synchronize() if dev != "cpu" else None
    el = time.time() - t0
    del t, model
    if dev != "cpu":
        torch.cuda.empty_cache()
    return {"move": mv.uci(), "visits": int(info["visits"]), "secs": el, "vps": info["visits"] / el,
            "evals": info["evals"], "batches": info["batches"], "top": [(u, n) for u, n, _ in info["top"]],
            "q": round(float(info["q"]), 6), "label": label}


slow = run(False)
fast = run(True)
print(f"{slow['label']}, {VISITS} visits, batch 48, {FEN}")
for name, r in (("autocast", slow), ("FAST_EVAL", fast)):
    print(f"  {name:>9}: {r['vps']:>7,.0f} visits/s  ({r['secs']:.2f}s, {r['evals']:,} evals in "
          f"{r['batches']:,} passes)")
print(f"  speed-up: {fast['vps'] / slow['vps']:.2f}x")
same_move = slow["move"] == fast["move"]
same_tree = slow["top"] == fast["top"]
print(f"  same move: {same_move} ({slow['move']} vs {fast['move']}), same root counts: {same_tree}, "
      f"q {slow['q']} vs {fast['q']}")
if not (same_move and same_tree):
    print("  NOT IDENTICAL: bfloat16 weights change the numbers, so this cannot go into a frozen "
          "version without re-sealing it.")
