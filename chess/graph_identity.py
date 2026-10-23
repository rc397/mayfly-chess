"""Check that a CUDA graph of the engine's autocast forward pass matches the eager pass bit for bit.

The engine runs fp32 weights under bfloat16 autocast with no graph, and much of a search is Python
launching that pass's kernels. FAST_EVAL's graph casts the weights to bfloat16 permanently, which
changes the numbers, so a sealed version cannot use it. This captures the same autocast pass with fp32
weights and the autocast cache off inside the graph (each replay casts the weights itself, to the same
bfloat16 values), one graph per batch size, and compares every output with the eager pass on batches
the graph was not captured on. It only prints.
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess
import torch
from mcts import MCTS, Node, pinned_search
from conversion_suite import load

perf = time.perf_counter
m = load("mayfly_v4.pt")
t = MCTS(m, device="cuda", visits=400)
t.apply(pinned_search("mayfly_v4.pt"))
t.open_temp = 0.0
fens = [chess.STARTING_FEN, "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R2Q1RK1 w - - 0 10"]
nodes, seen = [], set()
for f in fens:
    t.search(chess.Board(f), visits=400, reuse=False)
    q = [t.root]
    while q and len(nodes) < 160 * (fens.index(f) + 1) // len(fens) + 1:
        nd = q.pop(0)
        if nd is None or nd.key in seen:
            continue
        seen.add(nd.key)
        if nd.term is None:
            nodes.append(Node(nd.board.copy(stack=False), t.draw_plies))
        if nd.kids:
            q.extend(k for k in nd.kids if k is not None)
print("positions", len(nodes), flush=True)


def tensors(ns):
    prep = t._prepare(ns)
    L = lambda a: torch.tensor(a, dtype=torch.long, device="cuda")  # noqa: E731
    return (torch.tensor(prep["ids"], dtype=torch.long, device="cuda"),
            torch.tensor(prep["cast"], device="cuda"), L(prep["hm"]), L(prep["ep"]))


@torch.inference_mode()
def eager(x):
    # search() holds one autocast context for the whole search and _dispatch opens its own inside it
    with torch.autocast("cuda", torch.bfloat16):
        with torch.autocast("cuda", torch.bfloat16):
            out = m(*x)
    return {k: v.clone() for k, v in out.items()}


POOL = torch.cuda.graph_pool_handle()


@torch.inference_mode()
def capture(x):
    static = [a.clone() for a in x]
    with torch.autocast("cuda", torch.bfloat16):            # as inside a search
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
                for _ in range(3):
                    m(*static)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=POOL):
            with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
                out = m(*static)
    return {"g": g, "in": static, "out": out}


@torch.inference_mode()
def replay(G, x):
    for s, a in zip(G["in"], x):
        s.copy_(a)
    G["g"].replay()
    return {k: v.clone() for k, v in G["out"].items()}


def same(a, b):
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    worst = max(((a[k].float() - b[k].float()).abs().max().item(), k) for k in bad) if bad else (0.0, "")
    return bad, worst


ok = True
for k in (48, 47, 36, 24, 13, 1):
    batches = [nodes[i * k:(i + 1) * k] for i in range(len(nodes) // k)][:3]
    if len(batches) < 2:
        batches = [nodes[:k], nodes[-k:]]
    xs = [tensors(b) for b in batches]
    G = capture(xs[0])                       # captured on the first batch, tested on the others as well

    for j, x in enumerate(xs):
        r1, r2 = eager(x), eager(x)
        ctl, _ = same(r1, r2)
        gr = replay(G, x)
        bad, worst = same(r1, gr)
        ok &= not bad and not ctl
        print(f"batch {k:>2}, set {j}: eager repeats itself: {not ctl}; graph equals eager on all "
              f"{len(r1)} outputs: {not bad}" + (f" (differs in {bad}, worst {worst})" if bad else ""),
              flush=True)
    torch.cuda.synchronize()
    te, tg = [], []
    for _ in range(15):
        torch.cuda.synchronize(); t0 = perf(); eager(xs[0]); torch.cuda.synchronize(); te.append(perf() - t0)
        torch.cuda.synchronize(); t0 = perf(); replay(G, xs[0]); torch.cuda.synchronize(); tg.append(perf() - t0)
    print(f"   batch {k:>2} pass, eager {1e3 * statistics.median(te):6.2f} ms, graph "
          f"{1e3 * statistics.median(tg):6.2f} ms", flush=True)
print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND", flush=True)
