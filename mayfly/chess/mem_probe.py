"""Measure what one search node costs in memory, with the real model (Windows only).

    python mem_probe.py [visits] [version]
    python mem_probe.py fake [total] [step]      the same tree with random network outputs

Builds a tree from a rich middlegame, where nearly every visit makes a node with thirty or more
legal moves. Reports the process's private bytes before and after (what Windows charges against
its commit limit), the nodes counted by walking the tree, the bound the search uses for its memory
cap, and how long the tree takes to free. The eval cache is off for the first tree so the delta is
the tree alone, then on for a second search so the cache's share is visible too."""
import os, sys, time, gc, ctypes, contextlib
from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


_k32 = ctypes.WinDLL("kernel32")
_psapi = ctypes.WinDLL("psapi")
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]


def private_bytes():
    m = PMC()
    m.cb = ctypes.sizeof(PMC)
    _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(m), m.cb)
    return m.PrivateUsage


def count_nodes(root):
    seen, stack, n = set(), [root], 0
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        n += 1
        if x.kids:
            stack.extend(k for k in x.kids if k is not None)
    return n


FEN = "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N1P/PP1P1PP1/RNBQR1K1 b - - 0 9"


def deep_bytes(node, NodeT):
    """The Python-level bytes one node owns.

    That is the node, its board and board state, its move list and moves, its key and its arrays.
    Other nodes and shared immortal objects are not followed."""
    seen, stack, total = set(), [node], 0
    while stack:
        o = stack.pop()
        if id(o) in seen or isinstance(o, type) or sys.getrefcount(o) > 1_000_000:
            continue
        if isinstance(o, NodeT) and o is not node:
            continue
        seen.add(id(o))
        total += sys.getsizeof(o)
        stack.extend(gc.get_referents(o))
    return total


def fake_main(total, step):
    """The same tree without the network, to see what a node costs at millions of them.

    Random logits and values stand in for the model, so it runs on the CPU and never touches a GPU.
    A node's memory is its board, its moves and its arrays, none of which depend on the network's
    output."""
    import numpy as np
    import chess
    from chess_net import N_MOVES
    from mcts import MCTS
    rng = np.random.default_rng(1)
    t = MCTS(None, "cpu", batch=48, cache_size=0)
    t.book = False

    def fake_eval(nodes):
        pol = rng.standard_normal((len(nodes), N_MOVES)).astype(np.float32)
        val = rng.uniform(-0.2, 0.2, len(nodes)).astype(np.float32)
        qv = [rng.uniform(-0.5, 0.5, len(n.moves)).astype(np.float32) for n in nodes]
        return pol, val, qv

    t._evaluate = fake_eval
    board = chess.Board(FEN)
    t.search(board, visits=2000, reuse=False)
    t.root = None
    gc.collect()
    p0 = private_bytes()
    last_n, last_p, done = 0, p0, 0
    t0 = time.time()
    while done < total:
        t.search(board, visits=step, reuse=done > 0)
        done += step
        gc.collect()
        n, p = count_nodes(t.root), private_bytes()
        bound = t.tree_nodes(t.root) if hasattr(t, "tree_nodes") else None
        print(f"{n:>8} nodes: +{(p - p0) / 1e9:.2f} GB total, {(p - p0) / max(1, n):.0f} B/node average, "
              f"{(p - last_p) / max(1, n - last_n):.0f} B/node this step"
              + (f", bound {bound}" if bound is not None else "")
              + f", {time.time() - t0:.0f}s", flush=True)
        last_n, last_p = n, p
    f0 = time.perf_counter()
    t.root = None
    freed = time.perf_counter() - f0
    gc.collect()
    print(f"free: {freed:.2f}s for {last_n} nodes ({1e6 * freed / last_n:.2f} us/node); process still "
          f"holds {(private_bytes() - p0) / 1e9:.2f} GB for reuse", flush=True)
    # reuse, a second tree the same size should cost nothing new
    pr = private_bytes()
    done = 0
    while done < total:
        t.search(board, visits=step, reuse=done > 0)
        done += step
    gc.collect()
    print(f"regrown to {count_nodes(t.root)} nodes: +{(private_bytes() - pr) / 1e9:.2f} GB beyond "
          f"what the first tree left held", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "fake":
        total = int(sys.argv[2]) if len(sys.argv) > 2 else 600_000
        step = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
        return fake_main(total, step)
    visits = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    version = sys.argv[2] if len(sys.argv) > 2 else "mayfly_v3.pt"
    with contextlib.redirect_stdout(sys.stderr):
        import chess
        from play_web import load_version
        from mcts import MCTS, Node
        model, label, _ = load_version(version)
    dev = str(next(model.parameters()).device)
    board = chess.Board(FEN)
    t = MCTS(model, dev, batch=48, cache_size=0)              # the defaults, which are v4's
    t.book = False                                            # a book move makes no tree
    t.search(board, visits=2000, reuse=False)                 # CUDA, autocast, allocator warm
    t.root = None
    gc.collect()
    p0 = private_bytes()
    t0 = time.time()
    t.search(board, visits=visits // 2, reuse=False)
    s1 = time.time() - t0
    gc.collect()
    n1, p1 = count_nodes(t.root), private_bytes()
    t0 = time.time()
    t.search(board, visits=visits // 2, reuse=True)           # the same tree, grown further
    s2 = time.time() - t0
    gc.collect()
    n2, p2 = count_nodes(t.root), private_bytes()
    first = (p1 - p0) / max(1, n1)
    marginal = (p2 - p1) / max(1, n2 - n1)
    print(f"tree: {n1} nodes +{(p1 - p0) / 1e6:.0f} MB ({first:.0f} B/node), then {n2} nodes "
          f"+{(p2 - p1) / 1e6:.0f} MB more ({marginal:.0f} B/node marginal); "
          f"{(n2 - n1) / s2:.0f} visits/s", flush=True)
    import random
    nodes, stack = [], [t.root]
    while stack:
        x = stack.pop()
        nodes.append(x)
        if x.kids:
            stack.extend(k for k in x.kids if k is not None)
    sample = random.Random(1).sample(nodes, min(2000, len(nodes)))
    py = sum(deep_bytes(x, Node) for x in sample) / len(sample)
    moves = sum(len(x.moves) for x in sample) / len(sample)
    print(f"python-level: {py:.0f} B/node on average ({moves:.1f} legal moves a node); "
          f"process pays {marginal / py:.2f}x that", flush=True)
    bound = t.tree_nodes(t.root) if hasattr(t, "tree_nodes") else None
    if bound is not None:
        print(f"cap bound {bound} for {n2} counted nodes", flush=True)
    del nodes, sample, stack
    f0 = time.perf_counter()
    t.root = None
    freed = time.perf_counter() - f0
    gc.collect()
    p3 = private_bytes()
    print(f"free: {freed:.2f}s ({1e6 * freed / n2:.2f} us/node); process still holds "
          f"{(p3 - p0) / 1e6:.0f} of {(p2 - p0) / 1e6:.0f} MB for reuse", flush=True)
    # the eval cache, with entries sampled from a short search with it on
    c = MCTS(model, dev, batch=48, cache_size=200_000)
    c.book = False
    c.search(board, visits=20_000, reuse=False)
    keys = list(c.cache)[:2000]
    per = sum(deep_bytes(c.cache[k], Node) + sys.getsizeof(k) for k in keys) / max(1, len(keys))
    print(f"cache: {per:.0f} B/entry python-level, x200000 = {per * 2e5 / 1e6:.0f} MB "
          f"(~{per * 2e5 * marginal / py / 1e6:.0f} MB at the process rate), full at 200k and "
          f"never grows past it", flush=True)


if __name__ == "__main__":
    main()
