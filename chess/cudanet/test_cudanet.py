"""The hand-tuned CUDA backend checked against ChessNet, and timed against the other paths the search can run.
Needs a CUDA card. Nothing here runs at import.

  python test_cudanet.py numerics|speed|all --cnw X.cnw --ckpt Y.pt [--n 512] [--json out.json]

numerics  N positions (fixed ones from real play, en passant and castling cases, one legal move, 218 legal moves,
          clocks past 100, then random playouts), in batches of mixed sizes, I and M padded to each batch's widest
          row as _prepare pads them. The reference is ChessNet in float32 on the CPU (export_ts.load_v5 on the
          checkpoint the .cnw was made from) through _collect's arithmetic (export_ts.post, as cpp/netbackend.h).
          Reports the largest and mean absolute differences of pol, the prior (softmax of pol), val and qv at the
          legal moves, and how often the top move and the top 3 agree. With a CUDA torch, the bf16 autocast path
          the Python search plays (MCTS one_copy and the autocast graph, through mcts_cpp._evaluate) and the
          LibTorch trace (<ckpt>.cuda.ts) are measured against the same reference, for scale.
speed     evaluations a second at B = 1, 8, 16, 32, 48 for the CUDA backend, the Python one_copy graph path, the
          LibTorch trace (a CUDA graph per batch size, as cpp/torchnet.cpp runs it) and, when mayfly_cppsearch
          imports, the whole C++ search on that trace (tree and net, so not the same thing). Each after warm-up,
          over many repetitions, printed as a table.
all       both.
profile   where a pass's time goes, step by step (embed, the layer norms, each product, attention, gelu, the
          heads), at each of --sizes, the mean of --reps passes, and the whole pass as a CUDA graph with the
          evaluations a second it gives.
--json    the results as JSON.
--sizes and --mixed above 48 need env MAYFLY_CUDANET_MAX_BATCH at least as big (cudanet's buffers are sized by it).

Exit status 1 when a numerics pass line fails (TOL and the TOP* lines below).
"""
import argparse
import json
import os
import platform
import random
import sys
import time

# the autocast graph is the path v5 and v6 play (mcts.AUTOCAST_GRAPH_OK is read when mcts is imported)
os.environ.setdefault("AUTOCAST_GRAPH", "1")

import chess  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CHESS = os.path.dirname(HERE)
for _d in (HERE, os.path.join(CHESS, "cpp"), CHESS):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import export_ts  # noqa: E402
from cudanet_py import CudaNet  # noqa: E402

SIZES = (1, 8, 16, 32, 48)
MIXED = (48, 1, 16, 8, 33, 48, 3, 32, 47, 24)    # numerics batch sizes, cycled (--mixed)

# Pass lines for FP16 products with FP32 accumulation (cudanet_spec.h) against the float32 reference. FP16 keeps
# 11 significant bits where the bf16 the Python search plays keeps 8, so the backend should sit well inside the
# bf16 path's own distance from float32. When that path is measured too, the backend must also be no worse than it
# on every max and mean below.
TOL = {"pol": {"max": 0.10, "mean": 0.010},      # raw policy logits
       "prior": {"max": 0.010, "mean": 2e-4},     # softmax of each row's legal logits (temperature 1)
       "val": {"max": 0.010, "mean": 0.002},      # value, in [-1, 1]
       "qv": {"max": 0.020, "mean": 0.002}}       # per move q, in [-1, 1]
TIE_GAP = 0.20      # logits, a best move ahead of the second by less than this is a near tie and either order is fine
TOP1_MIN = 0.99     # rows whose best move is the reference's
TOP1_CLEAR_MIN = 1.0  # the same over rows without a near tie
TOP3_MIN = 0.97     # rows whose top 3 (as a set) is the reference's

# from real play (MayFly's Lichess games and the usual openings) and hand-picked special cases
FIXED = [
    chess.STARTING_FEN,
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 8",
    "r2q1rk1/1b2bppp/p2ppn2/1p6/3NP3/1BN1B3/PPP2PPP/R2Q1RK1 w - - 0 12",
    "2r2rk1/pp1bqppp/2n1pn2/3p4/2PP4/P1N1PN2/1P1BQPPP/2R2RK1 b - - 3 15",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1b2rk1/2q1bppp/p2p1n2/np2p3/3PP3/5N1P/PPBN1PP1/R1BQR1K1 b - - 0 12",
    "2kr3r/ppp2ppp/2n5/2b1p3/4P1b1/2NP1N2/PPP2PPP/R1B2RK1 w - - 2 11",
    "8/5pk1/6p1/8/3R4/6P1/5PKP/1r6 w - - 4 41",
    "8/8/4k3/3p4/3P4/4K3/8/8 w - - 10 60",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 30",
    "4r1k1/1p3ppp/p7/3p4/3P1P2/P3P3/1P4PP/2R3K1 b - - 0 25",
    # castling on both sides, black to move, partial rights
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 7 30",
    "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 8",
    # en passant whose capture is illegal (the pawn is pinned), so the encoder must leave ep at 0
    "8/8/8/K2pP2r/8/8/8/7k w - d6 0 1",
    # one legal move
    "k7/8/8/8/8/8/1q6/K7 w - - 0 1",
    "k7/1Q6/8/8/8/8/8/K7 b - - 0 1",
    # 218 legal moves (the most known), and another very wide one
    "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1",
    "3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1",
    # promotions and underpromotions, both colours, a capture promotion
    "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
    "8/P6k/8/8/8/8/6Kp/8 b - - 0 1",
    "1r5k/P7/8/8/8/8/8/K7 w - - 0 1",
    # clocks just under the cap and past it (hm is min(halfmove, 100))
    "8/8/4k3/8/8/4K3/4P3/8 w - - 99 120",
    "8/8/4k3/8/8/4K3/4P3/8 b - - 130 150",
]

# played out from the start, so the en passant and castling states are real ones
LINES = [
    "e4 d5 e5 f5",                      # white to move, exf6 e.p.
    "Nf3 e5 Nc3 e4 d4",                 # black to move, exd3 e.p.
    "e4 e6 e5 d5",                      # exd6 e.p.
    "d4 Nf6 d5 e5",                     # dxe6 e.p.
    "e4 Nf6 e5 d5",                     # exd6 e.p. beside a knight move
    "e4 c5 Nf3 d6 Bb5+ Bd7",            # white may castle short
    "d4 d5 Nc3 Nc6 Bf4 Bf5 Qd2 Qd7",    # white may castle long
    "e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6",    # white castled, black still may
    "e4 e5 Ke2 Ke7",                    # no rights left on either side
    "e4 e5 Nf3 Nf6 Rg1 Rg8 Rh1",        # white lost the short right, black to move
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3",
]


def _ok(b):
    return b.is_valid() and any(True for _ in b.legal_moves)


def positions(n, seed=0):
    """N FENs from the fixed list, the played lines, a few one-move positions and random playouts."""
    out = [f for f in FIXED if _ok(chess.Board(f))]
    for line in LINES:
        b = chess.Board()
        try:
            for san in line.split():
                b.push_san(san)
        except ValueError:
            print(f"  skipped line {line!r}: not legal")
            continue
        if _ok(b):
            out.append(b.fen())
    rng = random.Random(seed)
    ones, rand = [], []
    want_ones = max(2, n // 50)
    while len(out) + len(ones) + len(rand) < n or len(ones) < want_ones:
        b = chess.Board()
        for _ in range(rng.randint(0, 160)):
            moves = list(b.legal_moves)
            if not moves:
                break
            if len(moves) == 1 and len(ones) < want_ones:
                ones.append(b.fen())
            b.push(rng.choice(moves))
        if _ok(b):
            rand.append(b.fen())
        if len(rand) > 50 * n:          # never loop forever hunting one-move positions
            break
    out = out + ones + rand
    return out[:n]


def batch_sizes(n, mixed=MIXED):
    """mixed (MIXED by default), cycled, summing to n (the last one cut)."""
    out, k = [], 0
    while sum(out) < n:
        out.append(min(mixed[k % len(mixed)], n - sum(out)))
        k += 1
    return out


def cut(policy, b):
    """A (B, 4162) policy cut to each row's legal moves, (B, wide), 0 on padding (what the backend returns)."""
    B, wide = b["I"].shape
    P = np.zeros((B, wide), np.float32)
    for i, k in enumerate(b["widths"]):
        P[i, :k] = policy[i, b["I"][i, :k]]
    return P


# -------------------------------------------------------------------- the paths
def reference(model, batches):
    """ChessNet float32 on the CPU, then _collect's arithmetic (export_ts.post)."""
    mha = getattr(torch.backends, "mha", None)
    fast = mha.get_fastpath_enabled() if mha is not None and hasattr(mha, "get_fastpath_enabled") else None
    if fast is not None:
        mha.set_fastpath_enabled(False)      # the ordinary encoder path, as the CPU trace takes
    exp = export_ts.Export(model.float()).eval()
    out = []
    try:
        with torch.no_grad():
            for b in batches:
                pol, qval = exp(*export_ts.inputs(b, "cpu"))
                r = export_ts.post(pol, qval, b, "cpu")
                out.append((cut(r["policy"], b), r["val"].astype(np.float32), r["qv"].astype(np.float32)))
    finally:
        if fast is not None:
            mha.set_fastpath_enabled(fast)
    return out


def run_cuda(net, b):
    return net.evaluate(b["ids"], b["cast"], b["hm"], b["ep"], b["I"], b["M"].astype(np.uint8))


def make_tree(ckpt):
    """The Python path, an MCTS on the card with v6's defaults (one_copy, the autocast graph)."""
    import mcts
    model = export_ts.load_v5(ckpt, "cuda")
    tree = mcts.MCTS(model, device="cuda", batch=48)
    return tree, tree._oc_ok()


def run_tree(tree, b):
    """The network through the tree's own _prepare blob, _dispatch and _collect (mcts_cpp._evaluate)."""
    import mcts_cpp
    B, wide = b["I"].shape
    P, V, Q = mcts_cpp._evaluate(tree, B, wide, b["ids"].tobytes(), b["cast"].tobytes(), b["hm"].tobytes(),
                                 b["ep"].tobytes(), b["I"].tobytes(), b["M"].astype(np.uint8).tobytes(),
                                 b["widths"].astype(np.int32).tobytes())
    return (np.frombuffer(P, np.float32).reshape(B, wide), np.frombuffer(V, np.float32),
            np.frombuffer(Q, np.float32).reshape(B, wide))


class TraceRunner:
    """The LibTorch trace (export_ts "cuda") as cpp/torchnet.cpp runs it, the forward pass as a CUDA graph per batch
    size, then _collect's post-processing as torch ops on the batch's own I and M."""

    def __init__(self, ts_path):
        # no map location (cpp/net_backend2.cpp). The trace keeps MultiheadAttention's shape arithmetic on 0-dim CPU
        # constants, and mapping them to the card makes every int() a device sync, which breaks the capture.
        self.m = torch.jit.load(ts_path)
        self.m.to("cuda")
        self.m.eval()
        self.graphs = {}

    def _graph(self, B):
        if B in self.graphs:
            return self.graphs[B]
        dev = "cuda"
        ids = torch.zeros((B, 64), dtype=torch.long, device=dev)
        cast = torch.zeros((B, 4), dtype=torch.float32, device=dev)
        hm = torch.zeros(B, dtype=torch.long, device=dev)
        ep = torch.zeros(B, dtype=torch.long, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.m(ids, cast, hm, ep)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            pol, q = self.m(ids, cast, hm, ep)
        self.graphs[B] = (g, ids, cast, hm, ep, pol, q)
        return self.graphs[B]

    def __call__(self, b):
        with torch.inference_mode():
            B = b["ids"].shape[0]
            g, ids, cast, hm, ep, pol, q = self._graph(B)
            ids.copy_(torch.from_numpy(b["ids"]))
            cast.copy_(torch.from_numpy(b["cast"]))
            hm.copy_(torch.from_numpy(b["hm"]))
            ep.copy_(torch.from_numpy(b["ep"]))
            g.replay()
            I = torch.from_numpy(b["I"]).to("cuda")
            M = torch.from_numpy(b["M"].astype(bool)).to("cuda")
            pl = pol.float()
            qs = torch.sigmoid(q.float())
            lg = torch.gather(pl, 1, I)
            w = torch.softmax(lg.masked_fill(~M, float("-inf")), -1)
            qg = torch.gather(qs, 1, I)
            val = (w * qg).sum(1) * 2.0 - 1.0
            lg = lg.masked_fill(~M, 0.0)
            return lg.cpu().numpy(), val.cpu().numpy(), (qg * 2.0 - 1.0).cpu().numpy()


def find_ts(args):
    if args.ts:
        return args.ts if os.path.exists(args.ts) else None
    if args.ckpt:
        ts = os.path.splitext(args.ckpt)[0] + ".cuda.ts"
        return ts if os.path.exists(ts) else None
    return None


# -------------------------------------------------------------------- numerics
def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def metrics(ref, got, batches):
    """Differences at the legal moves only, and how often the best move and the top 3 agree."""
    d = {"pol": [], "prior": [], "val": [], "qv": []}
    rows = top1 = clear = clear_ok = top3 = 0
    nonfinite = pad_nonzero = 0
    for (rp, rv, rq), (gp, gv, gq), b in zip(ref, got, batches):
        gp, gv, gq = (np.asarray(a, np.float64) for a in (gp, gv, gq))
        nonfinite += int((~np.isfinite(gp)).sum() + (~np.isfinite(gv)).sum() + (~np.isfinite(gq)).sum())
        d["val"].append(np.abs(rv.astype(np.float64) - gv))
        for i, k in enumerate(b["widths"]):
            k = int(k)
            a, g = rp[i, :k].astype(np.float64), gp[i, :k]
            d["pol"].append(np.abs(a - g))
            d["prior"].append(np.abs(_softmax(a) - _softmax(g)))
            d["qv"].append(np.abs(rq[i, :k].astype(np.float64) - gq[i, :k]))
            pad_nonzero += int((gp[i, k:] != 0).sum())
            ra, ga = np.argsort(-a, kind="stable"), np.argsort(-g, kind="stable")
            rows += 1
            top1 += int(ra[0] == ga[0])
            gap = (a[ra[0]] - a[ra[1]]) if k > 1 else np.inf
            if gap > TIE_GAP:
                clear += 1
                clear_ok += int(ra[0] == ga[0])
            k3 = min(3, k)
            top3 += int(set(ra[:k3].tolist()) == set(ga[:k3].tolist()))
    out = {}
    for key, parts in d.items():
        v = np.concatenate(parts) if parts else np.zeros(1)
        v = np.where(np.isfinite(v), v, np.inf)
        out[key] = {"max": float(v.max()), "mean": float(v.mean()), "n": int(v.size)}
    out.update(rows=rows, top1=top1 / max(rows, 1), top1_clear=clear_ok / max(clear, 1), clear_rows=clear,
               top3=top3 / max(rows, 1), nonfinite=nonfinite, pad_nonzero=pad_nonzero)
    return out


def verdict(m, bf=None):
    """The pass lines this result fails, as plain sentences (empty when it passes)."""
    bad = []
    if m["nonfinite"]:
        bad.append(f"{m['nonfinite']} values are NaN or infinite")
    for key, lim in TOL.items():
        for stat, x in lim.items():
            if m[key][stat] > x:
                bad.append(f"{key} {stat} abs {m[key][stat]:.3g} > {x:g}")
            if bf is not None and m[key][stat] > bf[key][stat]:
                bad.append(f"{key} {stat} abs {m[key][stat]:.3g} worse than bf16's {bf[key][stat]:.3g}")
    if m["top1"] < TOP1_MIN:
        bad.append(f"top move agrees on {m['top1']:.4f} < {TOP1_MIN}")
    if m["top1_clear"] < TOP1_CLEAR_MIN:
        bad.append(f"top move without a near tie agrees on {m['top1_clear']:.4f} < {TOP1_CLEAR_MIN}")
    if m["top3"] < TOP3_MIN:
        bad.append(f"top 3 agrees on {m['top3']:.4f} < {TOP3_MIN}")
    return bad


def print_metrics(name, m):
    print(f"  {name}")
    for key in ("pol", "prior", "val", "qv"):
        print(f"    {key:6s} max abs {m[key]['max']:.3e}  mean abs {m[key]['mean']:.3e}  (n {m[key]['n']})")
    print(f"    top move {m['top1']:.4f}  (without near ties {m['top1_clear']:.4f} over {m['clear_rows']} rows)  "
          f"top 3 {m['top3']:.4f}  rows {m['rows']}  nonfinite {m['nonfinite']}  padding not 0 {m['pad_nonzero']}")


def numerics(args, fens):
    import mcts_cpp
    sizes = batch_sizes(len(fens), [int(x) for x in args.mixed.split(",")])
    batches = export_ts.make_batches(fens, sizes)
    widths = np.concatenate([b["widths"] for b in batches])
    print(f"numerics: {len(fens)} positions in {len(batches)} batches (sizes {sorted(set(sizes))}), "
          f"legal moves {int(widths.min())} to {int(widths.max())}, "
          f"{int((widths == 1).sum())} with one, ep set on {int(sum((b['ep'] > 0).sum() for b in batches))}")
    model = export_ts.load_v5(args.ckpt, "cpu")
    fp_ckpt = mcts_cpp.fingerprint(model)
    t0 = time.perf_counter()
    ref = reference(model, batches)
    print(f"  reference: float32 on the CPU, {time.perf_counter() - t0:.1f} s")
    del model

    res = {"positions": len(fens), "batches": len(batches), "sizes": sizes,
           "widths": {"min": int(widths.min()), "max": int(widths.max()), "ones": int((widths == 1).sum())}}
    net = CudaNet(args.cnw, args.device, args.dll)
    fp_cnw = net.fingerprint()
    res["fingerprint"] = {"cnw": fp_cnw, "ckpt": fp_ckpt, "match": fp_cnw == fp_ckpt}
    got = [run_cuda(net, b) for b in batches]
    again = [run_cuda(net, b) for b in batches]       # replays the captured graphs, which must give the same bits
    res["repeatable"] = all(all(np.array_equal(x, y) for x, y in zip(g1, g2)) for g1, g2 in zip(got, again))
    net.close()
    res["cuda"] = metrics(ref, got, batches)
    print_metrics("cudanet (fp16 products) vs float32", res["cuda"])

    bf = None
    want = set(args.paths.split(","))
    if torch.cuda.is_available():
        try:
            if "python_one_copy" not in want:
                raise RuntimeError("left out by --paths")
            tree, mode = make_tree(args.ckpt)
            res["bf16_mode"] = mode
            bf = metrics(ref, [run_tree(tree, b) for b in batches], batches)
            res["bf16"] = bf
            print_metrics(f"today's bf16 path (one_copy {mode}) vs float32", bf)
            del tree
        except Exception as e:
            res["bf16_error"] = f"{type(e).__name__}: {e}"
            print(f"  bf16 path not measured: {res['bf16_error']}")
        ts = find_ts(args) if "libtorch_trace" in want else None
        if ts:
            try:
                tr = TraceRunner(ts)
                res["libtorch"] = metrics(ref, [tr(b) for b in batches], batches)
                print_metrics(f"LibTorch trace {os.path.basename(ts)} vs float32", res["libtorch"])
                del tr
            except Exception as e:
                res["libtorch_error"] = f"{type(e).__name__}: {e}"
                print(f"  LibTorch trace not measured: {res['libtorch_error']}")
        torch.cuda.empty_cache()
    else:
        print("  no CUDA torch here: the bf16 path and the trace are not measured")

    bad = verdict(res["cuda"], bf)
    if not res["fingerprint"]["match"]:
        bad.insert(0, f"the .cnw fingerprint {fp_cnw} is not the checkpoint's {fp_ckpt}")
    if not res["repeatable"]:
        bad.append("two runs of the same batches differ")
    res["fails"] = bad
    res["pass"] = not bad
    print("  PASS" if not bad else "  FAIL: " + "; ".join(bad))
    return res


# -------------------------------------------------------------------- speed
def timed(fn, batches, warmup, reps):
    """Seconds a call, after warm-up, cycling through the batches."""
    for i in range(warmup):
        fn(batches[i % len(batches)])
    t0 = time.perf_counter()
    for i in range(reps):
        fn(batches[i % len(batches)])
    return (time.perf_counter() - t0) / reps


def cpp_search_speed(args, ts, sizes, fens):
    """The whole C++ search (mayfly_cppsearch on the trace) in evaluations a second, tree time included."""
    import mcts
    import mcts_cpp
    if mcts_cpp.mayfly_cppsearch is None:
        return None, f"no mayfly_cppsearch ({mcts_cpp.IMPORT_ERROR})"
    tree = mcts.MCTS(export_ts.load_v5(args.ckpt, "cuda"), device="cuda")
    out = {}
    starts = fens[:args.cpp_positions]
    for B in sizes:
        tree.batch = B
        cfg = mcts_cpp._settings(tree)
        cfg["use_tb"] = 0.0
        cfg["_ts"] = ts
        t = mcts_cpp.mayfly_cppsearch.Tree(cfg, None, lambda fen: None, ts, "cuda", True)

        def search(fen):
            t.reset()
            t.begin(fen, [], False)
            if t.needs_eval():
                t.expand_root()
            while t.step(float(args.visits), -1.0) == 0:
                pass
            s = t.stats()
            return s["evals"], s["batches"]

        search(starts[0])                   # warm-up, so graphs are captured and kernels loaded
        evals = batches = 0
        t0 = time.perf_counter()
        for fen in starts:
            e, n = search(fen)
            evals += e
            batches += n
        el = time.perf_counter() - t0
        out[B] = {"evals_s": evals / el, "ms_batch": 1000.0 * el / max(batches, 1),
                  "mean_batch": evals / max(batches, 1)}
        del t
    return out, None


def speed(args, fens):
    sizes = [int(x) for x in args.sizes.split(",")]
    sets = {B: export_ts.make_batches(fens, [B] * 4) for B in sizes}
    res = {"sizes": sizes, "warmup": args.warmup, "reps": args.reps, "paths": {}, "notes": {}}

    def run(name, fn):
        row = {}
        for B in sizes:
            s = timed(fn, sets[B], args.warmup, args.reps)
            row[B] = {"evals_s": B / s, "ms_batch": 1000.0 * s}
        res["paths"][name] = row

    want = set(args.paths.split(","))
    if "cudanet" in want:
        net = CudaNet(args.cnw, args.device, args.dll)
        run("cudanet", lambda b: run_cuda(net, b))
        net.close()
    if torch.cuda.is_available() and args.ckpt:
        try:
            if "python_one_copy" not in want:
                raise RuntimeError("left out by --paths")
            tree, mode = make_tree(args.ckpt)
            res["notes"]["python_one_copy"] = f"MCTS one_copy, _oc_ok() = {mode}, through mcts_cpp._evaluate"
            run("python_one_copy", lambda b: run_tree(tree, b))
            del tree
        except Exception as e:
            res["notes"]["python_one_copy"] = f"not measured: {type(e).__name__}: {e}"
        ts = find_ts(args)
        if ts:
            try:
                if "libtorch_trace" not in want:
                    raise RuntimeError("left out by --paths")
                tr = TraceRunner(ts)
                res["notes"]["libtorch_trace"] = f"{ts}, CUDA graph per batch size, post as torch ops"
                run("libtorch_trace", tr)
                del tr
            except Exception as e:
                res["notes"]["libtorch_trace"] = f"not measured: {type(e).__name__}: {e}"
            try:
                if "cpp_search" not in want:
                    raise RuntimeError("left out by --paths")
                row, why = cpp_search_speed(args, ts, sizes, fens)
                if row is None:
                    res["notes"]["cpp_search"] = why
                else:
                    res["paths"]["cpp_search"] = row
                    res["notes"]["cpp_search"] = (f"whole C++ search on the trace, {args.visits} visits from "
                                                  f"{args.cpp_positions} positions: tree time included, batches "
                                                  f"can be smaller than B (collisions)")
            except Exception as e:
                res["notes"]["cpp_search"] = f"not measured: {type(e).__name__}: {e}"
        else:
            res["notes"]["libtorch_trace"] = "no trace (--ts, or <ckpt>.cuda.ts beside the checkpoint)"
        torch.cuda.empty_cache()
    else:
        res["notes"]["python_one_copy"] = "no CUDA torch here, or no --ckpt"

    print(f"speed: evaluations a second (ms a batch), {args.warmup} warm-up then {args.reps} calls each")
    print(f"  {'path':18s}" + "".join(f"{'B=' + str(B):>18s}" for B in sizes))
    for name, row in res["paths"].items():
        print(f"  {name:18s}" + "".join(f"{row[B]['evals_s']:>10.0f} ({row[B]['ms_batch']:5.2f})" for B in sizes))
    if "cudanet" in res["paths"]:
        for name, row in res["paths"].items():
            if name != "cudanet":
                print(f"  cudanet / {name}: " + "  ".join(
                    f"B={B} x{res['paths']['cudanet'][B]['evals_s'] / row[B]['evals_s']:.2f}" for B in sizes))
    for k, v in res["notes"].items():
        print(f"  note {k}: {v}")
    return res


# -------------------------------------------------------------------- profile
def profile(args):
    """Where a pass's time goes, step by step, at each batch size (CudaNet.profile)."""
    sizes = [int(x) for x in args.sizes.split(",")]
    net = CudaNet(args.cnw, args.device, args.dll)
    rows = {B: net.profile(B, args.reps) for B in sizes}
    net.close()
    names = list(rows[sizes[0]])
    steps = [n for n in names if n != "graph_pass"]
    print(f"profile: ms a pass by step, the mean of {args.reps} passes (eager, an event after each step)")
    print(f"  {'step':12s}" + "".join(f"{'B=' + str(B):>16s}" for B in sizes))
    for n in steps:
        print(f"  {n:12s}" + "".join(f"{rows[B][n]:>8.3f} ({100 * rows[B][n] / sum(rows[B][s] for s in steps):4.1f}%)"
                                     for B in sizes))
    print(f"  {'eager sum':12s}" + "".join(f"{sum(rows[B][s] for s in steps):>16.3f}" for B in sizes))
    print(f"  {'graph pass':12s}" + "".join(f"{rows[B]['graph_pass']:>16.3f}" for B in sizes))
    print(f"  {'evals/s':12s}" + "".join(f"{1000.0 * B / rows[B]['graph_pass']:>16.0f}" for B in sizes))
    return {"reps": args.reps, "ms": {str(B): rows[B] for B in sizes},
            "env": {k: os.environ.get(k) for k in ("MAYFLY_CUDANET_MAX_BATCH", "MAYFLY_CUDANET_ACC16")}}


# -------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="cudanet against ChessNet, numerics and speed")
    ap.add_argument("mode", choices=("numerics", "speed", "all", "profile"))
    ap.add_argument("--cnw", required=True, help="the weight file (export_cudanet.py)")
    ap.add_argument("--ckpt", help="the checkpoint the .cnw was made from (numerics needs it)")
    ap.add_argument("--n", type=int, default=512, help="positions for numerics")
    ap.add_argument("--json", help="write the results here as JSON")
    ap.add_argument("--dll", help="cudanet.dll (default MAYFLY_CUDANET_DLL or chess/cudanet/build/cudanet.dll)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--ts", help="the LibTorch trace (default <ckpt>.cuda.ts)")
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--mixed", default=",".join(str(s) for s in MIXED),
                    help="numerics batch sizes, cycled (above 48 needs MAYFLY_CUDANET_MAX_BATCH)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--visits", type=int, default=800, help="cpp_search: visits a search")
    ap.add_argument("--cpp-positions", type=int, default=8, help="cpp_search: searches per batch size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paths", default="cudanet,python_one_copy,libtorch_trace,cpp_search",
                    help="which comparisons to run; the Python tree's pipeline thread breaks later graph captures "
                         "in the same process, so run libtorch_trace and cpp_search without python_one_copy")
    args = ap.parse_args(argv)
    if args.mode in ("numerics", "all") and not args.ckpt:
        ap.error("numerics needs --ckpt")

    biggest = max([48] + [int(x) for x in args.sizes.split(",")])
    fens = positions(max(args.n, biggest * 4), args.seed)   # speed takes four batches of each size
    out = {"test": "cudanet", "when": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
           "cnw": os.path.abspath(args.cnw), "ckpt": os.path.abspath(args.ckpt) if args.ckpt else None,
           "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else None,
           "tolerances": {"tol": TOL, "tie_gap": TIE_GAP, "top1": TOP1_MIN, "top1_clear": TOP1_CLEAR_MIN,
                          "top3": TOP3_MIN}}
    ok = True
    if args.mode in ("numerics", "all"):
        out["numerics"] = numerics(args, fens[:args.n])
        ok = out["numerics"]["pass"]
    if args.mode in ("speed", "all"):
        out["speed"] = speed(args, fens)
    if args.mode == "profile":
        out["profile"] = profile(args)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=str)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
