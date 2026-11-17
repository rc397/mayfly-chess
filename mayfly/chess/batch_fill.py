"""Measure how full the search's batches of 48 are and what one batch costs.

Terminals, tablebase hits, repetitions and cache hits never take a batch slot. What can still shrink or
waste a batch is the tail batch of a search, and duplicate positions inside one batch (two node objects
that transpose into the same position are both sent to the GPU). The second half microbenchmarks the
parts of one batch on a fixed set of boards, namely marshalling, host to device copy, the forward pass,
and pulling back either the full policy or only the legal-move slice.

    python chess/batch_fill.py [POSITIONS] [VISITS]
"""
import os, sys, glob, random, time, collections
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

E = os.environ.get
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
VISITS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PGN_DIR = E("PGN", os.path.join(HERE, "twic"))


def sample(n, seed=3):
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
    rng.shuffle(files)
    out = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while len(out) < n:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                mv = list(g.mainline_moves())
                if len(mv) < 30:
                    continue
                b = g.board()
                for m in mv[:rng.randrange(14, 40)]:
                    b.push(m)
                if not b.is_game_over():
                    out.append(b.fen())
        if len(out) >= n:
            break
    return out[:n]


if __name__ == "__main__":
    import numpy as np
    import torch
    from mcts import MCTS
    from play_web import load_version
    from chess_net import board_to_ids, clock_and_ep
    from moves import N_MOVES

    clock = time.perf_counter
    fens = sample(N_POS)
    model, label, _ = load_version(E("VERSION", "mayfly_v1.pt"))
    dev = str(next(model.parameters()).device)
    print(f"{len(fens)} positions, {VISITS} visits, batch 48, {label} on {dev}", flush=True)

    class Counted(MCTS):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.fill = collections.Counter()
            self.dups = 0
            self.slots = 0
            self.widths = 0

        def _evaluate(self, nodes):
            self.fill[len(nodes)] += 1
            self.slots += len(nodes)
            self.dups += len(nodes) - len(set(n.key for n in nodes))
            self.widths += sum(len(n.moves) for n in nodes)
            return super()._evaluate(nodes)

    fill, dups, slots, widths, batches = collections.Counter(), 0, 0, 0, 0
    for fen in fens:
        t = Counted(model, dev, batch=48, cache_size=20000)
        t.search(chess.Board(fen), visits=VISITS, reuse=False)
        fill.update(t.fill)
        dups += t.dups
        slots += t.slots
        widths += t.widths
        batches += sum(t.fill.values())

    print("")
    print(f"  {batches:,} batches, {slots:,} slots used, mean fill "
          f"{slots/max(batches,1):.1f} of 48 ({100*slots/max(batches*48,1):.1f}%)")
    print(f"  duplicate positions inside a batch: {dups:,} of {slots:,} slots "
          f"({100*dups/max(slots,1):.1f}%), each one a forward pass computed twice")
    print(f"  mean legal moves at a leaf: {widths/max(slots,1):.1f}")
    print("  how full the batches were:")
    for k in sorted(fill, reverse=True)[:6]:
        print(f"    {k:>3} leaves  {fill[k]:>6,} batches  {100*fill[k]/batches:>5.1f}%")
    small = sum(c for k, c in fill.items() if k < 48)
    print(f"    under 48: {small:,} batches ({100*small/max(batches,1):.1f}%), "
          f"which should be about one tail batch a search")

    # the microbenchmark, on one fixed batch of 48 real positions
    boards = [chess.Board(f) for f in fens]
    while len(boards) < 48:
        boards += [chess.Board(f) for f in fens]
    boards = boards[:48]
    W = int(round(widths / max(slots, 1))) or 40
    REP = 60

    def timed(fn):
        fn()
        torch.cuda.synchronize()
        t0 = clock()
        for _ in range(REP):
            fn()
        torch.cuda.synchronize()
        return (clock() - t0) / REP * 1000.0

    def marshal():
        e = [board_to_ids(b) for b in boards]
        return [clock_and_ep(b, x[2]) for b, x in zip(boards, e)]

    enc = [board_to_ids(b) for b in boards]
    ce = [clock_and_ep(b, e[2]) for b, e in zip(boards, enc)]
    ids_np = np.array([e[0] for e in enc])
    cast_np = np.array([e[1] for e in enc], np.float32)
    hm = [c[0] for c in ce]
    ep = [c[1] for c in ce]

    def h2d():
        torch.tensor(ids_np, dtype=torch.long, device=dev)
        torch.tensor(cast_np, device=dev)
        torch.tensor(hm, dtype=torch.long, device=dev)
        torch.tensor(ep, dtype=torch.long, device=dev)

    ids = torch.tensor(ids_np, dtype=torch.long, device=dev)
    cast = torch.tensor(cast_np, device=dev)
    hmt = torch.tensor(hm, dtype=torch.long, device=dev)
    ept = torch.tensor(ep, dtype=torch.long, device=dev)

    def forward():
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev != "cpu")):
            return model(ids, cast, hmt, ept)

    with torch.no_grad():
        out = forward()
        full = out["policy"].float()
        sl = torch.zeros((48, W), dtype=torch.long, device=dev)

        def d2h_full():
            return full.cpu().numpy()

        def d2h_slice():
            return torch.gather(full, 1, sl).cpu().numpy()

        t_marshal = timed(marshal)
        t_h2d = timed(h2d)
        t_fwd = timed(forward)
        t_full = timed(d2h_full)
        t_slice = timed(d2h_slice)

    tot = t_marshal + t_h2d + t_fwd + t_full
    print("")
    print(f"  one batch of 48, mean of {REP}, milliseconds")
    for name, ms, note in (
            ("marshalling", t_marshal, "board_to_ids + clock_and_ep, 48 boards, pure python"),
            ("H2D", t_h2d, "token ids and clocks across the bus"),
            ("forward", t_fwd, "the model under autocast bfloat16"),
            ("D2H full policy", t_full, f"48 x {N_MOVES} floats, what ships today"),
            ("D2H legal slice", t_slice, f"48 x {W} floats, what is actually read")):
        print(f"    {name:<17} {ms:>7.2f} ms  {100*ms/tot:>5.1f}%  {note}")
    print(f"    {'total as shipped':<17} {tot:>7.2f} ms")
    print(f"    replacing the full pull with the slice saves {t_full - t_slice:.2f} ms a batch, "
          f"{100*(t_full-t_slice)/tot:.1f}% of a batch")
