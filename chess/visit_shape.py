r"""visit_shape measures whether the leaf batch size changes where the root visits go.

A scheduling change such as async evaluation alters no value, only when a backup happens, so a bug shows
up as visits landing on different root moves. Batch 1 is the reference, the textbook sequential PUCT with
no virtual loss outstanding. Each larger batch is compared to it by total variation distance between the
root visit distributions (0 is identical, 1 is disjoint) and by whether the chosen move agrees. That gap
is the tolerance an async implementation has to come in under.

  python chess/visit_shape.py [POSITIONS] [VISITS]
  env: BATCHES=1,8,24,48,64  PGN (folder of PGN files, default paths.under("twic"))
"""
import os, sys, glob, random
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
VISITS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
BATCHES = [int(x) for x in E("BATCHES", "1,8,24,48,64").split(",")]
PGN_DIR = E("PGN", paths.under("twic"))


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
                if not b.is_game_over() and b.legal_moves.count() > 2:
                    out.append(b.fen())
        if len(out) >= n:
            break
    return out[:n]


def tvd(a, b):
    """Total variation distance between two visit vectors over the same move list."""
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return 1.0
    return float(0.5 * abs(a / sa - b / sb).sum())


def shape(model, dev, fens, batch, visits):
    """Root visit vector and chosen move for each position, at this batch size."""
    from mcts import MCTS
    out = []
    for fen in fens:
        t = MCTS(model, dev, batch=batch, cache_size=20000)
        board = chess.Board(fen)
        mv, _ = t.search(board, visits=visits, reuse=False)
        out.append((t.root.N.copy(), mv, int(t.root.sum_n), t.stats["batches"]))
    return out


if __name__ == "__main__":
    import numpy as np
    import torch
    from play_web import load_version

    fens = sample(N_POS)
    model, label, _ = load_version(E("VERSION", "mayfly_v1.pt"))
    dev = str(next(model.parameters()).device)
    print(f"{len(fens)} positions, {VISITS} visits, {label} on {dev}", flush=True)
    print(f"  reference arm is batch {BATCHES[0]}", flush=True)
    print("", flush=True)

    ref = shape(model, dev, fens, BATCHES[0], VISITS)
    print(f"  {'batch':>6} {'mean TVD':>9} {'worst':>7} {'same move':>10} {'visits':>8} "
          f"{'passes':>7}")
    print(f"  {BATCHES[0]:>6} {0.0:>9.3f} {0.0:>7.3f} {'reference':>10} "
          f"{sum(r[2] for r in ref)//len(ref):>8,} {sum(r[3] for r in ref)/len(ref):>7.1f}")
    for bs in BATCHES[1:]:
        got = shape(model, dev, fens, bs, VISITS)
        ds = [tvd(a[0], b[0]) for a, b in zip(ref, got)]
        same = sum(1 for a, b in zip(ref, got) if a[1] == b[1])
        print(f"  {bs:>6} {sum(ds)/len(ds):>9.3f} {max(ds):>7.3f} "
              f"{str(same) + '/' + str(len(got)):>10} "
              f"{sum(r[2] for r in got)//len(got):>8,} {sum(r[3] for r in got)/len(got):>7.1f}")

    print("")
    print("  TVD is half the summed absolute difference of the normalised root visit counts.")
    print("  Any async implementation has to sit at or under the batch-48 row on both columns,")
    print("  because async changes scheduling exactly the way a larger batch does: more descents")
    print("  in flight, more virtual loss outstanding when a selection is made.")
