r"""time_mcts times the parts of an MCTS search and reports how deep the batched leaves are.

cProfile misleads here. Its per-call overhead inflates cheap frequent functions, and it hides the CUDA
sync inside later calls such as .cpu(). So this uses manual timers on a subclass and leaves the search
code untouched.

  t_select          the PUCT argmax
  t_backup          pushing the value up and undoing virtual loss
  t_descend_other   the rest of a descent, leaf creation, board copy and terminal checks
  t_evaluate        the forward pass including the sync, since _evaluate ends on .cpu()

The leaf depth histogram shows whether collisions are structural. If nearly every leaf is a child of the
root, virtual loss cannot help. If leaves are spread deep, virtual loss is the right knob.

  python chess/time_mcts.py [POSITIONS] [VISITS] [BATCH]
"""
import os, sys, glob, random, time, collections
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
VISITS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 48
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
                if not b.is_game_over():
                    out.append(b.fen())
        if len(out) >= n:
            break
    return out[:n]


if __name__ == "__main__":
    import torch
    from mcts import MCTS
    from play_web import load_version

    clock = time.perf_counter

    class Timed(MCTS):
        """A subclass rather than a patch, so the search code itself is unchanged."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.t = collections.Counter()
            self.depths = collections.Counter()
            self.root_moves = []

        def _pick(self, *a, **k):
            # `_descend` calls `_pick`, not `_select`, so `_pick` is the one to time
            t0 = clock()
            try:
                return super()._pick(*a, **k)
            finally:
                self.t["select"] += clock() - t0

        def _backup(self, *a, **k):
            t0 = clock()
            try:
                return super()._backup(*a, **k)
            finally:
                self.t["backup"] += clock() - t0

        def _evaluate(self, nodes):
            t0 = clock()
            try:
                return super()._evaluate(nodes)
            finally:
                self.t["evaluate"] += clock() - t0

        def _descend(self, root):
            t0 = clock()
            got = super()._descend(root)
            self.t["descend_total"] += clock() - t0
            if got is not None:
                self.depths[len(got[0])] += 1
            return got

    fens = sample(N_POS)
    model, label, _ = load_version(E("VERSION", "mayfly_v1.pt"))
    dev = str(next(model.parameters()).device)
    print(f"{len(fens)} positions, {VISITS} visits, batch {BATCH}, {label} on {dev}", flush=True)

    total = collections.Counter()
    depths = collections.Counter()
    wall = 0.0
    n_root_moves = []
    for fen in fens:
        b = chess.Board(fen)
        n_root_moves.append(b.legal_moves.count())
        t = Timed(model, dev, batch=BATCH, cache_size=20000)
        t0 = clock()
        t.search(b, visits=VISITS, reuse=False)
        if dev != "cpu":
            torch.cuda.synchronize()
        wall += clock() - t0
        total.update(t.t)
        depths.update(t.depths)
        total["collisions"] += t.stats["collisions"]
        total["evals"] += t.stats["evals"]
        total["batches"] += t.stats["batches"]

    sel, bak = total["select"], total["backup"]
    dsc, ev = total["descend_total"], total["evaluate"]
    other = max(dsc - sel - bak, 0.0)          # backups from the cache-hit path sit outside descend
    unaccounted = max(wall - dsc - ev, 0.0)

    print("")
    print(f"  {wall:.2f}s of wall clock for {len(fens)} moves, "
          f"{wall/len(fens)*1000:.0f} ms a move, {len(fens)*VISITS/wall:,.0f} visits a second")
    print("")
    print(f"  {'t_evaluate':<18} {ev:>7.2f}s  {100*ev/wall:>5.1f}%   the forward pass and its sync")
    print(f"  {'t_select':<18} {sel:>7.2f}s  {100*sel/wall:>5.1f}%")
    print(f"  {'t_backup':<18} {bak:>7.2f}s  {100*bak/wall:>5.1f}%")
    print(f"  {'t_descend_other':<18} {other:>7.2f}s  {100*other/wall:>5.1f}%   "
          f"leaf creation, board copy, terminal checks")
    print(f"  {'unaccounted':<18} {unaccounted:>7.2f}s  {100*unaccounted/wall:>5.1f}%   "
          f"the search loop itself, expansion, cache")
    print(f"  python budget      {wall-ev:>7.2f}s  {100*(wall-ev)/wall:>5.1f}%")

    print("")
    got = sum(depths.values())
    print(f"  depth of the {got:,} leaves the batches collected "
          f"({total['collisions']:,} collisions, {total['evals']:,} evals, "
          f"{sum(n_root_moves)/len(n_root_moves):.1f} legal moves at the root)")
    run = 0
    for d in sorted(depths):
        run += depths[d]
        bar = "#" * max(1, round(40 * depths[d] / got))
        print(f"    ply {d:>3}  {depths[d]:>7,}  {100*depths[d]/got:>5.1f}%  "
              f"(cum {100*run/got:>5.1f}%)  {bar}")
