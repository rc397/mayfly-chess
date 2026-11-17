r"""How many leaves a search batch should hold on this GPU, and where a batch's time goes.

  python profile_batch.py            env VERSION (mayfly_v4.pt), VISITS (1600), BATCHES (16,24,32,48,64)

For each batch size the version's sealed search runs the same positions at a fixed visit count. The
table shows visits a second, the share of the search inside the network call (the rest is selection,
expansion and backup on the CPU, while the GPU waits), and the moves chosen, so a batch size that
changes moves is visible. On a shared GPU the absolute numbers vary, the comparison between sizes is
what matters.
"""
import os
import sys
import time

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from conversion_suite import load  # noqa: E402
from mcts import MCTS, pinned_search  # noqa: E402

E = os.environ.get
FENS = ["r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 10",
        "r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R2Q1RK1 w - - 0 12",
        "2r3k1/5ppp/p3p3/1p1bP3/3P4/P2B4/1P3PPP/2R3K1 b - - 1 28",
        "r1b2rk1/pp2qppp/2n1pn2/3p4/2PP4/P1N1PN2/1P3PPP/R2QKB1R w KQ - 1 10",
        "8/5k2/8/3R4/8/2K5/8/6r1 w - - 30 60"]


def main():
    version = E("VERSION", "mayfly_v4.pt")
    visits = int(E("VISITS", 1600))
    sizes = [int(x) for x in E("BATCHES", "16,24,32,48,64").split(",")]
    model = load(version)
    print(f"{version}, {visits} visits a position, {len(FENS)} positions, batch sizes {sizes}", flush=True)
    print(f"{'batch':>6} {'visits/s':>9} {'network share':>14} {'ms a batch':>11}  moves", flush=True)
    for bsz in sizes + sizes[:1]:                          # the first size again, as a warm repeat
        tree = MCTS(model, device="cuda", visits=visits, batch=bsz)
        tree.apply(pinned_search(version))
        tree.open_temp = 0.0
        inner = tree._evaluate
        spent = [0.0, 0]

        def timed(nodes, _inner=inner, _s=spent):
            t = time.perf_counter()
            r = _inner(nodes)
            _s[0] += time.perf_counter() - t
            _s[1] += 1
            return r

        tree._evaluate = timed
        moves, total, t0 = [], 0, time.perf_counter()
        for f in FENS:
            tree.reset() if hasattr(tree, "reset") else None
            mv, info = tree.search(chess.Board(f), visits=visits, reuse=False)
            moves.append(mv.uci())
            total += int(info.get("visits", visits))
        el = time.perf_counter() - t0
        print(f"{bsz:>6} {total / el:>9.0f} {100 * spent[0] / el:>13.0f}% "
              f"{1000 * spent[0] / max(spent[1], 1):>10.1f}  {' '.join(moves)}", flush=True)


if __name__ == "__main__":
    main()
