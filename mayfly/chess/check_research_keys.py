"""Check on the real network that the search keys tm_catchup, r50_damp and root_temp do what they say.

    python check_research_keys.py [version.pt]

1. Off is a no-op. v4's sealed search with tm_catchup 0, r50_damp 0 and root_temp 1 plays the same
   move and visit counts both times.
2. r50_damp shrinks a leaf's value toward the draw as the fifty-move counter climbs.
3. root_temp flattens the root's priors once, and a second search on the same root does not
   flatten them again.
4. tm_catchup stops a clock search early when the leader cannot be caught, and says so in stop.
"""
import os
import sys
import time

import chess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mcts import MCTS, pinned_search  # noqa: E402
import paths  # noqa: E402


def load(name):
    import torch
    from chess_net import ChessNet
    path = os.path.join(os.environ.get("VERSION_DIR", paths.under("versions")), name)
    ck = torch.load(path, map_location="cuda")
    c = ck.get("cfg", {})
    m = ChessNet(d=c.get("d", 1024), nhead=c.get("nhead", 16), layers=c.get("layers", 16),
                 dim_ff=int(c.get("ff_mult", 2.0) * c.get("d", 1024)),
                 qhead=bool(c.get("qhead", "qvalue.weight" in ck["model"])),
                 att=bool(c.get("att", "att.q.weight" in ck["model"])),
                 dist=int(c.get("dist", 64 if "vdist.weight" in ck["model"] else 0)),
                 mlh=bool(c.get("mlh", "mleft.0.weight" in ck["model"]))).cuda().eval()
    m.load_state_dict(ck["model"], strict=False)
    return m


def tree(model, name, **over):
    t = MCTS(model, device="cuda", visits=400)
    t.apply(pinned_search(name))
    for k, v in over.items():
        setattr(t, k, v)
    t.rng = np.random.default_rng(0)
    return t


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "mayfly_v4.pt"
    model = load(name)
    ok = True
    # 1. off is a no-op, the new keys at their sealed values give the same search twice
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    a = tree(model, name)
    mv_a, ia = a.search(chess.Board(fen), visits=400, reuse=False)
    b = tree(model, name)
    for k in ("tm_catchup", "r50_damp", "root_temp"):
        assert getattr(b, k) in (0.0, 1.0)
    mv_b, ib = b.search(chess.Board(fen), visits=400, reuse=False)
    same = mv_a == mv_b and ia["top"] == ib["top"]
    print(f"1. sealed v4 with the new keys off: {mv_a.uci()} both times, identical top-5: {same}")
    ok &= same
    # 2. r50 damping on a quiet rook ending with the counter at 80
    fen2 = "8/5k2/8/3R4/8/2K5/8/6r1 w - - 80 120"
    t0 = tree(model, name)
    t1 = tree(model, name, r50_damp=1.0)
    _, i0 = t0.search(chess.Board(fen2), visits=64, reuse=False)
    _, i1 = t1.search(chess.Board(fen2), visits=64, reuse=False)
    v0, v1 = t0.root.value, t1.root.value
    shrink = abs(v1) <= abs(v0) * (1 - 80 / 199) + 1e-4
    print(f"2. r50_damp 1.0 at counter 80: root value {v0:+.4f} -> {v1:+.4f}, shrunk by the rate: {shrink}")
    ok &= shrink
    # 3. root temperature applied once
    t2 = tree(model, name, root_temp=1.5)
    t2.search(chess.Board(fen), visits=64, reuse=False)
    p1 = t2.root.P.copy()
    t2.search(chess.Board(fen), visits=64, reuse=True)
    p2 = t2.root.P.copy()
    t3 = tree(model, name)
    t3.search(chess.Board(fen), visits=64, reuse=False)
    flatter = float(p1.max()) < float(t3.root.P.max())
    once = np.allclose(p1, p2)
    print(f"3. root_temp 1.5: top prior {t3.root.P.max():.3f} -> {p1.max():.3f} (flatter: {flatter}), "
          f"unchanged by a second search on the same root: {once}")
    ok &= flatter and once
    # 4. catch-up stop in an obvious recapture under a 6 s budget
    fen4 = "rnb1kbnr/pppp1ppp/8/4p3/4P1q1/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"   # unused, overwritten below
    fen4 = "rnbqkbnr/ppp2ppp/8/3pp3/4P3/5Q2/PPPP1PPP/RNB1KBNR w KQkq - 0 3"
    b4 = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 3")
    b4.push_uci("e5d4")                                    # white now recaptures on d4
    t4 = tree(model, name, tm_catchup=0.4, tm_extend=False, tm_stop=False)
    s0 = time.time()
    _, i4 = t4.search(b4, visits=10 ** 7, seconds=6.0, reuse=False)
    took = time.time() - s0
    print(f"4. tm_catchup 0.4 on a recapture with 6 s: stopped after {took:.2f}s, stop = {i4.get('stop')!r}, "
          f"{i4['visits']} visits, top {i4['top'][:2]}")
    t5 = tree(model, name, tm_extend=False, tm_stop=False)
    _, i5 = t5.search(b4, visits=10 ** 7, seconds=2.0, reuse=False)
    print(f"   without it, 2 s: stop = {i5.get('stop')!r}")
    ok &= i4.get("stop") == "catchup" and took < 6.0 and i5.get("stop") == "time"
    print("ALL PASS" if ok else "SOMETHING FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
