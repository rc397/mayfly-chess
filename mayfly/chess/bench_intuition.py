"""Test whether the network's first move, with no search, deals with a hanging piece.

Takes held-out positions where passing would lose two points or more, asks each network for its top
moves in one forward pass, and counts a move as a save if the opponent's best capture afterwards is
worth less than two. Material is scored two plies deep (the capture minus the cheapest recapture),
which is approximate but the same for both architectures.

    python chess/bench_intuition.py [N_POSITIONS]
"""
import os, sys, json
import chess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from play_race import load, think
import train_chess_race as R

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
THRESHOLD = 2.0
V = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9,
     chess.KING: 0}


def best_capture_gain(board):
    """Material the side to move wins with its best single capture, allowing for one recapture."""
    best = 0.0
    for m in board.legal_moves:
        if not board.is_capture(m):
            continue
        victim = board.piece_at(m.to_square)
        taken = V[victim.piece_type] if victim else 1.0      # en passant takes a pawn
        attacker = board.piece_at(m.from_square)
        t = board.copy(); t.push(m)
        # cheapest recapture on that square, if there is one
        recap = min((V[t.piece_at(r.from_square).piece_type]
                     for r in t.legal_moves if r.to_square == m.to_square), default=None)
        gain = taken - (V[attacker.piece_type] if recap is not None else 0.0)
        best = max(best, gain)
    return best


def threat_if_i_do_nothing(board):
    """What the opponent wins if the side to move passes, using python-chess's null move."""
    if board.is_check():
        return None                      # in check the whole position is forcing, so skip it
    t = board.copy()
    t.push(chess.Move.null())
    return best_capture_gain(t)


if __name__ == "__main__":
    gen = R.stream(True, 99)
    models = {a: load(a) for a in ("trop", "ctrl")}
    for a, (m, nm, v) in models.items():
        print(f"  {nm:11s} {m.num_params()/1e6:.3f}M params, held-out policy loss {v:.4f}")

    tested = 0
    hits = {a: {"top1": 0, "top3": 0} for a in models}
    scanned = 0
    print(f"\nscanning held-out positions for something hanging by {THRESHOLD:g} or more",
          flush=True)

    while tested < N and scanned < N * 60:
        recs = [next(gen) for _ in range(32)]
        for rec in recs:
            scanned += 1
            b = chess.Board(rec["fen"]) if "fen" in rec else None
            if b is None:
                continue
            before = threat_if_i_do_nothing(b)
            if before is None or before < THRESHOLD:
                continue
            tested += 1
            for a, (model, _, _) in models.items():
                moves, _ = think(model, b, topn=3)
                for rank, (mv, _) in enumerate(moves):
                    t = b.copy(); t.push(mv)
                    if best_capture_gain(t) < THRESHOLD:
                        if rank == 0:
                            hits[a]["top1"] += 1
                        hits[a]["top3"] += 1
                        break
            if tested >= N:
                break

    print(f"scanned {scanned} positions, {tested} had something hanging\n")
    print(f"{'model':14s} {'saves it with its top move':>28s} {'somewhere in its top 3':>25s}")
    print("-" * 70)
    out = {}
    for a, (_, nm, _) in models.items():
        t1 = hits[a]["top1"] / max(tested, 1)
        t3 = hits[a]["top3"] / max(tested, 1)
        out[a] = {"name": nm, "n": tested, "top1": t1, "top3": t3}
        print(f"{nm:14s} {t1*100:27.1f}% {t3*100:24.1f}%")
    json.dump(out, open(os.path.join(HERE, "bench_intuition.json"), "w"), indent=1)
    print("\nNo search was used. This is the move the network offers first.")
