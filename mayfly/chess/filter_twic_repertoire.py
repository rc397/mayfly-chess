"""Filter the TWIC archive to a chosen opening repertoire and write its middlegame positions.

Keeps games where both players are rated at least MIN_ELO and the ECO code is in the repertoire, and
writes the unique positions from plies 8 to 80 for annotate_fens.py to label. Parsing is CPU-heavy.

    python chess/filter_twic_repertoire.py [MIN_ELO=2400] [CAP=20000000]
    output chess/repertoire_fens.txt, unique FENs one per line
"""
import os, sys, glob
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__))
TWIC = os.path.join(HERE, "twic")
OUT = os.path.join(HERE, "repertoire_fens.txt")
MIN_ELO = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000_000
PLY_LO, PLY_HI = 8, 80          # skip the first few book moves and stop near move 40

def in_repertoire(eco):
    if not eco or len(eco) < 3:
        return False
    L = eco[0]
    try:
        n = int(eco[1:3])
    except ValueError:
        return False
    return ((L == "A" and 10 <= n <= 39) or      # English / 1.c4
            (L == "B" and 10 <= n <= 19) or      # Caro-Kann
            (L == "C" and 0 <= n <= 19) or       # French
            (L == "D" and 10 <= n <= 69) or      # QGD, QGA, Slav
            (L == "E" and 0 <= n <= 19))         # Catalan (E00 to E09), QID and Bogo (E10 to E19)


seen = set(); games = kept = 0
out = open(OUT, "w", encoding="utf-8")
for fp in sorted(glob.glob(os.path.join(TWIC, "*.pgn"))):
    try:
        f = open(fp, encoding="utf-8", errors="ignore")
    except Exception:
        continue
    while len(seen) < CAP:
        try:
            game = chess.pgn.read_game(f)
        except Exception:
            continue
        if game is None:
            break
        games += 1
        h = game.headers
        if not in_repertoire(h.get("ECO", "")):
            continue
        try:
            if min(int(h.get("WhiteElo", "0")), int(h.get("BlackElo", "0"))) < MIN_ELO:
                continue
        except ValueError:
            continue
        kept += 1
        board = game.board()
        for ply, mv in enumerate(game.mainline_moves()):
            if PLY_LO <= ply <= PLY_HI:
                fen = board.fen()
                if fen not in seen:
                    seen.add(fen); out.write(fen + "\n")
            board.push(mv)
            if ply > PLY_HI:
                break
    f.close()
    if len(seen) >= CAP:
        break
    print(f"  scanned {games} games, {kept} in-repertoire, {len(seen)} unique positions", flush=True)
out.close()
print(f"DONE: {kept} repertoire games (>= {MIN_ELO}) -> {len(seen)} unique FENs -> {OUT}", flush=True)
