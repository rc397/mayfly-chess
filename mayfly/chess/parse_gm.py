"""Parse master PGNs into human-style records for an experimental human head.

For each position, the move the featured player chose (the player the file is named for), tagged
with player, ECO, rating, result and source. Only the featured player's moves are kept, so the style
is theirs and not their opponents'. Files prefixed chesscom_ or lichess_ are tagged with that source.

  python chess/parse_gm.py
Input:  chess/gm_pgn/*.pgn    Output: chess/human/gm_<file>.jsonl
"""
import os, glob, json
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__))
GM = os.path.join(HERE, "gm_pgn"); OUT = os.path.join(HERE, "human"); os.makedirs(OUT, exist_ok=True)

def featured(fname):                       # "Capablanca.pgn" gives "capablanca", "lichess_X.pgn" gives "x"
    b = os.path.splitext(os.path.basename(fname))[0].lower()
    return b.replace("chesscom_", "").replace("lichess_", "")

def side_of(name, white, black):
    w, b = (white or "").lower(), (black or "").lower()
    if name in w: return chess.WHITE
    if name in b: return chess.BLACK
    return None

for fp in sorted(glob.glob(os.path.join(GM, "*.pgn"))):
    feat = featured(fp)
    bn = os.path.basename(fp).lower()
    src = "chesscom" if "chesscom_" in bn else ("lichess" if "lichess_" in bn else "gm")
    outp = os.path.join(OUT, f"gm_{os.path.splitext(os.path.basename(fp))[0]}.jsonl")
    nG = nP = 0
    with open(fp, encoding="utf-8", errors="ignore") as f, open(outp, "w", encoding="utf-8") as out:
        while True:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                continue
            if game is None:
                break
            h = game.headers
            fs = side_of(feat, h.get("White", ""), h.get("Black", ""))
            if fs is None:
                fs = "both"                 # a mixed collection, so keep both sides
            eco = h.get("ECO", "?"); res = h.get("Result", "*")
            r_w = 1.0 if res == "1-0" else (0.0 if res == "0-1" else 0.5)
            board = game.board()
            for mv in game.mainline_moves():
                stm = board.turn
                if fs == "both" or stm == fs:
                    try:
                        elo = int(h.get("WhiteElo" if stm == chess.WHITE else "BlackElo", "0"))
                    except ValueError:
                        elo = 0
                    r_stm = r_w if stm == chess.WHITE else 1.0 - r_w
                    out.write(json.dumps({"fen": board.fen(), "move": mv.uci(), "elo": elo,
                                          "eco": eco, "player": feat, "source": src,
                                          "result": r_stm}) + "\n"); nP += 1
                board.push(mv)
            nG += 1
            if nG % 2000 == 0:
                out.flush(); print(f"  [{feat}] {nG} games {nP} pos", flush=True)
    print(f"  [{feat}] DONE {nG} games {nP} pos -> {outp}", flush=True)
print("GM parse DONE -> chess/human/gm_*.jsonl", flush=True)
