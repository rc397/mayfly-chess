"""
Turn the Lichess Elite PGNs (2200+ rated human games) into human-style training records.

For each position, the move a strong human played, the rating of the side to move and the game
result, a Maia-style signal kept separate from Stockfish's action values. No engine, a fast CPU-only
parse.

  python chess/parse_lichess_human.py [MAX_GAMES_PER_FILE=500000] [SAMPLE=1.0]
Input:  chess/lichess_pgn/lichess_elite_*.pgn
Output: chess/human/human_<file>.jsonl   (fen, move, elo, result)
"""
import os, sys, glob, json, random
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__))
PGN_DIR = os.path.join(HERE, "lichess_pgn")
OUT = os.path.join(HERE, "human"); os.makedirs(OUT, exist_ok=True)
MAXG = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
SAMPLE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

def parse_file(pgn_path):
    base = os.path.splitext(os.path.basename(pgn_path))[0]
    outp = os.path.join(OUT, f"human_{base}.jsonl")
    rng = random.Random(1234)
    nG = nP = 0
    with open(pgn_path, encoding="utf-8", errors="ignore") as f, open(outp, "w", encoding="utf-8") as out:
        while nG < MAXG:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                continue
            if game is None:
                break
            h = game.headers
            try:
                we, be = int(h.get("WhiteElo", "0")), int(h.get("BlackElo", "0"))
            except ValueError:
                we = be = 0
            res = h.get("Result", "*")
            r_w = 1.0 if res == "1-0" else (0.0 if res == "0-1" else 0.5)
            board = game.board()
            for mv in game.mainline_moves():
                stm = board.turn                                  # True = White to move
                elo = we if stm == chess.WHITE else be
                r_stm = r_w if stm == chess.WHITE else (1.0 - r_w)
                if elo > 0 and (SAMPLE >= 1.0 or rng.random() < SAMPLE):
                    out.write(json.dumps({"fen": board.fen(), "move": mv.uci(),
                                          "elo": elo, "result": r_stm}) + "\n"); nP += 1
                board.push(mv)
            nG += 1
            if nG % 5000 == 0:
                out.flush()
                print(f"  [{base}] {nG} games, {nP} positions", flush=True)
    print(f"  [{base}] DONE {nG} games, {nP} positions -> {outp}", flush=True)

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
    print(f"parsing {len(files)} Elite PGNs (max {MAXG} games/file, sample {SAMPLE})", flush=True)
    for fp in files:
        parse_file(fp)
    print("human parse DONE -> chess/human/*.jsonl", flush=True)
