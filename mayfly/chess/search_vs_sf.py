"""
search_vs_sf plays SearchPlayer (alpha-beta on the value head) against Stockfish at a few Skill levels.

N games per level with alternating colours, using the opening book in repertoire.py when it is there.
Prints W/D/L and the score per level. The level where the score crosses 50% gives a rough Elo.

  python chess/search_vs_sf.py [N=4] [DEPTH=3] [LEVELS=0,2,4]
"""
import os, sys, torch, chess, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from search import SearchPlayer
try:
    from repertoire import book_move
except Exception:
    def book_move(b):
        return None

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 3
LEVELS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "2", "4"])]
device = "cuda" if torch.cuda.is_available() else "cpu"
SF = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
sp = SearchPlayer(device=device, depth=DEPTH, topk=5)
print(f"SearchPlayer depth {DEPTH} vs Stockfish skill {LEVELS}, {N} games each | {device}", flush=True)

eng = chess.engine.SimpleEngine.popen_uci(SF)
for lvl in LEVELS:
    eng.configure({"Skill Level": lvl})
    w = d = l = 0
    for g in range(N):
        board = chess.Board(); net_white = (g % 2 == 0)
        while not board.is_game_over():
            if (board.turn == chess.WHITE) == net_white:
                mv = book_move(board) or sp.best_move(board)
                if mv is None or mv not in board.legal_moves:
                    mv = sp.best_move(board)
                board.push(mv)
            else:
                board.push(eng.play(board, chess.engine.Limit(time=0.1)).move)
        r = board.result()
        if r == "1/2-1/2":
            d += 1
        elif (r == "1-0") == net_white:
            w += 1
        else:
            l += 1
        print(f"  skill {lvl} game {g+1}: {r} (net {'White' if net_white else 'Black'})", flush=True)
    print(f"SKILL {lvl}: {w}W {d}D {l}L  (score {(w + 0.5 * d) / N:.2f})", flush=True)
eng.quit()
print("DONE", flush=True)
