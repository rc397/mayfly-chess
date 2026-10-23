"""Generate position and Stockfish best-move pairs for distillation.

Games are played with varied openings and occasional non-best moves, so the positions span the whole
game, while the label is always Stockfish's best move at a fixed depth. Appends to chess/chess_data.txt,
one "FEN<tab>UCI" pair per line, so a re-run resumes.

    python chess/gen_chess_data.py [TARGET=300000] [DEPTH=11]
"""
import os, sys, random, time, chess, chess.engine
random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "chess_data.txt")
SF_PATH = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 11
MAXPLY = 80

eng = chess.engine.SimpleEngine.popen_uci(SF_PATH)
eng.configure({"Threads": 4, "Hash": 256})

n = 0
if os.path.exists(OUT):
    n = sum(1 for _ in open(OUT, encoding="utf-8"))
    print(f"resume: {n} existing pairs", flush=True)
fh = open(OUT, "a", encoding="utf-8")
t0 = time.time()
while n < TARGET:
    board = chess.Board()
    for ply in range(MAXPLY):
        if board.is_game_over():
            break
        try:
            infos = eng.analyse(board, chess.engine.Limit(depth=DEPTH), multipv=4)
        except Exception:
            break
        moves = [i["pv"][0] for i in infos if i.get("pv")]
        if not moves:
            break
        best = moves[0]
        fh.write(f"{board.fen()}\t{best.uci()}\n")          # the label is Stockfish's best move
        n += 1
        if n % 500 == 0:
            print(f"  {n}/{TARGET}  ({n/max(1,time.time()-t0):.1f} pos/s)", flush=True)
            fh.flush()
        if n >= TARGET:
            break
        # pick the move to play, varied so positions differ but still reasonable

        if ply < 10:
            nxt = random.choice(moves)                       # varied openings
        else:
            nxt = best if random.random() < 0.6 else random.choice(moves)
        board.push(nxt)
fh.close()
eng.quit()
print(f"DONE: {n} pairs -> {OUT}", flush=True)
