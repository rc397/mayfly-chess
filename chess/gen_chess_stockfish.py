"""Generate Stockfish move labels seeded from an opening repertoire, with a defensive move bias.

Openings come from repertoire.py. Among Stockfish's near-best moves the quiet one is preferred for
"our" side. Only our side's decisions are recorded, with the colour alternating across games, by
parallel Stockfish workers.

    python chess/gen_chess_stockfish.py [MAX_POS=800000] [MOVETIME=0.08] [WORKERS=6] [DEF_MARGIN=30]
    output chess/chess_data_sf_<wid>.txt (FEN<tab>UCI), which train_chess.py reads
"""
import os, sys, random, time
import chess, chess.engine
from multiprocessing import Process

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from repertoire import book_move

SF = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
MAX_POS    = int(sys.argv[1]) if len(sys.argv) > 1 else 800_000
MOVETIME   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
WORKERS    = int(sys.argv[3]) if len(sys.argv) > 3 else 6
DEF_MARGIN = int(sys.argv[4]) if len(sys.argv) > 4 else 30      # prefer a quiet move within this many cp of best
MAX_PLIES  = 80

def sf_move(eng, board, defensive):
    """Stockfish move. If defensive, prefer a quiet (non-capture, non-check) move within DEF_MARGIN cp of best."""
    try:
        info = eng.analyse(board, chess.engine.Limit(time=MOVETIME), multipv=(3 if defensive else 1))
    except Exception:
        return random.choice(list(board.legal_moves))
    if isinstance(info, dict):
        info = [info]
    cand = []
    for pv in info:
        if pv.get("pv"):
            sc = pv["score"].pov(board.turn).score(mate_score=100000)
            cand.append((pv["pv"][0], sc if sc is not None else -99999))
    if not cand:
        return random.choice(list(board.legal_moves))
    best_mv, best_sc = cand[0]
    if defensive:
        quiet = [mv for mv, sc in cand if best_sc - sc <= DEF_MARGIN
                 and not board.is_capture(mv) and not board.gives_check(mv)]
        if quiet:
            return quiet[0]
    return best_mv

def worker(wid, quota, seed):
    rng = random.Random(seed)
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    out = open(os.path.join(HERE, f"chess_data_sf_{wid}.txt"), "w", encoding="utf-8")
    n = games = 0; t0 = time.time()
    while n < quota:
        board = chess.Board(); our = rng.choice([chess.WHITE, chess.BLACK]); games += 1
        for _ in range(MAX_PLIES):
            if board.is_game_over():
                break
            if board.turn == our:                       # our move, book opening, else defensive Stockfish
                mv = book_move(board, rng) or sf_move(eng, board, defensive=True)
                out.write(f"{board.fen()}\t{mv.uci()}\n"); n += 1
            else:                                        # opponent, book transpositions sometimes, else Stockfish

                bm = book_move(board, rng)
                mv = bm if (bm and rng.random() < 0.5) else sf_move(eng, board, defensive=False)
            board.push(mv)
            if n >= quota:
                break
        if games % 50 == 0:
            print(f"  [w{wid}] {n}/{quota} pos | {games} games | {n/max(1,time.time()-t0):.1f} pos/s", flush=True)
    out.close(); eng.quit()
    print(f"  [w{wid}] DONE {n} positions", flush=True)

if __name__ == "__main__":
    per = max(1, MAX_POS // WORKERS)
    print(f"gen SF data: {MAX_POS} pos | {WORKERS} workers x {per} | movetime {MOVETIME}s | def_margin {DEF_MARGIN}cp", flush=True)
    procs = [Process(target=worker, args=(w, per, 1000 + w)) for w in range(WORKERS)]
    for p in procs: p.start()
    for p in procs: p.join()
    print("gen DONE -> chess/chess_data_sf_*.txt", flush=True)
