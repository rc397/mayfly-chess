"""Generate rich multi-task Stockfish labels, in the spirit of "Grandmaster-Level Chess Without Search".

For each position Stockfish gives the best move, the eval in centipawns and the top-K moves with their
evals (action values), and positional features are computed (material, mobility, king safety, pawn
structure, bishop pair). Openings come from Stockfish's own play with top-K sampling for variety, or
with --seed from repertoire.seed_lines_uci. Each worker runs a single-threaded Stockfish with a small
hash, writes JSONL, and resumes by counting its lines.

    python chess/gen_chess_rich.py [MAX_POS=5000000] [MOVETIME=0.10] [WORKERS=4] [TOPK=5] [--seed]
    output chess/rich/rich_<wid>.jsonl
"""
import os, sys, json, random, time
import chess, chess.engine
from multiprocessing import Process

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "rich")
os.makedirs(OUT, exist_ok=True)
SF = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SEED = "--seed" in sys.argv          # seed games from the repertoire (repertoire.seed_lines_uci)
MAX_POS  = int(ARGS[0]) if len(ARGS) > 0 else 5_000_000
MOVETIME = float(ARGS[1]) if len(ARGS) > 1 else 0.10
WORKERS  = int(ARGS[2]) if len(ARGS) > 2 else 4
TOPK     = int(ARGS[3]) if len(ARGS) > 3 else 5
MAX_PLIES = 100
SEEDS = []
if SEED:
    sys.path.insert(0, HERE)
    from repertoire import seed_lines_uci
    SEEDS = seed_lines_uci()
PV = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

def features(board):
    def material(c):
        return sum(PV[p.piece_type] for p in board.piece_map().values()
                   if p.color == c and p.piece_type != chess.KING)
    def king_safety(c):
        k = board.king(c)
        if k is None:
            return 0
        return sum(1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[k]) if board.is_attacked_by(not c, sq))
    def pawns(c):
        files = [0] * 8
        for sq in board.pieces(chess.PAWN, c):
            files[chess.square_file(sq)] += 1
        doubled = sum(n - 1 for n in files if n > 1)
        isolated = sum(n for f, n in enumerate(files) if n > 0
                       and (f == 0 or files[f - 1] == 0) and (f == 7 or files[f + 1] == 0))
        return [int(sum(files)), int(doubled), int(isolated)]
    return {
        "material": [material(chess.WHITE), material(chess.BLACK)],
        "stm_mobility": board.legal_moves.count(),
        "king_safety": [king_safety(chess.WHITE), king_safety(chess.BLACK)],
        "pawns": [pawns(chess.WHITE), pawns(chess.BLACK)],
        "bishop_pair": [int(len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2),
                        int(len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2)],
    }

def label(eng, board):
    info = eng.analyse(board, chess.engine.Limit(time=MOVETIME), multipv=TOPK)
    if isinstance(info, dict):
        info = [info]
    topk = []
    for pv in info:
        if pv.get("pv"):
            sc = pv["score"].pov(board.turn).score(mate_score=100000)
            topk.append([pv["pv"][0].uci(), int(sc if sc is not None else -99999)])
    if not topk:
        return None
    return {"fen": board.fen(), "best": topk[0][0], "cp": topk[0][1],
            "topk": topk, "feat": features(board)}

def worker(wid, quota, seed):
    rng = random.Random(seed)
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    try:
        eng.configure({"Threads": 1, "Hash": 64})
    except Exception:
        pass
    path = os.path.join(OUT, (f"rich_seed_{wid}.jsonl" if SEED else f"rich_{wid}.jsonl"))
    n = sum(1 for _ in open(path, encoding="utf-8")) if os.path.exists(path) else 0   # resume
    out = open(path, "a", encoding="utf-8")
    games = 0; t0 = time.time()
    while n < quota:
        board = chess.Board(); games += 1
        seedline = list(rng.choice(SEEDS)) if (SEED and SEEDS) else []; si = 0
        for ply in range(MAX_PLIES):
            if board.is_game_over():
                break
            rec = label(eng, board)
            if rec is None:
                break
            out.write(json.dumps(rec) + "\n"); n += 1
            if si < len(seedline):                       # follow the seeded opening line

                mv = chess.Move.from_uci(seedline[si]); si += 1
                if mv not in board.legal_moves:
                    mv = chess.Move.from_uci(rec["best"])
            else:
                moves = [m for m, _ in rec["topk"]]
                if ply < 8 or rng.random() < 0.15:       # exploration for position diversity
                    mv = chess.Move.from_uci(rng.choice(moves))
                else:
                    mv = chess.Move.from_uci(rec["best"])
            board.push(mv)
            if n >= quota:
                break
        if games % 25 == 0:
            out.flush()
            print(f"  [w{wid}] {n}/{quota} pos | {games} games | {n/max(1,time.time()-t0):.1f} pos/s", flush=True)
    out.close(); eng.quit()
    print(f"  [w{wid}] DONE {n} positions", flush=True)

if __name__ == "__main__":
    per = max(1, MAX_POS // WORKERS)
    print(f"gen RICH: {MAX_POS} pos | {WORKERS} workers x {per} | movetime {MOVETIME}s | topk {TOPK}", flush=True)
    procs = [Process(target=worker, args=(w, per, 7000 + w)) for w in range(WORKERS)]
    for p in procs: p.start()
    for p in procs: p.join()
    print("gen RICH DONE -> chess/rich/rich_*.jsonl", flush=True)
