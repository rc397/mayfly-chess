"""Label real tournament positions from TWIC with Stockfish, in gen_chess_rich.py's record format.

A corpus from Stockfish's own play never shows the model the positions humans actually reach. The Week
in Chess archives hold millions of titled classical games. At 5 ms a position the labels are much
shallower than earlier corpora, so they go to their own folder, which keeps shallow-but-many against
deep-but-few a measurable comparison.

    python chess/gen_gm_rich.py [MAX_POS] [MOVETIME] [WORKERS] [TOPK]
    python chess/gen_gm_rich.py 100000000 0.005 20 5
    env SRC=twic, OUT=rich5, MIN_ELO=2200, SKIP_PLIES=8, STRIDE=2

SKIP_PLIES drops the opening moves, which repeat across thousands of games. STRIDE labels every Nth
position, for more variety per unit of Stockfish time.
"""
import os, sys, glob, json, random, time
import chess, chess.engine, chess.pgn
from multiprocessing import Process

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
SRC = os.path.join(HERE, E("SRC", "twic"))
OUT = os.path.join(HERE, E("OUT", "rich5"))
MIN_ELO = int(E("MIN_ELO", 2200))
SKIP_PLIES = int(E("SKIP_PLIES", 8))
STRIDE = int(E("STRIDE", 2))
os.makedirs(OUT, exist_ok=True)
SF = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()

A = sys.argv[1:]
MAX_POS = int(A[0]) if len(A) > 0 else 100_000_000
MOVETIME = float(A[1]) if len(A) > 1 else 0.005
WORKERS = int(A[2]) if len(A) > 2 else 20
TOPK = int(A[3]) if len(A) > 3 else 5
PV = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def features(board):
    def material(c):
        return sum(PV[p.piece_type] for p in board.piece_map().values()
                   if p.color == c and p.piece_type != chess.KING)

    def king_safety(c):
        k = board.king(c)
        if k is None:
            return 0
        return sum(1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[k])
                   if board.is_attacked_by(not c, sq))

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


def strong_enough(game):
    """Titled-level classical only. Missing ratings are rejected rather than assumed."""
    h = game.headers
    try:
        w, b = int(h.get("WhiteElo", 0) or 0), int(h.get("BlackElo", 0) or 0)
    except ValueError:
        return False
    if min(w, b) < MIN_ELO:
        return False
    # blitz and rapid events leak into these archives, so drop those that say so

    ev = (h.get("Event", "") + " " + h.get("TimeControl", "")).lower()
    return not any(k in ev for k in ("blitz", "rapid", "bullet", "armageddon"))


def worker(wid, files, quota):
    """One Stockfish, single threaded, its own shard, appends so it can resume."""
    out_p = os.path.join(OUT, f"gm5_{wid}.jsonl")
    done = 0
    if os.path.exists(out_p):
        with open(out_p, encoding="utf-8", errors="ignore") as f:
            done = sum(1 for _ in f)
    if done >= quota:
        print(f"[w{wid}] already has {done:,}", flush=True)
        return
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    try:
        eng.configure({"Threads": 1, "Hash": 32})
    except Exception:
        pass
    n, skipped, t0 = done, 0, time.time()
    try:
        with open(out_p, "a", encoding="utf-8") as out:
            for fp in files:
                if n >= quota:
                    break
                with open(fp, encoding="utf-8", errors="ignore") as fh:
                    while n < quota:
                        try:
                            g = chess.pgn.read_game(fh)
                        except Exception:
                            continue
                        if g is None:
                            break
                        if not strong_enough(g):
                            skipped += 1
                            continue
                        board = g.board()
                        for ply, mv in enumerate(g.mainline_moves()):
                            if n >= quota:
                                break
                            if ply >= SKIP_PLIES and ply % STRIDE == 0 \
                                    and not board.is_game_over():
                                rec = label(eng, board)
                                if rec:
                                    out.write(json.dumps(rec) + "\n")
                                    n += 1
                                    if n % 20000 == 0:
                                        out.flush()
                                        rate = (n - done) / max(time.time() - t0, 1)
                                        print(f"[w{wid}] {n:,}  {rate:.0f}/s  "
                                              f"{skipped:,} games skipped", flush=True)
                            board.push(mv)
    finally:
        eng.quit()
    print(f"[w{wid}] done, {n:,} positions", flush=True)


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(SRC, "*.pgn")))
    if not files:
        raise SystemExit(f"no pgn in {SRC}")
    random.Random(0).shuffle(files)
    quota = MAX_POS // WORKERS
    print(f"{len(files)} archives in {os.path.basename(SRC)}, min Elo {MIN_ELO}, "
          f"every {STRIDE} plies after ply {SKIP_PLIES}")
    print(f"{WORKERS} workers x {quota:,} positions at {MOVETIME*1000:.0f}ms -> "
          f"{os.path.basename(OUT)}/")
    print(f"expect about {MAX_POS / (WORKERS * (1 / MOVETIME) * 0.85) / 3600:.1f} hours\n",
          flush=True)

    # each worker gets its own slice of archives, so they never read the same game
    procs = []
    for w in range(WORKERS):
        mine = files[w::WORKERS]
        p = Process(target=worker, args=(w, mine, quota), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    total = sum(sum(1 for _ in open(f, encoding="utf-8", errors="ignore"))
                for f in glob.glob(os.path.join(OUT, "*.jsonl")))
    print(f"\n{total:,} positions in {os.path.basename(OUT)}/")
