"""Label repertoire positions with Stockfish in the same rich format as gen_chess_rich.

Reads the unique FENs that filter_twic_repertoire.py pulled from master games in a chosen set of
openings, and writes best move, eval, top-K action values and positional features for each. Runs
parallel single-threaded Stockfish workers. Each worker skips FENs it already labelled, so an
interrupted run resumes.

    python chess/annotate_fens.py [MOVETIME=0.20] [WORKERS=14] [TOPK=5] [CAP=0]
    input chess/repertoire_fens.txt, output chess/rich/rich_rep_<wid>.jsonl, CAP=0 means every FEN
"""
import os, sys, json, time
import chess, chess.engine
from multiprocessing import Process

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "rich"); os.makedirs(OUT, exist_ok=True)
FENS = os.path.join(HERE, "repertoire_fens.txt")
SF = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
MOVETIME = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
WORKERS  = int(sys.argv[2]) if len(sys.argv) > 2 else 14
TOPK     = int(sys.argv[3]) if len(sys.argv) > 3 else 5
CAP      = int(sys.argv[4]) if len(sys.argv) > 4 else 0          # 0 = annotate every FEN
PV = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

def features(board):                                            # identical to gen_chess_rich.features
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

def label(eng, board):                                          # identical rich record to gen_chess_rich
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

def worker(wid, fens):
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    try:
        eng.configure({"Threads": 1, "Hash": 64})
    except Exception:
        pass
    path = os.path.join(OUT, f"rich_rep_{wid}.jsonl")
    done = set()                                                # skip FENs already labelled, so a run resumes
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["fen"])
            except Exception:
                continue
    out = open(path, "a", encoding="utf-8")
    t0 = time.time(); n = 0; skipped = len(done)
    for fen in fens:
        if fen in done:
            continue
        try:
            board = chess.Board(fen)
            rec = label(eng, board)
        except Exception:
            rec = None
        if rec is not None:
            out.write(json.dumps(rec) + "\n")
        n += 1
        if n % 200 == 0:
            out.flush()
            print(f"  [w{wid}] {skipped+n}/{skipped+len(fens)} | {n/max(1,time.time()-t0):.1f} fen/s", flush=True)
    out.close(); eng.quit()
    print(f"  [w{wid}] DONE {n} new ({skipped} already done)", flush=True)

if __name__ == "__main__":
    if not os.path.exists(FENS):
        raise SystemExit(f"no {FENS} -- run filter_twic_repertoire.py first")
    fens = [l.strip() for l in open(FENS, encoding="utf-8") if l.strip()]
    if CAP > 0 and len(fens) > CAP:
        step = max(1, len(fens) // CAP)
        fens = fens[::step][:CAP]                              # spread the sample across all games/years, not just the earliest
    print(f"annotate RICH (repertoire): {len(fens)} FENs | {WORKERS} workers | movetime {MOVETIME}s | topk {TOPK}", flush=True)
    procs = [Process(target=worker, args=(w, fens[w::WORKERS])) for w in range(WORKERS)]   # strided partition
    for p in procs: p.start()
    for p in procs: p.join()
    print("annotate RICH DONE -> chess/rich/rich_rep_*.jsonl", flush=True)
