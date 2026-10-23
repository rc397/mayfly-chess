"""Label tournament positions from TWIC with Stockfish straight into packed 112-byte records (gm shards).

Packed records are about a third the size of JSONL and skip the JSON parse and board rebuild at
training time. 30 ms a position was chosen because Stockfish at 5 ms measured 424 Elo weaker than at
50 ms, while at 30 ms it was only 123 Elo weaker, above where the data can take the model.

Each worker keeps its own Stockfish and shard, appends, and counts existing records on start, so a
kill and restart loses at most one record.

    python chess/gen_packed.py [TARGET] [MOVETIME] [WORKERS]
    python chess/gen_packed.py 5000000000 0.030 24
    env SRC=twic, OUT (shard folder), MIN_ELO=2200, SKIP_PLIES=8, STRIDE=2, SF (Stockfish path)
"""
import os, sys, glob, time, random
import numpy as np
import chess, chess.engine, chess.pgn
from multiprocessing import Process
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import board_to_ids, move_to_idx
from pack import DT, REC, encode_record
import paths

E = os.environ.get
SRC = os.path.join(HERE, E("SRC", "twic"))
OUT = E("OUT", paths.SHARDS)
MIN_ELO = int(E("MIN_ELO", 2200))
SKIP_PLIES = int(E("SKIP_PLIES", 8))
STRIDE = int(E("STRIDE", 2))
SF = E("SF") or open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
# argv is parsed at import time, so only numeric arguments are taken, and a script that imports
# this module for its helpers does not crash on its own arguments
def _num(v):
    try:
        float(v); return True
    except Exception:
        return False
A = [a for a in sys.argv[1:] if _num(a)]
TARGET = int(float(A[0])) if len(A) > 0 else 5_000_000_000
MOVETIME = float(A[1]) if len(A) > 1 else 0.030
WORKERS = int(A[2]) if len(A) > 2 else 24
# 512 records is 57 KB, written about every twenty seconds at 30 ms a position
FLUSH = 512
# an unbuffered write reaches the OS, not the disk, so fsync on a timer to survive a sudden reboot
FSYNC_SECONDS = 30
PV = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def features(board, flip):
    """Positional features, already swapped to the side-to-move view so training never flips."""
    def material(c):
        return sum(PV[p.piece_type] for p in board.piece_map().values()
                   if p.color == c and p.piece_type != chess.KING)

    def king_safety(c):
        k = board.king(c)
        return 0 if k is None else sum(
            1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[k]) if board.is_attacked_by(not c, sq))

    def pawns(c):
        f = [0] * 8
        for sq in board.pieces(chess.PAWN, c):
            f[chess.square_file(sq)] += 1
        doubled = sum(n - 1 for n in f if n > 1)
        isolated = sum(n for i, n in enumerate(f) if n > 0
                       and (i == 0 or f[i - 1] == 0) and (i == 7 or f[i + 1] == 0))
        return [sum(f), doubled, isolated]

    sw = (lambda a: [a[1], a[0]]) if flip else (lambda a: a)
    return {
        "material": sw([material(chess.WHITE), material(chess.BLACK)]),
        "stm_mobility": board.legal_moves.count(),
        "king_safety": sw([king_safety(chess.WHITE), king_safety(chess.BLACK)]),
        "pawns": sw([pawns(chess.WHITE), pawns(chess.BLACK)]),
        "bishop_pair": sw([int(len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2),
                           int(len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2)]),
    }


def strong_enough(game):
    h = game.headers
    try:
        w, b = int(h.get("WhiteElo", 0) or 0), int(h.get("BlackElo", 0) or 0)
    except ValueError:
        return False
    if min(w, b) < MIN_ELO:
        return False
    ev = (h.get("Event", "") + " " + h.get("TimeControl", "")).lower()
    return not any(k in ev for k in ("blitz", "rapid", "bullet", "armageddon"))


def worker(wid, files, quota):
    path = os.path.join(OUT, f"gm_{wid:02d}.bin")
    have = os.path.getsize(path) // REC if os.path.exists(path) else 0
    if have >= quota:
        print(f"[w{wid}] complete, {have:,}", flush=True)
        return
    # a partial record from a kill gets truncated rather than corrupting the shard
    if os.path.exists(path) and os.path.getsize(path) % REC:
        with open(path, "r+b") as f:
            f.truncate(have * REC)

    eng = chess.engine.SimpleEngine.popen_uci(SF)
    try:
        eng.configure({"Threads": 1, "Hash": 32})
    except Exception:
        pass
    buf = np.zeros(FLUSH, DT)
    k, n, t0, skipped = 0, have, time.time(), 0
    last_sync = time.time()
    try:
        with open(path, "ab", buffering=0) as out:
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
                            if ply >= SKIP_PLIES and ply % STRIDE == 0 and not board.is_game_over():
                                try:
                                    # no game= argument, so the hash table stays warm across a game,
                                    # which is several times faster and slightly deeper

                                    info = eng.analyse(board, chess.engine.Limit(time=MOVETIME),
                                                       multipv=5)
                                    if isinstance(info, dict):
                                        info = [info]
                                    ids, cast, flip = board_to_ids(board)
                                    idx, cps = [], []
                                    for pv in info:
                                        if not pv.get("pv"):
                                            continue
                                        sc = pv["score"].pov(board.turn).score(mate_score=100000)
                                        idx.append(move_to_idx(pv["pv"][0], flip))
                                        cps.append(sc if sc is not None else -99999)
                                    if idx:
                                        encode_record(ids, cast, idx, cps, cps[0],
                                                      features(board, flip),
                                                      ply=ply, aux_valid=True,
                                                      out=buf[k])
                                        k += 1; n += 1
                                        if k == FLUSH:
                                            out.write(buf.tobytes()); k = 0
                                            now = time.time()
                                            if now - last_sync >= FSYNC_SECONDS:
                                                out.flush(); os.fsync(out.fileno())
                                                last_sync = now
                                                r = (n - have) / max(now - t0, 1)
                                                print(f"[w{wid}] {n:,}  {r:.0f}/s  "
                                                      f"{skipped:,} games skipped", flush=True)
                                except Exception:
                                    pass
                            board.push(mv)
            if k:
                out.write(buf[:k].tobytes())
            out.flush(); os.fsync(out.fileno())
    finally:
        eng.quit()
    print(f"[w{wid}] done, {n:,} records", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.pgn")))
    if not files:
        raise SystemExit(f"no pgn in {SRC}")
    random.Random(0).shuffle(files)
    quota = TARGET // WORKERS
    print(f"{len(files)} archives, min Elo {MIN_ELO}, every {STRIDE} plies after ply {SKIP_PLIES}")
    print(f"{WORKERS} workers x {quota:,} records at {MOVETIME*1000:.0f}ms -> {OUT}")
    print(f"{TARGET * REC / 1e9:.0f} GB when complete", flush=True)

    procs = []
    for w in range(WORKERS):
        p = Process(target=worker, args=(w, files[w::WORKERS], quota))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    tot = sum(os.path.getsize(f) for f in glob.glob(os.path.join(OUT, "*.bin"))) // REC
    print(f"\n{tot:,} records, {tot * REC / 1e9:.1f} GB")
