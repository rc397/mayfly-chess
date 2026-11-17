"""Generate positions and labels with Stockfish self-play into packed records (sp shards).

Human games give far fewer positions than a large model wants, so positions are made. Each position
gets one 30 ms MultiPV 5 search, used both as the label and to choose the move, so generation costs
the same as labelling. Variety comes from four things.
    seeding    most games start from a random real position from the human corpus
    sampling   the move is drawn from the top five weighted by exp(cp / T)
    schedule   T starts high and decays, so openings diverge and endgames are played properly
    breadth    OPEN_RANDOM of games open from the start position with every legal move considered,
               weighted by its evaluation, so rare but sound openings are covered too
Sampling never picks a move more than MAX_LOSS centipawns below the best, so variety never comes from
blunders.

    python chess/gen_selfplay.py [TARGET] [MOVETIME] [WORKERS]
    python chess/gen_selfplay.py 5000000000 0.030 24
    env OUT (shard folder), SEEDS=twic, T_HI=120, T_LO=15, T_PLIES=24, MAX_LOSS=90,
        MAX_PLIES=200, STRIDE=1, PREFIX=sp, WID_OFFSET=0
"""
import os, sys, glob, time, random
import numpy as np
import chess, chess.engine, chess.pgn
from multiprocessing import Process
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import board_to_ids, move_to_idx
from pack import DT, REC, encode_record
from gen_packed import features, strong_enough, FLUSH, FSYNC_SECONDS
import paths

E = os.environ.get
OUT = E("OUT", paths.SHARDS)
SEEDS = os.path.join(HERE, E("SEEDS", "twic"))
T_HI, T_LO = float(E("T_HI", 120)), float(E("T_LO", 15))
T_PLIES = int(E("T_PLIES", 24))          # plies over which the temperature decays
MAX_LOSS = int(E("MAX_LOSS", 90))        # never sample a move this far below the best
MAX_PLIES = int(E("MAX_PLIES", 200))
STRIDE = int(E("STRIDE", 1))
SEED_SKIP = int(E("SEED_SKIP", 6))       # plies into a real game to start from, at least
# Real games follow popularity, so a model seeded only from them never sees rare first moves. This
# fraction of games starts from the initial position and spreads its first few plies widely.
OPEN_RANDOM = float(E("OPEN_RANDOM", 0.15))
OPEN_PLIES = int(E("OPEN_PLIES", 2))
# The wide opening is not uniform. It looks at every legal move and samples weighted by Stockfish's
# evaluation, with a temperature wide enough that an offbeat opening still gets real coverage.
OPEN_T = float(E("OPEN_T", 160))
OPEN_LOSS = int(E("OPEN_LOSS", 300))
OPEN_MULTIPV = int(E("OPEN_MULTIPV", 20))
SF = E("SF") or open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
# When several machines generate into one corpus, PREFIX keeps shard names apart and WID_OFFSET gives
# workers distinct random seeds.
PREFIX = E("PREFIX", "sp")
WID_OFFSET = int(E("WID_OFFSET", 0))
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


def temperature(ply):
    """High early so openings diverge, low later so the chess stays real."""
    if ply >= T_PLIES:
        return T_LO
    f = ply / max(T_PLIES, 1)
    return T_HI * (1 - f) + T_LO * f


def pick(moves, cps, ply, rng, t=None, cap=None):
    """Sample among the moves, weighted by evaluation, never below the cap.

    A good line gets traffic in proportion to how good it is and a losing one gets none. Passing t
    and cap overrides the ply schedule, which the wide opening uses to spread further.
    """
    cap = MAX_LOSS if cap is None else cap
    best = max(cps)
    ok = [i for i, c in enumerate(cps) if best - c <= cap]
    if len(ok) == 1:
        return moves[ok[0]]
    t = temperature(ply) if t is None else t
    w = [pow(2.718281828, (cps[i] - best) / t) for i in ok]
    tot = sum(w)
    r = rng.random() * tot
    for i, wi in zip(ok, w):
        r -= wi
        if r <= 0:
            return moves[i]
    return moves[ok[-1]]


def seed_positions(files, rng, want=4000):
    """Real positions to start games from, so the opening distribution is human."""
    out = []
    for fp in files:
        try:
            fh = open(fp, encoding="utf-8", errors="ignore")
        except Exception:
            continue
        while len(out) < want:
            try:
                g = chess.pgn.read_game(fh)
            except Exception:
                continue
            if g is None:
                break
            if not strong_enough(g):
                continue
            mv = list(g.mainline_moves())
            if len(mv) < SEED_SKIP + 4:
                continue
            b = g.board()
            cut = rng.randrange(SEED_SKIP, min(len(mv) - 2, 40))
            for m in mv[:cut]:
                b.push(m)
            if not b.is_game_over():
                out.append(b.fen())
        fh.close()
        if len(out) >= want:
            break
    return out


def worker(wid, files, quota):
    path = os.path.join(OUT, f"{PREFIX}_{wid:02d}.bin")
    have = os.path.getsize(path) // REC if os.path.exists(path) else 0
    if have >= quota:
        print(f"[w{wid}] complete, {have:,}", flush=True)
        return
    if os.path.exists(path) and os.path.getsize(path) % REC:
        with open(path, "r+b") as f:
            f.truncate(have * REC)

    rng = random.Random(1000 + wid + WID_OFFSET * 977)
    seeds = seed_positions(files, rng)
    if not seeds:
        print(f"[w{wid}] no seed positions", flush=True)
        return
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    try:
        eng.configure({"Threads": 1, "Hash": 32})
    except Exception:
        pass

    buf = np.zeros(FLUSH, DT)
    k, n, t0, last_sync, games = 0, have, time.time(), time.time(), 0
    dead, strikes, n_seen = False, 0, n
    try:
        with open(path, "ab", buffering=0) as out:
            while n < quota and not dead:
                wide = rng.random() < OPEN_RANDOM
                board = chess.Board() if wide else chess.Board(rng.choice(seeds))
                games += 1
                for ply in range(MAX_PLIES):
                    if n >= quota or board.is_game_over():
                        break
                    random_open = wide and ply < OPEN_PLIES
                    try:
                        # the wide opening needs to see every move, not just the best five
                        info = eng.analyse(board, chess.engine.Limit(time=MOVETIME),
                                           multipv=OPEN_MULTIPV if random_open else 5)
                    except Exception:
                        # Restart a dead Stockfish. If it will not start, or fails three times running
                        # with no record in between, the worker stops the normal way, writing its
                        # buffered records first.
                        strikes = strikes + 1 if n == n_seen else 1
                        n_seen = n
                        try:
                            eng.quit()
                        except Exception:
                            pass
                        if strikes >= 3:
                            print(f"[w{wid}] Stockfish failed three times running, stopping", flush=True)
                            dead = True
                            break
                        try:
                            eng = chess.engine.SimpleEngine.popen_uci(SF)
                            eng.configure({"Threads": 1, "Hash": 32})
                        except Exception as e:
                            print(f"[w{wid}] Stockfish will not start again ({type(e).__name__}: {e}), "
                                  f"stopping", flush=True)
                            dead = True
                        break
                    if isinstance(info, dict):
                        info = [info]
                    mvs, cps, idx = [], [], []
                    ids, cast, flip = board_to_ids(board)
                    for pv in info:
                        if not pv.get("pv"):
                            continue
                        sc = pv["score"].pov(board.turn).score(mate_score=100000)
                        mvs.append(pv["pv"][0])
                        cps.append(sc if sc is not None else -99999)
                        idx.append(move_to_idx(pv["pv"][0], flip))
                    if not mvs:
                        break
                    # A wide-opening position is still recorded with its true label, only the move
                    # played differs. Ply zero is skipped, since the start position is always the same.

                    if ply % STRIDE == 0 and not (random_open and ply == 0):
                        encode_record(ids, cast, idx, cps, cps[0], features(board, flip),
                                      ply=ply, aux_valid=True, out=buf[k])
                        k += 1; n += 1
                        if k == FLUSH:
                            out.write(buf.tobytes()); k = 0
                            now = time.time()
                            if now - last_sync >= FSYNC_SECONDS:
                                out.flush(); os.fsync(out.fileno())
                                last_sync = now
                                print(f"[w{wid}] {n:,}  "
                                      f"{(n-have)/max(now-t0,1):.0f}/s  {games:,} games",
                                      flush=True)
                    if random_open:
                        board.push(pick(mvs, cps, ply, rng, t=OPEN_T, cap=OPEN_LOSS))
                    else:
                        board.push(pick(mvs, cps, ply, rng))
            if k:
                out.write(buf[:k].tobytes())
            out.flush(); os.fsync(out.fileno())
    finally:
        try:
            eng.quit()
        except Exception:
            pass
    print(f"[w{wid}] done, {n:,} records from {games:,} games", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SEEDS, "*.pgn")))
    if not files:
        raise SystemExit(f"no seed pgn in {SEEDS}")
    random.Random(0).shuffle(files)
    quota = TARGET // WORKERS
    print(f"self-play from real seed positions, {len(files)} seed archives")
    print(f"temperature {T_HI} decaying to {T_LO} over {T_PLIES} plies, "
          f"never sampling {MAX_LOSS}cp below best")
    print(f"{OPEN_RANDOM*100:.0f}% of games open with {OPEN_PLIES} uniformly random plies, "
          f"so every opening gets covered")
    print(f"{WORKERS} workers x {quota:,} records at {MOVETIME*1000:.0f}ms -> {OUT}")
    print(f"{TARGET * REC / 1e9:.0f} GB when complete\n", flush=True)
    procs = []
    for w in range(WORKERS):
        p = Process(target=worker, args=(w, files[w::WORKERS], quota))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
