r"""sf_steps measures the Elo of ladder rungs past Skill 20 by playing Stockfish against itself.

Each rung past 20 is full-strength Stockfish with sqrt(2) times the previous rung's budget (21 = 141 ms,
22 = 200 ms, 23 = 283 ms at the ladder's 100 ms base). Games are played at fixed nodes rather than time,
so a busy CPU makes them slower but never different. BASE_NODES is the median node count of the ladder's
Stockfish at 100 ms with two threads and 256 MB of hash (elo_vs_sf.py) on the reference machine.

Each pair is one rung against the next, plus Skill 19 against Skill 20 for the size of one Skill level.
Each pair plays OPENINGS balanced TWIC positions (8 to 14 plies in, within 50 centipawns), once from each
side. Results go one JSON line a game into OUT/games.jsonl, so a rerun carries on and a failed game is
retried next run. `--report` prints each pair's score and Elo difference with a 95% interval.

  python chess/sf_steps.py [--report]
  env: SF (default stockfish on PATH)  BASE_NODES=236276  SF_THREADS=2  SF_HASH=256  OPENINGS=200  WORKERS=4
       OUT (default paths.under("sf_steps"))  PGN (default paths.under("twic"))
"""
import glob
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import chess
import chess.engine
import chess.pgn

import paths

E = os.environ.get
SF = E("SF", "stockfish")
BASE = int(E("BASE_NODES", 236276))
THREADS, HASH = int(E("SF_THREADS", 2)), int(E("SF_HASH", 256))     # as elo_vs_sf.py plays the ladder
OPENINGS = int(E("OPENINGS", 200))
WORKERS = int(E("WORKERS", 4))
OUT = E("OUT", paths.under("sf_steps"))
PGN = E("PGN", paths.under("twic"))
MAX_PLIES = 300
R2 = math.sqrt(2)
# (name, (skill, nodes) of the first side, (skill, nodes) of the second). The second should be the stronger.
PAIRS = [("skill19-vs-20", (19, BASE), (20, BASE)),
         ("rung20-vs-21", (20, BASE), (20, round(BASE * R2))),
         ("rung21-vs-22", (20, round(BASE * R2)), (20, BASE * 2)),
         ("rung22-vs-23", (20, BASE * 2), (20, round(BASE * 2 * R2)))]
GAMES = os.path.join(OUT, "games.jsonl")
BOOK = os.path.join(OUT, "openings.json")


def make_openings():
    """OPENINGS balanced positions from the TWIC games, chosen once and kept in OUT\\openings.json."""
    if os.path.exists(BOOK):
        return json.load(open(BOOK, encoding="utf-8"))
    rnd = random.Random(20261008)
    files = sorted(glob.glob(os.path.join(PGN, "*.pgn")))
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    seen, book = set(), []
    while len(book) < OPENINGS:
        with open(rnd.choice(files), encoding="utf-8", errors="replace") as fh:
            fh.seek(rnd.randrange(max(1, os.path.getsize(fh.name) - 20000)))
            fh.readline()
            game = None
            for _ in range(50):                 # skip to the start of the next game
                game = chess.pgn.read_game(fh)
                if game is None or len(list(game.mainline_moves())) >= 20:
                    break
        if game is None:
            continue
        board, plies = game.board(), rnd.randint(8, 14)
        for k, mv in enumerate(game.mainline_moves()):
            if k == plies:
                break
            board.push(mv)
        key = " ".join(board.fen().split()[:4])
        if board.is_game_over() or key in seen:
            continue
        info = eng.analyse(board, chess.engine.Limit(nodes=200_000))
        cp = info["score"].white().score(mate_score=100_000)
        if abs(cp) <= 50:
            seen.add(key)
            book.append(board.fen())
    eng.quit()
    os.makedirs(OUT, exist_ok=True)
    json.dump(book, open(BOOK, "w", encoding="utf-8"), indent=0)
    return book


def shut(eng):
    """Quit an engine, killing one that has not gone within python-chess's 10 s for a quit. On a busy CPU an
    engine can take longer than that to exit, and the TimeoutError would stop the whole run."""
    try:
        eng.quit()
    except Exception:
        eng.close()                             # closes the pipes and kills the process


def play(task):
    """play_game, with a failed game returned as an error line. main() leaves it off the file, so the run
    carries on and the next run plays that game again."""
    try:
        return play_game(task)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "pair": task[0], "k": task[1], "second_white": task[2]}


def play_game(task):
    """One game, from (pair name, opening index, True when the second side has White) to its result line. A
    fresh engine pair each game, so the hash never carries one game's search into the next."""
    name, k, second_white, fen = task
    _, a, b = next(p for p in PAIRS if p[0] == name)
    ea = eb = None
    try:
        ea = chess.engine.SimpleEngine.popen_uci(SF)
        eb = chess.engine.SimpleEngine.popen_uci(SF)
        ea.configure({"Threads": THREADS, "Hash": HASH, "Skill Level": a[0]})
        eb.configure({"Threads": THREADS, "Hash": HASH, "Skill Level": b[0]})
        board = chess.Board(fen)
        side_b = chess.WHITE if second_white else chess.BLACK
        while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
            e, nodes = (eb, b[1]) if board.turn == side_b else (ea, a[1])
            board.push(e.play(board, chess.engine.Limit(nodes=nodes)).move)
        res = board.result(claim_draw=True)
        s_white = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
        s_b = s_white if second_white else 1.0 - s_white
        return {"pair": name, "k": k, "second_white": second_white, "s_second": s_b, "plies": board.ply(),
                "result": res, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    finally:
        for eng in (ea, eb):
            if eng is not None:
                shut(eng)


def done_games():
    got = {}
    if os.path.exists(GAMES):
        for line in open(GAMES, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue                        # the torn last line of a stopped run
            got[(r["pair"], r["k"], r["second_white"])] = r
    return got


def report():
    games = done_games()
    print(f"Stockfish against itself at nodes ({BASE:,} = 100 ms of the ladder's Stockfish, "
          f"{THREADS} threads, {HASH} MB hash)")
    for name, a, b in PAIRS:
        rs = [r for (p, _, _), r in games.items() if p == name]
        if not rs:
            print(f"  {name:15s} no games yet")
            continue
        n = len(rs)
        w = sum(r["s_second"] == 1.0 for r in rs)
        d = sum(r["s_second"] == 0.5 for r in rs)
        s = (w + 0.5 * d) / n
        var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + (n - w - d) * s ** 2) / n
        se = math.sqrt(var / n)
        elo = lambda p: -400 * math.log10(1 / min(max(p, 1e-6), 1 - 1e-6) - 1)
        print(f"  {name:15s} {n:4d} games  second side {w}W {d}D {n - w - d}L = {100 * s:5.1f}%  "
              f"-> {elo(s):+5.0f} Elo ({elo(s - 1.96 * se):+.0f} to {elo(s + 1.96 * se):+.0f})")


def main():
    if "--report" in sys.argv:
        report()
        return
    os.makedirs(OUT, exist_ok=True)
    book = make_openings()
    have = done_games()
    tasks = [(name, k, sw, book[k]) for name, _, _ in PAIRS for k in range(len(book)) for sw in (True, False)
             if (name, k, sw) not in have]
    print(f"{len(tasks)} games to play ({len(have)} on disk), {WORKERS} at a time", flush=True)
    with mp.Pool(WORKERS) as pool, open(GAMES, "a", encoding="utf-8") as fh:
        for i, r in enumerate(pool.imap_unordered(play, tasks), 1):
            if "error" in r:
                print(f"  {r['pair']} opening {r['k']} failed, left for the next run: {r['error']}", flush=True)
                continue
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} games", flush=True)
    report()


if __name__ == "__main__":
    main()
