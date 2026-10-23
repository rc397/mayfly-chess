"""Lichess's game accuracy formulas (from lila), and a tool that applies them to the ladder's saved games.

accuracy_match.py imports win_percent and game_accuracy from here and is the fair measure, since it plays
every version and budget against the same opponent. The tool below scores the ladder's own games instead,
where opponents differ by budget, and writes rows with no protocol tag.

Every position of each ladder game is evaluated by Stockfish at a fixed node count. Each of MayFly's moves
gets Lichess's move accuracy from the drop in its win chance, and a game's accuracy is lila's blend (the
mean weighted by how much the position was swinging, averaged with the harmonic mean). A budget's
accuracy is the mean over its games.

    python chess/game_accuracy.py [PGN folder]
    env NODES (200000), WORKERS (8), OUT (chess/accuracy_results.jsonl),
        PER_CELL (0 for every game, 8 takes every third of a cell's 24 so each opening and colour is sampled),
        CACHE (one line a game, so a run resumes)

The PGN folder holds the ladder's files, elo_ladder_<stem>.games_<visits>_<level>.pgn, written by
elo_vs_sf.py. A candidate's stem (v5c_DATE) counts as its version. One row per version and budget is
appended to OUT.
"""
import glob
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import chess
import chess.engine
import chess.pgn

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
NODES = int(E("NODES", 200_000))
WORKERS = int(E("WORKERS", 8))
PER_CELL = int(E("PER_CELL", 0))           # 0 means every game of every cell
OUT = E("OUT", os.path.join(HERE, "accuracy_results.jsonl"))
CACHE = E("CACHE", paths.under("cardstats", "accuracy_cache.jsonl"))
NAME = re.compile(r"elo_ladder_(?P<stem>.+)\.games_(?P<visits>\d+)_(?P<level>\d+)\.pgn$")


def win_percent(cp):
    """Lichess's win chance for White, 0 to 100, from a White-side centipawn score held to +-1,000."""
    cp = max(-1000, min(1000, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(before, after):
    """lila's AccuracyPercent.fromWinPercents, both from the mover's side."""
    if after >= before:
        return 100.0
    raw = 103.1668100711649 * math.exp(-0.04354415386753951 * (before - after)) - 3.166924740191411
    return max(0.0, min(100.0, raw + 1))         # lila adds 1 for the analysis being imperfect


def game_accuracy(start_white, wins, side_white):
    """lila's AccuracyPercent.gameAccuracy for one side.

    wins are White's win chance at the start and after every ply. Each move is weighted by the spread of
    the win chances around it, and the result is the mean of that weighted mean and the harmonic mean."""
    n = len(wins) - 1
    if n < 2:
        return None
    size = max(2, min(8, n // 10))
    head = [wins[:size]] * max(0, min(size, len(wins)) - 2)
    windows = head + [wins[i:i + size] for i in range(len(wins) - size + 1)]
    weights = []
    for w in windows[:n]:
        m = sum(w) / len(w)
        sd = math.sqrt(sum((x - m) ** 2 for x in w) / len(w))
        weights.append(max(0.5, min(12.0, sd)))
    acc, wts = [], []
    for i in range(n):
        white_moved = (i % 2 == 0) == start_white
        if white_moved != side_white:
            continue
        prev, nxt = wins[i], wins[i + 1]
        a = move_accuracy(prev, nxt) if white_moved else move_accuracy(100 - prev, 100 - nxt)
        acc.append(a)
        wts.append(weights[i])
    if not acc:
        return None
    weighted = sum(a * w for a, w in zip(acc, wts)) / sum(wts)
    harmonic = len(acc) / sum(1 / max(1.0, a) for a in acc)
    return (weighted + harmonic) / 2, len(acc)


_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        path = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
        _ENGINE = chess.engine.SimpleEngine.popen_uci(path)
        _ENGINE.configure({"Threads": 1, "Hash": 64})
    return _ENGINE


def analyse(job):
    """(key, file, index) to (key, {accuracy, moves}) for MayFly's side of one game."""
    key, path, index = job
    with open(path, encoding="utf-8") as fh:
        for _ in range(index):
            chess.pgn.skip_game(fh)
        game = chess.pgn.read_game(fh)
    side_white = game.headers.get("White", "").startswith("MayFly")
    board = game.board()
    start_white = board.turn == chess.WHITE
    eng = _engine()
    wins = []

    def score(b):
        if b.is_checkmate():
            return -1000 if b.turn == chess.WHITE else 1000
        if b.is_stalemate() or b.is_insufficient_material():
            return 0
        info = eng.analyse(b, chess.engine.Limit(nodes=NODES), game=key)
        return info["score"].white().score(mate_score=1000)

    wins.append(win_percent(score(board)))
    for mv in game.mainline_moves():
        board.push(mv)
        wins.append(win_percent(score(board)))
    got = game_accuracy(start_white, wins, side_white)
    return key, ({"accuracy": round(got[0], 3), "moves": got[1]} if got else None)


def main():
    if os.name == "nt":
        # below normal priority, inherited by the workers and their engines, so hours of Stockfish do not slow
        # the machine's other jobs
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    folder = sys.argv[1] if len(sys.argv) > 1 else paths.under("cardstats", "ladder_pgns")

    files = sorted(p for p in glob.glob(os.path.join(folder, "*.pgn")) if NAME.search(os.path.basename(p)))
    done = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            r = json.loads(line)
            done[r["key"]] = r["value"]
    jobs, meta = [], {}
    for path in files:
        m = NAME.search(os.path.basename(path))
        stem, visits, level = m["stem"], int(m["visits"]), int(m["level"])
        cm = re.match(r"v(\d+)c_", stem)
        base = f"mayfly_v{cm.group(1)}.pt" if cm else f"{stem}.pt"
        with open(path, encoding="utf-8") as fh:
            count = 0
            while chess.pgn.read_headers(fh) is not None:
                count += 1
        stride = max(1, math.ceil(count / PER_CELL)) if PER_CELL else 1
        for i in range(0, count, stride):
            key = f"{os.path.basename(path)}#{i}@{NODES}"
            meta[key] = (base, visits, level)
            if key not in done:
                jobs.append((key, path, i))
    print(f"{len(files)} files, {len(meta)} games, {len(jobs)} to analyse at {NODES:,} nodes on {WORKERS} workers",
          flush=True)
    t0 = time.time()
    with Pool(WORKERS) as pool, open(CACHE, "a", encoding="utf-8") as cache:
        for n, (key, value) in enumerate(pool.imap_unordered(analyse, jobs), 1):
            done[key] = value
            cache.write(json.dumps({"key": key, "value": value}) + "\n")
            cache.flush()
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)} games, {(time.time() - t0) / 60:.1f} min", flush=True)

    groups = defaultdict(list)
    for key, (base, visits, level) in meta.items():
        v = done.get(key)
        if v:
            groups[(base, visits)].append((level, v["accuracy"]))
    when = time.strftime("%Y-%m-%d %H:%M")
    with open(OUT, "a", encoding="utf-8") as fh:
        for (base, visits), rows in sorted(groups.items()):
            accs = [a for _, a in rows]
            by_level = defaultdict(list)
            for level, a in rows:
                by_level[level].append(a)
            mean = sum(accs) / len(accs)
            sd = math.sqrt(sum((a - mean) ** 2 for a in accs) / max(1, len(accs) - 1))
            row = {"when": when, "base": base, "visits": visits, "games": len(accs), "accuracy": round(mean, 2),
                   "se": round(sd / math.sqrt(len(accs)), 2), "nodes": NODES,
                   "by_level": {str(k): round(sum(v) / len(v), 2) for k, v in sorted(by_level.items())}}
            fh.write(json.dumps(row) + "\n")
            print(f"{base} at {visits} visits: accuracy {mean:.1f}% over {len(accs)} games "
                  f"(+-{1.96 * row['se']:.1f}), Skill levels {min(by_level)} to {max(by_level)}", flush=True)


if __name__ == "__main__":
    main()
