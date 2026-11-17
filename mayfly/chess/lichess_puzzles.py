r"""Rated Lichess puzzles, and the puzzle rating a version solves them at.

A version plays a spread of rated puzzles from the Lichess puzzle database (database.lichess.org, CC0).
Its rating is the maximum likelihood fit of Lichess's expected score, 1 / (1 + 10^((puzzle - rating) / 400))
a puzzle, to the puzzles it solved, with a standard error from the same curve.

  python chess/lichess_puzzles.py build [N]       writes lichess_puzzles.json, N puzzles (default 20) in each
                                                  100-point band from 600 to 3200, env DB=<the .csv.zst>
  python chess/lichess_puzzles.py build_top [N]   N extra puzzles rated 3,000 or more
  python chess/lichess_puzzles.py score [VISITS]  env VERSION=mayfly_v5.pt [BASE=mayfly_v5.pt]
                                                  VISITS 0 is the bare network

Only settled puzzles are drawn (rating deviation at most 100, at least 100 plays, popularity 50 or more).
Stricter filters leave almost nothing rated 2,800 or more. A puzzle is solved when each of its moves is found
in turn, with the puzzle's own replies, or when a mate in one is played instead, which Lichess also accepts.
Every move is a fresh search. A score run appends one row to puzzle_results.jsonl.
"""
import csv
import io
import json
import math
import os
import random
import sys
import time

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paths  # noqa: E402

E = os.environ.get
SUITE = E("SUITE", os.path.join(HERE, "lichess_puzzles.json"))
RESULTS = E("RESULTS", os.path.join(HERE, "puzzle_results.jsonl"))
DB = E("DB", paths.under("lichess_puzzles", "lichess_db_puzzle.csv.zst"))
LO, HI, BAND = 600, 3200, 100
MAX_RD, MIN_PLAYS, MIN_POP = 100, 100, 50


def build(per_band):
    """Reservoir-sample per_band settled puzzles from every band, in one pass over the database."""
    import zstandard
    rng = random.Random(20261008)
    bands = {b: [] for b in range(LO, HI, BAND)}
    seen = {b: 0 for b in bands}
    scanned = kept = 0
    with open(DB, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for row in csv.DictReader(text):
            scanned += 1
            try:
                rating, rd = int(row["Rating"]), int(row["RatingDeviation"])
                plays, pop = int(row["NbPlays"]), int(row["Popularity"])
            except (KeyError, ValueError):
                continue
            if rd > MAX_RD or plays < MIN_PLAYS or pop < MIN_POP or not LO <= rating < HI:
                continue
            kept += 1
            b = LO + (rating - LO) // BAND * BAND
            seen[b] += 1
            item = {"id": row["PuzzleId"], "fen": row["FEN"], "moves": row["Moves"].split(), "rating": rating,
                    "rd": rd, "plays": plays, "themes": row["Themes"]}
            if len(bands[b]) < per_band:
                bands[b].append(item)
            else:
                j = rng.randrange(seen[b])
                if j < per_band:
                    bands[b][j] = item
    puzzles = [p for b in sorted(bands) for p in bands[b]]
    json.dump({"built": time.strftime("%Y-%m-%d %H:%M"), "db": os.path.basename(DB), "scanned": scanned,
               "settled": kept, "per_band": per_band, "bands": f"{LO}-{HI} by {BAND}",
               "filters": {"max_rd": MAX_RD, "min_plays": MIN_PLAYS, "min_popularity": MIN_POP},
               "puzzles": puzzles}, open(SUITE, "w", encoding="utf-8"), indent=0)
    thin = {b: len(v) for b, v in bands.items() if len(v) < per_band}
    print(f"{len(puzzles)} puzzles from {kept:,} settled of {scanned:,} -> {SUITE}"
          + (f"; bands short of {per_band}: {thin}" if thin else ""))


def build_top(n, path=None):
    """n extra puzzles rated 3,000 or more, which the main set has few of.

    Deviation at most 120 and 50 plays, since only about 1,150 that high exist."""
    import zstandard
    rng = random.Random(20261009)
    pool = []
    with open(DB, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for row in csv.DictReader(text):
            try:
                rating, rd, plays = int(row["Rating"]), int(row["RatingDeviation"]), int(row["NbPlays"])
            except (KeyError, ValueError):
                continue
            if rating >= 3000 and rd <= 120 and plays >= 50:
                pool.append({"id": row["PuzzleId"], "fen": row["FEN"], "moves": row["Moves"].split(),
                             "rating": rating, "rd": rd, "plays": plays, "themes": row["Themes"]})
    main = {p["id"] for p in json.load(open(SUITE, encoding="utf-8"))["puzzles"]}
    pool = [p for p in pool if p["id"] not in main]
    pick = sorted(rng.sample(pool, min(n, len(pool))), key=lambda p: p["rating"])
    out = path or os.path.join(HERE, "lichess_puzzles_top.json")
    json.dump({"built": time.strftime("%Y-%m-%d %H:%M"), "db": os.path.basename(DB), "pool": len(pool),
               "filters": {"min_rating": 3000, "max_rd": 120, "min_plays": 50}, "puzzles": pick},
              open(out, "w", encoding="utf-8"), indent=0)
    print(f"{len(pick)} puzzles rated {pick[0]['rating']} to {pick[-1]['rating']} from {len(pool)} -> {out}")


def fit(results):
    """The maximum-likelihood rating for [(puzzle rating, solved)], its standard error, and whether it is a bound
    (everything solved, or nothing)."""
    k = sum(s for _, s in results)
    expect = lambda r: sum(1.0 / (1.0 + 10 ** ((p - r) / 400.0)) for p, _ in results)
    if k == 0 or k == len(results):
        return (0.0 if k == 0 else 4000.0), None, True
    lo, hi = -1000.0, 5000.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expect(mid) < k else (lo, mid)
    r = (lo + hi) / 2
    c = math.log(10) / 400.0
    info = sum(c * c * q * (1 - q) for q in (1.0 / (1.0 + 10 ** ((p - r) / 400.0)) for p, _ in results))
    return r, 1.0 / math.sqrt(info), False


def score(visits):
    from mcts import MCTS, pinned_search, policy_move
    from play_web import load_version
    version = E("VERSION", "mayfly_v5.pt")
    base_name = E("BASE", version)
    base = pinned_search(base_name)
    assert base, f"no sealed search for {base_name}"
    puzzles = json.load(open(SUITE, encoding="utf-8"))["puzzles"]
    model, label, _ = load_version(version)
    dev = str(next(model.parameters()).device)

    def play(board):
        if visits <= 0:
            return policy_move(model, board, dev, base.get("policy_head", "cls"), castle=base.get("castle_prior", "king"))
        mv, _ = MCTS(model, dev, batch=48, cache_size=20000).apply(base).search(board, visits=visits, reuse=False)
        return mv

    t0, results, searches = time.time(), [], 0
    for i, p in enumerate(puzzles):
        board = chess.Board(p["fen"])
        board.push_uci(p["moves"][0])                   # the opponent's move that sets the puzzle
        solved = True
        for j in range(1, len(p["moves"]), 2):
            want = chess.Move.from_uci(p["moves"][j])
            mv = play(board)
            searches += 1
            if mv == want:
                board.push(mv)
                if j + 1 < len(p["moves"]):
                    board.push_uci(p["moves"][j + 1])   # the puzzle's own reply
                continue
            if mv is not None and mv in board.legal_moves:
                board.push(mv)
                solved = board.is_checkmate()           # another mate in one is accepted, as on Lichess
            else:
                solved = False
            break
        results.append((p["rating"], solved))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}: {sum(s for _, s in results)} solved", flush=True)
    rating, se, bound = fit(results)
    if not bound and rating < LO:       # below the easiest band the set can measure, reported as 0
        rating, se, bound = 0.0, None, True
    n, k = len(results), sum(s for _, s in results)
    lo, hi = (None, None) if se is None else (rating - 1.96 * se, rating + 1.96 * se)
    print(f"{label}, {visits} visits, base {base_name}: {k}/{n} solved -> puzzle rating "
          + (("above " + str(int(rating)) if k == n else f"0 (below the set's easiest band, {LO})") if bound else
             f"{rating:.0f} (95% {lo:.0f} to {hi:.0f})")
          + f", {searches} searches in {(time.time() - t0) / 60:.1f} min")
    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M"), "network": label, "base": base_name,
                             "suite": os.path.basename(SUITE),
                             "visits": visits, "solved": k, "n": n, "rate": round(k / n, 4),
                             "rating": round(rating), "se": None if se is None else round(se, 1),
                             "bound": bound, "searches": searches,
                             # one flag a puzzle in the suite's order, so any band can be read off later
                             "solved_each": "".join("1" if s else "0" for _, s in results)}) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif cmd == "build_top":
        build_top(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    else:
        score(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
