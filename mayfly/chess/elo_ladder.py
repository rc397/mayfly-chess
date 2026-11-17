"""Measure every frozen version at several search depths against one fixed Stockfish.

Each cell is a version at a visit count (0 is the bare network) against the same opponent, Stockfish
Skill Level 0 at 100 ms a move by default, so the whole grid reads against one yardstick. Cells run
one at a time, cheapest depth first across all versions, since parallel processes on one GPU were
measured at half the total throughput. Every finished cell is a JSON line in
elo_ladder_<version>.jsonl, written by elo_vs_sf, and cells already there are skipped, so a re-run
resumes. Each version plays its own sealed search (elo_vs_sf applies pinned_search).

    python chess/elo_ladder.py            run whatever is missing, then print the grid
    python chess/elo_ladder.py --table    print the grid from what is on disk
    env VERSIONS=mayfly_v1.pt,mayfly_v2.pt,mayfly_v3.pt, VISITS=0,100,400,1600, GAMES=24,
        LEVEL=0, SF_MS=100, PARALLEL=0 (1 runs one process per version, which is slower)
"""
import os, sys, json, time, subprocess, threading
HERE = os.path.dirname(os.path.abspath(__file__))

E = os.environ.get
VERSIONS = E("VERSIONS", "mayfly_v1.pt,mayfly_v2.pt,mayfly_v3.pt").split(",")
VISITS = [int(v) for v in E("VISITS", "0,100,400,1600").split(",")]
GAMES = int(E("GAMES", 24))
LEVEL = int(E("LEVEL", 0))
SF_MS = int(E("SF_MS", 100))
OUT = E("OUT", HERE)
# One at a time by default. Three processes on one GPU ran each search about six times slower than one
# alone, half the total throughput, since Windows time-slices the GPU coarsely and these searches are
# many small batched forwards. PARALLEL=1 runs one process per version.
PARALLEL = E("PARALLEL", "0") == "1"
# passed through to elo_vs_sf and part of a cell's protocol, off by default
PONDER = E("PONDER", "0") == "1"
# BOOK=b has elo_vs_sf play its second opening book under its own protocol tag, so a book-b row never
# counts as an ordinary ladder cell or the other way round
BOOK_B_TAG = "book2b-draw2"
BOOK_B = E("BOOK", "").lower() == "b"


def results_path(ver):
    return os.path.join(OUT, f"elo_ladder_{ver.replace('.pt', '')}.jsonl")


def done_cells(ver):
    """The (visits) already measured for this version under THIS protocol."""
    got = {}
    try:
        with open(results_path(ver), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                # blank, or the torn last line of a killed run
                # pondering is part of the protocol, since a pondering cell searches more
                if (r.get("games") == GAMES and r.get("level") == LEVEL
                        and r.get("sf_ms") == SF_MS
                        and (r.get("protocol") == BOOK_B_TAG) == BOOK_B
                        # a 0-visit cell is recorded without pondering whatever PONDER says
                        and bool(r.get("ponder", False)) == (PONDER and int(r["visits"]) > 0)):
                    got[int(r["visits"])] = r
    except FileNotFoundError:
        pass
    return got


def worker(ver):
    for v in sorted(VISITS):
        run_cell(ver, v)


# A failed cell fails the run. Failures are collected and the run exits 1 at the end, so the queue's
# retry resumes it game by game.
FAILED = []
# what the opponent is called, 100 and above is a UCI_Elo rating, below it a Skill Level
RUNG = (f"UCI_Elo {LEVEL}" if LEVEL >= 100 else
        f"rung {LEVEL} (Skill 20 with {float(E('RUNG_RATIO', 2 ** 0.5)) ** (LEVEL - 20):.2f}x the time)" if LEVEL > 20
        else f"Skill Level {LEVEL}")      # past Skill 20, elo_vs_sf.py gives Skill 20 more time a rung


def run_cell(ver, v):
    log = os.path.join(OUT, f"elo_ladder_{ver.replace('.pt', '')}.log")
    if v in done_cells(ver):
        print(f"  {ver} at {v} visits: already measured, skipping", flush=True)
        return
    env = dict(os.environ, VERSION=ver, LEVELS=str(LEVEL), SF_MS=str(SF_MS),
               # this cell's visits, not the ladder's list, since play_web reads VISITS as one
               # number at import
               VISITS=str(v),
               RESULTS=results_path(ver), PONDER=("1" if PONDER else "0"),
               # bounded so torch and numpy do not starve the Stockfish beside them

               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== {ver} at {v} visits, {GAMES} games, "
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        fh.flush()
        r = subprocess.run([sys.executable, os.path.join(HERE, "elo_vs_sf.py"),
                            str(GAMES), str(v)], env=env, cwd=HERE,
                           stdout=fh, stderr=subprocess.STDOUT)
    cell = done_cells(ver).get(v)
    if r.returncode != 0 or cell is None:
        print(f"  {ver} at {v} visits: FAILED (exit {r.returncode}), see {log}", flush=True)
        FAILED.append((ver, v))
        return
    print(f"  {ver} at {v:>4} visits: {100*cell['score']:5.1f}%  "
          f"{cell['w']}W {cell['d']}D {cell['l']}L  gap {cell['elo_gap']:+5.0f}  "
          f"{cell['secs_per_move']:.2f}s a move  ({(time.time()-t0)/60:.0f} min)", flush=True)


def table():
    import math
    grid = {ver: done_cells(ver) for ver in VERSIONS}
    names = [v.replace("mayfly_", "").replace(".pt", "") for v in VERSIONS]
    print("")
    print(f"against Stockfish {RUNG} at {SF_MS} ms a move, {GAMES} games a cell")
    print("score, and the Elo GAP to that Stockfish (Skill Level is not a rating scale)\n")
    print(f"  {'search':<16}" + "".join(f"{n:>22}" for n in names))
    for v in sorted(VISITS):
        label = "bare network" if v == 0 else f"{v} visits"
        cells = []
        for ver in VERSIONS:
            c = grid[ver].get(v)
            if c is None:
                cells.append(f"{'...':>22}")
                continue
            p, n = c["score"], c["games"]
            se = math.sqrt(max(p * (1 - p), 1e-9) / n)
            cells.append(f"{100*p:>7.1f}% {c['elo_gap']:>+5.0f} (±{196*se:>3.0f}pt)")
        print(f"  {label:<16}" + "".join(cells))
    print("")
    print("  seconds a move:")
    for v in sorted(VISITS):
        label = "bare network" if v == 0 else f"{v} visits"
        row = []
        for ver in VERSIONS:
            c = grid[ver].get(v)
            row.append(f"{c['secs_per_move']:>22.2f}" if c else f"{'...':>22}")
        print(f"  {label:<16}" + "".join(row))
    print("")
    print("  (±N pt) is the 95% interval on the SCORE in percentage points. At 24 games it is")
    print("  wide: two cells whose intervals overlap are not shown to differ.")


if __name__ == "__main__":
    if "--table" in sys.argv:
        table()
        sys.exit(0)
    todo = {ver: [v for v in VISITS if v not in done_cells(ver)] for ver in VERSIONS}
    print(f"{len(VERSIONS)} versions x {len(VISITS)} depths, {GAMES} games a cell, "
          f"Stockfish {RUNG} at {SF_MS} ms", flush=True)
    for ver, t in todo.items():
        print(f"  {ver}: {'nothing to do' if not t else 'to run at ' + ', '.join(map(str, t))}",
              flush=True)
    print("", flush=True)
    if PARALLEL:
        threads = [threading.Thread(target=worker, args=(ver,)) for ver in VERSIONS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for v in sorted(VISITS):
            for ver in VERSIONS:
                run_cell(ver, v)
    table()
    if FAILED:
        print(f"{len(FAILED)} cell(s) failed: {', '.join(f'{a} at {b}' for a, b in FAILED)}", flush=True)
        sys.exit(1)
