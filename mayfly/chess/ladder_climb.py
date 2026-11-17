r"""Raise Stockfish's skill level until the version stops winning.

Plays one level at a time, every budget at that level, and stops at the first level the version does
not beat, since the levels above it would only record the same result more slowly. Cells go into the
file elo_ladder.py writes, with the level recorded, so a measured cell is skipped and a stopped run
resumes where it left off.

  python chess/ladder_climb.py
  env: VERSION=mayfly_v4.pt  LEVELS=2,3,4,5,6,7,8  VISITS=0,400,1600,6400  GAMES=24
       SF_MS=100  STOP=0.5  OUT=<dir>  PONDER=0
"""
import os, sys, json, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))

E = os.environ.get
VERSION = E("VERSION", "mayfly_v4.pt")
LEVELS = [int(x) for x in E("LEVELS", "2,3,4,5,6,7,8").split(",")]
VISITS = E("VISITS", "0,400,1600,6400")
GAMES, SF_MS = E("GAMES", "24"), E("SF_MS", "100")
STOP = float(E("STOP", 0.5))
OUT = E("OUT", HERE)


def cells(level):
    """{visits: score} already measured for this version at this level."""
    path = os.path.join(OUT, f"elo_ladder_{VERSION.replace('.pt', '')}.jsonl")
    got = {}
    if not os.path.exists(path):
        return got
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        try:
            r = json.loads(line)
        except ValueError:
            continue                        # blank, or the torn last line of a killed run
        # Match the protocol elo_ladder skips on, so a row played at another SF_MS or with
        # pondering on is never what this climb is judged on. elo_vs_sf records ponder only for
        # a searched cell.
        if (int(r.get("level", -1)) == level and int(r.get("games", 0)) == int(GAMES)
                and int(r.get("sf_ms", -1)) == int(SF_MS)
                and r.get("protocol") != "book2b-draw2"    # the verdict's near-miss games, never a rung
                and bool(r.get("ponder", False))
                == (E("PONDER", "0") == "1" and int(r["visits"]) > 0)):
            got[int(r["visits"])] = float(r["score"])
    return got


print(f"{VERSION}: climbing levels {LEVELS} at {VISITS} visits, {GAMES} games a cell, "
      f"stopping under {100*STOP:.0f}%", flush=True)
for level in LEVELS:
    env = dict(os.environ, VERSIONS=VERSION, VISITS=VISITS, GAMES=GAMES, LEVEL=str(level),
               SF_MS=SF_MS, OUT=OUT, PONDER=E("PONDER", "0"))
    t0 = time.time()
    rc = subprocess.call([sys.executable, os.path.join(HERE, "elo_ladder.py")], env=env, cwd=HERE)
    got = cells(level)
    # Judge only on the budgets this climb was asked for, all of them. Another budget on file for
    # the same level must not decide it. A missing cell fails the climb (exit 1) so a retry can
    # resume it game by game.
    asked = sorted(int(x) for x in str(VISITS).split(",") if x.strip())
    missing = [v for v in asked if v not in got]
    if missing:
        print(f"  Skill {level}: no result at {', '.join(map(str, missing))} visits (exit {rc}); "
              f"stopping with a failure so the queue retries", flush=True)
        sys.exit(1)
    got = {v: got[v] for v in asked}
    deepest = max(asked)
    line = ", ".join(f"{v}: {100*s:.1f}%" for v, s in sorted(got.items()))
    print(f"  Skill {level} in {(time.time()-t0)/3600:.2f} h -> {line}", flush=True)
    if got[deepest] < STOP:
        print(f"  Skill {level} at {deepest} visits is under {100*STOP:.0f}%: this is where it "
              f"stops, and the levels above it are not worth the games.", flush=True)
        break
else:
    print("  every level on the list was beaten; give it more levels.", flush=True)
