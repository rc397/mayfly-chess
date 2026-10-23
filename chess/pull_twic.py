"""Pull the TWIC (The Week in Chess) archive, weekly zips of strong over-the-board games.

Each issue extracts to a PGN. Later steps use them as a source of realistic positions, for opening
suites, profiling and Stockfish labelling. Network only.

  python chess/pull_twic.py [START=920] [END=1650]
Output: chess/twic/*.pgn
"""
import os, sys, time, urllib.request, zipfile, io
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "twic"); os.makedirs(OUT, exist_ok=True)
UA = "Mozilla/5.0 (mayfly-chess-research)"
START = int(sys.argv[1]) if len(sys.argv) > 1 else 920
END = int(sys.argv[2]) if len(sys.argv) > 2 else 1650
ok = miss = 0
for n in range(START, END + 1):
    if any(f.startswith(f"twic{n}") for f in os.listdir(OUT)):
        continue
    try:
        data = urllib.request.urlopen(urllib.request.Request(
            f"https://theweekinchess.com/zips/twic{n}g.zip", headers={"User-Agent": UA}), timeout=60).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(OUT)
        ok += 1
        if ok % 25 == 0:
            print(f"  {ok} issues pulled (latest twic{n})", flush=True)
    except Exception:
        miss += 1
    time.sleep(0.2)
print(f"TWIC pull DONE: {ok} issues, {miss} missing/absent -> chess/twic/", flush=True)
