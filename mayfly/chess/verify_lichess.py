"""Check that each gm_pgn/lichess_*.pgn file belongs to a titled player and delete the namesakes.

A bare handle can belong to an untitled player with the same name. One profile request per handle. On a
network error the file is kept.
"""
import os, glob, json, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); GM = os.path.join(HERE, "gm_pgn")
UA = "Mozilla/5.0 (mayfly-chess-research)"
KEEP = {"GM", "IM", "FM", "WGM", "WIM", "WFM", "CM", "NM"}
kept = dropped = 0
for fp in sorted(glob.glob(os.path.join(GM, "lichess_*.pgn"))):
    h = os.path.basename(fp)[len("lichess_"):-4]
    try:
        prof = json.loads(urllib.request.urlopen(
            urllib.request.Request(f"https://lichess.org/api/user/{h}", headers={"User-Agent": UA}),
            timeout=30).read())
        title = (prof.get("title") or "").upper()
        err = False
    except Exception:
        title = ""; err = True
    if err or title in KEEP:
        print(f"KEEP {h}: title={title or ('ERR' if err else 'none')}", flush=True); kept += 1
    else:
        print(f"DROP {h}: untitled namesake", flush=True)
        try:
            os.remove(fp); dropped += 1
        except Exception:
            pass
    time.sleep(1)
print(f"verify DONE: kept {kept}, dropped {dropped}", flush=True)
