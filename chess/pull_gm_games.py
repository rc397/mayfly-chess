"""Download defensive and classical masters' games (pgnmentor.com per-player PGNs), and optionally
one chess.com account's games, as raw PGNs for an experimental human-style head. Network and light
CPU only. Parsing into training records is a separate step (parse_gm.py).

  python chess/pull_gm_games.py
  env: CHESSCOM_USER=<chess.com account to fetch, optional>
Output: chess/gm_pgn/*.pgn
"""
import os, sys, json, time, urllib.request, zipfile, io
HERE = os.path.dirname(os.path.abspath(__file__))
GM_DIR = os.path.join(HERE, "gm_pgn"); os.makedirs(GM_DIR, exist_ok=True)
UA = "Mozilla/5.0 (mayfly-chess-research)"

# defensive moderns, French specialists and classical masters
PLAYERS = ["Capablanca", "Petrosian", "Karpov", "Korchnoi", "Andersson", "Kramnik", "Adams",
           "Karjakin", "Leko", "Uhlmann", "Bareev", "Rubinstein", "Smyslov", "Botvinnik", "Nimzowitsch"]

def fetch(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout).read()

ok = 0
for p in PLAYERS:
    try:
        data = fetch(f"https://www.pgnmentor.com/players/{p}.zip")
        z = zipfile.ZipFile(io.BytesIO(data)); z.extractall(GM_DIR)
        print(f"OK   {p}: {len(data)//1024} KB {z.namelist()}", flush=True); ok += 1
    except Exception as e:
        print(f"MISS {p}: {e}", flush=True)
print(f"pgnmentor: {ok}/{len(PLAYERS)} players", flush=True)

def pull_chesscom(user):
    try:
        arch = json.loads(fetch(f"https://api.chess.com/pub/player/{user}/games/archives"))["archives"]
    except Exception as e:
        print(f"chess.com {user} archives MISS: {e}", flush=True); return
    out = os.path.join(GM_DIR, f"chesscom_{user}.pgn"); n = 0
    with open(out, "w", encoding="utf-8") as f:
        for a in arch:
            try:
                for g in json.loads(fetch(a)).get("games", []):
                    if g.get("pgn"):
                        f.write(g["pgn"] + "\n\n"); n += 1
            except Exception:
                continue
            time.sleep(0.25)
    print(f"chess.com {user}: {n} games -> {out}", flush=True)

if os.environ.get("CHESSCOM_USER"):
    pull_chesscom(os.environ["CHESSCOM_USER"])
print("GM pull DONE -> chess/gm_pgn/", flush=True)
