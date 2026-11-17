"""Download the Lichess Elite Database, games where both players are strong, already in PGN.

Reads the index at database.nikonoel.fr, downloads the latest N monthly files into chess/lichess_pgn/
and decompresses them to .pgn. Prints the start of the index if no data links are found.

    python chess/get_lichess_elite.py [N_MONTHS=1]
"""
import os, re, sys, requests

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "lichess_pgn")
os.makedirs(DEST, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
BASE = "https://database.nikonoel.fr/"

print(f"fetching index {BASE} ...", flush=True)
html = requests.get(BASE, timeout=30).text
links = re.findall(r'href=[\'"]?([^\'" >]+\.(?:pgn\.zst|pgn\.bz2|zip|zst))', html, re.I)
links = sorted(set(l for l in links if re.search(r'lichess_elite_\d{4}-\d{2}', l)))  # monthly files only

if not links:
    print("NO data links found. Index head so we can adjust:\n" + html[:1200])
    sys.exit(1)
print(f"found {len(links)} files; latest: {links[-5:]}", flush=True)

for t in links[-N:]:
    url = t if t.startswith("http") else BASE + t.lstrip("/")
    fn = os.path.join(DEST, os.path.basename(t))
    print(f"downloading {url} ...", flush=True)
    with requests.get(url, stream=True, timeout=1200) as r:
        r.raise_for_status()
        with open(fn, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print(f"  saved {os.path.getsize(fn)/1e6:.0f} MB; decompressing ...", flush=True)
    if fn.endswith(".zip"):
        import zipfile
        zipfile.ZipFile(fn).extractall(DEST)
    elif fn.endswith(".zst"):
        import zstandard
        with open(fn, "rb") as cf, open(fn[:-4], "wb") as of:
            zstandard.ZstdDecompressor().copy_stream(cf, of)
    elif fn.endswith(".bz2"):
        import bz2, shutil
        with bz2.open(fn, "rb") as cf, open(fn[:-4], "wb") as of:
            shutil.copyfileobj(cf, of)

pgns = [f for f in os.listdir(DEST) if f.endswith(".pgn")]
total = sum(os.path.getsize(os.path.join(DEST, f)) for f in pgns)
print(f"DONE: {len(pgns)} PGN file(s), {total/1e9:.2f} GB total -> {DEST}", flush=True)
