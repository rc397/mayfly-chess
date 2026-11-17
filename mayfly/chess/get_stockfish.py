"""Download the latest Stockfish Windows binary and check python-chess can drive it.

The binary goes to tools/stockfish at the repository root (not tracked), and its path is written to
chess/stockfish_path.txt, where the other scripts look for it. It ends by asking Stockfish for the
start position's best move as a smoke test.

    python chess/get_stockfish.py
"""
import os, io, glob, zipfile, requests

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(os.path.dirname(HERE), "tools", "stockfish")
os.makedirs(DEST, exist_ok=True)

rel = requests.get("https://api.github.com/repos/official-stockfish/Stockfish/releases/latest",
                   timeout=30).json()
asset = None
for a in rel.get("assets", []):
    n = a["name"].lower()
    if "windows" in n and "x86-64-avx2" in n and n.endswith(".zip"):
        asset = a
        break
if asset is None:                                   # fall back to any windows zip

    for a in rel.get("assets", []):
        if "windows" in a["name"].lower() and a["name"].endswith(".zip"):
            asset = a
            break
assert asset, "no Windows Stockfish asset found in latest release"
print(f"release {rel.get('tag_name')} -> downloading {asset['name']} ...", flush=True)
data = requests.get(asset["browser_download_url"], timeout=600).content
zipfile.ZipFile(io.BytesIO(data)).extractall(DEST)
exes = glob.glob(os.path.join(DEST, "**", "stockfish*.exe"), recursive=True)
assert exes, "no stockfish .exe after extract"
EXE = exes[0]
with open(os.path.join(HERE, "stockfish_path.txt"), "w") as f:
    f.write(EXE)
print(f"stockfish: {EXE}", flush=True)

# prove python-chess can drive it
import chess
import chess.engine
eng = chess.engine.SimpleEngine.popen_uci(EXE)
board = chess.Board()
res = eng.play(board, chess.engine.Limit(depth=14))
info = eng.analyse(board, chess.engine.Limit(depth=14))
eng.quit()
print(f"TEST OK: start-position best move (d14) = {res.move}  | eval = {info['score']}", flush=True)
