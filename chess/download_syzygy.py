"""Download the 3, 4 and 5 piece Syzygy tablebases (about 1 GB) into the data folder's syzygy/.

Tries known mirrors, reads the directory index and downloads each .rtbw and .rtbz file. Files already
present are skipped, so it resumes.

    python chess/download_syzygy.py
"""
import os, re, sys, urllib.request

import paths

BASES = [
    "http://tablebase.sesse.net/syzygy/3-4-5/",
    "https://tablebase.lichess.ovh/tables/standard/3-4-5/",
]
DEST = paths.under("syzygy")
os.makedirs(DEST, exist_ok=True)

base, files = None, []
for b in BASES:
    try:
        idx = urllib.request.urlopen(b, timeout=30).read().decode("utf-8", "ignore")
        fs = sorted(set(re.findall(r'href="([^"?]+\.rtb[wz])"', idx)))
        if fs:
            base, files = b, fs
            break
        print(f"  {b} -> listed but no .rtb files", flush=True)
    except Exception as e:
        print(f"  {b} -> {e}", flush=True)

if not files:
    print("no working tablebase mirror found; stopping."); sys.exit(1)
print(f"{len(files)} files from {base} -> {DEST}", flush=True)

for i, fn in enumerate(files, 1):
    name = os.path.basename(fn)
    out = os.path.join(DEST, name)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        continue
    urllib.request.urlretrieve(base + name, out)
    if i % 25 == 0 or i == len(files):
        mb = sum(os.path.getsize(os.path.join(DEST, f)) for f in os.listdir(DEST)) / 1e6
        print(f"  {i}/{len(files)}  ({mb:.0f} MB)", flush=True)
print(f"DONE: {len(files)} files in {DEST}", flush=True)
