r"""Pack one Kaggle TPU training session's slice of the shards (<KAGGLE_USER>/mayfly-train-data).

A session trains only on this slice (train_xla.py). A Kaggle dataset keeps at most 50 top-level
files, so the shards go up as one uncompressed zip, which Kaggle unpacks, with the route 1
down-weighting keys beside it. Kaggle limits the total size of private datasets, so keep slices small.

The slice holds ChessBench (av) and Leela (lz) shards drawn at random, half the Leela ones written
since MLH_SINCE so the moves-left head has rows, all of route 1 (dp) and the deep second labels (dh),
and a few Lichess deep evaluation (lv) and Lichess policy (lc) shards. Each SLICE_SEED draws a
different slice.

  python chess/make_train_data.py [--upload]
  env: KAGGLE_USER=<name>  SHARDS=<data>/shards  SLICE_DIR=<data>/kaggle/train_data  SLICE_SEED=1
       SLICE_AV=13 SLICE_LZ=30 SLICE_LV=2 SLICE_LC=2  MLH_SINCE=2026-09-24 19:10
       DP_DOWN_KEYS=<data>/selfplay/prefix_selfplay_keys.npy
"""
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

E = os.environ.get
SHARDS = E("SHARDS", paths.under("shards"))
OUT = os.path.normpath(E("SLICE_DIR", paths.under("kaggle", "train_data")))
SLUG = f"{E('KAGGLE_USER', '')}/mayfly-train-data"


def pick():
    rng = random.Random(int(E("SLICE_SEED", 1)))
    since = time.mktime(time.strptime(E("MLH_SINCE", "2026-09-24 19:10"), "%Y-%m-%d %H:%M"))
    fam = lambda f: sorted(glob.glob(os.path.join(SHARDS, f"{f}_*.bin")))  # noqa: E731
    lz = fam("lz")
    new = [p for p in lz if os.path.getmtime(p) >= since]
    old = [p for p in lz if os.path.getmtime(p) < since]
    n = int(E("SLICE_LZ", 30))
    take = rng.sample(new, min(len(new), n // 2))
    take += rng.sample(old, min(len(old), n - len(take)))
    out = {"av": rng.sample(fam("av"), int(E("SLICE_AV", 13))), "lz": take, "dp": fam("dp"), "dh": fam("dh"),
           "lv": rng.sample(fam("lv"), int(E("SLICE_LV", 2))), "lc": rng.sample(fam("lc"), int(E("SLICE_LC", 2)))}
    return out


def build():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    files = pick()
    total = sum(os.path.getsize(p) for v in files.values() for p in v)
    print(f"slice: {', '.join(f'{k} {len(v)}' for k, v in files.items())}, {total / 1e9:.1f} GB", flush=True)
    t0 = time.time()
    with zipfile.ZipFile(os.path.join(OUT, "shards.zip"), "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for v in files.values():
            for p in v:
                z.write(p, arcname="shards/" + os.path.basename(p))
    print(f"zipped in {time.time() - t0:.0f}s", flush=True)
    keys = E("DP_DOWN_KEYS", paths.under("selfplay", "prefix_selfplay_keys.npy"))
    if os.path.exists(keys):
        shutil.copy2(keys, OUT)
    json.dump({k: [os.path.basename(p) for p in v] for k, v in files.items()},
              open(os.path.join(OUT, "slice.json"), "w"), indent=1)
    json.dump({"title": "mayfly-train-data", "id": SLUG, "licenses": [{"name": "other"}]},
              open(os.path.join(OUT, "dataset-metadata.json"), "w"))
    return total


def upload():
    tok = os.path.expanduser(os.path.join("~", ".kaggle", "access_token"))
    env = dict(os.environ, KAGGLE_API_TOKEN=tok, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    k = lambda *a: subprocess.run([sys.executable, "-m", "kaggle", *a], capture_output=True, text=True,  # noqa: E731
                                  encoding="utf-8", errors="replace", env=env, timeout=4 * 3600)
    st = k("datasets", "status", SLUG)
    t0 = time.time()
    if st.returncode == 0 and "ready" in (st.stdout + st.stderr).lower():
        r = k("datasets", "version", "-p", OUT, "-m", f"slice {E('SLICE_SEED', 1)}", "--dir-mode", "skip")
    else:
        r = k("datasets", "create", "-p", OUT, "--dir-mode", "skip")
    tail = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and "%|" not in l][-3:]
    print(f"upload exit {r.returncode} in {time.time() - t0:.0f}s: {' | '.join(tail)[-300:]}", flush=True)
    return r.returncode == 0


if __name__ == "__main__":
    build()
    if "--upload" in sys.argv:
        print("uploaded" if upload() else "UPLOAD FAILED")
