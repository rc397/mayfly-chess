r"""Build the private Kaggle dataset <KAGGLE_USER>/mayfly-selfplay that kaggle_selfplay.py runs from.

Run it where the weights, the corpus shards and the Kaggle token are. Everything goes into one flat
folder, since a dataset keeps no subfolders. That is the self-play code, the native helpers' C
sources, the weights as chessnet_jax's npz (float32 with the config inside, so a TPU host needs no
torch), versions.json with the sealed search, and corpus_starts.txt, STARTS positions drawn the way
selfplay.random_corpus_board draws them. --upload then creates or versions the dataset.

  python chess/make_selfplay_dataset.py [--upload]
  env: KAGGLE_USER=<name>  SP_DATASET_DIR=<data>/kaggle/sp_dataset  SP_VERSION=mayfly_v6.pt
       SP_CKPT (default: that version in <data>/versions)
       STARTS=30000
"""
import json
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

E = os.environ.get
OUT = E("SP_DATASET_DIR", paths.under("kaggle", "sp_dataset"))
VDIR = E("VERSION_DIR", paths.under("versions"))
VERSION = E("SP_VERSION", "mayfly_v6.pt")      # the version that plays; written to sp_version.json
CKPT = E("SP_CKPT", os.path.join(VDIR, VERSION))
CODE = ["selfplay_remote.py", "selfplay.py", "mcts.py", "chess_net.py", "chessnet_jax.py", "encode.py", "moves.py",
        "lc0_moves.py", "book.py", "kaggle_selfplay.py", "tpu_bench.py", "paths.py", "mcts_cpp.py"]
NATIVE = ["movegen.c", "walk.c", "pick.c"]
SLUG = f"{E('KAGGLE_USER', '')}/mayfly-selfplay"


def build():
    os.makedirs(OUT, exist_ok=True)
    for f in CODE:
        if os.path.exists(os.path.join(HERE, f)):
            shutil.copy2(os.path.join(HERE, f), OUT)
    for f in NATIVE:
        shutil.copy2(os.path.join(HERE, "native", f), OUT)
    shutil.copy2(os.path.join(VDIR, "versions.json"), OUT)
    npz_name = os.path.splitext(VERSION)[0] + ".npz"
    # move other versions' weights out of the folder, or every upload would carry them
    old = os.path.normpath(OUT) + "_old_weights"
    for f in os.listdir(OUT):
        if f.endswith(".npz") and f != npz_name:
            os.makedirs(old, exist_ok=True)
            os.replace(os.path.join(OUT, f), os.path.join(old, f))
    npz = os.path.join(OUT, npz_name)
    if not os.path.exists(npz) or os.path.getmtime(npz) < os.path.getmtime(CKPT):
        from selfplay_remote import save_npz
        save_npz(CKPT, npz)
    starts = os.path.join(OUT, "corpus_starts.txt")
    if not os.path.exists(starts):
        import selfplay as sp
        rng = random.Random(20261018)
        n = int(E("STARTS", 30000))
        fens = set()
        t0 = time.time()
        while len(fens) < n and time.time() - t0 < 1800:
            fens.add(sp.random_corpus_board(rng).fen())
        if len(fens) < 1000:                  # random_corpus_board gives the start position when it finds no shards
            raise SystemExit(f"only {len(fens)} distinct corpus starts: are the shards at {sp.SHARDS}?")
        with open(starts, "w", encoding="utf-8") as fh:
            fh.write("".join(f + "\n" for f in sorted(fens)))
    with open(os.path.join(OUT, "sp_version.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": VERSION, "npz": npz_name}, fh)
    meta = {"title": "mayfly-selfplay", "id": SLUG, "licenses": [{"name": "other"}]}
    with open(os.path.join(OUT, "dataset-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    for f in sorted(os.listdir(OUT)):
        print(f"  {f:28s} {os.path.getsize(os.path.join(OUT, f)) / 1e6:9.2f} MB")


def upload():
    tok = os.path.expanduser(os.path.join("~", ".kaggle", "access_token"))
    env = dict(os.environ, KAGGLE_API_TOKEN=tok, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    k = lambda *a: subprocess.run([sys.executable, "-m", "kaggle", *a], capture_output=True, text=True,  # noqa: E731
                                  encoding="utf-8", errors="replace", env=env, timeout=3600)
    d = os.path.normpath(OUT)               # the CLI builds a cache file name from this path, and "/" in it breaks on Windows
    st = k("datasets", "status", SLUG)
    if st.returncode == 0 and "ready" in (st.stdout + st.stderr).lower():
        r = k("datasets", "version", "-p", d, "-m", time.strftime("self-play code %Y-%m-%d %H:%M"))
    else:
        r = k("datasets", "create", "-p", d)
    print((r.stdout or "")[-800:], (r.stderr or "")[-800:])
    return r.returncode == 0


if __name__ == "__main__":
    build()
    if "--upload" in sys.argv:
        print("uploaded" if upload() else "UPLOAD FAILED")
