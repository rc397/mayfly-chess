"""Build and upload the public Kaggle kit that volunteers run.

  --dataset   create/update rc7530/mayfly-selfplay-public: the self-play code, v6's weights, the Stockfish labeller,
              a pool of hard positions to label and Stockfish's Linux build
  --test      push a private test notebook on that dataset (labels for 40 minutes, no accelerator)
  --notebook  push the public notebook rc7530/mayfly-selfplay-volunteer

Kaggle sets a notebook's visibility when it's created, so the test and public notebooks are separate.
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

E = os.environ.get
USER = E("KAGGLE_USER", "rc7530")
BASE = E("KAGGLE_DIR", paths.under("kaggle"))
SRC = E("SP_DATASET_DIR", os.path.join(BASE, "sp_dataset"))
DS_DIR = E("VOL_DATASET_DIR", os.path.join(BASE, "sp_public"))
NB_ROOT = E("VOL_NOTEBOOK_DIR", os.path.join(BASE, "volunteer_nb"))
EXTRA = [os.path.join(HERE, "kaggle_kernel.py"), os.path.join(BASE, "label_pool.txt")]
# stored as .bin because Kaggle unpacks uploaded .gz files
SF_TGZ = os.path.join(BASE, "stockfish-linux-x86-64-universal.tar.gz")
SF_NAME = "stockfish-sf19-linux-x86-64.tgz.bin"
PUBLIC_DS = f"{USER}/mayfly-selfplay-public"

DESCRIPTION = """MayFly is a chess engine built from scratch (a transformer network with a C++ Monte Carlo tree
search), open source under GPL-3.0-or-later at https://github.com/rc397/mayfly-chess, playing on Lichess as
MayFlyEngine.

This dataset is what the volunteer notebook rc7530/mayfly-selfplay-volunteer runs from: the self-play code, MayFly
v6's network (weights as a float32 npz), the sealed search settings of every released version, starting positions,
the Stockfish labeller and a pool of hard positions to label. MayFly's code and weights are GPL-3.0-or-later;
Stockfish is GPL-3.0 (https://github.com/official-stockfish/Stockfish).

To help: open the notebook, Copy & Edit, then Save Version. With a TPU or GPU (Settings > Accelerator) it plays
MayFly self-play; without one it labels hard positions with Stockfish at depth 35 or more. Thank you!"""


def kaggle(*args):
    tok = os.path.expanduser(os.path.join("~", ".kaggle", "access_token"))
    env = dict(os.environ, KAGGLE_API_TOKEN=tok, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=3600)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.encode("ascii", "replace").decode()[-800:], flush=True)   # the rig's console isn't UTF-8
    return r.returncode, out


def dataset():
    os.makedirs(DS_DIR, exist_ok=True)
    for f in os.listdir(SRC):
        p = os.path.join(SRC, f)
        if os.path.isfile(p) and f != "dataset-metadata.json":
            shutil.copy2(p, DS_DIR)
    for p in EXTRA:
        if os.path.exists(p):
            shutil.copy2(p, DS_DIR)
        else:
            print(f"missing, left out: {p}")
    if os.path.exists(SF_TGZ):
        shutil.copy2(SF_TGZ, os.path.join(DS_DIR, SF_NAME))
    else:
        print(f"missing, left out: {SF_TGZ}")
    meta = {"title": "mayfly-selfplay-public", "id": PUBLIC_DS, "licenses": [{"name": "other"}],
            "subtitle": "MayFly chess engine: code and network for volunteers", "description": DESCRIPTION}
    with open(os.path.join(DS_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    d = os.path.normpath(DS_DIR)      # the CLI breaks on forward slashes in this path on Windows
    rc, out = kaggle("datasets", "status", PUBLIC_DS)
    if rc == 0 and "ready" in out.lower():
        rc, out = kaggle("datasets", "version", "-p", d, "-m", time.strftime("volunteer kit %Y-%m-%d %H:%M"))
    else:
        rc, out = kaggle("datasets", "create", "-p", d, "--public")
    return rc == 0 and "error" not in out.lower()


def notebook(slug, title, public):
    d = os.path.join(NB_ROOT, slug.split("/")[1])
    os.makedirs(d, exist_ok=True)
    code = open(os.path.join(HERE, "kaggle_selfplay.py"), encoding="utf-8").read()
    if not public:
        # the test only labels for 40 minutes so we get its output quickly
        assert code.count("LABEL_BUDGET = 11 * 3600 + 30 * 60") == 1
        code = code.replace("LABEL_BUDGET = 11 * 3600 + 30 * 60", "LABEL_BUDGET = 40 * 60")
    with open(os.path.join(d, "kaggle_selfplay.py"), "w", encoding="utf-8") as fh:
        fh.write(code)
    meta = {"id": slug, "title": title, "code_file": "kaggle_selfplay.py", "language": "python",
            "kernel_type": "script", "is_private": not public, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [PUBLIC_DS], "competition_sources": [], "kernel_sources": []}
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    rc, out = kaggle("kernels", "push", "-p", os.path.normpath(d))
    return rc == 0 and "error" not in out.lower()      # a refused push still exits 0


if __name__ == "__main__":
    if "--dataset" in sys.argv:
        ok = dataset()
    elif "--test" in sys.argv:
        ok = notebook(f"{USER}/mayfly-selfplay-volunteer-test", "MayFly Selfplay Volunteer Test", False)
    elif "--notebook" in sys.argv:
        ok = notebook(f"{USER}/mayfly-selfplay-volunteer", "MayFly Selfplay Volunteer", True)
    else:
        raise SystemExit("--dataset, --test or --notebook")
    print("done" if ok else "FAILED")
