r"""Build the code dataset a Kaggle TPU training session runs (<KAGGLE_USER>/mayfly-train-code).

It holds train_av.py and the modules it imports, train_xla.py, kaggle_train.py, and train_env.json,
the main run's settings from autostart.json less the machine specific ones (DROP), which
kaggle_train.py sets for the TPU. So the TPU trains with the same recipe.

  python chess/make_train_code.py [--upload]
  env: KAGGLE_USER=<name>  TRAIN_CODE_DIR=<data>/kaggle/train_code
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

OUT = os.path.normpath(os.environ.get("TRAIN_CODE_DIR", paths.under("kaggle", "train_code")))
SLUG = f"{os.environ.get('KAGGLE_USER', '')}/mayfly-train-code"
CODE = ["train_jax.py", "train_xla.py", "train_av.py", "chess_net.py", "data_av.py", "data_packed.py", "pack.py", "pack_av.py",
        "precision.py", "aside.py", "moves.py", "encode.py", "lc0_moves.py", "kaggle_train.py", "pidfile.py",
        "kaggle_kernel.py",                   # the spare cores' labeller (kaggle_train.start_labelling)
        "paths.py"]                           # data locations, imported by the modules above
DROP = {"CKPT_DIR", "DP_DOWN_KEYS", "COMPILE", "PRECISION", "RESUME", "VERSION_DIR", "SHARDS"}


def build():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    for f in CODE:
        if os.path.exists(os.path.join(HERE, f)):
            shutil.copy2(os.path.join(HERE, f), OUT)
    env = json.load(open(os.path.join(HERE, "autostart.json"), encoding="utf-8-sig"))["env"]
    env = {k: v for k, v in env.items() if k not in DROP}
    json.dump(env, open(os.path.join(OUT, "train_env.json"), "w"), indent=1)
    json.dump({"title": "mayfly-train-code", "id": SLUG, "licenses": [{"name": "other"}]},
              open(os.path.join(OUT, "dataset-metadata.json"), "w"))
    print(f"{len(os.listdir(OUT))} files; the run's settings: {env}")


def upload():
    tok = os.path.expanduser(os.path.join("~", ".kaggle", "access_token"))
    env = dict(os.environ, KAGGLE_API_TOKEN=tok, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    k = lambda *a: subprocess.run([sys.executable, "-m", "kaggle", *a], capture_output=True, text=True,  # noqa: E731
                                  encoding="utf-8", errors="replace", env=env, timeout=1800)
    st = k("datasets", "status", SLUG)
    if st.returncode == 0 and "ready" in (st.stdout + st.stderr).lower():
        r = k("datasets", "version", "-p", OUT, "-m", time.strftime("code %Y-%m-%d %H:%M"))
    else:
        r = k("datasets", "create", "-p", OUT)
    return r.returncode == 0


if __name__ == "__main__":
    build()
    if "--upload" in sys.argv:
        print("uploaded" if upload() else "UPLOAD FAILED")
