"""
train_real runs one long training run of the tropical model or its transformer control.

Short screening runs saw about 1.3 positions per parameter and the crawl rule stopped them early on noisy
plateaus. This run uses d384 by 6 layers (12.3M parameters), a fixed long budget with the crawl rule
effectively off, cosine decay only over the final tenth, and a resumable checkpoint at every evaluation.
It keeps MIX_NORM, the one ablation arm that beat the baseline across three seeds (0.087 nats against a
seed spread of 0.014). Selectivity is left free.

  python chess/train_real.py trop            the tropical model
  python chess/train_real.py ctrl            the transformer control, same budget
  env: ITERS=600000  D=384  LAYERS=6  HOURS=24
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
ARCH = sys.argv[1] if len(sys.argv) > 1 else "trop"
ITERS = E("ITERS", "600000")
D = E("D", "384")
LAYERS = E("LAYERS", "6")
HOURS = E("HOURS", "24")
BSZ = E("BSZ", "128")

COMMON = {
    "ITERS": ITERS, "VAL_EVERY": "2000", "VAL_BATCHES": "20", "RESUME": "1",
    # crawl rule effectively off, the wall clock ends the run
    "CRAWL_WINDOW": "60", "CRAWL_THRESH": "0.005",
    "MAX_HOURS": HOURS, "SEED": "0",
}
ARCHCFG = {
    # Each architecture at its own swept optimum. DFF_H makes the tropical trunk match the control's
    # 2x width. The default formula assumes a 4x control and would give tropical 40 percent more parameters.
    "trop": {"ARCH": "trop", "TAG": "real_trop", "LR": "3e-3", "T": "0.3", "H": "4", "SC": "3.0",
             "MIX_NORM": "1", "DFF_H": E("DFF_H", "120")},
    "ctrl": {"ARCH": "ctrl", "TAG": "real_ctrl", "LR": "3e-4", "NHEAD": "16", "FF_MULT": "2.0",
             "DROPOUT": "0.0"},
}

if ARCH not in ARCHCFG:
    raise SystemExit("usage: train_real.py [trop|ctrl]")

env = dict(os.environ)
env.update(COMMON)
env.update(ARCHCFG[ARCH])
cfg = ARCHCFG[ARCH]

print(f"real training run: {ARCH}")
print(f"  d{D} x {LAYERS} layers, batch {BSZ}")
print(f"  {int(ITERS):,} iterations cap = {int(ITERS) * int(BSZ):,} positions")
print(f"  {HOURS}h wall clock cap, checkpoint every 2000 iters, resume on")
print(f"  {' '.join(f'{k}={v}' for k, v in cfg.items() if k not in ('ARCH', 'TAG'))}")
print(flush=True)

log = os.path.join(HERE, f"real_{ARCH}.log")
with open(log, "a", encoding="utf-8") as f:
    subprocess.run([sys.executable, os.path.join(HERE, "train_chess_long.py"),
                    ITERS, BSZ, D, LAYERS], cwd=HERE, env=env, stdout=f,
                   stderr=subprocess.STDOUT)
print(f"finished, see {os.path.basename(log)}")
