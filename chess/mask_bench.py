r"""Measure what the legal move mask in pack.batch_tensors costs the policy loader.

Times batch_tensors with and without the mask on real records from the shards, since the cost
depends on the number of legal moves, and compares the rate with NEED, what the trainer consumes.

  python chess/mask_bench.py [BATCH_SIZE] [BATCHES]      env: SHARDS=<data>/shards
"""
import os, sys, time, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pack as P
from moves import N_MOVES
import paths as _paths

SH = os.environ.get("SHARDS", _paths.SHARDS)
BSZ = int(sys.argv[1]) if len(sys.argv) > 1 else 32
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 60
NEED = 5.13                      # policy batches a second the trainer currently consumes

paths = [p for p in sorted(glob.glob(os.path.join(SH, "*.bin")))
         if os.path.basename(p).split("_")[0] in ("lz", "lc")]
rng = np.random.default_rng(0)
pool = []
for p in rng.choice(paths, min(8, len(paths)), replace=False):
    mm = P.open_shard(p)
    if len(mm) < 5000:
        continue
    i = int(rng.integers(0, len(mm) - 4096))
    pool.append(np.array(mm[i:i + 4096]))
rows_all = np.concatenate(pool)
print(f"{len(rows_all):,} real records from {len(pool)} shards, batch {BSZ}")

def run(legal):
    batches = [rows_all[i * BSZ:(i + 1) * BSZ] for i in range(NB)]
    P.batch_tensors(batches[0], N_MOVES, 100.0, legal)          # warm
    t0 = time.perf_counter()
    out = None
    for b in batches:
        out = P.batch_tensors(b, N_MOVES, 100.0, legal)
    return (time.perf_counter() - t0) / NB, out

t_off, _ = run(False)
t_on, out = run(True)
m = out["legal"]
print("")
print(f"  {'without the mask':<24} {t_off*1000:>8.2f} ms a batch   {1/t_off:>8.1f} batches/s")
print(f"  {'with the mask':<24} {t_on*1000:>8.2f} ms a batch   {1/t_on:>8.1f} batches/s")
print(f"  {'the mask itself':<24} {(t_on-t_off)*1000:>8.2f} ms a batch   "
      f"{(t_on-t_off)*1e6/BSZ:>8.0f} us a position")
print("")
print(f"  mean legal moves a position {m.sum(1).mean():.1f}, "
      f"so the softmax goes from 5-of-{N_MOVES} to 5-of-{m.sum(1).mean():.0f}")
print(f"  the trainer needs {NEED:.2f} batches a second")
print(f"  headroom without {1/t_off/NEED:>6.0f}x,  with {1/t_on/NEED:>6.0f}x")
if 1 / t_on > NEED * 20:
    print("")
    print(f"  THE MASK FITS. At {1/t_on:.0f} batches a second against {NEED:.2f} needed it is not")
    print(f"  close to being the constraint, and it runs on the filler thread, overlapped.")
else:
    print("")
    print(f"  TOO EXPENSIVE at this batch size. {1/t_on:.1f} against {NEED:.2f} needed.")
