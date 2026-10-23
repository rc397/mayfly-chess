r"""
Set the flags byte on shards written before it existed.

The flags byte was carved out of reserved space after some Lichess shards were written, so they read
as auxiliary-invalid although they carry real king safety and pawn structure. The record size is
unchanged, so this edits the byte in place.

  python chess/patch_flags.py <dir> <prefix> <0|1>
"""
import os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pack import DT, REC

d, prefix, val = sys.argv[1], sys.argv[2], int(sys.argv[3])
paths = sorted(glob.glob(os.path.join(d, prefix + "*.bin")))
print(f"{len(paths)} shards matching {prefix}*, setting flags={val}")
total = 0
for p in paths:
    n = os.path.getsize(p) // REC
    mm = np.memmap(p, dtype=DT, mode="r+")
    mm["flags"] = val
    mm.flush()
    del mm
    total += n
    print(f"  {os.path.basename(p)}: {n:,} records")
print(f"{total:,} records patched")
