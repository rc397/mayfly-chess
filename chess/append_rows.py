"""Append label shards shipped from other labelling machines to the training corpus, each exactly once.

local_label.py copies each shard into incoming_rows under a unique name, <machine>_<generation>_<file>, and
runs this. Each new file is checked for whole 40-byte rows and piece ids and move indices in range (a torn file
with bad move indices crashes the trainer with a device-side assert), then appended to dp_<machine>NNNN.bin
through pack_av's ShardWriter. Appended names are recorded, so a shard sent twice goes in once. Processed files
move to incoming_rows/done, rejected ones to incoming_rows/rejected. Nothing is deleted.

    python append_rows.py        env IN (incoming folder), OUT (shard folder)
"""
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from moves import N_MOVES          # 4,162, which is 4,096 from-to pairs and 66 underpromotions
from pack_av import DT_AV, REC_AV, ShardWriter
import paths

E = os.environ.get
IN = E("IN", os.path.join(paths.SELFPLAY, "incoming_rows"))
OUT = E("OUT", paths.SHARDS)
LEDGER = os.path.join(IN, "appended.txt")
LOCK = os.path.join(IN, ".lock")
MACHINES = ("lap", "lapc", "sp")          # machine tags that local_label.py puts in shard names


def bad(data):
    """Why these bytes are not whole, sane dp rows, or None."""
    if not data or len(data) % REC_AV:
        return f"{len(data)} bytes is not a whole number of {REC_AV}-byte rows"
    rows = np.frombuffer(data, DT_AV)
    if int(rows["mv"].max()) >= N_MOVES:
        return f"a move index of {int(rows['mv'].max())}"
    ids = rows["ids4"]
    if int((ids & 0x0F).max()) > 12 or int((ids >> 4).max()) > 12:
        return "a piece id past 12"
    return None


def main():
    os.makedirs(os.path.join(IN, "done"), exist_ok=True)
    os.makedirs(os.path.join(IN, "rejected"), exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        if time.time() - os.path.getmtime(LOCK) < 600:
            print("another append_rows is running", flush=True)
            return 1
        os.utime(LOCK, None)            # a lock older than ten minutes was left by a run that died
    try:
        done = set(open(LEDGER, encoding="utf-8").read().split()) if os.path.exists(LEDGER) else set()
        added = {}
        for p in sorted(glob.glob(os.path.join(IN, "*.bin"))):
            name = os.path.basename(p)
            machine = name.split("_", 1)[0]
            if name not in done:
                data = open(p, "rb").read()
                why = bad(data) if machine in MACHINES else f"no machine called {machine!r}"
                if why:
                    print(f"rejected {name}: {why}", flush=True)
                    os.replace(p, os.path.join(IN, "rejected", name))
                    continue
                # A "_dh_" shard holds deeper second labels of MayFly's mistakes (deep_label.py HARD=1). It goes
                # to its own dh_<machine> series, which the trainer does not read. Everything else goes to dp_.
                prefix = "dh" if "_dh_" in name else "dp"
                sw = ShardWriter(OUT, prefix, 4_000_000, tag=machine)
                sw.write_raw(data)
                sw.close()
                with open(LEDGER, "a", encoding="utf-8") as fh:
                    fh.write(name + "\n")
                label = machine if prefix == "dp" else f"{machine} (deep second labels)"
                added[label] = added.get(label, 0) + len(data) // REC_AV
            os.replace(p, os.path.join(IN, "done", name))
        print("appended " + (", ".join(f"{n:,} rows from {m}" for m, n in added.items()) or "nothing new"), flush=True)
        return 0
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
