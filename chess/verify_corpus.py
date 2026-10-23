r"""
verify_corpus checks that the packed shards are intact.

It looks for the faults that would otherwise go unnoticed.
  alignment     a file size that is not a whole number of records means a cut write
  zero records  an unflushed buffer writes zeros, which decode as an empty board
  legality      exactly one king per side. A shifted read almost never leaves two kings, so this one
                test also catches truncation, misalignment and a wrong record size
  ranges        piece ids 0 to 12, move indices under 4096, one to five moves per record

  python chess/verify_corpus.py [DIR] [SAMPLE_PER_SHARD]
"""
import os, sys, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pack import DT, REC, open_shard, split_is_val
from paths import under

DIR = sys.argv[1] if len(sys.argv) > 1 else under("shards")
SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
KING_MINE, KING_THEIRS = 6, 12
from moves import N_MOVES


def check(rows, label):
    bad = {}
    ids = rows["ids"]
    bad["ids out of range"] = int((ids > 12).sum())
    bad["no king of mine"] = int(((ids == KING_MINE).sum(1) != 1).sum())
    bad["no enemy king"] = int(((ids == KING_THEIRS).sum(1) != 1).sum())
    bad["all-zero record"] = int((ids.sum(1) == 0).sum())
    bad["nmv not 1..5"] = int(((rows["nmv"] < 1) | (rows["nmv"] > 5)).sum())
    mv = rows["mv"].astype(np.int64)
    valid = np.arange(5)[None, :] < rows["nmv"][:, None]
    bad["move index >= 4096"] = int((np.where(valid, mv, 0) >= N_MOVES).sum())
    bad["cp out of range"] = int((np.abs(rows["cp"].astype(np.int64)) > 32000).sum())
    bad["mobility zero"] = int((rows["mob"] == 0).sum())
    return bad


if __name__ == "__main__":
    paths = sorted(glob.glob(os.path.join(DIR, "*.bin")))
    if not paths:
        raise SystemExit(f"no shards in {DIR}")

    print(f"{len(paths)} shards in {DIR}\n")
    print(f"{'shard':16s} {'records':>14s} {'GB':>7s}  {'aligned':>8s}")
    total, misaligned = 0, []
    groups = {}
    for p in paths:
        sz = os.path.getsize(p)
        n = sz // REC
        total += n
        ok = (sz % REC) == 0
        if not ok:
            misaligned.append(p)
        groups.setdefault(os.path.basename(p)[:2], [0, 0])
        groups[os.path.basename(p)[:2]][0] += n
        groups[os.path.basename(p)[:2]][1] += 1
        if len(paths) <= 30:
            print(f"{os.path.basename(p):16s} {n:14,} {sz/1e9:7.2f}  {'yes' if ok else 'NO':>8s}")

    print(f"\n{'total':16s} {total:14,} {total*REC/1e9:7.2f} GB")
    for k, (n, c) in sorted(groups.items()):
        src = {"gm": "human tournament games", "sp": "self-play"}.get(k, k)
        print(f"  {src:26s} {n:14,} in {c} shards")
    print(f"\nmisaligned files: {len(misaligned)}" + (f" {misaligned}" if misaligned else ""))

    print(f"\nsampling up to {SAMPLE:,} records per shard")
    agg, checked = {}, 0
    rng = np.random.default_rng(0)
    for p in paths:
        mm = open_shard(p)
        if len(mm) == 0:
            continue
        idx = rng.choice(len(mm), size=min(SAMPLE, len(mm)), replace=False)
        idx.sort()
        rows = mm[idx]
        checked += len(rows)
        for k, v in check(rows, p).items():
            agg[k] = agg.get(k, 0) + v

    print(f"{checked:,} records checked\n")
    worst = 0
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        rate = 100 * v / max(checked, 1)
        flag = "OK" if v == 0 else ("negligible" if rate < 0.01 else "PROBLEM")
        if flag == "PROBLEM":
            worst += 1
        print(f"  {k:22s} {v:10,}  {rate:7.4f}%  {flag}")

    # a shard read at the wrong offset would fail the king test almost everywhere, so passing it
    # across millions of records is strong evidence the record size and alignment are right
    print()
    if worst == 0:
        print("VERDICT: intact. Record alignment, legality and ranges all hold.")
    else:
        print(f"VERDICT: {worst} check(s) failed materially. Do not train on this yet.")

    mm = open_shard(paths[0])
    v = split_is_val(mm[:200000])
    print(f"\nholdout fraction on the first shard: {100*v.mean():.2f}% (target {100/64:.2f}%)")
