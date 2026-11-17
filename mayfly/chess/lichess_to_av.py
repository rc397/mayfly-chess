r"""The Lichess deep evaluations as action-value records for the action-value heads.

The Lichess cloud evaluations (fetch_lichess_evals.py) are deep Stockfish searches, a median 25 million
nodes a position, with 3 to 5 moves scored in each. This writes each scored move as one action-value
record (pack_av's 40-byte format), read by the trainer's AVLoader as the lv_* shards.

Only Lichess rows are read. The lz_NNN shards also hold Leela records, whose scores are visit shares,
not evaluations. Lichess rows carry flags & 1 and nothing else does, so that bit is the filter.
Centipawns become a win probability by ChessBench's logistic (deep_label.win_prob), the same curve
as every other action value, and a clamped mate (20,000 or more) is certain.

Older shards store White's scores with Black to move. pack.batch_tensors turns them round at load
time by the same test used here (a Lichess record whose scores rise from first to last). Correctly
written records never rise.

Rows of one position sit together, so the trainer reads these thinned (LV_THIN). The holdout hashes
the board, so a validation position of the policy corpus stays validation here. A shard whose output
already exists is skipped.

    python chess/lichess_to_av.py [MAX_SHARDS]      env: SRC=<data>/shards OUT=same PREFIX=lv
"""
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pack import DT, castle_by_king           # noqa: E402
from pack_av import DT_AV, pack_ids           # noqa: E402
import paths                                  # noqa: E402

E = os.environ.get
SRC = E("SRC", paths.under("shards"))
OUT = E("OUT", SRC)
PREFIX = E("PREFIX", "lv")
K_CHESSBENCH = 0.00368208                     # deep_label.K_CHESSBENCH, ChessBench's own constant
MATE = 20000


def win_prob(cp):
    """ChessBench's curve, vectorised, with a clamped mate score certain."""
    p = 1.0 / (1.0 + np.exp(-K_CHESSBENCH * np.clip(cp, -MATE, MATE)))
    return np.where(cp >= MATE, 1.0, np.where(cp <= -MATE, 0.0, p))


def convert(rows):
    """Policy records (pack.DT) of the Lichess set to action-value records, one per scored move.

    Returns the records and a tally. Only the Lichess rows are read (flags & 1). Their scores may
    be White's, and an action value needs the sign, so a row whose side to move cannot be read off
    its scores is dropped rather than guessed (equal non-zero scores, or scores that neither fall
    nor rise).
    """
    lich = (rows["flags"] & 1) > 0
    rows = rows[lich]
    nmv = rows["nmv"].astype(np.int64)
    valid = np.arange(5)[None, :] < nmv[:, None]
    cp5 = rows["cp5"].astype(np.float64)
    rise, live = np.diff(cp5, axis=1), valid[:, 1:]
    falls = np.all((rise <= 0) | ~live, axis=1)
    rises = np.all((rise >= 0) | ~live, axis=1)
    tie = falls & rises
    level = np.all((cp5 == 0) | ~valid, axis=1)            # a tie at 0.00 reads the same both ways
    inv = rises & ~tie                                     # Black to move, stored as White's
    keep = (falls & ~tie) | inv | (tie & level)
    tally = {"lichess": int(lich.sum()), "turned": int((inv & keep).sum()),
             "dropped_tie": int((tie & ~level).sum()), "dropped_mixed": int((~falls & ~rises).sum())}
    rows, nmv, valid, cp5, inv = rows[keep], nmv[keep], valid[keep], cp5[keep], inv[keep]
    cp5 = np.where(inv[:, None], -cp5, cp5)
    mv = castle_by_king(rows["ids"], rows["mv"])          # the castle on the king's own square
    ri, ci = np.nonzero(valid)
    out = np.zeros(len(ri), DT_AV)
    ids = rows["ids"][ri].astype(np.uint8)
    out["ids4"] = (ids[:, 0::2] | (ids[:, 1::2] << 4)).astype(np.uint8)
    c = rows["cast"][ri].astype(np.int64)
    out["cast"] = (c[:, 0] | (c[:, 1] << 1) | (c[:, 2] << 2) | (c[:, 3] << 3)).astype(np.uint8)
    out["mv"] = mv[ri, ci]
    out["q"] = np.round(win_prob(cp5[ri, ci]) * 65535).astype(np.uint16)
    # the clock and en passant bits carry over exactly as the policy record had them
    out["flags"] = (rows["flags"][ri] & 6).astype(np.uint8)
    out["hm"] = rows["hm"][ri]
    out["ep"] = rows["ep"][ri]
    return out, tally


def has_lichess(path):
    """A cheap check whether this shard holds any Lichess rows."""
    mm = np.memmap(path, dtype=DT, mode="r")
    return bool(np.any(np.asarray(mm["flags"][::997]) & 1))


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    src = [p for p in sorted(glob.glob(os.path.join(SRC, "lz_[0-9][0-9][0-9]*.bin"))) if has_lichess(p)]
    if limit:
        src = src[:limit]
    print(f"{len(src)} shards hold Lichess rows; converting into {PREFIX}_*.bin", flush=True)
    t0, total = time.time(), {"lichess": 0, "turned": 0, "dropped_tie": 0, "dropped_mixed": 0, "out": 0}
    for p in src:
        name = os.path.basename(p).replace("lz_", PREFIX + "_", 1)
        dst = os.path.join(OUT, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        out, tally = convert(np.fromfile(p, dtype=DT))
        with open(dst + ".tmp", "wb") as fh:
            out.tofile(fh)
        os.replace(dst + ".tmp", dst)
        for k, v in tally.items():
            total[k] += v
        total["out"] += len(out)
        print(f"  {name}: {tally['lichess']:,} Lichess positions -> {len(out):,} action values "
              f"({tally['turned']:,} turned round, {tally['dropped_tie'] + tally['dropped_mixed']:,} "
              f"dropped), {time.time() - t0:.0f}s", flush=True)
    print(f"done this run: {total}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
