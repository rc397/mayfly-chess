r"""MayFly's own search targets as action-value shards (st_*), for training on the search's values.

The record is pack_selfplay.py's (pack_av's 40 bytes, the position, one move and a win probability for the side
to move), holding the search's value for each move it searched (qa, mapped from [-1, 1] to [0, 1]), what the
tree believed after its visits. The trainer reads them as one more slice of the action-value batch (train_av.py
ST_SHARE).

What is kept, and the settings to tune.
  a move     only with at least max(MIN_VISITS, MIN_SHARE x the position's visits) visits, since a move seen a
             few times has the mean of a few leaf values
  a position at most DEDUP_MAX times, since games often start from the initial position and openings repeat
  visits     MIN_TOTAL skips positions searched with fewer visits in all
  outcomes   OUTCOMES=1 adds pack_selfplay's flagged game-result record (bit 3). Off, because train_av has no
             reader for that bit and would train it as an action value

  python chess/pack_search.py "<data>/selfplay/targets_lap*.jsonl;<other>/targets.jsonl.gz" OUT_DIR
  env: MIN_VISITS=8  MIN_SHARE=0.02  DEDUP_MAX=2  MIN_TOTAL=0  OUTCOMES=0  SHARD_MB=256  PREFIX=st
"""
import glob
import gzip
import hashlib
import json
import os
import sys

import chess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from encode import board_to_ids, clock_and_ep
from moves import move_to_idx
from pack_av import DT_AV, REC_AV, encode_av
import paths

E = os.environ.get
MIN_VISITS = int(E("MIN_VISITS", 8))
MIN_SHARE = float(E("MIN_SHARE", 0.02))
DEDUP_MAX = int(E("DEDUP_MAX", 2))
MIN_TOTAL = int(E("MIN_TOTAL", 0))
OUTCOMES = E("OUTCOMES", "0") == "1"
SHARD_MB = int(E("SHARD_MB", 256))
PREFIX = E("PREFIX", "st")
Z_FLAG = 8


def rows_of(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else paths.under("selfplay", "targets_lap*.jsonl")
    out = sys.argv[2] if len(sys.argv) > 2 else paths.under("search_targets")
    files = []
    for pat in src.split(";"):
        files += sorted(glob.glob(pat)) if any(c in pat for c in "*?") else [pat]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit(f"no targets at {src}")
    os.makedirs(out, exist_ok=True)
    per_shard = max(1, SHARD_MB * 1024 * 1024 // REC_AV)
    buf = np.zeros(per_shard, DT_AV)
    at, shard = 0, 0
    seen = {}
    stats = {"positions": 0, "kept_positions": 0, "records": 0, "dupes": 0, "thin": 0, "moves_dropped": 0,
             "outcomes": 0}

    def flush():
        nonlocal at, shard
        if not at:
            return
        path = os.path.join(out, f"{PREFIX}_{shard:04d}.bin")
        buf[:at].tofile(path + ".part")
        os.replace(path + ".part", path)
        print(f"  {os.path.basename(path)}: {at:,} records", flush=True)
        shard += 1
        at = 0

    for path in files:
        for row in rows_of(path):
            stats["positions"] += 1
            n, qa = row.get("n") or {}, row.get("qa") or {}
            total = sum(n.values())
            if total < MIN_TOTAL:
                stats["thin"] += 1
                continue
            key = hashlib.blake2b(" ".join(row["fen"].split()[:4]).encode(), digest_size=8).digest()
            if seen.get(key, 0) >= DEDUP_MAX:
                stats["dupes"] += 1
                continue
            seen[key] = seen.get(key, 0) + 1
            board = chess.Board(row["fen"])
            ids, cast, flip = board_to_ids(board)
            hm, ep = clock_and_ep(board, flip)
            legal = {m.uci(): m for m in board.legal_moves}
            floor = max(MIN_VISITS, MIN_SHARE * total)
            wrote = 0
            for uci, visits in n.items():
                if visits < floor or uci not in legal or uci not in qa:
                    stats["moves_dropped"] += 1
                    continue
                if at >= per_shard:
                    flush()
                mv = legal[uci]
                encode_av(ids, cast, move_to_idx(mv, flip), (float(qa[uci]) + 1.0) * 0.5,
                          (1 if mv.promotion else 0) | 6, buf[at], hm, ep)
                at += 1
                wrote += 1
            z, played = row.get("z"), row.get("played")
            if OUTCOMES and z is not None and played in legal:
                if at >= per_shard:
                    flush()
                mv = legal[played]
                encode_av(ids, cast, move_to_idx(mv, flip), float(z), (1 if mv.promotion else 0) | 6 | Z_FLAG,
                          buf[at], hm, ep)
                at += 1
                wrote += 1
                stats["outcomes"] += 1
            stats["records"] += wrote
            stats["kept_positions"] += 1 if wrote else 0
    flush()
    meta = dict(stats, source=files, min_visits=MIN_VISITS, min_share=MIN_SHARE, dedup_max=DEDUP_MAX,
                min_total=MIN_TOTAL, outcomes=OUTCOMES, shards=shard, prefix=PREFIX,
                distinct_positions=len(seen))
    json.dump(meta, open(os.path.join(out, f"{PREFIX}_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1)[:1500])


if __name__ == "__main__":
    main()
