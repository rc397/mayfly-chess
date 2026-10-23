r"""Turn self-play's search-improved targets into shards the trainer reads.

Self-play targets come from the model's own search, which is stronger than the bare network, so
training on them can move the network past a fixed teacher, as in AlphaZero and Leela.

The output is the existing 40-byte action-value record (pack_av), so the trainer reads it unchanged.
One record per move the search visited.

    mv   the move, in the side to move's frame, as in ChessBench's records
    q    the search's mean value for that move, mapped from [-1, 1] to [0, 1], where ChessBench's
         q is Stockfish's win probability

MIN_VISITS drops moves with too few visits to have a meaningful mean.

The game outcome rides in the same file on a spare flag bit. Bit 3 means "this record's q is the game
result for the side to move, not an action value for the move named", with one such record a
position carrying the move played and z as its value. The trainer does not read bit 3 yet, so it
would train that record as one more, noisier action value.

  python chess/pack_selfplay.py [TARGETS_JSONL] [OUT_DIR]
  env: MIN_VISITS=3  SHARD_MB=256  MIN_PLY=0
"""
import os, sys, json, glob
import numpy as np
import chess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from encode import board_to_ids, clock_and_ep
from moves import move_to_idx
from pack_av import DT_AV, REC_AV, encode_av
import paths

E = os.environ.get
# Bit 0 promotion, bit 1 clock real, bit 2 en passant real, and bit 3 means this record's q is a
# game result for the side to move rather than an action value for the move it names.
Z_FLAG = 8
MIN_VISITS = int(E("MIN_VISITS", 3))
MIN_PLY = int(E("MIN_PLY", 0))
SHARD_MB = int(E("SHARD_MB", 256))
SRC = sys.argv[1] if len(sys.argv) > 1 else paths.under("selfplay", "targets.jsonl")
OUT = sys.argv[2] if len(sys.argv) > 2 else paths.under("selfplay", "shards")


def rows_of(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def records(row, buf, at):
    """Write this position's visited moves into buf, returning how many were written.

    The board is encoded once per position, and the records differ only in mv and q, the same
    shape as the ChessBench corpus this trains beside.
    """
    board = chess.Board(row["fen"])
    ids, cast, flip = board_to_ids(board)
    hm, ep = clock_and_ep(board, flip)
    legal = {m.uci(): m for m in board.legal_moves}
    n = row.get("n") or {}
    qa = row.get("qa") or {}
    wrote = 0
    for uci, visits in n.items():
        if visits < MIN_VISITS or uci not in legal or uci not in qa:
            continue
        if at + wrote >= len(buf):
            break
        mv = legal[uci]
        # the search's value is in [-1, 1] for the side to move, the record holds a win probability
        q = (float(qa[uci]) + 1.0) * 0.5
        # Bits 1 and 2, the clock and the en passant square come from a real FEN, as deep_label
        # marks them. Without them the loader reads hm as unknown (101) and ep as none.
        flags = (1 if mv.promotion else 0) | 6
        encode_av(ids, cast, move_to_idx(mv, flip), q, flags, buf[at + wrote], hm, ep)
        wrote += 1
    # The outcome record, one a position, flagged so the trainer can tell it apart. It names the
    # move played, and its q is the result of the game from here.
    z = row.get("z")
    played = row.get("played")
    if z is not None and played in legal and at + wrote < len(buf):
        mv = legal[played]
        encode_av(ids, cast, move_to_idx(mv, flip), float(z),
                  (1 if mv.promotion else 0) | 6 | Z_FLAG, buf[at + wrote], hm, ep)
        wrote += 1
    return wrote


def main():
    files = sorted(glob.glob(SRC)) if any(c in SRC for c in "*?") else [SRC]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        raise SystemExit(f"no targets at {SRC}")
    os.makedirs(OUT, exist_ok=True)
    per_shard = max(1, SHARD_MB * 1024 * 1024 // REC_AV)
    buf = np.zeros(per_shard, DT_AV)
    at = shard = 0
    positions = written = skipped = 0
    outcomes = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
    zs = []

    def flush():
        nonlocal at, shard
        if not at:
            return
        path = os.path.join(OUT, f"selfplay_{shard:04d}.bin")
        tmp = path + ".part"
        buf[:at].tofile(tmp)
        os.replace(tmp, path)
        print(f"  {os.path.basename(path)}: {at:,} records", flush=True)
        shard += 1
        at = 0

    for path in files:
        for row in rows_of(path):
            if row.get("ply", 0) < MIN_PLY:
                skipped += 1
                continue
            positions += 1
            outcomes[row.get("result", "*")] = outcomes.get(row.get("result", "*"), 0) + 1
            if row.get("z") is not None:
                zs.append(float(row["z"]))
            got = records(row, buf, at)
            at += got
            written += got
            if at >= per_shard - 64:
                flush()
    flush()

    # A sidecar describing the shards, so later readers know where they came from.
    meta = {"source": files, "positions": positions, "records": written, "skipped": skipped,
            "z_flag": Z_FLAG,
            "min_visits": MIN_VISITS, "outcomes": outcomes,
            "z_mean": round(float(np.mean(zs)), 4) if zs else None,
            "z_known": len(zs), "shards": shard}
    with open(os.path.join(OUT, "selfplay_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"\n{positions:,} positions -> {written:,} records in {shard} shard(s)")
    print(f"  {written / max(positions, 1):.1f} moves a position at {MIN_VISITS}+ visits")
    print(f"  results {outcomes}, z known for {len(zs):,} rows"
          + (f", mean {np.mean(zs):.3f}" if zs else ""))


if __name__ == "__main__":
    main()
