r"""pack_selfplay_visits turns MayFly's own self-play positions into policy records whose target is the search's
visit spread, the way Leela's training data is built.

selfplay.py's TARGETS_OUT rows (the visit count `n` of every root move, the root's `q`, the move played and the
game's result `z`) become 112-byte packed records (pack.DT) in sv_*.bin shards:

  policy      the five most visited moves, cp5 = 100 ln(p / p_best), which batch_tensors' softmax at temperature 100
              turns back into the visit shares (lc0_convert's mapping for Leela's records)
  value       cp from the root's q (the value target, unused while the main run's W_V is 0)
  plies left  from the finished game (flags bit 8); a game cut off by a restart has none
  features    the real positional features where gen_packed imports, else Leela's (aux off)

The trainer reads them as one more slice (train_av SV_SHARE, 0 by default), so nothing changes until a test says so.

  python chess/pack_selfplay_visits.py "GLOB[;GLOB...]" OUT_DIR
  python chess/pack_selfplay_visits.py --check OUT_DIR       decode some records back and compare with the visits
  env: MIN_TOTAL=600 (a search with fewer visits in all is skipped: keeps 800-visit searches, which often end a few
       short, and drops the 200-visit quick moves)  DEDUP_MAX=2 (copies of one position kept)
       SHARD=4000000  PREFIX=sv  TAG=
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import chess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from encode import board_to_ids, clock_and_ep  # noqa: E402
from moves import move_to_idx  # noqa: E402
from pack import DT, REC, encode_record  # noqa: E402

E = os.environ.get
MIN_TOTAL = int(E("MIN_TOTAL", 600))
DEDUP_MAX = int(E("DEDUP_MAX", 2))
SHARD = int(E("SHARD", 4_000_000))
PREFIX, TAG = E("PREFIX", "sv"), E("TAG", "")
FLUSH = 4096
PV = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
try:
    from gen_packed import features                  # needs stockfish_path.txt beside it
except Exception:
    features = None


def leela_features(board, flip):
    """What lc0_convert writes for Leela's records: material, bishop pairs and pawn counts, nothing else."""
    me, them = (chess.BLACK, chess.WHITE) if flip else (chess.WHITE, chess.BLACK)
    mat = lambda c: sum(PV.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == c)  # noqa: E731
    return {"material": [mat(me), mat(them)], "stm_mobility": min(board.legal_moves.count(), 255),
            "king_safety": [0, 0], "pawns": [[len(board.pieces(chess.PAWN, me)), 0, 0],
                                             [len(board.pieces(chess.PAWN, them)), 0, 0]],
            "bishop_pair": [int(len(board.pieces(chess.BISHOP, me)) >= 2),
                            int(len(board.pieces(chess.BISHOP, them)) >= 2)]}


def rows_of(patterns):
    """Every target row, in file order, from the ;-separated globs."""
    for pat in patterns.split(";"):
        for p in sorted(glob.glob(pat)):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue                  # a line cut off by a kill


def record_args(r, plies_left):
    """One target row to encode_record's arguments, or None when it teaches nothing."""
    n = {m: int(v) for m, v in r.get("n", {}).items() if int(v) > 0}
    total = sum(n.values())
    if total < MIN_TOTAL or len(n) < 2:
        return None                               # a short search, or a forced move
    board = chess.Board(r["fen"])
    ids, cast, flip = board_to_ids(board)
    hm, ep = clock_and_ep(board, flip)
    top = sorted(n, key=lambda m: -n[m])[:5]
    idx = [move_to_idx(chess.Move.from_uci(m), flip) for m in top]
    best = n[top[0]]
    cps = [int(round(max(-32000.0, 100.0 * math.log(n[m] / best)))) for m in top]
    q = min(0.999, max(-0.999, float(r.get("q", 0.0))))
    cp = int(max(-32000, min(32000, -400.0 * math.log10(2.0 / (q + 1.0) - 1.0))))
    feat, aux = (features(board, flip), True) if features else (leela_features(board, flip), False)
    return (ids, cast, idx, cps, cp, feat, int(r.get("ply", 0)), aux, hm, ep, True, True, plies_left)


def pack(patterns, out):
    os.makedirs(out, exist_ok=True)
    games = defaultdict(list)
    order = []
    for r in rows_of(patterns):
        g = r.get("game", "")
        if g not in games:
            order.append(g)
        games[g].append(r)
    seen = defaultdict(int)
    buf = np.zeros(FLUSH, DT)
    k = written = shard = in_shard = skipped = dup = 0
    f = None

    def flush():
        nonlocal k, f, shard, in_shard
        if not k:
            return
        if f is None:
            f = open(os.path.join(out, f"{PREFIX}_{TAG}{shard:03d}.bin"), "wb")
        f.write(buf[:k].tobytes())
        in_shard += k
        k = 0
        if in_shard >= SHARD:
            f.close()
            f, shard, in_shard = None, shard + 1, 0

    for g in order:
        rows = games[g]
        done = rows[-1].get("z") is not None and rows[-1].get("result", "*") != "*"
        last = max(int(x.get("ply", 0)) for x in rows)
        for r in rows:
            key = " ".join(r["fen"].split()[:4])          # the position, not its move counters
            if seen[key] >= DEDUP_MAX:
                dup += 1
                continue
            a = record_args(r, (last + 1 - int(r.get("ply", 0))) if done else None)
            if a is None:
                skipped += 1
                continue
            seen[key] += 1
            encode_record(*a, out=buf[k])
            k += 1
            written += 1
            if k == FLUSH or in_shard + k >= SHARD:
                flush()
    flush()
    if f is not None:
        f.close()
    print(f"{len(order):,} games, {written:,} records to {out} ({PREFIX}_{TAG}*.bin), {skipped:,} skipped "
          f"(fewer than {MIN_TOTAL} visits or one move), {dup:,} repeats over {DEDUP_MAX}; "
          f"features {'real' if features else 'Leela-style'}", flush=True)


def check(out, n=2000):
    """Decode records back through batch_tensors and compare the policy target with the visit shares it came from."""
    from pack import batch_tensors, legal_mask
    from chess_net import N_MOVES
    p = sorted(glob.glob(os.path.join(out, f"{PREFIX}_*.bin")))[0]
    rows = np.fromfile(p, dtype=DT, count=n)
    bt = batch_tensors(rows, N_MOVES, 100.0)
    mask = legal_mask(rows, N_MOVES)
    pol = bt["policy"]
    illegal = float((pol * ~mask).sum() / len(rows))
    err = []
    for i, r in enumerate(rows):
        m = r["mv"][:r["nmv"]]
        w = np.exp(r["cp5"][:r["nmv"]] / 100.0)
        err.append(np.abs(pol[i, m] / pol[i, m].sum() - w / w.sum()).max())
    print(f"{len(rows):,} records from {os.path.basename(p)}: policy mass on illegal moves {illegal:.4f}, "
          f"largest share error against the visits {max(err):.4f}, plies left on {(rows['flags'] & 8 > 0).mean():.0%}",
          flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        check(sys.argv[2])
    else:
        pack(sys.argv[1], sys.argv[2])
