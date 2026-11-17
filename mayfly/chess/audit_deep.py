"""Check the packed 112-byte corpora for silent data bugs.

These are bugs that do not throw and do not show in a loss curve or a record count, such as mirrored
boards, auxiliary fields that are zero in one source and real in another, train and validation splits
that disagree across sources, and policy targets that do not sum to one.

    python chess/audit_deep.py [DIR]
"""
import os, sys, glob, collections
import numpy as np
import chess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pack import DT, REC, open_shard, batch_tensors, split_is_val
from chess_net import move_to_idx, N_MOVES
import paths as data_paths    # the loops below use `paths` for shard lists

DIR = sys.argv[1] if len(sys.argv) > 1 else data_paths.SHARDS
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


# Only the 112-byte packed corpora. The 40-byte action-value shards have a different dtype and would
# read as noise here.
PACKED = ("lz", "lc", "gm", "sp")


def sources(d):
    out = collections.defaultdict(list)
    skipped = set()
    for p in sorted(glob.glob(os.path.join(d, "*.bin"))):
        pre = os.path.basename(p).split("_")[0]
        if pre not in PACKED:
            skipped.add(pre)
            continue
        out[pre].append(p)
    if skipped:
        print(f"  (skipping {', '.join(sorted(skipped))}: not the 112 byte record)")
    return out


def sample(paths, n, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    per = max(n // max(len(paths), 1), 1)
    for p in paths:
        mm = open_shard(p)
        if len(mm) == 0:
            continue
        k = min(per, len(mm))
        rows.append(mm[np.sort(rng.choice(len(mm), k, replace=False))])
    return np.concatenate(rows) if rows else np.zeros(0, DT)


print("1. THE MOVE SPACE\n")
# From-to pairs cannot name the promotion piece, so 66 indices were added for underpromotions. What
# matters is that no two moves legal in the same position share an index. A piece move and a queen
# promotion on the same squares do share one, but only one of them can be legal at a time.
from lc0_moves import policy_index
idxs = [move_to_idx(chess.Move.from_uci(u), False) for u in policy_index]
by_idx = collections.defaultdict(list)
for u, i in zip(policy_index, idxs):
    by_idx[i].append(u)
collided = {k: v for k, v in by_idx.items() if len(v) > 1}


def can_coexist(ucis):
    """Could two of these be legal in one position? Only if they want the same piece there."""
    return sum(1 for u in ucis if len(u) > 4) > 1 or sum(1 for u in ucis if len(u) == 4) > 1


bad = {k: v for k, v in collided.items() if can_coexist(v)}
check("no two simultaneously legal moves share an index", not bad,
      f"{len(collided)} indices shared, {len(bad)} by moves that could both be legal")
check("underpromotions are representable",
      all(move_to_idx(chess.Move.from_uci(u), False) >= 4096
          for u in policy_index if len(u) > 4 and u[4] != "q"),
      f"{N_MOVES} indices, {N_MOVES - 4096} of them for underpromotion")

print("\n2. POLICY TARGETS PER SOURCE\n")
src = sources(DIR)
for name, paths in src.items():
    rows = sample(paths, 6000, seed=1)
    if not len(rows):
        continue
    b = batch_tensors(rows, N_MOVES)
    s = b["policy"].sum(1)
    check(f"[{name}] policy rows sum to 1", bool(np.allclose(s, 1.0, atol=1e-3)),
          f"min {s.min():.4f} max {s.max():.4f}")
    # a record claiming five moves whose indices collide has fewer real entries than it says
    nz = (b["policy"] > 0).sum(1)
    claimed = rows["nmv"].astype(int)
    check(f"[{name}] distinct moves match nmv", bool((nz <= claimed).all() and
                                                     (nz >= 1).all()),
          f"mean claimed {claimed.mean():.2f}, mean distinct {nz.mean():.2f}")
    v = b["value"]
    check(f"[{name}] value targets in range", bool(np.isfinite(v).all() and
                                                   (v >= 0).all() and (v <= 1).all()),
          f"min {v.min():.3f} max {v.max():.3f} mean {v.mean():.3f}")

print("\n3. TRAIN AND VALIDATION ISOLATION ACROSS SOURCES\n")
# The split hashes the board, so a position in two corpora must land on the same side. A castling
# encoding that differs between converters would break that.
keysets = {}
for name, paths in src.items():
    rows = sample(paths, 40000, seed=2)
    if not len(rows):
        continue
    v = split_is_val(rows)
    keysets[name] = {
        "val": {r["ids"].tobytes() + r["cast"].tobytes() for r in rows[v]},
        "train": {r["ids"].tobytes() + r["cast"].tobytes() for r in rows[~v]},
        "rate": float(v.mean()),
    }
    check(f"[{name}] holdout rate near 1/64", 0.010 < v.mean() < 0.023, f"{100*v.mean():.2f}%")
names = list(keysets)
for i, a in enumerate(names):
    for b2 in names[i+1:]:
        bad = keysets[a]["val"] & keysets[b2]["train"]
        bad2 = keysets[b2]["val"] & keysets[a]["train"]
        check(f"{a} validation disjoint from {b2} training", not (bad or bad2),
              f"{len(bad) + len(bad2)} positions on both sides")

print("\n4. THE AUXILIARY FLAG\n")
for name, paths in src.items():
    rows = sample(paths, 8000, seed=3)
    if not len(rows):
        continue
    f = (rows["flags"] & 1).mean()
    expect = 0.0 if name == "lc" else 1.0
    check(f"[{name}] aux flag is {expect:.0f}", abs(f - expect) < 0.01, f"{100*f:.1f}% valid")
    if expect == 1.0:
        king = rows["king"].astype(float)
        check(f"[{name}] king safety is not all zeros", king.sum() > 0,
              f"mean {king.mean():.2f}")

print("\n5. BOARD LEGALITY PER SOURCE\n")
PIECE = {1: chess.PAWN, 2: chess.KNIGHT, 3: chess.BISHOP, 4: chess.ROOK,
         5: chess.QUEEN, 6: chess.KING}
for name, paths in src.items():
    rows = sample(paths, 1200, seed=4)
    if not len(rows):
        continue
    legal = tested = 0
    for r in rows:
        bd = chess.Board(None)
        for sq in range(64):
            v = int(r["ids"][sq])
            if v:
                bd.set_piece_at(sq, chess.Piece(PIECE[v if v <= 6 else v - 6],
                                                chess.WHITE if v <= 6 else chess.BLACK))
        bd.turn = chess.WHITE
        try:
            lm = {(m.from_square, m.to_square) for m in bd.legal_moves}
        except Exception:
            continue
        if not lm:
            continue
        tested += 1
        k = int(r["nmv"])
        legal += sum(1 for i in range(k) if divmod(int(r["mv"][i]), 64) in lm) / max(k, 1)
    pct = 100 * legal / max(tested, 1)
    check(f"[{name}] policy moves legal in their own position", pct > 95,
          f"{pct:.1f}% over {tested} positions")

print("\n6. DUPLICATES WITHIN AND ACROSS SOURCES\n")
allkeys = {}
for name, paths in src.items():
    rows = sample(paths, 30000, seed=5)
    if not len(rows):
        continue
    ks = [r["ids"].tobytes() for r in rows]
    u = len(set(ks))
    check(f"[{name}] mostly distinct positions", u / len(ks) > 0.80,
          f"{100*u/len(ks):.1f}% unique in sample")
    allkeys[name] = set(ks)
names = list(allkeys)
for i, a in enumerate(names):
    for b2 in names[i+1:]:
        ov = len(allkeys[a] & allkeys[b2])
        print(f"  note  {a} and {b2} share {ov} positions in these samples "
              f"({100*ov/min(len(allkeys[a]), len(allkeys[b2])):.2f}%)")

print("\n" + "=" * 66)
if FAILS:
    print(f"{len(FAILS)} CHECK(S) FAILED:")
    for f in FAILS:
        print(f"  - {f}")
else:
    print("every check passed")
