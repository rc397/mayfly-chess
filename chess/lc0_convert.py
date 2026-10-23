r"""
Read Leela v6 training records and write them in MayFly's packed format.

A v6 record is 8,356 bytes, so only the 112-byte packed form is practical at billions of positions.

Two things can go wrong silently. Leela stores 104 bitboards, and the first 12 are the current
board (our pawn, knight, bishop, rook, queen, king, then theirs), already from the side to move's
view, so nothing is flipped here. A wrong plane order gives legal looking but wrong boards, so
decode_planes is checked against a hand-encoded position below. The 1858 policy probabilities
follow Leela's fixed move list, which chess/lc0_moves.py generates.

These targets are Leela's visit distributions, not Stockfish evaluations. They are turned into
pseudo centipawns only to fit the record, and the softmax in batch_tensors recovers the visit
distribution. Mixing them with Stockfish data means training on two teachers.

  python chess/lc0_convert.py <tar file or directory> [MAX]
  env: OUT=<data>/shards  PREFIX=lc  SHARD=4000000  MIN_VISITS=100  TAG=
"""
import os, sys, glob, gzip, io, tarfile, struct, time
import numpy as np
import chess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import board_to_ids, move_to_idx, N_MOVES
from pack import DT, REC, encode_record
from lc0_moves import policy_index
import paths

E = os.environ.get
OUT = E("OUT", paths.under("shards"))
PREFIX = E("PREFIX", "lc")
# A shard tag namespaces this process's output. Sharder writes {prefix}_{tag}{index:03d}.bin and
# resumes from that tag's own sequence, so several converters can share one directory without a
# lock. lc0_pool uses it.
TAG = E("TAG", "")
SHARD = int(E("SHARD", 4_000_000))
MIN_VISITS = int(E("MIN_VISITS", 100))
FLUSH = 4096
V6_SIZE = 8356
PV_TAKE = 5

# offsets into the v6 record, from the format documentation
OFF_VERSION, OFF_INPUT = 0, 4
OFF_PROBS = 8                      # float32[1858]
OFF_PLANES = OFF_PROBS + 1858 * 4  # uint64[104]
OFF_CAST = OFF_PLANES + 104 * 8    # us_ooo, us_oo, them_ooo, them_oo
OFF_STM = OFF_CAST + 4
OFF_RULE50 = OFF_STM + 1
OFF_INVARIANCE = OFF_RULE50 + 1
OFF_BESTQ = OFF_INVARIANCE + 2 + 4         # dummy, then root_q, then best_q
# Then root_d, best_d, root_m, best_m and plies_left, the real number of plies to the end of the
# game, which is the moves-left head's label. It falls by one from record to record within a game.
OFF_PLIES = OFF_BESTQ + 20
OFF_VISITS = 8356 - 4 - 4 - 4               # visits, then played_idx/best_idx, policy_kld, reserved

MOVE_IDX = np.full(1858, -1, np.int32)
for i, uci in enumerate(policy_index):
    try:
        MOVE_IDX[i] = move_to_idx(chess.Move.from_uci(uci), False)
    except Exception:
        pass


# Two Leela move names need the board to read. A plain seventh to eighth rank pawn move is Leela's
# knight promotion (its list has d7d8, d7d8q, d7d8r, d7d8b and no d7d8n), while move_to_idx would read
# it as a queen promotion. Castling is written king takes rook (e1h1, e1a1), where python-chess and
# the search use e1g1 and e1c1.
_KNIGHT_PROMO, _CASTLE = {}, {}
for _i, _u in enumerate(policy_index):
    if len(_u) == 4 and _u[1] == "7" and _u[3] == "8":
        _KNIGHT_PROMO[_i] = move_to_idx(chess.Move(chess.parse_square(_u[:2]), chess.parse_square(_u[2:4]),
                                                   promotion=chess.KNIGHT), False)
    if _u in ("e1h1", "e1a1"):
        _CASTLE[_i] = move_to_idx(chess.Move.from_uci("e1g1" if _u == "e1h1" else "e1c1"), False)


def lc0_move_idx(i, ids):
    """Leela policy index i as MayFly's move index, reading the board where the name is ambiguous."""
    if i in _KNIGHT_PROMO and ids[chess.parse_square(policy_index[i][:2])] == 1:      # our pawn
        return _KNIGHT_PROMO[i]
    if i in _CASTLE and ids[chess.E1] == 6:                                            # our king
        return _CASTLE[i]
    return int(MOVE_IDX[i])


# trainingdata.cc writes every plane through ReverseBitsInBytes, which mirrors each rank. Undoing it
# takes legal decoded policy moves from about 10 percent to 99.8 percent. The format documentation
# does not mention it.
_REV = np.array([int(f"{b:08b}"[::-1], 2) for b in range(256)], np.uint8)


def _unreverse(v):
    b = np.frombuffer(np.uint64(v).tobytes(), np.uint8)
    return int(np.frombuffer(_REV[b].tobytes(), "<u8")[0])


def decode_planes(planes):
    """The 12 current-position piece planes to MayFly's 64 piece ids.

    planes[0:6]  our pawn, knight, bishop, rook, queen, king
    planes[6:12] theirs, same order
    Ids follow chess_net: 1..6 mine, 7..12 theirs, which is exactly this order.
    """
    ids = np.zeros(64, np.uint8)
    for p in range(12):
        bb = _unreverse(planes[p])
        while bb:
            sq = (bb & -bb).bit_length() - 1
            ids[sq] = p + 1
            bb &= bb - 1
    return ids


def read_record(buf):
    """One v6 record to the arguments encode_record wants, or None if unusable."""
    if len(buf) < V6_SIZE:
        return None
    probs = np.frombuffer(buf, np.float32, 1858, OFF_PROBS)
    planes = np.frombuffer(buf, "<u8", 104, OFF_PLANES)
    cast = np.frombuffer(buf, np.uint8, 4, OFF_CAST)
    best_q = struct.unpack_from("<f", buf, OFF_BESTQ)[0]
    # A few records have a non-finite Q. Drop the record, not the whole archive.
    if not np.isfinite(best_q):
        return None
    # A search of a handful of visits is mostly its prior. The visit count is the uint32 16 bytes
    # from the end. OFF_VISITS above points at played_idx, four bytes later, so it is not used here.
    if struct.unpack_from("<I", buf, V6_SIZE - 16)[0] < MIN_VISITS:
        return None
    # Leela carries the fifty move counter but not the en passant square in any readable form, so that
    # stays marked unknown rather than guessed.
    rule50 = int(buf[OFF_RULE50])

    ids = decode_planes(planes)
    if (ids == 6).sum() != 1 or (ids == 12).sum() != 1:
        return None                                  # not a legal board, drop it

    good = np.where((probs > 0) & (MOVE_IDX >= 0))[0]
    if len(good) < 2:
        return None
    top = good[np.argsort(-probs[good])[:PV_TAKE]]
    if not np.all(np.isfinite(probs[top])):
        return None                                  # same reasoning, for the visit shares
    idx = [lc0_move_idx(int(i), ids) for i in top]
    # visit share to a pseudo centipawn, monotone in probability, so the softmax over cp5 in
    # batch_tensors reproduces an ordering faithful to Leela's visit distribution
    p = probs[top].astype(np.float64)
    p = np.clip(p / max(p.sum(), 1e-9), 1e-6, 1.0)
    cps = np.clip(np.log(p / p[0]) * 100.0, -32000, 32000).astype(int).tolist()

    q = float(np.clip(best_q, -0.999, 0.999))
    cp = int(np.clip(-400.0 * np.log10(2.0 / (q + 1.0) - 1.0), -32000, 32000))

    plies = struct.unpack_from("<f", buf, OFF_PLIES)[0]
    plies = float(plies) if np.isfinite(plies) and 0.0 <= plies < 65535.0 else None

    # castling order in MayFly's format is own kingside, own queenside, theirs kingside, theirs queenside
    c = [int(cast[1]), int(cast[0]), int(cast[3]), int(cast[2])]
    feat = {
        "material": [0, 0], "stm_mobility": 0, "king_safety": [0, 0],
        "pawns": [[0, 0, 0], [0, 0, 0]], "bishop_pair": [0, 0],
    }
    PVAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}
    feat["material"] = [sum(PVAL.get(int(v), 0) for v in ids if 1 <= v <= 5),
                        sum(PVAL.get(int(v) - 6, 0) for v in ids if 7 <= v <= 11)]
    feat["bishop_pair"] = [int((ids == 3).sum() >= 2), int((ids == 9).sum() >= 2)]
    feat["pawns"] = [[int((ids == 1).sum()), 0, 0], [int((ids == 7).sum()), 0, 0]]
    feat["stm_mobility"] = min(len(good), 255)
    return ids, c, idx, cps, cp, feat, rule50, plies


def records_in(tar_path):
    """Every v6 record inside a training tar, whose members are gzipped chunks."""
    # A partly downloaded archive ends mid-member. Stop cleanly there and keep what was read.
    try:
        tf = tarfile.open(tar_path, "r")
    except Exception:
        return
    try:
        for m in tf:
            if not m.isfile():
                continue
            try:
                f = tf.extractfile(m)
                if f is None:
                    continue
                raw = f.read()
            except Exception:
                break                      # truncated member, archive ends here
            if m.name.endswith(".gz"):
                try:
                    raw = gzip.decompress(raw)
                except Exception:
                    continue
            for off in range(0, len(raw) - V6_SIZE + 1, V6_SIZE):
                yield raw[off:off + V6_SIZE]
    finally:
        tf.close()


if __name__ == "__main__":
    # Check first that a hand-built position survives encode and decode.
    b = chess.Board()
    ids_ref, cast_ref, flip = board_to_ids(b)
    raw = np.zeros(104, np.uint64)
    for sq, pc in b.piece_map().items():
        plane = (pc.piece_type - 1) + (0 if pc.color == b.turn else 6)
        raw[plane] |= np.uint64(1) << np.uint64(sq)
    planes = np.array([_unreverse(int(x)) for x in raw], "<u8")   # as lc0 would write it
    got = decode_planes(planes)
    assert np.array_equal(got, np.array(ids_ref, np.uint8)), \
        f"plane decoding disagrees with board_to_ids\n{got}\n{ids_ref}"
    print("plane decoding matches board_to_ids on the start position")
    print(f"policy map: {int((MOVE_IDX >= 0).sum())} of 1858 moves map into the {N_MOVES} space")

    src = sys.argv[1] if len(sys.argv) > 1 else None
    if not src:
        print("\nno tar given, verification only")
        sys.exit(0)
    paths = sorted(glob.glob(os.path.join(src, "*.tar"))) if os.path.isdir(src) else [src]
    print(f"\n{len(paths)} archive(s)")
    from fetch_lichess_evals import Sharder
    sh = Sharder(OUT, PREFIX, SHARD, TAG)
    kept = seen = 0
    t0 = time.time()
    for p in paths:
        for buf in records_in(p):
            seen += 1
            r = read_record(buf)
            if r is None:
                continue
            ids, c, idx, cps, cp, feat, rule50, plies = r
            sh.write((ids, c, idx, cps, cp, feat, 0, False, rule50, 0, True, False, plies))
            kept += 1
            if kept % 200000 == 0:
                print(f"  {kept:,} kept of {seen:,}  {kept/(time.time()-t0):.0f}/s", flush=True)
    sh.close()
    print(f"\n{kept:,} written from {seen:,} records ({100*kept/max(seen,1):.0f}% keep)")
