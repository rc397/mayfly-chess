"""
The on-disk record format for the policy corpus.

Positions are encoded once, at generation time, into fixed-width binary records, and training
memory-maps the files and slices them, with no JSON, no python-chess and no per-batch Python loop.
Parsing JSON and rebuilding boards would otherwise cost a large share of every training step.

  112 bytes per record, so 5 billion positions take 560 GB.

Layout, all little endian.

  offset  size  field
       0    64  ids          uint8   piece per square, side-to-move canonical
      64     4  cast         uint8   castling rights, 0 or 1
      68    10  mv           uint16  top-5 move indices into the 4096 move space
      78    10  cp5          int16   centipawns for each of those, side-to-move POV
      88     2  cp           int16   centipawns for the best move
      90     2  material     uint8   [mine, theirs], already side-swapped
      92     2  king         uint8
      94     6  pawns        uint8   [total, doubled, isolated] each side
     100     1  mob          uint8   side-to-move mobility
     101     2  bishop       uint8   bishop pair flags
     103     1  nmv          uint8   how many of the five slots are real
     104     2  ply          uint16  ply number within its game, so game boundaries survive
     106     1  flags        uint8   bit 0 aux real, bit 1 clock real, bit 2 en passant real
     107     1  hm           uint8   halfmove clock, which decides whether a won position is a draw
     108     1  ep           uint8   en passant target as square plus one, 0 none, already mirrored
     109     2  mleft        uint16  plies left in the game (Leela's plies_left), valid if flags & 8
     111     1  _pad         uint8   reserved

Leela's format carries no king safety or pawn structure, so those fields are zero in Leela shards
while other sources write real values. Bit 0 of flags lets the loss ignore records that do not have
the auxiliary features, instead of training the heads on contradictory zeros.

The ply field supports test-time training, which needs consecutive positions from one game. Records
are written in game order, so a drop in ply marks a new game.

Everything is stored from the side to move's point of view, matching board_to_ids, so no flipping
happens at training time.
"""
import os
import numpy as np

REC = 112
DT = np.dtype([
    ("ids", np.uint8, 64),
    ("cast", np.uint8, 4),
    ("mv", "<u2", 5),
    ("cp5", "<i2", 5),
    ("cp", "<i2"),
    ("material", np.uint8, 2),
    ("king", np.uint8, 2),
    ("pawns", np.uint8, 6),
    ("mob", np.uint8),
    ("bishop", np.uint8, 2),
    ("nmv", np.uint8),
    ("ply", "<u2"),
    ("flags", np.uint8),
    # The clock and the en passant square came out of the padding, so the record size did not
    # change and older shards still parse. Their bytes are zero, and the validity bits in flags make
    # an old record say "unknown" rather than something false.
    ("hm", np.uint8),
    ("ep", np.uint8),
    # Moves left, for the moves-left head. Two of the three reserved bytes, so the record stays 112
    # and older shards parse unchanged. flags bit 8 says the value is real. Only Leela's records carry
    # it, since the ply field was never filled in the other sources.
    ("mleft", "<u2"),
    ("_pad", np.uint8, 1),
])
assert DT.itemsize == REC, f"record is {DT.itemsize} bytes, expected {REC}"

CLIP = 32000                      # mate scores are 100000 in the raw labels, clamp to fit int16


def encode_record(fen_ids, cast, mv_idx, cps, cp, feat_sw, ply=0, aux_valid=True,
                  hm=0, ep=0, clock_valid=False, ep_valid=False, mleft=None, out=None):
    """Fill one record. feat_sw is the already-side-swapped feature dict.

    ep is the en passant target as square plus one, zero meaning none, and it must already be
    mirrored when black is to move, exactly like the board.

    The clock matters, since the same board can be winning at halfmove 10 and a dead draw at
    halfmove 99.
    """
    r = np.zeros(1, DT)[0] if out is None else out
    r["ids"][:] = fen_ids
    r["cast"][:] = [int(bool(c)) for c in cast]
    n = min(len(mv_idx), 5)
    r["nmv"] = n
    r["mv"][:n] = mv_idx[:n]
    r["cp5"][:n] = np.clip(cps[:n], -CLIP, CLIP)
    r["cp"] = int(np.clip(cp, -CLIP, CLIP))
    r["material"][:] = np.clip(feat_sw["material"], 0, 255)
    r["king"][:] = np.clip(feat_sw["king_safety"], 0, 255)
    pw = feat_sw["pawns"]
    r["pawns"][:] = np.clip([pw[0][0], pw[0][1], pw[0][2], pw[1][0], pw[1][1], pw[1][2]], 0, 255)
    r["mob"] = min(int(feat_sw["stm_mobility"]), 255)
    r["bishop"][:] = [int(bool(x)) for x in feat_sw["bishop_pair"]]
    r["ply"] = min(int(ply), 65535)
    r["hm"] = min(int(hm), 255)
    r["ep"] = int(ep) & 0xFF
    # always written, so a reused output buffer never carries the last record's value
    r["mleft"] = 0 if mleft is None else min(65535, max(0, int(round(float(mleft)))))
    r["flags"] = ((1 if aux_valid else 0) | (2 if clock_valid else 0) | (4 if ep_valid else 0)
                  | (8 if mleft is not None else 0))
    return r


def open_shard(path, mode="r"):
    """Memory-map a shard as a record array. Nothing is read until it is indexed.

    Sized from the file, whole records only. A converter may have just created a shard or be partway
    through a record, so an empty file is an empty array and a torn tail is left unread until it is
    complete.
    """
    n = os.path.getsize(path) // REC
    return np.memmap(path, dtype=DT, mode=mode, shape=(n,)) if n else np.zeros(0, DT)


class ShardReader:
    """A shard for the loaders, its length, and a run of rows read when asked, nothing held between.

    A memmap kept open holds a mapped section, and Windows then refuses to truncate or move the file,
    which breaks a converter's rewind (lc0_pool._rewind, ShardWriter.rewind) and torn-tail repair
    (fetch_lichess_evals.Sharder._open). Reading each run with fromfile returns the same rows and
    lets go of the file at once. A shard moved away since the listing reads as empty.
    """

    def __init__(self, path, dtype=DT):
        self.path, self.dt = path, np.dtype(dtype)
        try:
            self.n = os.path.getsize(path) // self.dt.itemsize
        except FileNotFoundError:
            self.n = 0

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        if isinstance(k, slice):
            a, b, _ = k.indices(self.n)
        else:
            a = int(k) + (self.n if int(k) < 0 else 0)
            b = a + 1
        try:
            rows = np.fromfile(self.path, dtype=self.dt, count=max(0, b - a),
                               offset=a * self.dt.itemsize)
        except FileNotFoundError:
            rows = np.zeros(0, self.dt)
        if isinstance(k, slice):
            return rows
        return rows[0] if len(rows) else np.zeros(1, self.dt)[0]


def shard_cap(paths, readers):
    """A frozen corpus, for runs that must read the same data (A/B arms run hours apart).

    SHARD_CAP names a JSON of {shard file name: rows}. Only those shards are read, each cut to that
    many rows, however the folder grows meanwhile. Without it the loaders read the corpus as it is
    at every refresh. The cut is a length, nothing is read. Unset, nothing changes."""
    cap = os.environ.get("SHARD_CAP")
    if not cap:
        return paths, readers
    import json
    rows = json.load(open(cap, encoding="utf-8"))
    keep = []
    for p, r in zip(paths, readers):
        n = rows.get(os.path.basename(p))
        if n is not None:
            r.n = min(r.n, int(n))
            keep.append((p, r))
    return [p for p, _ in keep], [r for _, r in keep]


# Pieces 1..6 are the side to move's, 7..12 are the opponent's, in the order board_to_ids uses.
_SYM = (None, 1, 2, 3, 4, 5, 6)     # chess.PAWN..chess.KING are 1..6, so the map is the identity


def _board(ids, cast, ep, ep_valid):
    """One record back to a python-chess board, for questions only chess can answer.

    Everything in the record is from the side to move's point of view, as board_to_ids stores it,
    so the pieces go on with white to move and no flipping happens.

      castling   the record carries four flags, own kingside, own queenside, theirs kingside,
                 theirs queenside. Without them a castling move would read as illegal.
      en passant the record stores the target square plus one, zero meaning none, with a validity
                 bit so a corpus without it says "unknown". Leela's rows lose their en passant
                 captures, which is rare, and since the target's own moves are added to the mask
                 later it never hides a move being trained on.
    """
    import chess
    b = chess.Board(None)
    for sq in range(64):
        v = int(ids[sq])
        if v:
            b.set_piece_at(sq, chess.Piece(v if v <= 6 else v - 6,
                                           chess.WHITE if v <= 6 else chess.BLACK))
    b.turn = chess.WHITE
    r = 0
    if cast[0]:
        r |= chess.BB_H1
    if cast[1]:
        r |= chess.BB_A1
    if cast[2]:
        r |= chess.BB_H8
    if cast[3]:
        r |= chess.BB_A8
    b.castling_rights = r
    b.castling_rights = b.clean_castling_rights()   # drop rights whose king or rook has moved
    if ep_valid and ep:
        b.ep_square = int(ep) - 1
    b.clear_stack()
    return b


def legal_mask(rows, n_moves):
    """Which of the n_moves can actually be played, one row per record.

    The policy head softmaxes over all 4,162 moves, and without a mask much of its probability
    lands on moves that cannot be played. Masking turns a 5-of-4,162 problem into 5-of-35.

    This is the one Python loop over records in this file. It runs in the loader's filler thread,
    overlapped with the training step.
    """
    from moves import move_to_idx
    n = len(rows)
    out = np.zeros((n, n_moves), np.bool_)
    ids, cast, ep, fl = rows["ids"], rows["cast"], rows["ep"], rows["flags"]
    for i in range(n):
        b = _board(ids[i], cast[i], int(ep[i]), bool(fl[i] & 4))
        row = out[i]
        for mv in b.legal_moves:
            j = move_to_idx(mv, False)
            if 0 <= j < n_moves:
                row[j] = True
    return out


# Castling as the index the search reads, the king's own two-square move. Moves are from * 64 + to
# in the side to move's frame, and e1 is square 4.
OO_AS_ROOK, OO_AS_KING = 4 * 64 + 7, 4 * 64 + 6        # e1h1 263 becomes e1g1 262
OOO_AS_ROOK, OOO_AS_KING = 4 * 64 + 0, 4 * 64 + 2      # e1a1 256 becomes e1c1 258
OWN_KING = 6


def castle_by_king(ids, mv):
    """Every castle named the way python-chess's legal_moves names it, e1g1 and e1c1.

    Older Lichess and Leela shards write castling as king takes rook, O-O on 263 and O-O-O on 256,
    while legal_moves, and with it the search and the legality mask, uses 262 and 258. Training on
    the old indices puts castling's prior where the search never reads it.

    Fixed here at load, so every shard on disk and the frozen validation sets are corrected without
    rewriting them. With the side to move's king on e1, a move to h1 or a1 can only be a castle, so
    the check is safe, and a rook on e1 moving to h1 is left alone. It is idempotent, since a
    correctly written record has nothing on 263 or 256 with its king on e1.
    """
    k = (np.asarray(ids)[:, 4] == OWN_KING)[:, None]
    mv = np.asarray(mv).astype(np.int64)
    mv = np.where(k & (mv == OO_AS_ROOK), OO_AS_KING, mv)
    return np.where(k & (mv == OOO_AS_ROOK), OOO_AS_KING, mv)


def batch_tensors(rows, n_moves, temp=100.0, legal=False):
    """A slice of records to the arrays the model wants. Pure numpy, no Python loop over records.

    The policy target is a softmax over the top-5 centipawn scores, the same soft target the JSON
    path built, so models trained either way are comparable.
    """
    n = len(rows)
    ids = np.ascontiguousarray(rows["ids"]).astype(np.int64)
    cast = np.ascontiguousarray(rows["cast"]).astype(np.float32)

    cp5 = rows["cp5"].astype(np.float32)
    valid = np.arange(5)[None, :] < rows["nmv"][:, None]
    # Older Lichess records store White's scores, so with Black to move the best move carries the
    # lowest score. Their moves come best first for the side to move, so a Lichess record (aux real)
    # whose scores rise from first to last is one of those and is turned round here. Leela's records
    # have aux clear and a correctly written record never rises, so neither is touched.
    rise, live = np.diff(cp5, axis=1), valid[:, 1:]
    inv = (((rows["flags"] & 1) > 0) & np.all((rise >= 0) | ~live, axis=1)
           & np.any((rise > 0) & live, axis=1))
    cp5 = np.where(inv[:, None], -cp5, cp5)
    cp5 = np.where(valid, cp5, -np.inf)
    w = np.exp((cp5 - cp5.max(axis=1, keepdims=True)) / temp)
    w = np.where(valid, w, 0.0)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-9)

    policy = np.zeros((n, n_moves), np.float32)
    rowi = np.repeat(np.arange(n), 5)
    # castle_by_king before the mask is built, so the union below never sees 263 or 256 for a castle
    mv = castle_by_king(rows["ids"], rows["mv"])
    np.add.at(policy, (rowi, mv.reshape(-1)), w.reshape(-1))

    # Clamp before exponentiating. Mate scores are stored at +-32000, and 10 to the power of 80
    # overflows float32.
    cp = rows["cp"].astype(np.float32)
    cp = np.clip(np.where(inv, -cp, cp), -2000.0, 2000.0)
    value = 1.0 / (1.0 + np.power(10.0, -cp / 400.0))
    pw = rows["pawns"].astype(np.float32)
    # 101 is the bucket for "this corpus did not carry a clock", which is not the same thing as a
    # clock of zero and must not be folded into it.
    hm = np.where(rows["flags"] & 2, np.minimum(rows["hm"], 100), 101).astype(np.int64)
    ep = np.where(rows["flags"] & 4, rows["ep"], 0).astype(np.int64)
    out = {
        "ids": ids, "cast": cast, "hm": hm, "ep": ep, "policy": policy, "value": value,
        "material": rows["material"].astype(np.float32) / 39.0,
        "king": rows["king"].astype(np.float32) / 8.0,
        "pawns": pw / 8.0,
        "mob": rows["mob"].astype(np.float32) / 50.0,
        "bishop": rows["bishop"].astype(np.float32),
        # 1 where the auxiliary features are real, 0 where the source did not have them
        "aux": (rows["flags"] & 1).astype(np.float32),
        # Plies to the end of the game and whether the record carries it. An array saved before the
        # field existed (such as a frozen validation set) has no mleft column, and counts as
        # unlabelled.
        "mleft": (rows["mleft"].astype(np.float32) if "mleft" in rows.dtype.names
                  else np.zeros(len(rows), np.float32)),
        "mlv": (((rows["flags"] & 8) > 0).astype(np.float32) if "mleft" in rows.dtype.names
                else np.zeros(len(rows), np.float32)),
    }
    if legal:
        m = legal_mask(rows, n_moves)
        # The target is always reachable. A target move the decoded board calls illegal would get
        # zero probability under the mask, an infinite cross-entropy, so every target move is added
        # to the mask.
        m |= policy > 0
        out["legal"] = m
    return out


def split_is_val(rows, holdout=64):
    """Hold out by position, not by line number.

    Chess positions repeat, so a split by line puts some held-out positions in training too. Hashing
    the board means a position lands on one side of the split, however often it occurs.
    """
    h = np.zeros(len(rows), np.uint64)
    ids = rows["ids"].astype(np.uint64)
    for i in range(64):                       # FNV-1a over the 64 squares, vectorised across rows
        h = (h ^ ids[:, i]) * np.uint64(0x100000001B3)
    h ^= rows["cast"].astype(np.uint64) @ np.array([1, 2, 4, 8], np.uint64)
    return (h % np.uint64(holdout)) == 0


if __name__ == "__main__":
    import os, tempfile
    print(f"record {DT.itemsize} bytes")
    for n, label in ((2.8e9, "2.8B"), (5e9, "5B"), (10e9, "10B")):
        print(f"  {label:>5s} positions -> {n * REC / 1e12:.2f} TB as JSON would be "
              f"{n * 312 / 1e12:.2f} TB, packed is {n * REC / 1e9:.0f} GB")

    rng = np.random.default_rng(0)
    n = 5000
    arr = np.zeros(n, DT)
    arr["ids"] = rng.integers(0, 13, (n, 64), dtype=np.uint8)
    arr["cast"] = rng.integers(0, 2, (n, 4), dtype=np.uint8)
    arr["mv"] = rng.integers(0, 4096, (n, 5)).astype("<u2")
    arr["cp5"] = rng.integers(-500, 500, (n, 5)).astype("<i2")
    arr["cp"] = arr["cp5"][:, 0]
    arr["nmv"] = rng.integers(1, 6, n, dtype=np.uint8)
    arr["ply"] = rng.integers(0, 120, n).astype("<u2")
    arr["flags"] = rng.integers(0, 2, n, dtype=np.uint8)
    arr["material"] = rng.integers(0, 40, (n, 2), dtype=np.uint8)
    arr["mob"] = rng.integers(0, 50, n, dtype=np.uint8)

    p = os.path.join(tempfile.gettempdir(), "packtest.bin")
    arr.tofile(p)
    mm = open_shard(p)
    assert len(mm) == n and np.array_equal(mm["ids"], arr["ids"])
    print(f"\nround trip through {os.path.getsize(p)/1e6:.1f} MB: ok")

    b = batch_tensors(mm[:256], 4096)
    rowsum = b["policy"].sum(1)
    assert np.allclose(rowsum, 1.0, atol=1e-4), f"policy rows sum to {rowsum[:3]}"
    assert b["ids"].shape == (256, 64) and b["value"].shape == (256,)
    print(f"batch: policy {b['policy'].shape} rows sum to 1.0, value in "
          f"[{b['value'].min():.3f}, {b['value'].max():.3f}]")
    print(f"aux-valid share in the batch: {100*b['aux'].mean():.0f}%")

    v = split_is_val(mm)
    print(f"holdout {100*v.mean():.2f}% of records (target {100/64:.2f}%)")
    # the same position must always land on the same side, however often it appears
    dup = np.concatenate([mm[:100], mm[:100]])
    a, c = split_is_val(dup)[:100], split_is_val(dup)[100:]
    assert np.array_equal(a, c), "identical positions split differently"
    print("identical positions always land on the same side of the split")
