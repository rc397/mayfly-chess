r"""The record format for action values.

The policy corpus stores a position and the teacher's view of its best few moves. ChessBench stores
a position, one move and the win probability of playing it, for every legal move. The records
arrive shuffled, so the moves of one position are scattered, but a network that outputs a value for
every move can be trained one move at a time, with the loss masked to the index the record names.

The board is packed to nibbles, 40 bytes against the 72 the source uses.

  ids4   32 bytes   two squares per byte, low nibble first
  cast    1 byte    four castling bits, own side in the low two
  mv      2 bytes   MayFly's from-to index, already in the side to move's frame
  q       2 bytes   win probability for the side to move, scaled to 65,535
  flags   1 byte    bit 0 promotion, bit 1 clock real, bit 2 en passant real
  hm      1 byte    halfmove clock, which decides whether a won position is actually a draw
  ep      1 byte    en passant target as square plus one, zero for none, already mirrored
"""
import os
import numpy as np

REC_AV = 40
DT_AV = np.dtype([
    ("ids4", np.uint8, 32),
    ("cast", np.uint8),
    ("mv", "<u2"),
    ("q", "<u2"),
    ("flags", np.uint8),
    ("hm", np.uint8),
    ("ep", np.uint8),
])
assert DT_AV.itemsize == REC_AV, DT_AV.itemsize


def pack_ids(ids):
    """64 piece ids, each 0 to 12, to 32 bytes."""
    a = np.asarray(ids, np.uint8)
    return (a[0::2] | (a[1::2] << 4)).astype(np.uint8)


def unpack_ids(ids4):
    """(n, 32) packed bytes back to (n, 64) piece ids."""
    ids4 = np.atleast_2d(ids4)
    out = np.empty((len(ids4), 64), np.uint8)
    out[:, 0::2] = ids4 & 0x0F
    out[:, 1::2] = ids4 >> 4
    return out


def encode_av(ids, cast, mv, q, flags, out, hm=0, ep=0):
    out["ids4"] = pack_ids(ids)
    out["cast"] = (int(cast[0]) | (int(cast[1]) << 1) | (int(cast[2]) << 2) | (int(cast[3]) << 3))
    out["mv"] = mv
    out["q"] = int(np.clip(q, 0.0, 1.0) * 65535 + 0.5)
    out["flags"] = flags
    out["hm"] = min(int(hm), 255)
    out["ep"] = int(ep) & 0xFF


def open_shard(path):
    n = os.path.getsize(path) // REC_AV
    return np.memmap(path, DT_AV, "r", shape=(n,)) if n else np.zeros(0, DT_AV)


def batch_tensors_av(rows):
    """Packed rows to the arrays a training step wants."""
    ids = unpack_ids(rows["ids4"]).astype(np.int64)
    c = rows["cast"]
    cast = np.stack([(c >> i) & 1 for i in range(4)], 1).astype(np.float32)
    mv = rows["mv"].astype(np.int64)
    q = rows["q"].astype(np.float32) / 65535.0
    # 101 is the bucket for a missing clock, kept apart from a real clock of zero
    hm = np.where(rows["flags"] & 2, np.minimum(rows["hm"], 100), 101).astype(np.int64)
    ep = np.where(rows["flags"] & 4, rows["ep"], 0).astype(np.int64)
    return ids, cast, mv, q, hm, ep


class ShardWriter:
    """Append-only output shards that can be rewound.

    A converter killed part way through a source shard leaves a tail of records belonging to work
    that will be repeated, so the byte length at the last clean boundary is recorded and the file is
    truncated back to it on restart. Without that, a crash would write those records twice.
    """

    def __init__(self, out, prefix, per_shard=8_000_000, flush=8192, tag=""):
        self.out, self.prefix, self.per, self.tag = out, prefix, per_shard, tag
        os.makedirs(out, exist_ok=True)
        self.buf = np.zeros(flush, DT_AV)
        self.k = 0
        self.f = None
        self.i, self.n = self._resume()

    def _resume(self):
        i = 0
        while True:
            p = self.path(i)
            if not os.path.exists(p):
                return i, 0
            n = os.path.getsize(p) // REC_AV
            if n < self.per:
                if os.path.getsize(p) % REC_AV:
                    # A row torn by a kill. Appending after it would put every later row out of step,
                    # so it is cut off. Where a reader still maps the shard and Windows refuses the
                    # cut, the shard is left short (readers ignore a partial tail) and writing goes
                    # on in the next one.
                    try:
                        with open(p, "r+b") as f:
                            f.truncate(n * REC_AV)
                    except OSError:
                        i += 1
                        continue
                return i, n
            i += 1

    def path(self, i=None):
        # The worker letter goes after the underscore so every stripe reads as one source, since
        # the loader takes the prefix as everything before the first underscore.
        return os.path.join(self.out,
                            f"{self.prefix}_{self.tag}{self.i if i is None else i:04d}.bin")

    def mark(self):
        """Where a clean restart should truncate back to."""
        return {"shard": self.i, "bytes": self.n * REC_AV}

    def rewind(self, mark):
        if not mark:
            return 0
        self.flush()
        dropped = 0
        for i in range(mark["shard"], self.i + 1):
            p = self.path(i)
            if not os.path.exists(p):
                continue
            keep = mark["bytes"] if i == mark["shard"] else 0
            have = os.path.getsize(p)
            if have > keep:
                dropped += (have - keep) // REC_AV
                with open(p, "r+b") as f:
                    f.truncate(keep)
        if self.f:
            self.f.close(); self.f = None
        self.i, self.n = mark["shard"], mark["bytes"] // REC_AV
        return dropped

    def write(self, ids, cast, mv, q, flags=0):
        if self.f is None:
            self.f = open(self.path(), "ab", buffering=0)
        encode_av(ids, cast, mv, q, flags, self.buf[self.k])
        self.k += 1
        self.n += 1
        if self.k == len(self.buf):
            self.flush()
        if self.n >= self.per:
            self.close(); self.i += 1; self.n = 0

    def write_block(self, block):
        """A whole numpy block of DT_AV rows at once."""
        self.write_raw(block.tobytes())

    def write_raw(self, buf):
        """Already-encoded record bytes, a whole block at a time.

        The converter builds records with struct.pack straight into a bytearray, because a numpy
        call per record costs more than the parsing does. Split across the shard boundary rather
        than letting one block run past it, or the cap stops meaning anything.
        """
        assert len(buf) % REC_AV == 0, len(buf)
        self.flush()
        off = 0
        while off < len(buf):
            if self.f is None:
                self.f = open(self.path(), "ab", buffering=0)
            take = min((self.per - self.n) * REC_AV, len(buf) - off)
            self.f.write(buf[off:off + take])
            self.n += take // REC_AV
            off += take
            if self.n >= self.per:
                self.close(); self.i += 1; self.n = 0

    def flush(self):
        if self.f is not None and self.k:
            self.f.write(self.buf[:self.k].tobytes()); self.k = 0

    def close(self):
        if self.f is None:
            return
        self.flush()
        self.f.flush(); os.fsync(self.f.fileno()); self.f.close(); self.f = None
