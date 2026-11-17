"""Convert the Lichess evaluation database into packed 112-byte policy shards (the lz corpus).

About 410 million Stockfish-evaluated positions, CC0, 22 GB compressed, with a median search depth of
32 and several principal variations per position, which map onto the top-five policy target. The
archive is downloaded once (resumably) and then decompressed and converted on the fly. The line offset
is recorded as it goes, so a restart skips what is already converted, and with GUARD on every record
is checked against a hash of what is already on disk, so a wrong offset cannot write duplicates.

Dropped are positions whose evaluations have fewer than MIN_PVS variations or depth under MIN_DEPTH,
positions already at game over, and positions with impossible material. Mate scores are clamped to
fit the record.

    python chess/fetch_lichess_evals.py [MAX_POSITIONS]
    env OUT (shard folder), PREFIX=lz, MIN_PVS=3, MIN_DEPTH=20, SHARD=4000000,
        SKIP=-1 (lines to skip, -1 reads the recorded offset, 0 starts from the top),
        GUARD=1 (check every record against what is already written),
        RESUMES=40 (times to reopen the stream after a broken connection),
        LOCAL=1 (download the archive first rather than streaming it), LOCAL_PATH,
        STRIPE, STRIPE_ID, TAG (parallel workers on interleaved lines)
"""
import os, sys, io, json, time, glob, contextlib
import numpy as np
import requests, zstandard
import chess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import board_to_ids, move_to_idx, clock_and_ep
from pack import DT, REC, encode_record
from gen_packed import features
from fetch import fetch
import paths

E = os.environ.get
URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
OUT = E("OUT", paths.SHARDS)
PREFIX = E("PREFIX", "lz")
MIN_PVS = int(E("MIN_PVS", 3))
MIN_DEPTH = int(E("MIN_DEPTH", 20))
SHARD = int(E("SHARD", 4_000_000))
SKIP = int(E("SKIP", -1))              # -1 means read the recorded offset
GUARD = E("GUARD", "1") == "1"
RESUMES = int(E("RESUMES", 40))
LOCAL = E("LOCAL", "1") == "1"


def _flag(name, default):
    """A named command line flag, or the environment, or the default."""
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


# Converting is one core's work, so several workers take interleaved lines, worker k taking every
# line whose count modulo STRIPE is k. Skipping a line is far cheaper than converting it, and every
# worker reads the same local archive, which the OS cache serves after the first pass.
STRIPE = int(_flag("--stripe", E("STRIPE", 1)))
STRIPE_ID = int(_flag("--id", E("STRIPE_ID", 0)))
TAG = _flag("--tag", E("TAG", ""))
LOCAL_PATH = E("LOCAL_PATH", paths.under("lichess_db_eval.jsonl.zst"))
FLUSH = 4096
def _num(v):
    try:
        float(v); return True
    except Exception:
        return False
_A = [a for a in sys.argv[1:] if _num(a)]
MAX_POS = int(float(_A[0])) if _A else 500_000_000


PROGRESS = os.path.join(OUT, f".{PREFIX}{TAG}.progress.json")
GLOBAL_PROGRESS = os.path.join(OUT, f".{PREFIX}.progress.json")


def load_offset():
    """Lines already converted, from the recorded offset. Zero when there is nothing recorded."""
    try:
        return int(json.load(open(PROGRESS))["seen"])
    except Exception:
        return 0


def save_offset(seen, kept):
    """Record progress, never less than is already recorded.

    A connection that breaks during the skip phase leaves `seen` short of where the run really got
    to, and writing that back would move the offset backwards on every reconnection.
    """
    have = load_offset()
    if int(seen) < have:
        return
    tmp = PROGRESS + ".tmp"
    json.dump({"seen": int(seen), "kept": int(kept),
               "at": time.strftime("%Y-%m-%d %H:%M:%S")}, open(tmp, "w"), indent=1)
    os.replace(tmp, PROGRESS)              # atomic, so the offset is never half written


def _fnv(raw):
    """FNV-1a over each row of a (n, k) byte array.

    Fast on large blocks. For a single record the numpy call overhead dominates, so _fnv_one below
    does that case in plain Python.
    """
    h = np.zeros(len(raw), np.uint64)
    for c in range(raw.shape[1]):
        h = (h ^ raw[:, c].astype(np.uint64)) * np.uint64(0x100000001B3)
    return h


_M64 = (1 << 64) - 1


def _fnv_one(b):
    """The same hash over one bytes object, without numpy in the way."""
    h = 0
    for c in b:
        h = ((h ^ c) * 0x100000001B3) & _M64
    return h


KEYBYTES = 70          # the 64 piece ids, 4 castling flags and the first principal variations


def existing_keys(out_dir, prefix):
    """A sorted hash of every record already written under this prefix.

    Read in blocks rather than mapped whole, to keep memory use low.
    """
    hs = []
    for path in sorted(glob.glob(os.path.join(out_dir, f"{prefix}_*.bin"))):
        with open(path, "rb") as f:
            while True:
                b = f.read(REC * 500_000)
                if not b:
                    break
                a = np.frombuffer(b[:len(b) // REC * REC], np.uint8).reshape(-1, REC)[:, :KEYBYTES]
                hs.append(_fnv(a))
    return np.sort(np.concatenate(hs)) if hs else np.zeros(0, np.uint64)


def already_have(h, keys):
    i = int(np.searchsorted(keys, h))
    return i < len(keys) and keys[i] == h


@contextlib.contextmanager
def open_source():
    """The archive, from local disk if we can have it, otherwise straight off the network.

    Downloading once is worth the 22 GB. The converter reads far slower than the link delivers, and
    such a slow stream gets closed, with every break costing a re-skip. fetch resumes the download
    itself with HTTP Range, which Lichess honours.
    """
    if LOCAL:
        print(f"  fetching the archive to {LOCAL_PATH} (resumable, about 22 GB)", flush=True)
        p = fetch(URL, LOCAL_PATH)
        if p:
            print(f"  have it: {os.path.getsize(p)/1e9:.1f} GB on disk, "
                  f"the network is now out of the loop", flush=True)
            f = open(p, "rb")
            try:
                yield f
            finally:
                f.close()
            return
        print("  could not get a local copy, falling back to streaming", flush=True)
    r = requests.get(URL, stream=True, timeout=300)
    try:
        r.raise_for_status()
        yield r.raw
    finally:
        r.close()


def mate_to_cp(m):
    """A forced mate in n, as a centipawn number the record can hold."""
    return (32000 - abs(int(m)) * 10) * (1 if int(m) > 0 else -1)


def convert(rec):
    """One Lichess line to one packed record, or None if it is not usable."""
    evals = rec.get("evals") or []
    if not evals:
        return None
    # Pick the deepest evaluation that has enough variations, not simply the deepest, since a position
    # is often analysed once deep with a single line and again shallower with five.
    usable = [e for e in evals if len(e.get("pvs") or []) >= MIN_PVS]
    if not usable:
        return None
    best = max(usable, key=lambda e: e.get("depth", 0))
    if best.get("depth", 0) < MIN_DEPTH:
        return None
    pvs = best["pvs"]
    try:
        board = chess.Board(rec["fen"])
    except Exception:
        return None
    if board.is_game_over():
        return None
    ids, cast, flip = board_to_ids(board)
    idx, cps = [], []
    for pv in pvs[:5]:
        line = pv.get("line", "").split()
        if not line:
            continue
        try:
            # parse_uci rather than from_uci plus a legality test, since python-chess accepts the
            # king-takes-rook castle e1h1 while legal_moves names it e1g1. parse_uci normalises the
            # castle and raises on anything illegal, which the except below skips.
            mv = board.parse_uci(line[0])
            idx.append(move_to_idx(mv, flip))
        except Exception:
            continue
        # Lichess scores are White's and the records are the side to move's. pack.batch_tensors fixes
        # older records at load time, these are written the right way.
        sc = int(pv["cp"]) if pv.get("cp") is not None else mate_to_cp(pv.get("mate", 1))
        cps.append(sc if board.turn == chess.WHITE else -sc)
    if len(idx) < MIN_PVS:
        return None
    hm, ep = clock_and_ep(board, flip)
    feat = features(board, flip)
    # Lichess analyses any FEN a user types, so drop positions with more material than a real game holds

    if max(feat["material"]) > 45:
        return None
    return ids, cast, idx, cps, feat, hm, ep


class Sharder:
    """Append-only shards of a fixed record count, so a kill costs at most one buffer."""

    def __init__(self, out, prefix, per_shard, tag=""):
        self.out, self.prefix, self.per, self.tag = out, prefix, per_shard, tag
        os.makedirs(out, exist_ok=True)
        self.n = 0
        self.i = self._resume()
        self.f = None
        self.buf = np.zeros(FLUSH, DT)
        self.k = 0
        self.last_sync = time.time()

    def _resume(self):
        i = 0
        while True:
            p = os.path.join(self.out, f"{self.prefix}_{self.tag}{i:03d}.bin")
            if not os.path.exists(p):
                return i
            sz = os.path.getsize(p)
            if sz // REC < self.per:
                self.n = sz // REC
                return i
            i += 1

    def _open(self):
        p = os.path.join(self.out, f"{self.prefix}_{self.tag}{self.i:03d}.bin")
        if os.path.exists(p) and os.path.getsize(p) % REC:
            with open(p, "r+b") as f:
                f.truncate((os.path.getsize(p) // REC) * REC)
        self.f = open(p, "ab", buffering=0)

    def write(self, rowargs):
        if self.f is None:
            self._open()
        encode_record(*rowargs, out=self.buf[self.k])
        self.k += 1
        self.n += 1
        if self.k == FLUSH:
            self.f.write(self.buf.tobytes()); self.k = 0
            if time.time() - self.last_sync > 30:
                self.f.flush(); os.fsync(self.f.fileno()); self.last_sync = time.time()
        if self.n >= self.per:
            self.close(); self.i += 1; self.n = 0

    def close(self):
        if self.f is None:
            return
        if self.k:
            self.f.write(self.buf[:self.k].tobytes()); self.k = 0
        self.f.flush(); os.fsync(self.f.fileno()); self.f.close(); self.f = None


if __name__ == "__main__":
    print(f"streaming {URL}")
    print(f"keeping positions with depth >= {MIN_DEPTH} and >= {MIN_PVS} principal variations")
    print(f"writing {PREFIX}_*.bin to {OUT}\n", flush=True)

    sh = Sharder(OUT, PREFIX, SHARD, tag=TAG)
    skip = load_offset() if SKIP < 0 else SKIP
    if STRIPE > 1:
        # never start before the single-worker run got to, or the stripes would redo converted lines
        try:
            skip = max(skip, int(json.load(open(GLOBAL_PROGRESS))["seen"]))
        except Exception:
            pass
        print(f"worker {STRIPE_ID} of {STRIPE}, taking every {STRIPE}th line from {skip:,}",
              flush=True)
    # striped workers take disjoint lines beyond everything converted, so the guard is not needed
    use_guard = GUARD and STRIPE == 1
    keys = existing_keys(OUT, PREFIX) if use_guard else np.zeros(0, np.uint64)
    scratch = np.zeros(1, DT)
    if use_guard:
        print(f"guarding against {len(keys):,} records already on disk", flush=True)

    kept = seen = dupes = 0
    drop = {"shallow": 0, "few pvs": 0, "unusable": 0}
    t0 = time.time()
    attempt = 0
    stop = False
    # a long stream will be interrupted, so reopen and skip to the recorded line rather than stop
    while not stop and attempt <= RESUMES:
        if skip:
            print(f"opening at line {skip:,}, which is already converted", flush=True)
        try:
            with open_source() as src:
                reader = zstandard.ZstdDecompressor().stream_reader(src)
                lines = io.TextIOWrapper(reader, encoding="utf-8")
                seen = 0
                if skip:
                    ts = time.time()
                    for seen, _ in enumerate(lines, 1):
                        if seen >= skip:
                            break
                    print(f"skipped {seen:,} lines in {time.time()-ts:.0f}s", flush=True)
                    t0 = time.time()
                for line in lines:
                    seen += 1
                    if kept >= MAX_POS:
                        stop = True
                        break
                    if STRIPE > 1 and seen % STRIPE != STRIPE_ID:
                        continue
                    # print on lines read, not records written, so a stretch of duplicates still shows

                    if seen % 2_000_000 == 0:
                        save_offset(seen, kept)
                        el = time.time() - t0
                        print(f"  line {seen:,}  {kept:,} kept  {dupes:,} already had  "
                              f"{seen/max(el,1):,.0f} lines/s  {el/3600:.2f}h", flush=True)
                    try:
                        rec = json.loads(line)
                    except Exception:
                        drop["unusable"] += 1
                        continue
                    out = convert(rec)
                    if out is None:
                        evals = rec.get("evals") or []
                        usable = [e for e in evals if len(e.get("pvs") or []) >= MIN_PVS]
                        if not usable:
                            drop["few pvs"] += 1
                        elif max(e.get("depth", 0) for e in usable) < MIN_DEPTH:
                            drop["shallow"] += 1
                        else:
                            drop["unusable"] += 1
                        continue
                    ids, cast, idx, cps, feat, hm, ep = out
                    # the Lichess FEN has no clock fields, so the clock is unknown, not zero
                    args = (ids, cast, idx, cps, cps[0], feat, 0, True, hm, ep,
                            len(rec["fen"].split()) > 4, True)
                    if use_guard and len(keys):
                        encode_record(*args, out=scratch[0])
                        h = np.uint64(_fnv_one(scratch.tobytes()[:KEYBYTES]))
                        if already_have(h, keys):
                            dupes += 1
                            continue
                    sh.write(args)
                    kept += 1
                    if kept % 200000 == 0:
                        el = time.time() - t0
                        save_offset(seen, kept)
                        extra = f", {dupes:,} already had" if dupes else ""
                        print(f"  {kept:,} kept of {seen:,} seen  {kept/el:.0f}/s  "
                              f"{el/3600:.2f}h  ({100*kept/seen:.0f}% keep rate{extra})",
                              flush=True)
                else:
                    stop = True                      # the archive ended, not the connection
        except KeyboardInterrupt:
            print("\ninterrupted")
            stop = True
        except Exception as e:
            attempt += 1
            save_offset(seen, kept)
            skip = max(skip, seen)
            wait = min(15 * attempt, 120)
            print(f"  stream broke at line {seen:,} ({type(e).__name__}), "
                  f"reopening in {wait}s [{attempt}/{RESUMES}]", flush=True)
            sh.close()
            time.sleep(wait)
    sh.close()
    save_offset(seen, kept)
    print(f"\n{kept:,} positions written, {seen:,} lines read, "
          f"{(time.time()-t0)/3600:.2f}h")
    if dupes:
        print(f"  skipped, already on disk {dupes:,}")
    for k, v in drop.items():
        print(f"  dropped, {k:10s} {v:,}")
