"""Download DeepMind's ChessBench action-value corpus and pack it into 40-byte action-value shards.

ChessBench is the data behind "Grandmaster-Level Chess Without Search", every legal move in positions
from ten million Lichess games scored by Stockfish 16 at 50 ms. Shard sizes vary from 16 MB to 1.7 GB,
and the full release is about 1.4 TB and roughly 18 billion state-action pairs.

A .bag file is records, then an index of little-endian int64 end offsets, then eight bytes giving where
the index starts. Each record is an Apache Beam tuple, a length byte and the FEN, a length byte and the
UCI move, then the win probability as a big-endian double with no length prefix. Records are globally
shuffled, which is fine because the action-value head is trained one move at a time (see pack_av).
Underpromotions are kept, and counted in the log.

    python chess/chessbench.py [MAX_SHARDS]
    python chess/chessbench.py verify [file.bag]     check the fast parser against python-chess
    env RAW (download folder), OUT (shard folder), PREFIX=av,
        STRIDE=1, OFFSET=0, TAG= to stripe the shard list across parallel converters,
        STOP_AT=0 to stop once all workers together have this many pairs,
        MIN_FREE_GB=250, AHEAD=1, VERIFY=0, KEEP_RAW=0
"""
import os, sys, json, time, struct, threading, queue, shutil, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pack_av import REC_AV, ShardWriter
from moves import PROMO_IDX, UCI_PROMO
from fetch import fetch, load_manifest
import paths

E = os.environ.get
BASE = "https://storage.googleapis.com/searchless_chess/data/train"
N_SHARDS = 2148
RAW = E("RAW", paths.under("cb_raw"))
OUT = E("OUT", paths.SHARDS)
PREFIX = E("PREFIX", "av")
MIN_FREE_GB = int(E("MIN_FREE_GB", 250))
AHEAD = int(E("AHEAD", 1))
KEEP_RAW = E("KEEP_RAW", "0") == "1"
# Converting is about three times slower than downloading, so several converters run side by side on
# interleaved shard numbers, each with its own state file and output letter, each resumable.
def _flag(name, default):
    """A named command line flag, or the environment, or the default.

    Flags on the command line make parallel converters distinguishable in the process list.
    """
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


STRIDE = int(_flag("--stride", E("STRIDE", 1)))
OFFSET = int(_flag("--offset", E("OFFSET", 0)))
TAG = _flag("--tag", E("TAG", ""))
# Shard sizes vary a lot, so workers stop on the total pair count from every worker's ledger rather
# than on a shard count.
STOP_AT = float(_flag("--stop-at", E("STOP_AT", 0)))
STATE = os.path.join(OUT, f".{PREFIX}{TAG or ''}.state.json")


def url_for(i):
    return f"{BASE}/action_value-{i:05d}-of-{N_SHARDS:05d}_data.bag"


# A fast FEN parser. python-chess would take minutes per shard for what is a string walk. It matches
# the encoder in chess_net, and the verify mode checks that on real records.
_PC = {}
for _i, _c in enumerate(b"PNBRQK"):
    _PC[_c] = (_i + 1, True)
for _i, _c in enumerate(b"pnbrqk"):
    _PC[_c] = (_i + 1, False)
_SLASH, _ZERO, _SPACE = 47, 48, 32


def fen_to_ids(fen):
    """FEN bytes to the canonical board, castling bits and flip that board_to_ids would give.

    Ids are from the side to move's point of view: 1 to 6 are its own pieces in the order pawn,
    knight, bishop, rook, queen, king, and 7 to 12 are the opponent's. When black is to move the
    board is mirrored vertically, which is square xor 56.

    The board comes back already packed to nibbles, which is cheaper here than afterwards.
    """
    sp1 = fen.index(_SPACE)
    stm_w = fen[sp1 + 1] == 119                      # 'w'
    flip = not stm_w
    ids4 = bytearray(32)
    sq = 56                                          # the board field starts at rank 8
    for c in fen[:sp1]:
        if c == _SLASH:
            sq -= 16
        elif 49 <= c <= 56:                          # '1' to '8'
            sq += c - _ZERO
        else:
            t, w = _PC[c]
            s = (sq ^ 56) if flip else sq
            ids4[s >> 1] |= (t if w == stm_w else t + 6) << ((s & 1) << 2)
            sq += 1
    sp2 = fen.index(_SPACE, sp1 + 2)
    sp3 = fen.index(_SPACE, sp2 + 1)
    cf = fen[sp2 + 1:sp3]
    wk, wq = 75 in cf, 81 in cf                      # 'K', 'Q'
    bk, bq = 107 in cf, 113 in cf                    # 'k', 'q'
    bits = ((wk | wq << 1 | bk << 2 | bq << 3) if stm_w
            else (bk | bq << 1 | wk << 2 | wq << 3))
    # The en passant square mirrors with the board. Stored as square plus one so zero means none.
    try:
        sp4 = fen.index(_SPACE, sp3 + 1)
        epf = fen[sp3 + 1:sp4]
        if epf[0] == 45:                             # '-'
            ep = 0
        else:
            e = (epf[0] - 97) + ((epf[1] - 49) << 3)
            ep = ((e ^ 56) if flip else e) + 1
        sp5 = fen.index(_SPACE, sp4 + 1)
        hm = 0
        for c in fen[sp4 + 1:sp5]:
            hm = hm * 10 + (c - _ZERO)
    except (ValueError, IndexError):
        return ids4, bits, flip, 0, 0, False
    return ids4, bits, flip, min(hm, 255), ep, True


def move_index(mv, flip):
    """UCI move bytes to our move index, underpromotions included."""
    f = (mv[0] - 97) + ((mv[1] - 49) << 3)
    t = (mv[2] - 97) + ((mv[3] - 49) << 3)
    if flip:
        f ^= 56
        t ^= 56
    if len(mv) > 4 and mv[4] != 113:                 # anything but 'q'
        return PROMO_IDX.get((f, t, UCI_PROMO.get(mv[4])), (f << 6) | t)
    return (f << 6) | t


def bag_records(path):
    """Every record in a .bag, as slices of one mapping of the file.

    The index at the end gives every record's end offset, so nothing has to be scanned for
    delimiters and the file is never held in memory as a whole.
    """
    size = os.path.getsize(path)
    mm = np.memmap(path, np.uint8, "r")
    (index_start,) = struct.unpack("<Q", mm[size - 8:size].tobytes())
    n = (size - 8 - index_start) // 8
    ends = np.frombuffer(mm[index_start:index_start + n * 8].tobytes(), "<i8")
    view = memoryview(mm)
    start = 0
    for e in ends:
        yield view[start:e]
        start = int(e)


_DOUBLE = struct.Struct(">d")
_TAIL = struct.Struct("<HHBBB")          # move index, win probability, flags, clock, ep


def convert_shard(path, writer, block=200_000):
    """One .bag to packed action-value records. Returns (kept, dropped_underpromotion, seconds).

    Records are built straight into a bytearray, since a numpy call per record costs more than the
    parsing.
    """
    t0 = time.time()
    out = bytearray()
    kept = under = k = 0
    for rec in bag_records(path):
        b = bytes(rec)
        fl = b[0]
        mo = 1 + fl
        mv = b[mo + 1:mo + 1 + b[mo]]
        try:
            ids4, bits, flip, hm, ep, ok = fen_to_ids(b[1:mo])
        except (KeyError, ValueError, IndexError):
            continue
        idx = move_index(mv, flip)
        if len(mv) > 4 and mv[4] != 113:
            under += 1                               # counted, and kept
        q = _DOUBLE.unpack_from(b, len(b) - 8)[0]
        out += ids4
        out.append(bits)
        flags = (1 if len(mv) > 4 else 0) | (6 if ok else 0)
        out += _TAIL.pack(idx, int(min(max(q, 0.0), 1.0) * 65535 + 0.5), flags, hm, ep)
        kept += 1
        k += 1
        if k == block:
            writer.write_raw(bytes(out)); out.clear(); k = 0
    if k:
        writer.write_raw(bytes(out))
    return kept, under, time.time() - t0


def free_gb(p):
    try:
        return shutil.disk_usage(os.path.dirname(p) or p).free / 1e9
    except Exception:
        return 1e9


def all_pairs():
    """Action values packed by every worker, from the ledgers on disk."""
    n = 0
    for p in glob.glob(os.path.join(OUT, f".{PREFIX}*.state.json")):
        try:
            n += json.load(open(p)).get("pairs", 0)
        except Exception:
            pass
    return n


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {"done": [], "pairs": 0, "under": 0, "mark": None}


def save_state(s):
    tmp = STATE + ".tmp"
    json.dump(s, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


if __name__ == "__main__":
    if E("VERIFY", "0") == "1" or (len(sys.argv) > 1 and sys.argv[1] == "verify"):
        import chess
        from chess_net import board_to_ids, move_to_idx
        src = sys.argv[2] if len(sys.argv) > 2 else sorted(
            [os.path.join(RAW, f) for f in os.listdir(RAW) if f.endswith(".bag")])[0]
        print(f"checking the fast parser against python-chess on {os.path.basename(src)}")
        from pack_av import unpack_ids
        bad = {"board": 0, "castling": 0, "flip": 0, "move index": 0, "illegal move": 0,
               "clock": 0, "en passant": 0, "fen too short": 0}
        n = under = 0
        qs = []
        for rec in bag_records(src):
            b = bytes(rec)
            fl = b[0]
            fen = b[1:1 + fl]
            mo = 1 + fl
            mv = b[mo + 1:mo + 1 + b[mo]]
            ids4, bits, flip, hm, ep, ok = fen_to_ids(fen)
            ids = list(unpack_ids(np.frombuffer(bytes(ids4), np.uint8))[0])
            bd = chess.Board(fen.decode())
            rids, rcast, rflip = board_to_ids(bd)
            cast = [float((bits >> i) & 1) for i in range(4)]
            if ids != rids:
                bad["board"] += 1
                if bad["board"] < 3:
                    print(f"  board  {fen.decode()}")
            if cast != rcast:
                bad["castling"] += 1
                if bad["castling"] < 3:
                    print(f"  castling  {fen.decode()}  got {cast} want {rcast}")
            if flip != rflip:
                bad["flip"] += 1
            m = chess.Move.from_uci(mv.decode())
            if m not in bd.legal_moves:
                bad["illegal move"] += 1
            if len(mv) == 4 or mv[4] == 113:
                want = move_to_idx(chess.Move(m.from_square, m.to_square), flip)
                if move_index(mv, flip) != want:
                    bad["move index"] += 1
            else:
                under += 1
            if ok:
                if hm != bd.halfmove_clock and bd.halfmove_clock < 255:
                    bad["clock"] = bad.get("clock", 0) + 1
                want_ep = 0 if bd.ep_square is None else (
                    (chess.square_mirror(bd.ep_square) if flip else bd.ep_square) + 1)
                if ep != want_ep:
                    bad["en passant"] = bad.get("en passant", 0) + 1
                    if bad["en passant"] < 3:
                        print(f"  en passant  {fen.decode()}  got {ep} want {want_ep}")
            else:
                bad["fen too short"] = bad.get("fen too short", 0) + 1
            qs.append(_DOUBLE.unpack_from(b, len(b) - 8)[0])
            n += 1
            if n >= 20000:
                break
        q = np.array(qs)
        print("")
        print(f"  {n:,} records checked, {under:,} underpromotions")
        print(f"  win probability: min {q.min():.4f} max {q.max():.4f} mean {q.mean():.4f}, "
              f"{100*np.mean((q >= 0) & (q <= 1)):.1f}% inside [0, 1]")
        for k, v in bad.items():
            print(f"  {'PASS' if v == 0 else 'FAIL'}  {k}: {v} wrong")
        raise SystemExit(0 if not any(bad.values()) else 1)

    MAX = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else N_SHARDS
    os.makedirs(RAW, exist_ok=True)
    st = load_state()
    writer = ShardWriter(OUT, PREFIX, tag=TAG)
    # a converter killed mid shard leaves a tail of records that will be redone, so cut it off first
    dropped = writer.rewind(st.get("mark"))
    if dropped:
        print(f"rewound {dropped:,} records from an interrupted shard", flush=True)

    # The stride keeps workers in one run apart. Every worker also reads every ledger, so a run with
    # a different stride does not convert a shard twice.
    have = set()
    for other in glob.glob(os.path.join(OUT, f".{PREFIX}*.state.json")):
        try:
            have |= set(json.load(open(other)).get("done", []))
        except Exception:
            pass
    todo = [i for i in range(OFFSET, MAX, STRIDE)
            if f"action_value-{i:05d}-of-{N_SHARDS:05d}_data.bag" not in have]
    print(f"{len(st['done'])} shards converted by this worker, "
          f"{len(have)} by all workers, {len(todo)} to go")
    print(f"{st['pairs']:,} action values packed so far "
          f"({REC_AV * st['pairs'] / 1e9:.1f} GB)\n", flush=True)

    q = queue.Queue(maxsize=AHEAD)
    stop = threading.Event()

    def downloader():
        for i in todo:
            if stop.is_set():
                return
            u = url_for(i)
            name = u.rsplit("/", 1)[-1]
            while free_gb(RAW) < MIN_FREE_GB and not stop.is_set():
                print(f"  only {free_gb(RAW):.0f} GB free, waiting", flush=True)
                time.sleep(300)
            t0 = time.time()
            p = fetch(u, os.path.join(RAW, name), quiet=True)
            if p:
                mb = os.path.getsize(p) / 1e6
                el = time.time() - t0
                print(f"  fetched {name} {mb:.0f} MB in {el:.0f}s "
                      f"({mb * 8 / max(el, 1):.0f} Mbps)", flush=True)
                q.put((name, p))
            else:
                print(f"  {name}: failed, skipping", flush=True)
        q.put(None)

    dl = threading.Thread(target=downloader, daemon=True)
    dl.start()

    t_start = time.time()
    try:
        while True:
            try:
                item = q.get(timeout=60)
            except queue.Empty:
                # the downloader puts None when it finishes, so if it died instead, stop rather
                # than wait forever

                if not dl.is_alive():
                    print("  the downloader is gone and sent no end marker, stopping",
                          flush=True)
                    break
                continue
            if item is None:
                break
            name, p = item
            if STOP_AT and all_pairs() >= STOP_AT:
                print(f"  target of {STOP_AT:,.0f} action values reached, stopping", flush=True)
                stop.set()
                os.remove(p)
                break
            st["mark"] = writer.mark()
            save_state(st)
            kept, under, el = convert_shard(p, writer)
            writer.flush()
            if not KEEP_RAW:
                os.remove(p)
            st["done"].append(name)
            st["pairs"] += kept
            st["under"] += under
            st["mark"] = writer.mark()
            save_state(st)
            done = len(st["done"])
            print(f"  packed {kept:,} in {el:.0f}s ({kept/max(el,1):,.0f}/s), "
                  f"{under:,} underpromotions | "
                  f"{done}/{N_SHARDS} shards here, "
                  f"{all_pairs():,} across all workers "
                  f"({100*all_pairs()/(STOP_AT or 10e9):.2f}% of target)", flush=True)
    except KeyboardInterrupt:
        stop.set()
        print("\ninterrupted")
    finally:
        writer.close()
        st["mark"] = writer.mark()
        save_state(st)
    print(f"\n{st['pairs']:,} action values, {len(st['done'])} shards, "
          f"{(time.time()-t_start)/3600:.2f}h this run")
