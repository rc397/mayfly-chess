"""Convert Stockfish's long time control games from Fishtest (official-stockfish/fishtest_pgns) to action-value rows.

Every move in these PGNs carries Stockfish's evaluation and depth from the mover's side, as in
{-0.91/21 1.749s}. Each position gives one row in deep_label's format, the move Stockfish played valued
on ChessBench's win-probability curve (a mate as certain). In one sample the median depth was 23 against
about 13 for ChessBench's labels. There is only one move per position, so these fs_* rows can deepen the
action-value slice but do not cover bad moves.

WORKERS processes each turn one PGN file into a temporary run of rows. The parent appends each finished
run to the fs series and records the file in a ledger, so a stopped run resumes. A position repeated
within a game is written once.

    python chess/fishtest_to_av.py      env SRC (PGN folder), OUT (shard folder), WORKERS=4, MIN_DEPTH=0, WORK
"""
import glob
import gzip
import math
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

E = os.environ.get
SRC = E("SRC", paths.under("fishtest", "raw"))
OUT = E("OUT", paths.SHARDS)
PREFIX = E("PREFIX", "fs")
WORKERS = int(E("WORKERS", 4))
MIN_DEPTH = int(E("MIN_DEPTH", 0))
TMP = os.path.join(E("WORK", paths.under("fishtest")), "tmp")
LEDGER = os.path.join(E("WORK", paths.under("fishtest")), "converted.txt")
K_CHESSBENCH = 0.00368208                 # deep_label.win_prob's curve
EVAL = re.compile(r"^\s*([+-]?)(M?)(\d+(?:\.\d+)?)/(\d+)")


def win_prob(sign, mate, number):
    """The mover's chance of winning from a fishtest comment's evaluation (pawns, or M for a mate)."""
    if mate:
        return 0.0 if sign == "-" else 1.0
    cp = float(number) * 100.0 * (-1.0 if sign == "-" else 1.0)
    return 1.0 / (1.0 + math.exp(-K_CHESSBENCH * cp))


def convert(path):
    """One PGN file to a temporary run of rows. Returns (path, run file, games, rows, skipped)."""
    import chess.pgn
    from encode import board_to_ids, clock_and_ep
    from moves import move_to_idx
    from pack_av import DT_AV, encode_av
    os.makedirs(TMP, exist_ok=True)
    run = os.path.join(TMP, os.path.basename(path).replace(".pgn.gz", ".rows"))
    games = nrows = skipped = 0
    buf = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh, open(run, "wb") as out:
        while True:
            try:
                g = chess.pgn.read_game(fh)
            except Exception as e:             # a torn file, keep what was read rather than spin on the error

                print(f"  {os.path.basename(path)}: stopped reading at game {games:,}: {type(e).__name__}", flush=True)
                break
            if g is None:
                break
            games += 1
            b = g.board()
            seen = set()
            for node in g.mainline():
                m = EVAL.match(node.comment) if node.comment else None
                if m and int(m.group(4)) >= MIN_DEPTH:
                    key = b._transposition_key()
                    if key in seen:
                        skipped += 1
                    else:
                        seen.add(key)
                        ids, cast, flip = board_to_ids(b)
                        hm, ep = clock_and_ep(b, flip)
                        row = np.zeros(1, DT_AV)[0]
                        encode_av(ids, cast, move_to_idx(node.move, flip), win_prob(m.group(1), m.group(2), m.group(3)),
                                  (1 if node.move.promotion else 0) | 6, row, hm=hm, ep=ep)
                        buf.append(row.tobytes())
                else:
                    skipped += 1
                b.push(node.move)
            if len(buf) >= 100_000:
                out.write(b"".join(buf))
                nrows += len(buf)
                buf = []
        if buf:
            out.write(b"".join(buf))
            nrows += len(buf)
    return path, run, games, nrows, skipped


def main():
    from pack_av import ShardWriter
    done = set(open(LEDGER, encoding="utf-8").read().split()) if os.path.exists(LEDGER) else set()
    files = sorted(p for p in glob.glob(os.path.join(SRC, "**", "*.pgn.gz"), recursive=True)
                   if os.path.basename(p) not in done)
    print(f"{len(files)} PGN files to convert ({len(done)} done before), {WORKERS} workers, min depth {MIN_DEPTH}",
          flush=True)
    t0 = time.time()
    tot_games = tot_rows = 0
    with mp.Pool(WORKERS) as pool:
        for path, run, games, nrows, skipped in pool.imap_unordered(convert, files):
            sw = ShardWriter(OUT, PREFIX, 4_000_000)
            with open(run, "rb") as fh:
                while True:
                    chunk = fh.read(40 * 1_000_000)
                    if not chunk:
                        break
                    sw.write_raw(chunk)
            sw.close()
            os.remove(run)
            with open(LEDGER, "a", encoding="utf-8") as fh:
                fh.write(os.path.basename(path) + "\n")
            tot_games += games
            tot_rows += nrows
            el = time.time() - t0
            print(f"  {os.path.basename(path)}: {games:,} games, {nrows:,} rows ({skipped:,} moves without an "
                  f"evaluation or repeated); total {tot_games:,} games, {tot_rows:,} rows, "
                  f"{tot_games / max(el, 1):.0f} games a second, {el / 3600:.2f}h", flush=True)
    print(f"done: {tot_games:,} games, {tot_rows:,} rows in {(time.time() - t0) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
