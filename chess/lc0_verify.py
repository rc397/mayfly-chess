r"""Check that lc0_convert really reads a Leela run's archives.

lc0_convert slices the decompressed stream at V6_SIZE = 8356 bytes. A run with another record size
would be misread without any error, so this measures two rates on one archive.

  board rate    fraction of records that decode to a board with exactly one king a side
  legal rate    of those, fraction whose top policy move is legal on the decoded board

A stride mismatch collapses the board rate. A plane order or policy index mismatch leaves the board
rate high and collapses the legal rate (mirrored ranks once gave 24 percent). A known good run such
as test80 scores about 99 percent. A local .tar path skips the download.

  python chess/lc0_verify.py <run or .tar> [SAMPLE]
  env: RAW=<data>/lc0_raw
"""
import os, sys, re, glob, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

import paths

E = os.environ.get
RUN = sys.argv[1] if len(sys.argv) > 1 else "test91"
SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
RAW = E("RAW", paths.under("lc0_raw"))
BASE = "https://storage.lczero.org/files/training_data"


if __name__ == "__main__":
    import requests
    import numpy as np
    import chess
    from fetch import fetch
    from lc0_convert import (records_in, decode_planes, V6_SIZE,
                         OFF_PROBS, OFF_PLANES)
    from lc0_moves import policy_index
    from chess_net import board_to_ids

    os.makedirs(RAW, exist_ok=True)
    # A local path skips the download. Check a known good run first, so a low score elsewhere can
    # be trusted to be the run and not the verifier.
    if os.path.exists(RUN) and RUN.endswith(".tar"):
        dest = RUN
        print(f"  local archive: {os.path.basename(dest)}", flush=True)
        _skip_fetch = True
    else:
        _skip_fetch = False
    idx = requests.get(f"{BASE}/{RUN}/", timeout=120).text if not _skip_fetch else ""
    if not _skip_fetch:
        tars = sorted(set(re.findall(r'href="([^"]+\.tar)"', idx)))
        if not tars:
            raise SystemExit(f"no archives listed for {RUN}")
        # The newest archive, from the network whose policy would be learned.
        name = tars[-1]
        url = name if name.startswith("http") else f"{BASE}/{RUN}/{name}"
        dest = os.path.join(RAW, os.path.basename(name))
        if not os.path.exists(dest):
            print(f"fetching one {RUN} archive: {os.path.basename(name)}", flush=True)
            fetch(url, dest)
    print(f"  {os.path.basename(dest)}  {os.path.getsize(dest)/1e9:.2f} GB", flush=True)

    # Decoded ids to a python-chess board. The ids are already in the side to move's frame, so
    # they go on an empty board with white to move.
    SYM = {1: chess.PAWN, 2: chess.KNIGHT, 3: chess.BISHOP, 4: chess.ROOK,
           5: chess.QUEEN, 6: chess.KING}

    def to_board(ids):
        b = chess.Board(None)
        for sq in range(64):
            v = int(ids[sq])
            if v == 0:
                continue
            if 1 <= v <= 6:
                b.set_piece_at(sq, chess.Piece(SYM[v], chess.WHITE))
            elif 7 <= v <= 12:
                b.set_piece_at(sq, chess.Piece(SYM[v - 6], chess.BLACK))
        b.turn = chess.WHITE
        b.clear_stack()
        return b

    # Decoded here rather than through read_record, which maps to MayFly's move index. The test
    # checks Leela's own move against Leela's own board.
    seen = boards = legal = 0
    t0 = time.time()
    for buf in records_in(dest):
        seen += 1
        if len(buf) >= V6_SIZE:
            probs = np.frombuffer(buf, np.float32, 1858, OFF_PROBS)
            planes = np.frombuffer(buf, "<u8", 104, OFF_PLANES)
            ids = decode_planes(planes)
            if (ids == 6).sum() == 1 and (ids == 12).sum() == 1:
                boards += 1
                b = to_board(ids)
                j = int(np.argmax(probs))
                if 0 <= j < len(policy_index):
                    try:
                        mv = chess.Move.from_uci(policy_index[j])
                        legal += int(mv in b.legal_moves)
                    except Exception:
                        pass
        if seen >= SAMPLE:
            break

    el = time.time() - t0
    print("")
    print(f"  {seen:,} records read in {el:.0f}s, stride {V6_SIZE} bytes")
    br = 100.0 * boards / max(seen, 1)
    lr = 100.0 * legal / max(boards, 1)
    print(f"  {boards:,} decoded to a board with exactly one king a side ({br:.1f}%)")
    print(f"  {legal:,} of those had Leela's OWN TOP POLICY MOVE legal on it ({lr:.1f}%)")
    print(f"  test80 scores about 99% on the second number; the mirrored-rank bug scored 24%")
    print("")
    if boards / max(seen, 1) > 0.80:
        print(f"  THE STRIDE IS RIGHT for {RUN}: {100*boards/max(seen,1):.1f}% of records decode.")
        print(f"  This run writes the same v6 record size the converter expects.")
    elif boards / max(seen, 1) > 0.30:
        print(f"  SUSPICIOUS: {100*boards/max(seen,1):.1f}% decode. Either the stride is right and")
        print(f"  this run is noisier, or something subtler is wrong. Do not convert on this.")
    else:
        print(f"  THE STRIDE IS WRONG for {RUN}: only {100*boards/max(seen,1):.1f}% decode, against")
        print(f"  the ~99% test80 gives. This run does not use the 8,356-byte v6 record, and")
        print(f"  converting it would fill the corpus with noise that passes every range check.")
