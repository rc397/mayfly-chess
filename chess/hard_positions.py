"""Find the self-play positions where MayFly's own move was a mistake, and queue them for a deeper label.

deep_label.py's first label scores the top moves of each self-play position. Positions in its queue are
followed by the next position of the same game, so the move MayFly played is the legal move joining the
two, and after the sampled opening plies it is MayFly's own pick. A move is judged by its distance from
the best, not by whether it is in the top labelled moves, since those can range from winning to losing.
There are three kinds of line.
    deep    a clear mistake, at least GAP below the best, or outside the labelled moves when the last of
            them is already GAP down. Relabelled deeper, with MayFly's own move valued too.
    check   outside the labelled moves while they are close together, so its value is unknown. A short
            search of that move alone measures it, and it goes deep only if it is GAP down.
    crit    an only-move position, the best at least CRIT ahead of the second, relabelled deeper even when
            MayFly found the move, but only when nothing else is waiting.

A queue line is "FEN|uci|kind|best", the position, MayFly's move (empty when it is not its own pick), the
kind, and the first label's best value, which a check compares against. Nothing here imports torch, since
deep_label's parent imports this between batches.
"""
import numpy as np
import chess

from encode import board_to_ids
from moves import move_to_idx
from pack_av import DT_AV, pack_ids, unpack_ids

GAP = 0.05          # win chance below the best that makes a move a mistake
CRIT = 0.20         # the best this far ahead of the second makes an only-move position
OWN_FROM = 11       # the fullmove number from which a self-play move is MayFly's own pick (TEMP_PLIES = 20)


def board_hash(rows):
    """data_av.board_hash_av without data_av's torch import, and kept identical to it.

    FNV over the 64 piece ids from zero, with the castling byte folded in. Every row of one position
    shares the value."""
    ids = unpack_ids(rows["ids4"]).astype(np.uint64)
    h = np.zeros(len(rows), np.uint64)
    for i in range(64):
        h = (h ^ ids[:, i]) * np.uint64(0x100000001B3)
    h ^= rows["cast"].astype(np.uint64)
    return h


def position_hash(board):
    """board_hash for a live board, encoded exactly as deep_label encodes the rows it writes."""
    ids, cast, _ = board_to_ids(board)
    r = np.zeros(1, DT_AV)
    r["ids4"] = pack_ids(ids)
    r["cast"] = int(cast[0]) | (int(cast[1]) << 1) | (int(cast[2]) << 2) | (int(cast[3]) << 3)
    return int(board_hash(r)[0])


def labels_by_position(rows):
    """{position hash: {move index: win probability for the side to move}} for a block of label rows."""
    out = {}
    if len(rows):
        for h, mv, q in zip(board_hash(rows).tolist(), rows["mv"].tolist(), (rows["q"] / 65535.0).tolist()):
            out.setdefault(h, {})[mv] = q
    return out


def played_move(board, nxt_fen):
    """The legal move that turns `board` into `nxt_fen`, or None at a new game, a chunk's edge or a repeat."""
    try:
        nxt = chess.Board(nxt_fen)
    except ValueError:
        return None
    target = nxt.board_fen()
    for m in board.legal_moves:
        board.push(m)
        same = board.board_fen() == target and board.turn == nxt.turn
        board.pop()
        if same:
            return m
    return None


def classify(board, lab, move):
    """The kind of line for one labelled position, or None.

    `lab` is {move index: value}. `move` is MayFly's own move, or None when it is unknown or not its own pick."""
    qs = sorted(lab.values(), reverse=True)
    best = qs[0]
    if move is not None:
        q = lab.get(move_to_idx(move, board.turn == chess.BLACK))
        if q is not None:
            if best - q >= GAP:
                return "deep"
        elif best - qs[-1] >= GAP:
            return "deep"                   # worse than the last labelled move, already a mistake
        else:
            return "check"
    if len(qs) > 1 and best - qs[1] >= CRIT:
        return "crit"
    return None


def find(fens, lo, hi, rows):
    """The queue lines for the hard positions among fens[lo:hi], whose first labels are in `rows`.

    `rows` are the label rows written by the batch that covered them. Returns (lines, counts)."""
    labels = labels_by_position(rows)
    lines = []
    counts = {"scanned": 0, "deep": 0, "check": 0, "crit": 0}
    for i in range(lo, min(hi, len(fens))):
        fen = fens[i]
        if not fen or "|" in fen:
            continue
        try:
            b = chess.Board(fen)
        except ValueError:
            continue
        lab = labels.get(position_hash(b))
        if not lab:
            continue                        # a repeat, labelled in an earlier batch
        counts["scanned"] += 1
        move = None
        if b.fullmove_number >= OWN_FROM and i + 1 < len(fens) and fens[i + 1]:
            move = played_move(b, fens[i + 1])
        kind = classify(b, lab, move)
        if kind:
            counts[kind] += 1
            lines.append(f"{fen}|{move.uci() if move else ''}|{kind}|{max(lab.values()):.4f}")
    return lines, counts


def parse(line):
    """(fen, move uci, kind, best) from a queue line. A plain FEN gives ("FEN", "", "", None)."""

    fen, _, rest = line.partition("|")
    parts = rest.split("|") if rest else []
    uci = parts[0] if len(parts) > 0 else ""
    kind = parts[1] if len(parts) > 1 else ""
    try:
        best = float(parts[2]) if len(parts) > 2 and parts[2] else None
    except ValueError:
        best = None
    return fen.strip(), uci.strip(), kind.strip(), best
