"""Encode a board as the network's input numbers, without importing torch.

Kept apart from chess_net so worker processes that only encode boards (deep_label runs one per core)
do not each load PyTorch, which costs over a gigabyte of committed memory per process. chess_net
re-exports everything here.
"""
import chess

PIECE_ORD = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6}

def board_to_ids(board):
    """Canonical (side to move's view) board to 64 piece ids (0 empty, 1-6 mine, 7-12 theirs) and castling[4]."""
    flip = (board.turn == chess.BLACK)
    ids = [0] * 64
    for sq, pc in board.piece_map().items():
        s = chess.square_mirror(sq) if flip else sq
        ids[s] = PIECE_ORD[pc.piece_type] + (0 if pc.color == board.turn else 6)
    stm, opp = board.turn, not board.turn
    cast = [board.has_kingside_castling_rights(stm), board.has_queenside_castling_rights(stm),
            board.has_kingside_castling_rights(opp), board.has_queenside_castling_rights(opp)]
    return ids, [float(c) for c in cast], flip

def clock_and_ep(board, flip):
    """The halfmove clock bucket and the en passant target for a live board.

    The square is mirrored with the board when black is to move. It is reported only when an en
    passant capture is legal now, not after every double pawn push. python-chess sets ep_square on
    every double push but its fen() prints a dash unless a capture exists, and the corpora are FENs,
    so the FEN convention is used.
    """
    hm = min(int(board.halfmove_clock), 100)
    if not board.has_legal_en_passant():
        return hm, 0
    ep = board.ep_square
    return hm, (chess.square_mirror(ep) if flip else ep) + 1


# The batch encoder. board_to_ids costs 10 to 20 microseconds a board in Python, a few percent of a long
# search. Everything it reads is already in the node's transposition key (six piece bitboards, the two
# colours, the turn, castling rights and the legal en passant square), so encode_keys decodes a whole
# batch of keys with numpy and returns exactly what board_to_ids and clock_and_ep return, which selftest
# checks. Standard chess only (not chess960), on a little-endian host.
import struct
import sys

import numpy as np

assert sys.byteorder == "little"
_KEY10 = struct.Struct("<10Q")       # P N B R Q K, white occupied, black occupied, turn, castling rights
_W6 = np.arange(1, 7, dtype=np.uint8).reshape(1, 6, 1)
_ONE, _R1 = np.uint64(1), np.uint64(0xFF)
_R8 = np.uint64(0xFF) << np.uint64(56)


def _castle_bits(A, mover):
    """Kingside and queenside castling rights for both sides, as python-chess computes them.

    A right needs a king of that colour on its back rank and a castling-rights bit on that rank above
    it (kingside) or below it (queenside). A is (B,10) uint64, mover (B,) intp with 0 for white to move.
    Returns (B,4) float32, own K, own Q, their K, their Q."""
    kings, cr = A[:, 5], A[:, 9]
    K = np.stack([kings & A[:, 6] & _R1, kings & A[:, 7] & _R8], 1)      # (B,2), white then black
    C = np.stack([cr & _R1, cr & _R8], 1)
    with np.errstate(over="ignore"):                                     # a king on h8 makes 2K wrap to 0
        ok = K != 0
        ks = ok & ((C & ~((K << _ONE) - _ONE)) != 0)
        qs = ok & ((C & (K - _ONE)) != 0)
    r = np.arange(A.shape[0])
    o = 1 - mover
    return np.stack([ks[r, mover], qs[r, mover], ks[r, o], qs[r, o]], 1).astype(np.float32)


def encode_keys(keys, hms):
    """What board_to_ids and clock_and_ep give for every board in the batch, from
    Board._transposition_key() tuples and the boards' halfmove clocks.

    Returns ids (B,64) int64, cast (B,4) float32, hm (B,) int64 and ep (B,) int64."""
    B = len(keys)
    pk = _KEY10.pack
    A = np.frombuffer(b"".join([pk(k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9])
                                for k in keys]), np.uint64).reshape(B, 10)
    white = A[:, 8].astype(bool)
    black = ~white
    bb = np.empty((B, 7), np.uint64)
    bb[:, :6] = A[:, :6]
    bb[:, 6] = np.where(white, A[:, 7], A[:, 6])           # the pieces of the side NOT to move
    if black.any():
        bb[black] = bb[black].byteswap()                   # bit s to bit s ^ 56, the mover's own frame
    bits = np.unpackbits(bb.view(np.uint8).reshape(B, 7, 8), axis=2, bitorder="little")     # (B,7,64)
    ids = ((bits[:, :6] * _W6).sum(1, dtype=np.uint8) + 6 * bits[:, 6]).astype(np.int64)
    cast = _castle_bits(A, black.astype(np.intp))
    hm = np.minimum(np.array(hms, np.int64), 100)
    ep = np.zeros(B, np.int64)
    eps = [k[10] for k in keys]
    if eps.count(None) != B:                               # a capture is legal, mirror it like the board

        for i, e in enumerate(eps):
            if e is not None:
                ep[i] = (e ^ 56 if black[i] else e) + 1
    return ids, cast, hm, ep
