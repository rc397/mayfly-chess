r"""The move space, in one place and without torch.

The index a move gets is shared by the network, every converter and every audit. chess_net
re-exports it, so converters can number moves without importing torch.

    from*64 + to                for 4,096 ordinary moves and queen promotions
    4,096 to 4,161              for the 66 underpromotions

A from-square and a to-square cannot say which piece a pawn became, so underpromotions get their own
indices. Promotions run from the seventh rank to the eighth in the canonical frame, straight ahead or
one file either way, which is 22 pairs, times three pieces beyond the queen, so 66 indices follow
4,095. No older index moves, so shards written under the old numbering stay valid.

One ambiguity is left on purpose. A rook moving a7 to a8 shares an index with a pawn promoting to a
queen there. They can never both be legal, so the legal mask separates them.
"""
import chess

N_FROMTO = 64 * 64
UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

PROMO_IDX, IDX_PROMO = {}, {}
_n = N_FROMTO
for _f in range(48, 56):                      # the seventh rank, side to move's frame
    for _t in (_f + 7, _f + 8, _f + 9):       # capture left, push, capture right
        if not (56 <= _t <= 63) or abs((_t % 8) - (_f % 8)) > 1:
            continue                          # off the board, or wrapped around a file
        for _p in UNDERPROMOTIONS:
            PROMO_IDX[(_f, _t, _p)] = _n
            IDX_PROMO[_n] = (_f, _t, _p)
            _n += 1
N_MOVES = _n

# the byte a UCI string uses for each underpromotion, for converters that parse bytes directly
UCI_PROMO = {110: chess.KNIGHT, 98: chess.BISHOP, 114: chess.ROOK, 113: chess.QUEEN}


def move_to_idx(move, flip):
    f, t = move.from_square, move.to_square
    if flip:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    if move.promotion and move.promotion != chess.QUEEN:
        return PROMO_IDX.get((f, t, move.promotion), f * 64 + t)
    return f * 64 + t


def idx_to_move(idx, board):
    flip = (board.turn == chess.BLACK)
    idx = int(idx)
    under = IDX_PROMO.get(idx)
    if under is not None:
        f, t, promo = under
        if flip:
            f, t = chess.square_mirror(f), chess.square_mirror(t)
        return chess.Move(f, t, promotion=promo)
    f, t = divmod(idx, 64)
    if flip:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    pc = board.piece_at(f)
    promo = chess.QUEEN if (pc and pc.piece_type == chess.PAWN
                            and chess.square_rank(t) in (0, 7)) else None
    return chess.Move(f, t, promotion=promo)


if __name__ == "__main__":
    print(f"{N_FROMTO} from-to indices plus {len(PROMO_IDX)} underpromotions = {N_MOVES}")
    assert N_MOVES == 4096 + 66, N_MOVES
    assert len(PROMO_IDX) == len(IDX_PROMO) == 66

    # Every promotion available on one square must land on its own index and come back unchanged,
    # from both sides of the board.
    for fen, flip in (("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", False),
                      ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", True)):
        b = chess.Board(fen)
        seen = {}
        for m in b.legal_moves:
            if not m.promotion:
                continue
            i = move_to_idx(m, flip)
            assert i not in seen, f"{m.uci()} collides with {seen[i]}"
            seen[i] = m.uci()
            assert idx_to_move(i, b) == m, (i, idx_to_move(i, b).uci(), m.uci())
        print(f"  {'black' if flip else 'white'}: {len(seen)} distinct promotion indices, "
              f"all round tripping")

    # Report how Leela's move list maps onto these indices, and any index several moves share.
    try:
        from lc0_moves import policy_index
    except ImportError:
        raise SystemExit(0)
    import collections
    idxs = [move_to_idx(chess.Move.from_uci(u), False) for u in policy_index]
    dupe = {k: v for k, v in collections.Counter(idxs).items() if v > 1}
    carried = sum(dupe.values())
    print(f"  Leela's 1,858 moves land on {len(set(idxs))} indices, "
          f"{len(dupe)} of them shared by {carried} moves")
    for k in sorted(dupe)[:4]:
        names = [u for u in policy_index if move_to_idx(chess.Move.from_uci(u), False) == k]
        print(f"    index {k}: {names}")
