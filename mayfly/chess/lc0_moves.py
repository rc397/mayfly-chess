"""The 1,858 moves of Leela Chess Zero's policy head, in Leela's order, so Leela's training data can be mapped onto
MayFly's moves.

For each square from a1, b1 and so on to h8, every queen-line and knight move from it, targets in the same square
order. Then the promotions from the seventh rank to queen, rook and bishop. In the first part a plain move onto the
last rank stands for a knight promotion, as in Leela.
"""
FILES = "abcdefgh"


def _name(sq):
    return FILES[sq % 8] + str(sq // 8 + 1)


def _targets(sq):
    f, r = sq % 8, sq // 8
    out = set()
    for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        x, y = f + df, r + dr
        while 0 <= x < 8 and 0 <= y < 8:
            out.add(y * 8 + x)
            x, y = x + df, y + dr
    for df, dr in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
        x, y = f + df, r + dr
        if 0 <= x < 8 and 0 <= y < 8:
            out.add(y * 8 + x)
    return sorted(out)


def _build():
    moves = [_name(s) + _name(t) for s in range(64) for t in _targets(s)]
    for f in range(8):
        for tf in (f - 1, f, f + 1):
            if 0 <= tf < 8:
                moves += [FILES[f] + "7" + FILES[tf] + "8" + p for p in "qrb"]
    return moves


policy_index = _build()
