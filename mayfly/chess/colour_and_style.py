"""Two checks on the policy corpus, colour recoverability and pawn-move style.

The board is mirrored into the side to move's frame, so the network cannot see its colour directly.
Colour may still be inferable from which positions occur, since chess is symmetric in its rules but
not in its distribution. A probe tries to predict the original colour (ply parity) from the canonical
board. High accuracy means the network can condition on colour implicitly.

The second check measures how often the corpus's own best move is a pawn move, or a rook-pawn move,
for each source, to see whether a strong pawn-pushing tendency in the model comes from the data.

    python chess/colour_and_style.py [N_PER_SOURCE]
    env SHARDS (shard folder)
"""
import os, sys, glob, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pack import DT, open_shard
from moves import IDX_PROMO
import paths

E = os.environ.get
SHARDS = E("SHARDS", paths.SHARDS)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000


def load(prefix, n, seed=0):
    """Rows from this source's shards, sampled across files rather than from one of them."""
    paths = sorted(glob.glob(os.path.join(SHARDS, prefix + "_*.bin")))
    if not paths:
        return None
    rng = np.random.default_rng(seed)
    per = max(n // len(paths), 1)
    out = []
    for p in paths:
        mm = open_shard(p)
        if not len(mm):
            continue
        k = min(per, len(mm))
        out.append(mm[np.sort(rng.choice(len(mm), k, replace=False))])
        if sum(len(o) for o in out) >= n:
            break
    return np.concatenate(out)[:n] if out else None


def from_square(idx):
    u = IDX_PROMO.get(int(idx))
    return u[0] if u else int(idx) // 64


def style(rows):
    """What piece does the corpus's own best move move, in the mover's frame?"""
    ids = rows["ids"]
    f = np.array([from_square(i) for i in rows["mv"][:, 0]])
    piece = ids[np.arange(len(rows)), f]
    is_pawn = piece == 1                      # PIECE_ORD[PAWN] == 1, and 1-6 are the mover's
    is_rook_pawn = is_pawn & np.isin(f % 8, (0, 7))
    return 100.0 * is_rook_pawn.mean(), 100.0 * is_pawn.mean()


if __name__ == "__main__":
    print("=== TWO: what does the corpus itself play ===")
    print("  (MayFly v1 measured at 37.8% rook pawn / 63.8% any pawn on real positions,")
    print("   Stockfish at 6.8% / 25.2%, the titled player at 5.2% / 22.5%)")
    print("")
    sets = {}
    for prefix, name in (("lz", "Lichess + Stockfish"), ("lc", "Leela"),
                         ("gm", "our tournament games"), ("sp", "our self play")):
        rows = load(prefix, N)
        if rows is None or not len(rows):
            print(f"  {name:<22} no shards")
            continue
        sets[prefix] = rows
        rp, pw = style(rows)
        plies = rows["ply"]
        print(f"  {name:<22} n={len(rows):>7,}   rook pawn {rp:5.1f}%   any pawn {pw:5.1f}%"
              f"   ply range {plies.min()}-{plies.max()}, all zero: {bool((plies == 0).all())}")

    print("")
    print("=== ONE: can the colour be recovered from the canonical board ===")
    usable = {k: v for k, v in sets.items() if not (v["ply"] == 0).all()}
    if not usable:
        print("  no source records a usable ply, so there is no colour label to learn from")
        raise SystemExit(0)

    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for prefix, rows in usable.items():
        ids = torch.from_numpy(rows["ids"].astype(np.int64))
        cast = torch.from_numpy(rows["cast"].astype(np.float32))
        y = torch.from_numpy((rows["ply"] % 2).astype(np.int64))      # 0 white, 1 black
        n = len(y)
        cut = int(n * 0.8)
        share = 100.0 * float(y.float().mean())
        print(f"  {prefix}: {n:,} positions, {share:.1f}% of them black to move "
              f"(a coin flip scores {max(share, 100-share):.1f}%)")

        for name, make in (("linear probe", lambda: nn.Sequential(nn.Flatten(), nn.Linear(64 * 13 + 4, 2))),
                           ("2-layer probe", lambda: nn.Sequential(nn.Flatten(), nn.Linear(64 * 13 + 4, 256),
                                                                   nn.ReLU(), nn.Linear(256, 2)))):
            torch.manual_seed(0)
            m = make().to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            oh = torch.nn.functional.one_hot(ids, 13).float().reshape(n, -1)
            X = torch.cat([oh, cast], 1)
            Xtr, ytr = X[:cut].to(dev), y[:cut].to(dev)
            Xte, yte = X[cut:].to(dev), y[cut:].to(dev)
            for ep in range(240):
                i = torch.randint(0, len(ytr), (4096,), device=dev)
                loss = nn.functional.cross_entropy(m(Xtr[i]), ytr[i])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                acc = 100.0 * (m(Xte).argmax(1) == yte).float().mean().item()
            print(f"    {name:<16} held-out accuracy {acc:5.2f}%")
        print("")
