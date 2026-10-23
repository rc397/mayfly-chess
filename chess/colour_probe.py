"""Probe whether the side to move can be recovered from the canonical, mirrored board.

The board is mirrored into the side to move's frame, so a White-to-move position and the mirror of a
Black-to-move position are identical. Chess is symmetric in its rules but not in which positions
occur, so colour may still be inferable. This uses real master games from TWIC, where colour is known.
A probe near 50% means colour is lost. Well above means the network can condition on it implicitly.
The split is by game, since positions in one game are not independent, and a shuffled-label control
must fall to chance or the probe is broken.

    python chess/colour_probe.py [N_POSITIONS]
    env PGN (folder of PGN files, default chess/twic), STRIDE=3
"""
import os, sys, glob, random
import numpy as np
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import board_to_ids

E = os.environ.get
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
PGN_DIR = E("PGN", os.path.join(HERE, "twic"))

STRIDE = int(E("STRIDE", 3))          # odd, so it lands on both colours


def harvest(n, seed=0):
    """Positions from real games, tagged with the colour to move and which game they came from."""
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
    rng.shuffle(files)
    ids_all, cast_all, y_all, game_all = [], [], [], []
    g_id = 0
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while len(y_all) < n:
                try:
                    g = chess.pgn.read_game(fh)
                except Exception:
                    break
                if g is None:
                    break
                moves = list(g.mainline_moves())
                if len(moves) < 20:
                    continue
                b = g.board()
                for ply, m in enumerate(moves):
                    # past the book, and an odd stride so both colours are sampled
                    if ply >= 8 and ply % STRIDE == 0 and not b.is_game_over():
                        ids, cast, _ = board_to_ids(b)
                        ids_all.append(ids)
                        cast_all.append(cast)
                        y_all.append(1 if b.turn == chess.BLACK else 0)
                        game_all.append(g_id)
                    b.push(m)
                g_id += 1
        if len(y_all) >= n:
            break
    return (np.array(ids_all[:n], np.int64), np.array(cast_all[:n], np.float32),
            np.array(y_all[:n], np.int64), np.array(game_all[:n], np.int64))


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    print(f"harvesting up to {N:,} positions from real games, every {STRIDE} plies", flush=True)
    ids, cast, y, game = harvest(N)
    print(f"  {len(y):,} positions from {len(set(game.tolist())):,} games, "
          f"{100*y.mean():.1f}% black to move", flush=True)

    # split BY GAME so no game has positions on both sides of the line
    games = sorted(set(game.tolist()))
    cut_game = games[int(len(games) * 0.8)]
    tr, te = game < cut_game, game >= cut_game
    print(f"  train {tr.sum():,} positions, held out {te.sum():,}, split by game", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    oh = torch.nn.functional.one_hot(torch.from_numpy(ids), 13).float().reshape(len(y), -1)
    X = torch.cat([oh, torch.from_numpy(cast)], 1)
    Y = torch.from_numpy(y)

    def run(name, Xtr, ytr, Xte, yte, hidden=0, steps=400):
        torch.manual_seed(0)
        d = Xtr.shape[1]
        m = (nn.Linear(d, 2) if not hidden else
             nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 2))).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)
        for _ in range(steps):
            i = torch.randint(0, len(ytr), (8192,), device=dev)
            loss = nn.functional.cross_entropy(m(Xtr[i]), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = 100.0 * (m(Xte).argmax(1) == yte).float().mean().item()
        print(f"  {name:<34} {acc:6.2f}%")
        return acc

    base = 100.0 * max(y[te].mean(), 1 - y[te].mean())
    print("")
    print(f"  {'always guess the commoner class':<34} {base:6.2f}%   <- the number to beat")
    run("linear probe on the canonical board", X[tr], Y[tr], X[te], Y[te])
    run("2-layer probe, 256 hidden", X[tr], Y[tr], X[te], Y[te], hidden=256)

    # a control, the same probe on scrambled labels must collapse to chance or the probe is broken
    gen = np.random.default_rng(0)
    Ys = torch.from_numpy(gen.permutation(y))
    run("CONTROL, labels shuffled", X[tr], Ys[tr], X[te], Ys[te], hidden=256)
