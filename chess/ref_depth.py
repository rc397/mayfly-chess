r"""Separate the model's error from the noise in a shallow Stockfish reference.

The leaf value's correlation with Stockfish at a shallow depth mixes two things, how much the model
fails to learn from its teacher (ChessBench, 50 ms a move) and how noisy the shallow reference is
itself. Every position is scored twice, shallow and deep, and three correlations come out.

  r(model, shallow)   the usual measurement
  r(model, deep)      the model against a better yardstick
  r(shallow, deep)    the ceiling, since the shallow reference cannot measure the model more
                      reliably than it agrees with the deep one

A model that tracks the shallow reference better than the deep one has learned its teacher,
mistakes included, so deeper labels (deep_label.py) are the lever rather than more labels.

  python chess/ref_depth.py [POSITIONS] [SHALLOW_MS] [DEEP_MS]
"""
import os, sys, math
import chess, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from value_calib import sample, win_prob, pearson, spearman

E = os.environ.get
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 600
SHALLOW = int(sys.argv[2]) if len(sys.argv) > 2 else 80
DEEP = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
BATCH = int(E("BATCH", 64))


if __name__ == "__main__":
    import numpy as np
    import torch
    from play_web import load_version
    from chess_net import board_to_ids, clock_and_ep
    from moves import move_to_idx

    fens = sample(N_POS, seed=29)          # a different seed to the calibration set
    model, label, _ = load_version(E("VERSION", "mayfly_v2.pt"))
    dev = str(next(model.parameters()).device)
    print(f"{len(fens)} positions, {label} on {dev}", flush=True)
    print(f"  shallow reference {SHALLOW} ms, deep reference {DEEP} ms, "
          f"the model's own teacher is 50 ms a move", flush=True)

    sf_path = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 4, "Hash": 512})

    def score(board, ms):
        info = eng.analyse(board, chess.engine.Limit(time=ms / 1000.0))
        cp = info["score"].relative.score(mate_score=10000)
        return None if cp is None else win_prob(max(-2000, min(2000, cp)))

    shal, deep, keep = [], [], []
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        try:
            a, d = score(b, SHALLOW), score(b, DEEP)
        except Exception:
            continue
        if a is None or d is None:
            continue
        shal.append(a)
        deep.append(d)
        keep.append(fen)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(fens)} scored", flush=True)
    eng.quit()
    print(f"  {len(keep)} positions scored at both depths", flush=True)

    vpw = []
    with torch.no_grad():
        for i in range(0, len(keep), BATCH):
            chunk = [chess.Board(f) for f in keep[i:i + BATCH]]
            enc = [board_to_ids(b) for b in chunk]
            ce = [clock_and_ep(b, e[2]) for b, e in zip(chunk, enc)]
            ids = torch.tensor(np.array([e[0] for e in enc]), dtype=torch.long, device=dev)
            cast = torch.tensor(np.array([e[1] for e in enc], np.float32), device=dev)
            hm = torch.tensor([c[0] for c in ce], dtype=torch.long, device=dev)
            ep = torch.tensor([c[1] for c in ce], dtype=torch.long, device=dev)
            with torch.autocast("cuda", torch.bfloat16, enabled=(dev != "cpu")):
                out = model(ids, cast, hm, ep)
            q = torch.sigmoid(out["qvalue"].float())
            pl = out["policy"].float()
            widths = [b.legal_moves.count() for b in chunk]
            wide = max(widths)
            I = np.zeros((len(chunk), wide), np.int64)
            M = np.zeros((len(chunk), wide), bool)
            for j, b in enumerate(chunk):
                flip = b.turn == chess.BLACK
                idx = [move_to_idx(m, flip) for m in b.legal_moves]
                I[j, :len(idx)] = idx
                M[j, :len(idx)] = True
            It, Mt = torch.from_numpy(I).to(dev), torch.from_numpy(M).to(dev)
            w = torch.softmax(torch.gather(pl, 1, It).masked_fill(~Mt, float("-inf")), -1)
            vpw += (w * torch.gather(q, 1, It)).sum(1).cpu().numpy().tolist()

    print("")
    print(f"  {'pair':<34} {'r':>8} {'rho':>8}")
    rows = [(f"model against {SHALLOW} ms", vpw, shal),
            (f"model against {DEEP} ms", vpw, deep),
            (f"{SHALLOW} ms against {DEEP} ms  <- the ceiling", shal, deep)]
    got = {}
    for name, a, b in rows:
        r, rho = pearson(a, b), spearman(a, b)
        got[name] = r
        print(f"  {name:<34} {r:>+8.3f} {rho:>+8.3f}")

    ceil = got[f"{SHALLOW} ms against {DEEP} ms  <- the ceiling"]
    m_sh = got[f"model against {SHALLOW} ms"]
    m_dp = got[f"model against {DEEP} ms"]
    print("")
    print(f"  The shallow reference explains {100*ceil**2:.0f}% of the deep one's variance, so as a")
    print(f"  yardstick it is imperfect and some of the model's apparent error was the ruler.")
    if ceil > 1e-6:
        print(f"  Attenuation-corrected, the model against the {SHALLOW} ms reference would read "
              f"{m_sh/ceil:+.3f} if the reference were exact.")
    print(f"  Measured directly against {DEEP} ms it reads {m_dp:+.3f} against {m_sh:+.3f}.")
    print("")
    if m_dp > m_sh + 0.03:
        print("  THE MODEL IS BETTER THAN THE ORIGINAL NUMBER SAID. Part of r = 0.574 was the")
        print("  yardstick, and a deeper teacher has information the 50 ms corpus does not.")
    elif m_dp < m_sh - 0.03:
        print("  The model tracks the SHALLOW reference more closely than the deep one, which is")
        print("  what a model saturated on a 50 ms teacher looks like: it has learned its teacher,")
        print("  including its teacher's mistakes. Deeper labels are the lever, not more of them.")
    else:
        print("  Depth of reference barely matters, so r = 0.574 is the model and the original")
        print("  conclusion stands unchanged.")
