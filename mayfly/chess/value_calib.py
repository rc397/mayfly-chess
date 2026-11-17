r"""value_calib compares the network's leaf values with Stockfish and writes a calibration map.

The search does not read the value head. With `value_from_q=True` (the MCTS default) `_evaluate` scores a
leaf as the policy-weighted mean of the action-value head over the legal moves, because that head is
trained on every ChessBench record while the value head sees only the lz and lc slice. Three candidates
are measured against the same Stockfish reference.

  v_head    sigmoid(out["value"])                    not used by the search
  v_pw      policy-weighted mean action value        what the search uses
  v_max     max action value over legal moves        the minimax identity, for comparison

A flat line means no signal. A sloped but narrow cloud is a scale problem, fixable without retraining
because PUCT only cares about the ratio of Q to U. A wide unordered cloud means the head is confidently
wrong. The binned table separates these, which a correlation alone cannot.

  python chess/value_calib.py [POSITIONS] [SF_MS]
"""
import os, sys, glob, random, math
import chess, chess.pgn, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SF_MS = int(sys.argv[2]) if len(sys.argv) > 2 else 80
PGN_DIR = E("PGN", paths.under("twic"))
BATCH = int(E("BATCH", 64))


def sample(n, seed=11):
    """Positions from real games, spread across the whole game rather than the opening."""
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
    rng.shuffle(files)
    out = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while len(out) < n:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                mv = list(g.mainline_moves())
                if len(mv) < 24:
                    continue
                b = g.board()
                cut = rng.randrange(8, len(mv) - 2)
                for m in mv[:cut]:
                    b.push(m)
                if not b.is_game_over() and b.legal_moves.count() >= 2:
                    out.append(b.fen())
        if len(out) >= n:
            break
    return out[:n]


def win_prob(cp):
    """Centipawns to win probability, the usual logistic. Side-to-move point of view."""
    return 1.0 / (1.0 + 10.0 ** (-cp / 400.0))


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = sum((x - ma) ** 2 for x in a)
    sbb = sum((y - mb) ** 2 for y in b)
    return sab / math.sqrt(saa * sbb) if saa > 0 and sbb > 0 else 0.0


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def spread(name, v, ref):
    v = list(v)
    s = sorted(v)
    n = len(s)
    mean = sum(v) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / n)
    p = lambda q: s[min(n - 1, int(q * n))]
    return (f"  {name:<8} mean {mean:>6.3f}  sd {sd:>6.3f}  "
            f"p5 {p(0.05):>6.3f}  p50 {p(0.50):>6.3f}  p95 {p(0.95):>6.3f}  "
            f"range {s[-1]-s[0]:>5.3f}   r {pearson(v, ref):>+6.3f}  rho {spearman(v, ref):>+6.3f}")


if __name__ == "__main__":
    import numpy as np
    import torch
    from play_web import load_version
    from chess_net import board_to_ids, clock_and_ep
    from moves import move_to_idx

    fens = sample(N_POS)
    model, label, _ = load_version(E("VERSION", "mayfly_v1.pt"))
    dev = str(next(model.parameters()).device)
    print(f"{len(fens)} positions, Stockfish at {SF_MS} ms, {label} on {dev}", flush=True)

    sf_path = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 4, "Hash": 512})
    ref, keep = [], []
    for fen in fens:
        try:
            info = eng.analyse(chess.Board(fen), chess.engine.Limit(time=SF_MS / 1000.0))
            cp = info["score"].relative.score(mate_score=10000)
            if cp is None:
                continue
            ref.append(win_prob(max(-2000, min(2000, cp))))
            keep.append(fen)
        except Exception:
            continue
    eng.quit()
    print(f"  {len(keep)} positions scored by Stockfish", flush=True)

    vh, vpw, vmax = [], [], []
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
            vh += torch.sigmoid(out["value"].float()).cpu().numpy().tolist()
            # exactly the transformation _evaluate applies, so this is the number the tree backs up
            widths = [b.legal_moves.count() for b in chunk]
            wide = max(widths)
            I = np.zeros((len(chunk), wide), np.int64)
            M = np.zeros((len(chunk), wide), bool)
            for j, b in enumerate(chunk):
                flip = b.turn == chess.BLACK
                idx = [move_to_idx(m, flip) for m in b.legal_moves]
                I[j, :len(idx)] = idx
                M[j, :len(idx)] = True
            It = torch.from_numpy(I).to(dev)
            Mt = torch.from_numpy(M).to(dev)
            gq = torch.gather(q, 1, It)
            w = torch.softmax(torch.gather(pl, 1, It).masked_fill(~Mt, float("-inf")), -1)
            vpw += (w * gq).sum(1).cpu().numpy().tolist()
            vmax += gq.masked_fill(~Mt, -1.0).max(1).values.cpu().numpy().tolist()

    print("")
    print("  all three on the same [0,1] win-probability scale as the Stockfish reference")
    print(spread("stockfish", ref, ref))
    print(spread("v_head", vh, ref))
    print(spread("v_pw", vpw, ref))
    print(spread("v_max", vmax, ref))

    print("")
    print("  Stockfish decile against the mean model value in that decile.")
    print("  A flat column is no signal. A rising but narrow column is a scale problem.")
    order = sorted(range(len(ref)), key=lambda i: ref[i])
    k = max(len(order) // 10, 1)
    print(f"    {'SF bin':>14} {'n':>5} {'SF mean':>8} {'v_head':>8} {'v_pw':>8} {'v_max':>8}")
    for d in range(10):
        sl = order[d * k:(d + 1) * k] if d < 9 else order[9 * k:]
        if not sl:
            continue
        m = lambda v: sum(v[i] for i in sl) / len(sl)
        lo, hi = ref[sl[0]], ref[sl[-1]]
        print(f"    {lo:>6.3f}-{hi:<7.3f} {len(sl):>5} {m(ref):>8.3f} {m(vh):>8.3f} "
              f"{m(vpw):>8.3f} {m(vmax):>8.3f}")

    # ---- the calibration map, written out for the search to use
    # Quantile matching, not a regression. A regression would shrink the spread further where the
    # correlation is weak, and the problem is a band that is too narrow, not a bias. Matching
    # quantiles restores the spread and is monotone, so it never reorders siblings. It only changes
    # how much Q counts against U.
    import json as _json
    K = 33
    sv = sorted(vpw)
    sr = sorted(ref)
    xs, ys = [], []
    for i in range(K):
        q = i / (K - 1.0)
        j = min(len(sv) - 1, int(round(q * (len(sv) - 1))))
        x = sv[j] * 2.0 - 1.0                      # [0,1] win probability -> the tree's [-1,1]
        y = sr[j] * 2.0 - 1.0
        if xs and x <= xs[-1] + 1e-9:              # np.interp needs xs strictly increasing
            continue
        xs.append(x)
        ys.append(y)
    out = os.path.join(HERE, "value_calib.json")
    with open(out, "w", encoding="utf-8") as fh:
        _json.dump({"xs": xs, "ys": ys, "n": len(ref), "version": label,
                    "sf_ms": SF_MS, "knots": len(xs)}, fh, indent=1)
    print("")
    print(f"  wrote {out}: {len(xs)} knots from {len(ref)} paired samples")
    print(f"    v_pw spans {xs[0]:+.3f} to {xs[-1]:+.3f}, stretched onto "
          f"{ys[0]:+.3f} to {ys[-1]:+.3f}")
    mid = [(x, y) for x, y in zip(xs, ys) if -0.2 <= x <= 0.2]
    if len(mid) >= 2:
        print(f"    through the middle, {mid[0][0]:+.3f}..{mid[-1][0]:+.3f} becomes "
              f"{mid[0][1]:+.3f}..{mid[-1][1]:+.3f}, a "
              f"{(mid[-1][1]-mid[0][1])/max(mid[-1][0]-mid[0][0], 1e-9):.1f}x stretch")

    print("")
    lo10 = order[:k]
    hi10 = order[-k:]
    for name, v in (("v_head", vh), ("v_pw", vpw), ("v_max", vmax)):
        sep = sum(v[i] for i in hi10) / len(hi10) - sum(v[i] for i in lo10) / len(lo10)
        sd = math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v))
        print(f"  {name:<8} separates the worst tenth from the best tenth by {sep:>+6.3f}, "
              f"which is {sep/sd if sd else 0:>5.2f} of its own standard deviation")
