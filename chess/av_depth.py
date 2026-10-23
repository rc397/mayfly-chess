"""Measure whether deeper Stockfish gives different action values, or just slower ones.

ref_depth.py found that position evaluations barely change between 80 ms and 1500 ms. A shallow search
can still get the top move right while mis-ranking the moves below it, and the tree reads all of them
(v_pw is a policy-weighted mean over every legal move). So this compares the full action-value vector,
the inferior moves alone, and best-move agreement between a shallow and a deep MultiPV search over every
legal move. The defaults, 50 ms against 500 ms a move, match the ChessBench teacher against deep_label.

    python chess/av_depth.py [POSITIONS] [SHALLOW_MS_PER_MOVE] [DEEP_MS_PER_MOVE]
"""
import os, sys, math
import chess, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from value_calib import sample, win_prob, pearson, spearman

E = os.environ.get
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SHALLOW = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
DEEP = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0


def action_values(eng, board, ms_per_move):
    """Win probability for every legal move, from the side to move's point of view."""
    n = board.legal_moves.count()
    infos = eng.analyse(board, chess.engine.Limit(time=ms_per_move * n / 1000.0), multipv=n)
    out = {}
    for info in infos:
        pv = info.get("pv")
        if not pv or "score" not in info:
            continue
        cp = info["score"].pov(board.turn).score(mate_score=10000)
        if cp is None:
            continue
        out[pv[0].uci()] = win_prob(max(-2000, min(2000, cp)))
    return out


if __name__ == "__main__":
    fens = sample(N_POS, seed=41)
    sf_path = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 4, "Hash": 512})
    print(f"{len(fens)} positions, MultiPV over every legal move", flush=True)
    print(f"  shallow {SHALLOW:.0f} ms a move (the ChessBench teacher), "
          f"deep {DEEP:.0f} ms a move ({DEEP/SHALLOW:.0f}x)", flush=True)

    all_s, all_d = [], []
    inf_s, inf_d = [], []
    same_best = 0
    n_ok = 0
    worst = []
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        try:
            s = action_values(eng, b, SHALLOW)
            d = action_values(eng, b, DEEP)
        except Exception as e:
            print(f"    skipped: {type(e).__name__}: {e}", flush=True)
            continue
        shared = [k for k in s if k in d]
        if len(shared) < 3:
            continue
        n_ok += 1
        best_s = max(shared, key=lambda k: s[k])
        best_d = max(shared, key=lambda k: d[k])
        same_best += int(best_s == best_d)
        for k in shared:
            all_s.append(s[k])
            all_d.append(d[k])
            if k != best_d:
                inf_s.append(s[k])
                inf_d.append(d[k])
        gap = max(abs(s[k] - d[k]) for k in shared)
        worst.append((gap, fen, best_s, best_d))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(fens)} positions", flush=True)
    eng.quit()

    print("")
    print(f"  {n_ok} positions, {len(all_s):,} action values compared")
    print(f"  {'pair':<36} {'r':>8} {'rho':>8} {'n':>8}")
    print(f"  {'every legal move':<36} {pearson(all_s, all_d):>+8.3f} "
          f"{spearman(all_s, all_d):>+8.3f} {len(all_s):>8,}")
    if inf_s:
        print(f"  {'inferior moves only':<36} {pearson(inf_s, inf_d):>+8.3f} "
              f"{spearman(inf_s, inf_d):>+8.3f} {len(inf_s):>8,}")
    print(f"  the two depths pick the same best move in {100*same_best/max(n_ok,1):.1f}% "
          f"of positions")
    mad = sum(abs(a - b) for a, b in zip(all_s, all_d)) / max(len(all_s), 1)
    print(f"  mean absolute difference in win probability: {mad:.4f}")

    # The mean hides the tail. A small mean could be every label slightly off, or most exact and a few
    # far off, and the few are the tactical positions worth getting right. The pairs are saved to disk.
    import json as _json
    dif = sorted(abs(a - b) for a, b in zip(all_s, all_d))
    q = lambda f: dif[min(len(dif) - 1, int(f * len(dif)))]
    print("")
    print(f"  the distribution of |difference|, not just its mean")
    print(f"    {'p50':>6} {'p75':>6} {'p90':>6} {'p95':>6} {'p99':>6} {'max':>6}")
    print(f"    {q(.50):>6.4f} {q(.75):>6.4f} {q(.90):>6.4f} {q(.95):>6.4f} "
          f"{q(.99):>6.4f} {dif[-1]:>6.4f}")
    over = lambda t: 100.0 * sum(1 for x in dif if x > t) / max(len(dif), 1)
    print(f"    share of action values that moved by more than "
          f"0.03: {over(0.03):.1f}%   0.05: {over(0.05):.1f}%   0.10: {over(0.10):.1f}%")
    with open(os.path.join(HERE, "av_depth_pairs.json"), "w", encoding="utf-8") as fh:
        _json.dump({"shallow": all_s, "deep": all_d, "shallow_ms": SHALLOW,
                    "deep_ms": DEEP, "n_positions": n_ok}, fh)
    p99 = q(.99)
    print("")
    if p99 < 0.05:
        print(f"  THE TAIL IS THIN. p99 of {p99:.4f} means depth barely matters anywhere, not just")
        print(f"  on average. POSITION_SECONDS can go to 2.")
    elif p99 < 0.10:
        print(f"  p99 of {p99:.4f}: a small set of positions is depth-sensitive. 2 s is defensible,")
        print(f"  5 s is the cautious call.")
    else:
        print(f"  p99 of {p99:.4f}: a real tail. Depth matters for a few percent of values, and")
        print(f"  those are the ones worth getting right. POSITION_SECONDS should go to 5, not 2.")

    print("")
    worst.sort(reverse=True)
    print("  the five positions where the depths disagreed most on some move")
    for gap, fen, bs, bd in worst[:5]:
        print(f"    {gap:.3f}  best {bs} -> {bd}   {fen[:46]}")

    print("")
    r_inf = pearson(inf_s, inf_d) if inf_s else 0.0
    if r_inf > 0.97:
        print("  DEEPER LABELS ADD ALMOST NOTHING, inferior moves included. The teacher the model")
        print("  already has is near exact, the gap to it is the model's own, and deep_label is")
        print("  spending cores on a target it can barely improve.")
    elif r_inf > 0.90:
        print("  Deeper labels move the inferior moves a little. Whether that is worth twenty-four")
        print("  cores depends on how much of the model's error lives in those moves.")
    else:
        print("  THE INFERIOR MOVES ARE WHERE THE DEPTH GOES. The shallow teacher mis-ranks what")
        print("  it is not focused on, the tree reads exactly those values through v_pw, and")
        print("  deep_label is justified.")
