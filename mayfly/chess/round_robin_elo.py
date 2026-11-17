r"""round_robin_elo puts every round robin player on one Elo scale with error bars.

It reads each pairing's game log (round_robin.py writes one line a game, so pairings in progress count too)
and fits the Elo model by maximum likelihood, where player i expects 1 / (1 + 10^((Rj - Ri) / 400)) a game
against player j. ANCHOR is fixed at zero. The 95% intervals come from refitting on games resampled within
each pairing. Only games played under PROTOCOL count (round_robin.py's own by default), so games from
another book or rule set stay out.

  python chess/round_robin_elo.py [folder]      env ANCHOR (v1_0)  BOOT (200)  PROTOCOL (round_robin.PROTOCOL)
"""
import glob
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
FOLDER = sys.argv[1] if len(sys.argv) > 1 else paths.under("abq", "round_robin")
ANCHOR = os.environ.get("ANCHOR", "v1_0")
BOOT = int(os.environ.get("BOOT", 200))
PROTOCOL = os.environ.get("PROTOCOL") or __import__("round_robin").PROTOCOL


def load(folder):
    """{(a, b): [A's score in each game]} from the pairings' game logs (file names a__b.jsonl), PROTOCOL's only."""
    games = defaultdict(list)
    for path in glob.glob(os.path.join(folder, "*__*.jsonl")):
        a, b = os.path.basename(path)[:-len(".jsonl")].split("__")
        seen = {}
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (r.get("key") or "").split("|")[4:5] != [PROTOCOL]:
                continue
            seen[(r.get("key"), r["g"])] = float(r["s_a"])   # a replayed game counts once
        games[(a, b)] += list(seen.values())
    return games


K = np.log(10) / 400                                # Elo points to natural log-odds
PRIOR_SD = 10000.0


def fit(players, pairs, tol=1e-6, max_steps=100):
    """Maximum-likelihood ratings for {(a, b): (points for a, games)}, the anchor at 0.

    Newton's method run to convergence. A fixed number of gradient steps stopped short when players were far
    apart, which put the bootstrap intervals below their own estimates. A wide Gaussian prior (PRIOR_SD Elo)
    keeps a player who has won every game finite and barely moves anyone else."""
    idx = {p: k for k, p in enumerate(players)}
    m = len(players)
    anchor = idx.get(ANCHOR)
    free = np.array([k for k in range(m) if k != anchor])
    ia = np.array([idx[a] for a, _ in pairs])
    ib = np.array([idx[b] for _, b in pairs])
    pts = np.array([pairs[k][0] for k in pairs], dtype=float)
    n = np.array([pairs[k][1] for k in pairs], dtype=float)

    def loglik(r):
        d = K * (r[ia] - r[ib])
        return float(np.sum(-pts * np.logaddexp(0, -d) - (n - pts) * np.logaddexp(0, d))
                     - np.sum(r ** 2) / (2 * PRIOR_SD ** 2))

    r = np.zeros(m)
    for _ in range(max_steps):
        p = 1 / (1 + np.exp(-K * (r[ia] - r[ib])))
        g = -r / PRIOR_SD ** 2
        np.add.at(g, ia, K * (pts - n * p))
        np.add.at(g, ib, -K * (pts - n * p))
        w = K * K * n * p * (1 - p)
        h = -np.eye(m) / PRIOR_SD ** 2
        np.add.at(h, (ia, ia), -w)
        np.add.at(h, (ib, ib), -w)
        np.add.at(h, (ia, ib), w)
        np.add.at(h, (ib, ia), w)
        step = np.zeros(m)
        step[free] = np.linalg.solve(h[np.ix_(free, free)], -g[free])
        base, t = loglik(r), 1.0
        while loglik(r + t * step) < base and t > 1e-6:      # halve until the step goes uphill
            t /= 2
        r = r + t * step
        if np.max(np.abs(t * step)) < tol:
            break
    if anchor is None:
        r -= r.mean()
    return dict(zip(players, r))


def main():
    games = load(FOLDER)
    if not any(games.values()):
        print(f"no games in {FOLDER} yet")
        return
    players = sorted({p for ab, s in games.items() if s for p in ab})
    pairs = {ab: (sum(s), len(s)) for ab, s in games.items() if s}
    est = fit(players, pairs)
    boots = defaultdict(list)
    for _ in range(BOOT):
        res = {ab: (sum(x), len(x)) for ab, s in games.items() if s
               for x in [[random.choice(s) for _ in s]]}
        for p, v in fit(players, res).items():
            boots[p].append(v)
    total = defaultdict(lambda: [0.0, 0])
    for (a, b), (pts, n) in pairs.items():
        total[a][0] += pts
        total[a][1] += n
        total[b][0] += n - pts
        total[b][1] += n
    print(f"{sum(n for _, n in pairs.values())} games in {len(pairs)} pairings ({PROTOCOL}); {ANCHOR} = 0")
    print(f"{'player':<10}{'Elo':>8}{'95% interval':>18}{'games':>8}{'score':>8}")
    for p in sorted(players, key=lambda q: -est[q]):
        lo, hi = np.percentile(boots[p], [2.5, 97.5])
        pts, n = total[p]
        print(f"{p:<10}{est[p]:>8.0f}{f'{lo:.0f} to {hi:.0f}':>18}{n:>8}{100 * pts / n:>7.1f}%")


if __name__ == "__main__":
    main()
