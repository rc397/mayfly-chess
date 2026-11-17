r"""sweep_search grids c_puct, virtual loss and FPU, scored on agreement with Stockfish and on collisions.

Q spans about +-0.05 in this model while U at 100 visits is around 0.3, so c_puct decides whether the
search uses its own evaluations at all. Collisions are high even though most leaves sit at ply 2 or
deeper, and virtual loss is what spreads the descents. The two interact, so they are swept together.
FPU sets what an unvisited move is worth, which decides whether an unvisited sibling wins the race
against a visited move's penalty. Stockfish's move for each position is computed once and reused.

  python chess/sweep_search.py [POSITIONS] [VISITS]
  env: SF_MS=300  VERSION=mayfly_v1.pt  PGN (default paths.under("twic"))  CPUCTS  VLS  FPUS
"""
import os, sys, glob, random, itertools, time
import chess, chess.pgn, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
VISITS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SF_MS = int(E("SF_MS", 300))
PGN_DIR = E("PGN", paths.under("twic"))
CPUCTS = [float(x) for x in E("CPUCTS", "0.75,1.0,1.25,1.5,1.75").split(",")]
VLS = [float(x) for x in E("VLS", "1.0,1.5,2.0").split(",")]
FPUS = [float(x) for x in E("FPUS", "0.33,0.44,0.55").split(",")]


def sample(n, seed=5):
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
                if len(mv) < 30:
                    continue
                b = g.board()
                for m in mv[:rng.randrange(14, 45)]:
                    b.push(m)
                if not b.is_game_over() and b.legal_moves.count() > 2:
                    out.append(b.fen())
        if len(out) >= n:
            break
    return out[:n]


if __name__ == "__main__":
    import torch
    from mcts import MCTS
    from play_web import load_version

    fens = sample(N_POS)
    model, label, _ = load_version(E("VERSION", "mayfly_v1.pt"))
    dev = str(next(model.parameters()).device)

    sf_path = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 4, "Hash": 512})
    ref = {}
    for fen in fens:
        try:
            ref[fen] = eng.play(chess.Board(fen), chess.engine.Limit(time=SF_MS / 1000.0)).move
        except Exception:
            pass
    eng.quit()
    fens = [f for f in fens if f in ref]
    print(f"{len(fens)} positions, {VISITS} visits, Stockfish at {SF_MS} ms computed once", flush=True)
    print(f"  grid: c_puct {CPUCTS} x virtual loss {VLS} x fpu {FPUS} "
          f"= {len(CPUCTS)*len(VLS)*len(FPUS)} cells", flush=True)
    print("", flush=True)

    print(f"  {'c_puct':>7} {'vl':>5} {'fpu':>5} {'agree':>7} {'coll/eval':>10} {'evals':>8} "
          f"{'s/move':>7}")
    rows = []
    for c, vl, fp in itertools.product(CPUCTS, VLS, FPUS):
        hits = coll = evals = 0
        t0 = time.time()
        for fen in fens:
            t = MCTS(model, dev, batch=48, c_puct=c, fpu=fp, cache_size=20000)
            t.vl = vl
            mv, _ = t.search(chess.Board(fen), visits=VISITS, reuse=False)
            hits += int(mv == ref[fen])
            coll += t.stats["collisions"]
            evals += t.stats["evals"]
        el = time.time() - t0
        agree = 100.0 * hits / len(fens)
        per = coll / max(evals, 1)
        rows.append((agree, -per, c, vl, fp, evals, el / len(fens)))
        cur = abs(c - 1.75) < 1e-9 and abs(vl - 1.0) < 1e-9 and abs(fp - 0.33) < 1e-9
        print(f"  {c:>7.2f} {vl:>5.1f} {fp:>5.2f} {agree:>6.1f}% {per:>10.2f} {evals:>8,} "
              f"{el/len(fens):>7.2f}{'  <- current' if cur else ''}", flush=True)

    rows.sort(reverse=True)
    a, negper, c, vl, fp, evals, spm = rows[0]
    print("")
    print(f"  best agreement:    c_puct={c} vl={vl} fpu={fp} at {a:.1f}%, "
          f"{-negper:.2f} collisions per eval")
    byc = sorted(rows, key=lambda r: r[1], reverse=True)[0]
    print(f"  fewest collisions: c_puct={byc[2]} vl={byc[3]} fpu={byc[4]} at {-byc[1]:.2f} "
          f"per eval, {byc[0]:.1f}% agreement")
    # mean agreement along each axis
    import statistics as st
    for name, i, vals in (("c_puct", 2, CPUCTS), ("vl", 3, VLS), ("fpu", 4, FPUS)):
        means = [st.mean([r[0] for r in rows if abs(r[i] - v) < 1e-9]) for v in vals]
        print(f"  agreement by {name:<7}: " +
              "  ".join(f"{v}={m:.1f}%" for v, m in zip(vals, means)) +
              f"   spread {max(means)-min(means):.1f} points")
