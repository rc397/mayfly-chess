r"""search_value measures where the tree search improves on the bare policy move.

Expert iteration trains on what the search chose, so a search that is worse than the policy in quiet
positions would teach the model to be worse. For each master position this counts fixes (the policy
missed Stockfish's move and search found it) and breaks (the policy had it and search moved off it),
separately for tactical positions (Stockfish's move is a capture or check) and quiet ones.

  python chess/search_value.py [N]
  env: VERSION=mayfly_v1.pt  SF_MS=500  VISIT_LADDER=100,400,1600  PGN (default paths.under("twic"))
"""
import os, sys, glob, random, collections
import chess, chess.pgn, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SF_MS = int(E("SF_MS", 500))
LADDER = [int(x) for x in E("VISIT_LADDER", "100,400,1600").split(",")]
PGN_DIR = E("PGN", paths.under("twic"))


def sample_positions(n, seed=1):
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
    rng.shuffle(files)
    out = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while len(out) < n:
                try:
                    g = chess.pgn.read_game(fh)
                except Exception:
                    break
                if g is None:
                    break
                moves = list(g.mainline_moves())
                if len(moves) < 24:
                    continue
                k = rng.randrange(12, min(len(moves) - 1, 70))
                b = g.board()
                for m in moves[:k]:
                    b.push(m)
                if b.is_game_over() or b.legal_moves.count() < 2:
                    continue
                out.append(b.fen())
        if len(out) >= n:
            break
    return out[:n]


if __name__ == "__main__":
    import torch
    from play_race import think
    from play_web import load_version
    from mcts import MCTS

    print(f"sampling {N} middlegame positions", flush=True)
    fens = sample_positions(N)
    model, label, _ = load_version(E("VERSION", "mayfly_v1.pt"))
    dev = str(next(model.parameters()).device)
    sf_path = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 4, "Hash": 512})

    rows = []
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        try:
            best = eng.play(b, chess.engine.Limit(time=SF_MS / 1000.0)).move
        except Exception:
            continue
        moves, _ = think(model, b)
        if not moves:
            continue
        r = {"fen": fen, "sf": best, "policy": moves[0][0],
             # tactical is decided by Stockfish's own move, so the split does not depend on the model
             "tactical": b.is_capture(best) or b.gives_check(best)}
        for v in LADDER:
            tree = MCTS(model, dev, batch=48, cache_size=20000)
            mv, _ = tree.search(b, visits=v, reuse=False)
            r[v] = mv
        rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(fens)}", flush=True)
    eng.quit()

    tac = [r for r in rows if r["tactical"]]
    qui = [r for r in rows if not r["tactical"]]
    print("")
    print(f"{len(rows)} positions, Stockfish at {SF_MS} ms")
    print(f"  {len(tac)} tactical (its best move is a capture or a check), {len(qui)} quiet")
    print("")

    def agree(rs, key):
        return 100.0 * sum(1 for r in rs if r[key] == r["sf"]) / max(len(rs), 1)

    print("  agreement with Stockfish's move")
    print(f"    {'':<16} {'all':>8} {'tactical':>10} {'quiet':>8}")
    print(f"    {'policy head':<16} {agree(rows,'policy'):>7.1f}% {agree(tac,'policy'):>9.1f}% "
          f"{agree(qui,'policy'):>7.1f}%")
    for v in LADDER:
        print(f"    {str(v) + ' visits':<16} {agree(rows,v):>7.1f}% {agree(tac,v):>9.1f}% "
              f"{agree(qui,v):>7.1f}%")

    print("")
    print("  what search DID to the policy's answer")
    for v in LADDER:
        for name, rs in (("all", rows), ("tactical", tac), ("quiet", qui)):
            fixes = sum(1 for r in rs if r["policy"] != r["sf"] and r[v] == r["sf"])
            breaks = sum(1 for r in rs if r["policy"] == r["sf"] and r[v] != r["sf"])
            changed = sum(1 for r in rs if r[v] != r["policy"])
            net = fixes - breaks
            print(f"    {str(v) + ' visits':<12} {name:<9} changed {100*changed/max(len(rs),1):5.1f}% "
                  f"of moves   fixes {fixes:>3}   breaks {breaks:>3}   net {net:+4d}")
        print("")
