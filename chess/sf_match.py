"""
sf_match plays two UCI engines against each other from unbalanced openings.

Two strong engines from the start position draw almost every game. As in Fishtest, the openings are
positions from the labelled corpus with an edge of about one pawn, so one side has to convert or hold.
Each position is played twice with colours swapped to cancel the book's own bias.

  python chess/sf_match.py [GAMES] [MS_PER_MOVE]
  env: A=<path to engine A, default stockfish on PATH>  B=<path to engine B>  EDGE_LO=80  EDGE_HI=110
"""
import os, sys, json, glob, math, random, time, itertools
import chess, chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
MS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
EDGE_LO, EDGE_HI = int(E("EDGE_LO", 80)), int(E("EDGE_HI", 110))
A_PATH = E("A", "stockfish")
B_PATH = E("B", open(os.path.join(HERE, "stockfish_path.txt")).read().strip())
MAX_MOVES = 160


def book(n):
    """Positions with about a one pawn edge, from the labelled corpus."""
    rng = random.Random(5)
    files = sorted(glob.glob(os.path.join(HERE, "rich", "*.jsonl")))
    rng.shuffle(files)
    out = []
    for fp in files:
        for line in itertools.islice(open(fp, encoding="utf-8"), 60000):
            if len(out) >= n:
                return out
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not (EDGE_LO <= abs(r["cp"]) <= EDGE_HI):
                continue
            b = chess.Board(r["fen"])
            if b.is_game_over() or b.fullmove_number > 30:
                continue
            out.append(r["fen"])
    return out


def play(white, black, fen, ms):
    b = chess.Board(fen)
    while not b.is_game_over(claim_draw=True) and b.fullmove_number < MAX_MOVES:
        eng = white if b.turn == chess.WHITE else black
        try:
            b.push(eng.play(b, chess.engine.Limit(time=ms / 1000)).move)
        except Exception:
            return 0.5
    r = b.result(claim_draw=True)
    return 1.0 if r == "1-0" else 0.0 if r == "0-1" else 0.5


def elo(score, n):
    p = min(max(score, 0.5 / (n + 1)), 1 - 0.5 / (n + 1))
    return -400.0 * math.log10(1.0 / p - 1.0)


if __name__ == "__main__":
    fens = book(PAIRS)
    if not fens:
        raise SystemExit("no book positions found in that evaluation band")
    ea = chess.engine.SimpleEngine.popen_uci(A_PATH); ea.configure({"Threads": 1, "Hash": 64})
    eb = chess.engine.SimpleEngine.popen_uci(B_PATH); eb.configure({"Threads": 1, "Hash": 64})
    na, nb = ea.id.get("name", "A"), eb.id.get("name", "B")
    print(f"{na}  vs  {nb}")
    print(f"{len(fens)} positions at {EDGE_LO} to {EDGE_HI} centipawns, each played both ways, "
          f"{MS}ms per move\n", flush=True)

    sa = decisive = 0.0
    t0 = time.time()
    for i, fen in enumerate(fens):
        sa += play(ea, eb, fen, MS)                 # A as White
        r2 = play(eb, ea, fen, MS)                  # same position, engines swapped
        sa += 1.0 - r2
        decisive += (r2 != 0.5)
        if (i + 1) % 25 == 0:
            g = (i + 1) * 2
            print(f"  {g:4d} games  A scores {100*sa/g:5.1f}%  {time.time()-t0:5.0f}s", flush=True)
    ea.quit(); eb.quit()

    g = len(fens) * 2
    p = sa / g
    se = math.sqrt(max(p * (1 - p), 1e-9) / g)
    print(f"\n{na} scores {100*p:.1f}% over {g} games")
    print(f"Elo difference {elo(p, g):+.0f}  (95% interval about {elo(max(p-1.96*se,1e-4), g):+.0f} "
          f"to {elo(min(p+1.96*se,1-1e-4), g):+.0f})")
    print(f"{100*decisive/len(fens):.0f}% of positions were decisive rather than drawn both ways")
    json.dump({"a": na, "b": nb, "games": g, "score_a": p, "elo": elo(p, g), "ms": MS},
              open(os.path.join(HERE, "sf_match.json"), "w"), indent=1)
