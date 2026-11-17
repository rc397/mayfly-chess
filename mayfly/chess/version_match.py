r"""version_match plays two frozen versions against each other with the real tree and reports Elo.

The search is deterministic, so from one position the two versions would play the same game every time.
Each pair of games starts from a real position eight to fourteen plies into a master game, played once
with each colour. A game that reaches the ply cap is adjudicated by Stockfish rather than called a draw.
Elo comes with a 95% interval from the standard error of the score, about plus or minus 90 Elo at 60
games.

  python chess/version_match.py [GAMES] [VISITS]
  env: A=mayfly_v1.pt  B=mayfly_v2.pt  SF_MS=200  MAX_PLIES=200
"""
import os, sys, glob, random, math, time
import chess, chess.pgn, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
VISITS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
A_NAME = E("A", "mayfly_v1.pt")
B_NAME = E("B", "mayfly_v2.pt")
SF_MS = int(E("SF_MS", 200))
MAX_PLIES = int(E("MAX_PLIES", 200))
# openings from chess/twic if it is there, else from the data root
PGN_DIR = E("PGN", os.path.join(HERE, "twic") if os.path.isdir(os.path.join(HERE, "twic"))
            else paths.under("twic"))


def openings(n, seed=17):
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
    if not files:
        raise SystemExit(f"no openings: no .pgn files in {PGN_DIR} (set PGN=...)")
    rng.shuffle(files)
    out, seen = [], set()
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
                for m in mv[:rng.randrange(8, 15)]:
                    b.push(m)
                f = b.fen()
                if b.is_game_over() or f in seen:
                    continue
                seen.add(f)
                out.append(f)
        if len(out) >= n:
            break
    return out[:n]


def elo(score):
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


if __name__ == "__main__":
    import torch
    from mcts import MCTS, pinned_search
    from play_web import load_version

    pairs = GAMES // 2
    book = openings(pairs)
    a_model, a_label, _ = load_version(A_NAME)
    b_model, b_label, _ = load_version(B_NAME)
    dev = str(next(a_model.parameters()).device)
    print(f"{A_NAME} against {B_NAME}", flush=True)
    print(f"  A = {a_label}", flush=True)
    print(f"  B = {b_label}", flush=True)
    print(f"  {len(book)} openings x 2 colours = {2*len(book)} games, {VISITS} visits a move, "
          f"on {dev}", flush=True)

    sf_path = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 4, "Hash": 512})

    def adjudicate(board):
        """Stockfish's verdict from WHITE's point of view, as a score in {0, 0.5, 1}."""
        try:
            info = eng.analyse(board, chess.engine.Limit(time=SF_MS / 1000.0))
            cp = info["score"].white().score(mate_score=10000)
        except Exception:
            return 0.5
        if cp is None:
            return 0.5
        return 1.0 if cp > 300 else (0.0 if cp < -300 else 0.5)

    def play(white_model, black_model, fen, white_is_a=True):
        """One game. Returns White's score and the number of plies."""
        board = chess.Board(fen)
        # each side plays with the search its own version was frozen with
        trees = {chess.WHITE: MCTS(white_model, dev, batch=48, cache_size=20000).apply(
                     pinned_search(A_NAME if white_is_a else B_NAME)),
                 chess.BLACK: MCTS(black_model, dev, batch=48, cache_size=20000).apply(
                     pinned_search(B_NAME if white_is_a else A_NAME))}
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
            mv, _ = trees[board.turn].search(board, visits=VISITS, reuse=True)
            if mv is None or mv not in board.legal_moves:
                break
            board.push(mv)
            plies += 1
        if board.is_game_over(claim_draw=True):
            r = board.result(claim_draw=True)
            return ({"1-0": 1.0, "0-1": 0.0}.get(r, 0.5), plies, "played out")
        return (adjudicate(board), plies, "adjudicated")

    a_score = 0.0
    w = d = l = 0
    adj = 0
    t0 = time.time()
    print("")
    print(f"  {'#':>4} {'opening':>8} {'A as':>6} {'result':>7} {'plies':>6} {'how':>12} "
          f"{'A score':>8}")
    n = 0
    for i, fen in enumerate(book):
        for a_is_white in (True, False):
            n += 1
            if a_is_white:
                ws, plies, how = play(a_model, b_model, fen, True)
                s = ws
            else:
                ws, plies, how = play(b_model, a_model, fen, False)
                s = 1.0 - ws
            a_score += s
            w += s == 1.0
            d += s == 0.5
            l += s == 0.0
            adj += how == "adjudicated"
            print(f"  {n:>4} {i+1:>8} {'White' if a_is_white else 'Black':>6} "
                  f"{('win' if s == 1 else 'draw' if s == 0.5 else 'loss'):>7} {plies:>6} "
                  f"{how:>12} {a_score:>8.1f}", flush=True)
    eng.quit()

    el = time.time() - t0
    score = a_score / n
    se = math.sqrt(max(score * (1 - score), 1e-9) / n)
    lo, hi = max(0.001, score - 1.96 * se), min(0.999, score + 1.96 * se)
    print("")
    print(f"  {n} games in {el/60:.1f} minutes, {adj} adjudicated")
    print(f"  A ({A_NAME}) scored {a_score:.1f} of {n}: {w} wins, {d} draws, {l} losses")
    print(f"  score {100*score:.1f}%  ->  A is {elo(score):+.0f} Elo against B")
    print(f"  95% interval {100*lo:.1f}% to {100*hi:.1f}%, which is "
          f"{elo(lo):+.0f} to {elo(hi):+.0f} Elo")
    if lo < 0.5 < hi:
        print(f"  THE INTERVAL CONTAINS 50%, so {n} games did not separate these two versions.")
    else:
        print(f"  The interval excludes 50%, so the difference is real at this sample size.")
