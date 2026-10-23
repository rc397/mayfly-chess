r"""search_match plays one network against itself with two different searches and reports the Elo.

The network, openings, colours and adjudicator are the same on both sides, so only the search differs.
Agreement with Stockfish on a few dozen positions is too noisy to see changes of this size, so this plays
games and prints the interval.

  python chess/search_match.py "power_p=2.0" [GAMES] [VISITS]
  python chess/search_match.py "mcgs=1" 60 200
  python chess/search_match.py "power_p=2.0,mcgs=1,early_stop=1" 80 200

There is no pondering here. A ponder carries an unequal number of visits into the next search, and this
harness compares two searches at an identical budget. `elo_vs_sf.py` ponders by default and records it.

A is the baseline and B is the baseline plus the options given, so a positive Elo means the change helped.
Options are set on the MCTS object after construction. BASE=mayfly_v3.pt starts both sides from v3's
sealed search, so a run keeps its meaning when the defaults change. A_OPTS adds settings to the baseline
side only. Without BASE both sides start from the constructor's defaults.

  env: VERSION=mayfly_v3.pt  BASE=mayfly_v3.pt  A_OPTS=  RESULTS=search_match.jsonl  SF_MS=200
"""
import os, sys, glob, random, math, time
import chess, chess.pgn, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
# Parsed defensively, because clock_match imports this module for its openings and Elo helpers, and
# the importer's arguments are not this script's.
def _arg(i, default, cast=str):
    try:
        return cast(sys.argv[i])
    except (IndexError, ValueError, TypeError):
        return default


OPTS = _arg(1, "power_p=2.0")
GAMES = _arg(2, 60, int)
VISITS = _arg(3, 200, int)
SF_MS = int(E("SF_MS", 200))
MAX_PLIES = int(E("MAX_PLIES", 200))
# openings from chess/twic if it is there, else from the data root
PGN_DIR = E("PGN", os.path.join(HERE, "twic") if os.path.isdir(os.path.join(HERE, "twic"))
            else paths.under("twic"))


def parse(spec):
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        v = v.strip()
        try:
            out[k] = int(v) if v.isdigit() or (v[:1] == "-" and v[1:].isdigit()) else float(v)
        except ValueError:
            out[k] = v
    return out


def openings(n, seed=23):
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
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
    import torch, json
    from mcts import MCTS, pinned_search
    from play_web import load_version

    opts = parse(OPTS)
    a_opts = parse(E("A_OPTS", ""))
    base_name = E("BASE", "")
    base = pinned_search(base_name) if base_name else {}
    if base_name and not base:
        raise SystemExit(f"BASE={base_name} has no sealed search")
    pairs = GAMES // 2
    # OPENING_SEED picks a different set of openings, for confirming a tuned winner on unseen ones.
    # 23 is the default set.
    book = openings(pairs, seed=int(E("OPENING_SEED", 23)))
    model, label, _ = load_version(E("VERSION", "mayfly_v2.pt"))
    dev = str(next(model.parameters()).device)

    print(f"{label} against itself, {VISITS} visits a move", flush=True)
    print(f"  baseline = {base_name + chr(39) + 's sealed search' if base_name else 'the defaults'}",
          flush=True)
    print(f"  A = baseline" + (f" + {a_opts}" if a_opts else ""), flush=True)
    print(f"  B = baseline + {opts}", flush=True)
    print(f"  {len(book)} openings x 2 colours = {2*len(book)} games on {dev}", flush=True)

    sf_path = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    # python-chess keeps Stockfish on a non-daemon thread until it exits, so an uncaught exception
    # would leave the interpreter waiting forever. The engine is killed as shutdown begins.
    import threading
    def _kill_engine(e=eng):
        try:
            e.transport.kill()
        except Exception:
            pass
    threading._register_atexit(_kill_engine)
    eng.configure({"Threads": 4, "Hash": 512})

    def build(is_b):
        t = MCTS(model, dev, batch=48, cache_size=20000).apply(base)
        for k, v in (opts if is_b else a_opts).items():
            if not hasattr(t, k):
                raise SystemExit(f"MCTS has no option {k!r}")
            setattr(t, k, v)
        return t

    def adjudicate(board):
        try:
            cp = eng.analyse(board, chess.engine.Limit(time=SF_MS / 1000.0))["score"].white()
            cp = cp.score(mate_score=10000)
        except (chess.engine.EngineTerminatedError, TimeoutError):
            raise                           # a dead or stuck Stockfish must not score draws
        except Exception:
            return 0.5
        if cp is None:
            return 0.5
        return 1.0 if cp > 300 else (0.0 if cp < -300 else 0.5)

    def play(fen, b_is_white):
        board = chess.Board(fen)
        trees = {chess.WHITE: build(b_is_white), chess.BLACK: build(not b_is_white)}
        nodes = {True: 0, False: 0}
        plies = 0
        # Only a draw that has arisen ends a game, as in elo_vs_sf. claim_draw would also end it when
        # the side to move merely could repeat, even with a mate in one on the board.
        over = lambda b: b.is_game_over() or b.is_repetition(3) or b.is_fifty_moves()
        while not over(board) and plies < MAX_PLIES:
            side_is_b = (board.turn == chess.WHITE) == b_is_white
            t = trees[board.turn]
            mv, info = t.search(board, visits=VISITS, reuse=True)
            nodes[side_is_b] += info.get("evals", 0)
            if mv is None or mv not in board.legal_moves:
                break
            board.push(mv)
            plies += 1
        if over(board):
            r = board.result() if board.is_game_over() else "1/2-1/2"
            ws = {"1-0": 1.0, "0-1": 0.0}.get(r, 0.5)
            how = "played out"
        else:
            ws, how = adjudicate(board), "adjudicated"
        return (ws if b_is_white else 1.0 - ws), plies, how, nodes

    b_score = 0.0
    w = d = l = 0
    ev_a = ev_b = 0
    t0 = time.time()
    print("")
    print(f"  {'#':>4} {'B as':>6} {'result':>7} {'plies':>6} {'how':>12} {'B score':>8}")
    # Resumable. With GAMES_LOG set every finished game is appended there, and a rerun replays the
    # log instead of the games. The openings are seeded, so game k is the same opening and colour.
    glog = E("GAMES_LOG", "")
    played = {}
    # The log must be this experiment's, or two experiments would mix into one number.
    spec = json.dumps({"base": base_name, "a": a_opts, "b": opts, "visits": VISITS}, sort_keys=True)
    if glog and os.path.exists(glog):
        for line in open(glog, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue                    # blank, or the torn last line of a killed run
            if r.get("spec", spec) != spec:
                raise SystemExit(f"{glog} was written by a different experiment. It has "
                                 f"{r.get('spec')}, this is {spec}. Move it aside, or point "
                                 f"GAMES_LOG somewhere else.")
            played[int(r["k"])] = r
        if played:
            print(f"  resuming: {len(played)} games already in {glog}", flush=True)
    k = 0
    for i, fen in enumerate(book):
        for b_white in (True, False):
            k += 1
            if k in played:
                r = played[k]
                sc, plies, how = float(r["score"]), int(r["plies"]), r["how"]
                nodes = {True: int(r["evals_b"]), False: int(r["evals_a"])}
            else:
                sc, plies, how, nodes = play(fen, b_white)
                if glog:
                    with open(glog, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"k": k, "spec": spec, "b_white": b_white,
                                             "score": sc, "plies": plies, "how": how,
                                             "evals_b": nodes[True],
                                             "evals_a": nodes[False]}) + chr(10))
            b_score += sc
            w += sc == 1.0
            d += sc == 0.5
            l += sc == 0.0
            ev_b += nodes[True]
            ev_a += nodes[False]
            print(f"  {k:>4} {'White' if b_white else 'Black':>6} "
                  f"{('win' if sc == 1 else 'draw' if sc == 0.5 else 'loss'):>7} {plies:>6} "
                  f"{how:>12} {b_score:>8.1f}", flush=True)
    eng.quit()

    score = b_score / k
    # Wilson, not Wald. At a shutout Wald's error is zero and the clamps put the low end above the high.
    z2 = 1.96 ** 2
    mid = (score + z2 / (2 * k)) / (1 + z2 / k)
    half = 1.96 * math.sqrt(score * (1 - score) / k + z2 / (4 * k * k)) / (1 + z2 / k)
    lo, hi = max(0.001, mid - half), min(0.999, mid + half)
    print("")
    print(f"  {k} games in {(time.time()-t0)/60:.1f} minutes")
    print(f"  B ({opts}) scored {b_score:.1f} of {k}: {w} wins, {d} draws, {l} losses")
    print(f"  score {100*score:.1f}%  ->  B is {elo(score):+.0f} Elo against the baseline")
    print(f"  95% interval {elo(lo):+.0f} to {elo(hi):+.0f} Elo")
    print(f"  network evaluations: baseline {ev_a:,}, B {ev_b:,} "
          f"({100*(ev_b-ev_a)/max(ev_a,1):+.1f}%)")
    if lo < 0.5 < hi:
        print(f"  THE INTERVAL CONTAINS 50%: {k} games did not separate these two searches.")
    else:
        print(f"  The interval excludes 50%, so the difference is real at this sample size.")
    # one row per run, so results are recorded rather than copied from the console
    with open(E("RESULTS", os.path.join(HERE, "search_match.jsonl")), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M"), "network": label,
                             "base": base_name or "defaults", "a": a_opts, "b": opts,
                             "games": k, "visits": VISITS, "w": w, "d": d, "l": l,
                             "score": round(score, 4), "elo": round(elo(score)),
                             "lo": round(elo(lo)), "hi": round(elo(hi)),
                             "evals_a": ev_a, "evals_b": ev_b,
                             "minutes": round((time.time() - t0) / 60, 1)}) + chr(10))
