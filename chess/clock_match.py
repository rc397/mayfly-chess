"""Match two search settings, or two versions, with a game clock instead of fixed visits.

Under fixed visits, time-saving settings such as early_stop cannot help, since the saved time is
thrown away. Here each side gets a total time for the game. The base allocation is the time left over
an estimate of the moves left, a side may stretch or shrink that, and unused time stays in its bank.
A side that runs out plays on at a floor rather than losing on time. Visits per move are reported
beside the score, since winning by finding more visits in the same seconds is a different claim from
winning per visit.

    python chess/clock_match.py "early_stop=1" [GAMES] [GAME_SECONDS]
    env VERSION=mayfly_v2.pt, SF_MS=200, MAX_PLIES=200, FLOOR_MS=40
        A_OPTS     settings for the baseline side, so two non-default searches can be compared
        BASE       both sides start from that version's sealed search, as in search_match
        OPTS, GAMES, GAME_SECONDS   read from the environment when no arguments are given, as the
                   A/B queue runs it
        RESULTS=clock_match.jsonl   one row per finished run
        B_VERSION  B plays that network with that version's sealed search (OPTS on top), while A
                   keeps VERSION and BASE
        GAMES_LOG  per-game log, so a crashed run resumes
"""
import os, sys, glob, random, math, time
import chess, chess.pgn, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from search_match import parse, openings, elo

E = os.environ.get
OPTS = sys.argv[1] if len(sys.argv) > 1 else E("OPTS", "early_stop=1")
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else int(E("GAMES", 40))
GAME_SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else float(E("GAME_SECONDS", 30.0))
SF_MS = int(E("SF_MS", 200))
MAX_PLIES = int(E("MAX_PLIES", 200))
FLOOR = float(E("FLOOR_MS", 40)) / 1000.0


def allocate(remaining, ply):
    """The classic allocation, what is left over an estimate of how many moves are still to come."""
    left = max(12.0, 50.0 - ply / 2.0)
    return max(FLOOR, remaining / left)


if __name__ == "__main__":
    import json
    import torch
    from mcts import MCTS, pinned_search
    from play_web import load_version

    opts = parse(OPTS)
    a_opts = parse(E("A_OPTS", ""))
    # with BASE set both sides start from that version's sealed search, otherwise from the defaults
    base_name = E("BASE", "")
    base = pinned_search(base_name) if base_name else {}
    if base_name and not base:
        raise SystemExit(f"BASE={base_name} has no sealed search")
    # Another network for B, each side with its own weights and sealed search. version_match.py
    # asks this at fixed visits, which undersells a search that wins per second.
    b_name = E("B_VERSION", "")
    b_base = pinned_search(b_name) if b_name else base
    if b_name and not b_base:
        raise SystemExit(f"B_VERSION={b_name} has no sealed search")
    book = openings(GAMES // 2, seed=53)
    model, label, _ = load_version(E("VERSION", "mayfly_v2.pt"))
    model_b, label_b = (load_version(b_name)[:2] if b_name else (model, label))
    dev = str(next(model.parameters()).device)
    print(f"{label} against {label_b if b_name else 'itself'} on a clock, "
          f"{GAME_SECONDS:.0f} s a side a game", flush=True)
    print(f"  A = baseline" + (f" + {a_opts}" if a_opts else ""), flush=True)
    print(f"  B = " + (f"{label_b}, its own sealed search" if b_name else "baseline")
          + (f" + {opts}" if opts else ""), flush=True)
    print(f"  {len(book)} openings x 2 colours = {2*len(book)} games on {dev}", flush=True)

    sf_path = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    # python-chess keeps Stockfish on a non-daemon thread, so an uncaught exception would leave the
    # interpreter waiting forever. The engine is killed as shutdown begins.
    import threading
    def _kill_engine(e=eng):
        try:
            e.transport.kill()
        except Exception:
            pass
    threading._register_atexit(_kill_engine)
    eng.configure({"Threads": 4, "Hash": 512})

    def build(is_b):
        t = MCTS(model_b if is_b else model, dev, batch=48, cache_size=20000).apply(
            b_base if is_b else base)
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
        return 0.5 if cp is None else (1.0 if cp > 300 else (0.0 if cp < -300 else 0.5))

    def play(fen, b_is_white):
        board = chess.Board(fen)
        trees = {chess.WHITE: build(b_is_white), chess.BLACK: build(not b_is_white)}
        bank = {chess.WHITE: GAME_SECONDS, chess.BLACK: GAME_SECONDS}
        vis = {True: 0, False: 0}
        moves = {True: 0, False: 0}
        plies = 0
        # only a draw that has arisen ends a game, since claim_draw would also end it when the side
        # to move merely could repeat
        over = lambda b: b.is_game_over() or b.is_repetition(3) or b.is_fifty_moves()
        while not over(board) and plies < MAX_PLIES:
            side = board.turn
            is_b = (side == chess.WHITE) == b_is_white
            budget = allocate(bank[side], plies)
            t0 = time.time()
            mv, info = trees[side].search(board, visits=10_000_000,
                                          seconds=budget, reuse=True)
            spent = time.time() - t0
            bank[side] = max(0.0, bank[side] - spent)
            # the search's visit count holds the carried subtree after the ponder discount, so the
            # discounted carry is what comes off
            carry = int(round(int(info.get("carried", 0)) * float(getattr(trees[side], "ponder_discount", 1.0))))
            vis[is_b] += int(info.get("visits", 0)) - carry
            moves[is_b] += 1
            if mv is None or mv not in board.legal_moves:
                break
            board.push(mv)
            plies += 1
        if over(board):
            r = board.result() if board.is_game_over() else "1/2-1/2"
            ws, how = {"1-0": 1.0, "0-1": 0.0}.get(r, 0.5), "played out"
        else:
            ws, how = adjudicate(board), "adjudicated"
        return (ws if b_is_white else 1.0 - ws), plies, how, vis, moves

    b_score = 0.0
    w = d = l = 0
    vis_a = vis_b = mv_a = mv_b = 0
    t0 = time.time()
    print("")
    print(f"  {'#':>4} {'B as':>6} {'result':>7} {'plies':>6} {'how':>12} "
          f"{'A v/mv':>8} {'B v/mv':>8} {'B score':>8}")
    # Resumable through GAMES_LOG, as search_match is. The openings are seeded, so game k is the
    # same opening and colour on every run.
    glog = E("GAMES_LOG", "")
    spec = json.dumps({"version": E("VERSION", "mayfly_v2.pt"), "base": base_name, "a": a_opts,
                       "b": opts, "b_version": b_name, "seconds": GAME_SECONDS, "floor": FLOOR,
                       "sf_ms": SF_MS, "max_plies": MAX_PLIES}, sort_keys=True)
    played = {}
    if glog and os.path.exists(glog):
        for line in open(glog, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue                    # blank, or the torn last line of a killed run
            if r.get("spec") != spec:
                raise SystemExit(f"{glog} was written by a different experiment. It has "
                                 f"{r.get('spec')}, this is {spec}. Move it aside, or point "
                                 f"GAMES_LOG somewhere else.")
            played[int(r["k"])] = r
        if played:
            print(f"  resuming: {len(played)} games already in {glog}", flush=True)
    k = 0
    for fen in book:
        for b_white in (True, False):
            k += 1
            if k in played:
                r = played[k]
                sc, plies, how = float(r["score"]), int(r["plies"]), r["how"]
                vis = {True: int(r["vis_b"]), False: int(r["vis_a"])}
                moves = {True: int(r["mv_b"]), False: int(r["mv_a"])}
            else:
                sc, plies, how, vis, moves = play(fen, b_white)
                if glog:
                    with open(glog, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"k": k, "spec": spec, "b_white": b_white,
                                             "score": sc, "plies": plies, "how": how,
                                             "vis_b": vis[True], "vis_a": vis[False],
                                             "mv_b": moves[True], "mv_a": moves[False]})
                                 + chr(10))
            b_score += sc
            w += sc == 1.0
            d += sc == 0.5
            l += sc == 0.0
            vis_b += vis[True]
            vis_a += vis[False]
            mv_b += moves[True]
            mv_a += moves[False]
            print(f"  {k:>4} {'White' if b_white else 'Black':>6} "
                  f"{('win' if sc == 1 else 'draw' if sc == 0.5 else 'loss'):>7} {plies:>6} "
                  f"{how:>12} {vis[False]//max(moves[False],1):>8,} "
                  f"{vis[True]//max(moves[True],1):>8,} {b_score:>8.1f}", flush=True)
    eng.quit()

    score = b_score / k
    # Wilson interval, since Wald's error is zero at a shutout

    z2 = 1.96 ** 2
    mid = (score + z2 / (2 * k)) / (1 + z2 / k)
    half = 1.96 * math.sqrt(score * (1 - score) / k + z2 / (4 * k * k)) / (1 + z2 / k)
    lo, hi = max(0.001, mid - half), min(0.999, mid + half)
    a_vpm = vis_a / max(mv_a, 1)
    b_vpm = vis_b / max(mv_b, 1)
    print("")
    print(f"  {k} games in {(time.time()-t0)/60:.1f} minutes, {GAME_SECONDS:.0f} s a side")
    print(f"  B ({label_b if b_name else opts}) scored {b_score:.1f} of {k}: {w} wins, {d} draws, "
          f"{l} losses")
    print(f"  score {100*score:.1f}%  ->  B is {elo(score):+.0f} Elo, "
          f"95% interval {elo(lo):+.0f} to {elo(hi):+.0f}")
    print("")
    print(f"  visits a move, which is the mechanism rather than the outcome:")
    print(f"    baseline {a_vpm:>8,.0f}")
    print(f"    B        {b_vpm:>8,.0f}   ({100*(b_vpm-a_vpm)/max(a_vpm,1):+.1f}%)")
    if lo < 0.5 < hi:
        print(f"  THE INTERVAL CONTAINS 50%: {k} games did not separate them on score.")
        if b_vpm > a_vpm * 1.05:
            print(f"  But it DID buy {100*(b_vpm-a_vpm)/max(a_vpm,1):.0f}% more visits in the same "
                  f"seconds, and visits is the lever that works.")
    row = {"when": time.strftime("%Y-%m-%d %H:%M"), "harness": "clock_match", "network": label,
           "base": base_name or "defaults", "a": a_opts, "b": opts, "games": k,
           **({"network_b": label_b, "b_version": b_name} if b_name else {}),
           "seconds": GAME_SECONDS, "w": w, "d": d, "l": l, "score": round(score, 4),
           "elo": round(elo(score)), "lo": round(elo(lo)), "hi": round(elo(hi)),
           "visits_a": round(a_vpm), "visits_b": round(b_vpm),
           "minutes": round((time.time() - t0) / 60, 1)}
    with open(E("RESULTS", os.path.join(HERE, "clock_match.jsonl")), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + chr(10))
