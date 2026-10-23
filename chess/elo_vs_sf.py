"""Measure MayFly's strength against Stockfish at fixed levels, at one search budget.

The result depends on the search budget, so it is always reported with the visits per move (the board's
default of 400 by default). Several levels are played, since a shutout only bounds the gap. Levels of
1320 and up use UCI_LimitStrength with that UCI_Elo, and an estimate per level is
SF_Elo + 400*log10(S/(1-S)), combined with weights by how far from a shutout each was. Levels up to 20
are Skill Levels, which are not an Elo scale, so those rows report only a score. Levels 21 to 99 are
Skill 20 with more time per move (RUNG_RATIO per rung). Twelve standard openings are built in, each
played from both sides, since the search picks moves by argmax.

    python chess/elo_vs_sf.py [GAMES_PER_LEVEL] [VISITS]
    env VERSION=mayfly_v2.pt, LEVELS=0,3,1320, SF_MS=100, MAX_PLIES=200, RESULTS (jsonl path),
        OPTS (search settings "k=v,k=v" on top of the seal), PONDER=0, BOOK (b for the second book)
    VISITS 0 plays the bare network, above 0 the version's sealed search.
"""
import os, sys, threading, math, time
import chess, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

E = os.environ.get
# the queue runner passes no arguments to a script it does not know, so GAMES and VISITS can be set instead
PER_LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else int(E("GAMES", 20))
VISITS = int(sys.argv[2]) if len(sys.argv) > 2 else int(E("VISITS", 400))
# Search settings on top of the seal, "k=v,k=v", for testing changes against an opponent that punishes
# blunders. Part of the cell key, so a cell with settings never resumes into one without.
OPTS = E("OPTS", "")
LEVELS = [int(x) for x in E("LEVELS", "0,3,1320").split(",")]
SF_MS = int(E("SF_MS", 100))
MAX_PLIES = int(E("MAX_PLIES", 200))

# Twelve standard openings, eight to ten plies in. Named so a result can be read by eye.
BOOK = [
    ("Ruy Lopez",        "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4"),
    ("Italian",          "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"),
    ("Sicilian Najdorf", "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("French",           "rnbqkbnr/pp3ppp/4p3/2pp4/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4"),
    ("Caro-Kann",        "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 1 3"),
    ("Queen's Gambit",   "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
    ("Slav",             "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"),
    ("King's Indian",    "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4"),
    ("Nimzo-Indian",     "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 3 4"),
    ("English",          "rnbqkbnr/pppp1ppp/8/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR b KQkq - 1 2"),
    ("Scandinavian",     "rnb1kbnr/ppp1pppp/8/q7/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 4"),
    ("Pirc",             "rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 1 3"),
]
# Each opening is played twice in a row, once from each side. The protocol tag in cell_key keeps results
# from older book versions apart.
PROTOCOL = "book2-draw2"
# A second book of twelve other openings, for release_verdict.py's extension games when a candidate's
# score lands just under its bar. Replaying the first book would mostly repeat the same games. It runs
# under its own protocol tag (BOOK=b), and other readers of the ladder results leave those rows out.
BOOK_B = [
    ("Scotch",            "r1bqkbnr/pppp1ppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4"),
    ("Petroff",           "rnbqkb1r/ppp2ppp/3p4/8/4n3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 5"),
    ("Sicilian Dragon",   "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("Sicilian Taimanov", "r1bqkbnr/pp1p1ppp/2n1p3/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 1 5"),
    ("Alekhine",          "rnbqkb1r/ppp1pppp/3p4/3nP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 4"),
    ("Vienna",            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3"),
    ("London",            "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3"),
    ("Grunfeld",          "rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4"),
    ("Dutch",             "rnbqkb1r/ppppp2p/5np1/5p2/3P4/6P1/PPP1PPBP/RNBQK1NR w KQkq - 0 4"),
    ("Benoni",            "rnbqkb1r/pp3ppp/3p1n2/2pP4/8/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 6"),
    ("Catalan",           "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/6P1/PP2PPBP/RNBQK1NR b KQkq - 1 4"),
    ("Queen's Indian",    "rnbqkb1r/p1pp1ppp/1p2pn2/8/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4"),
]
PROTOCOL_B = "book2b-draw2"
if E("BOOK", "").lower() == "b":
    BOOK, PROTOCOL = BOOK_B, PROTOCOL_B
# Rungs past Skill 20. Skill 20 is Stockfish with no handicap, so a level from 21 to 99 is Skill 20 given
# RUNG_RATIO times the rung below's time a move (21 is SF_MS x 1.41, 22 x 2, 23 x 2.83, 24 x 4). Measured
# with sf_steps.py, a rung is about +31 Elo. Levels of 100 and up stay UCI_Elo.
RUNG_RATIO = float(E("RUNG_RATIO", math.sqrt(2)))


def move_ms(target):
    """Stockfish's time a move at this level, SF_MS, or for a rung past Skill 20 SF_MS times RUNG_RATIO a rung."""
    return SF_MS * RUNG_RATIO ** (target - 20) if 20 < target < 100 else SF_MS


def rung_name(target):
    if target >= 100:
        return f"UCI_Elo {target}"
    if target > 20:
        return f"rung {target} (Skill 20 at {move_ms(target):.0f} ms)"
    return f"Skill Level {target}"


def _kill(engine):
    try:
        engine.transport.kill()
    except Exception:
        pass


def game_over(board):
    """Over by the rules, or by a threefold or fifty-move state that has actually arisen. Not
    claim_draw, which would end a game when the side to move merely could claim."""
    return board.is_game_over() or board.is_repetition(3) or board.is_fifty_moves()


def game_result(board):
    if board.is_game_over():
        return board.result()
    return "1/2-1/2"


def elo_gap(score):
    """How much stronger the scorer is, in Elo, from a match score."""
    score = min(max(score, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(score / (1.0 - score))


if __name__ == "__main__":
    import json
    import torch
    from mcts import MCTS, pinned_search, policy_move
    from play_web import load_version

    VERSION = E("VERSION", "mayfly_v2.pt")
    model, label, _ = load_version(VERSION)
    dev = str(next(model.parameters()).device)
    # each version plays the search it was sealed with, not today's defaults
    SEALED = pinned_search(VERSION)
    if OPTS:
        for kv in filter(None, (x.strip() for x in OPTS.split(","))):
            k, _, v = kv.partition("=")
            SEALED[k.strip()] = {"true": True, "false": False}.get(v.strip().lower(), v.strip())
    # Pondering is off by default, since a pondering cell searches VISITS on top of what it carried
    # and would mix two budgets in one table. PONDER=1 is recorded in the row and the cell key.
    PONDER = E("PONDER", "0") == "1"
    if VISITS > 0 and not SEALED:
        raise SystemExit(f"{VERSION} has no sealed search; refusing to measure it on today's")
    RESULTS = E("RESULTS", "")
    sf_path = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()

    print(f"{label} at {VISITS} visits a move, on {dev}", flush=True)
    if VISITS > 0:
        print(f"  sealed search: tree_policy={SEALED.get('tree_policy')}  vl={SEALED.get('vl')}  "
              f"fpu_root={SEALED.get('fpu_root')}", flush=True)
    else:
        print(f"  no search: the bare network's top legal move", flush=True)
    print(f"  against Stockfish at {LEVELS} "
          f"(up to 20 = Skill Level, 21 to 99 = Skill 20 with more time, 1320+ = UCI_Elo)", flush=True)
    print(f"  {PER_LEVEL} games a level, {len(BOOK)} openings, colours alternated", flush=True)
    print("", flush=True)

    results = []
    for target in LEVELS:
        rated = target >= 100
        eng = chess.engine.SimpleEngine.popen_uci(sf_path)
        # python-chess keeps an engine on a non-daemon thread, so an uncaught exception would leave
        # the interpreter waiting forever. Every engine opened here is killed as shutdown begins.
        threading._register_atexit(lambda e=eng: _kill(e))
        try:
            if rated:
                eng.configure({"Threads": 2, "Hash": 256,
                               "UCI_LimitStrength": True, "UCI_Elo": target})
            else:
                eng.configure({"Threads": 2, "Hash": 256, "Skill Level": min(target, 20)})
        except Exception as e:
            print(f"  cannot set level {target}: {type(e).__name__}: {e}", flush=True)
            eng.quit()
            continue

        score = 0.0
        w = d = l = 0
        # Resumable per game. Each finished game is logged first, so a crash costs one game. The key
        # holds everything that defines the cell, including the Stockfish build and whether tablebases
        # were present, so a log from a different protocol is never replayed into this one.
        sf_name = eng.id.get("name", "?")
        import mcts as _mcts
        tb_on = bool(_mcts.tablebase() is not None and SEALED.get("use_tb", True))
        cell_key = (f"{VERSION}|{VISITS}|{target}|{PER_LEVEL}|{SF_MS}|{int(bool(PONDER))}|"
                    f"{PROTOCOL}|{sf_name}|tb{int(tb_on)}" + (f"|{OPTS}" if OPTS else "")
                    + (f"|r{RUNG_RATIO:.4f}" if 20 < target < 100 else ""))
        cell_log = (os.path.splitext(RESULTS)[0] + f".games_{VISITS}_{target}.jsonl") if RESULTS             else ""
        done_games = {}
        if cell_log and os.path.exists(cell_log):
            for line in open(cell_log, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                # blank, or the torn last line of a killed run
                if r.get("cell") == cell_key:
                    done_games[int(r["g"])] = r
            if done_games:
                print(f"  resuming this cell: {len(done_games)} games already played", flush=True)
        print(f"  --- Stockfish at {rung_name(target)} ---", flush=True)
        for g in range(PER_LEVEL):
            name, fen = BOOK[(g // 2) % len(BOOK)]
            if g in done_games:
                r = done_games[g]
                s_prev = float(r["s"])
                score += s_prev
                w += s_prev == 1.0
                d += s_prev == 0.5
                l += s_prev == 0.0
                secs_total = (secs_total + r["secs"]) if g else r["secs"]
                moves_total = (moves_total + r["moves"]) if g else r["moves"]
                carried_total = (carried_total + r.get("carried", 0)) if g else r.get("carried", 0)
                continue
            mf_white = (g % 2 == 0)
            board = chess.Board(fen)
            tree = MCTS(model, dev, batch=48, cache_size=20000).apply(SEALED)
            plies = 0
            mf_moves, mf_secs, mf_carried = 0, 0.0, 0
            ponder_thread = None

            def stop_ponder():
                if ponder_thread is not None:
                    tree.stop()
                    ponder_thread.join(timeout=10)
                    if ponder_thread.is_alive():
                        # still walking the tree and the next search clears the flag, so wait it
                        # out as uci.py and the board do
                        ponder_thread.join()
            while not game_over(board) and plies < MAX_PLIES:
                mf_turn = (board.turn == chess.WHITE) == mf_white
                if mf_turn:
                    stop_ponder()
                    ponder_thread = None
                    t0m = time.perf_counter()
                    if VISITS > 0:
                        mv, info = tree.search(board, visits=VISITS, reuse=True)
                        mf_carried += int(info.get("carried", 0))
                    else:
                        # the seal's castle_prior too, since v4 keeps O-O at e1h1 and read at e1g1
                        # the bare network never castles
                        mv = policy_move(model, board, dev, SEALED.get("policy_head", "cls"),
                                         castle=SEALED.get("castle_prior", "king"))
                    mf_secs += time.perf_counter() - t0m
                    mf_moves += 1
                else:
                    # a dead or stuck Stockfish ends the run rather than drawing the game, so the
                    # queue's retry replays the game against a fresh engine
                    mv = eng.play(board, chess.engine.Limit(time=move_ms(target) / 1000.0)).move
                if mv is None or mv not in board.legal_moves:
                    break
                board.push(mv)
                plies += 1
                # think on the opponent's time as the board does, the next search reuses the tree
                if PONDER and VISITS > 0 and mf_turn and not game_over(board):
                    tree.abort.clear()
                    ponder_thread = threading.Thread(
                        target=tree.ponder, args=(board.copy(), 1_000_000), daemon=True)
                    ponder_thread.start()
            stop_ponder()
            if game_over(board):
                r = game_result(board)
                ws = {"1-0": 1.0, "0-1": 0.0}.get(r, 0.5)
            else:
                ws = 0.5                       # hit the ply cap, call it drawn rather than guess
            s = ws if mf_white else 1.0 - ws
            score += s
            w += s == 1.0
            d += s == 0.5
            l += s == 0.0
            print(f"    {g+1:>3} {name:<18} MayFly as {'White' if mf_white else 'Black':<5} "
                  f"{('win' if s == 1 else 'draw' if s == 0.5 else 'loss'):>4}  {plies:>3} plies"
                  f"  {mf_secs/max(mf_moves,1):6.2f}s a move"
                  + (f", carried {mf_carried/max(mf_moves,1):5.0f}" if PONDER and VISITS else "")
                  + f"   running {score:>5.1f}/{g+1}", flush=True)
            secs_total = secs_total + mf_secs if g else mf_secs
            moves_total = moves_total + mf_moves if g else mf_moves
            carried_total = carried_total + mf_carried if g else mf_carried
            if cell_log:
                with open(cell_log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"cell": cell_key, "g": g, "s": s, "plies": plies,
                                         "secs": mf_secs, "moves": mf_moves,
                                         "carried": mf_carried}) + chr(10))
                # the game itself as PGN beside its result, for logging only
                try:
                    import chess.pgn
                    pg = chess.pgn.Game.from_board(board)
                    me = f"MayFly {os.path.splitext(os.path.basename(VERSION))[0]} ({VISITS} visits)"
                    opp = f"Stockfish {rung_name(target)}"
                    pg.headers["Event"] = f"MayFly ladder: {opp}, {move_ms(target):.0f} ms a move"
                    pg.headers["Round"] = str(g + 1)
                    pg.headers["Opening"] = name              # the book line the game started from
                    pg.headers["White"], pg.headers["Black"] = ((me, opp) if mf_white else (opp, me))
                    pg.headers["Result"] = {1.0: "1-0", 0.0: "0-1"}.get(ws, "1/2-1/2")
                    pg.headers["Date"] = time.strftime("%Y.%m.%d")
                    with open(os.path.splitext(cell_log)[0] + ".pgn", "a", encoding="utf-8") as fh:
                        fh.write(str(pg) + chr(10) + chr(10))
                except Exception as e:                       # a lost PGN must never cost a cell
                    print(f"    (pgn not written: {type(e).__name__}: {e})", flush=True)
        eng.quit()

        pct = score / PER_LEVEL
        se = math.sqrt(max(pct * (1 - pct), 1e-9) / PER_LEVEL)
        if rated and score in (0.0, float(PER_LEVEL)):
            # a shutout is a bound, not a rating, so report the exact one-sided 97.5% bound
            edge = 0.025 ** (1.0 / PER_LEVEL)
            est = None
            if score > 0:
                lo, hi = target + elo_gap(edge), None
                print(f"    scored {score:.1f}/{PER_LEVEL}: a shutout, MayFly above {lo:.0f} "
                      f"(97.5% one-sided)", flush=True)
            else:
                lo, hi = None, target + elo_gap(1.0 - edge)
                print(f"    scored 0/{PER_LEVEL}: shut out, MayFly below {hi:.0f} "
                      f"(97.5% one-sided)", flush=True)
        elif rated:
            est = target + elo_gap(pct)
            lo = target + elo_gap(max(pct - 1.96 * se, 1e-6))
            hi = target + elo_gap(min(pct + 1.96 * se, 1 - 1e-6))
            print(f"    scored {score:.1f}/{PER_LEVEL} = {100*pct:.1f}%  ->  "
                  f"MayFly about {est:.0f} ({lo:.0f} to {hi:.0f})", flush=True)
        else:
            est = lo = hi = None
            print(f"    scored {score:.1f}/{PER_LEVEL} = {100*pct:.1f}%  "
                  f"(Skill Level is not an Elo scale, so no rating from this row)", flush=True)
        results.append((target, rated, pct, est, lo, hi, w, d, l))
        if RESULTS:
            row = {"version": VERSION, "visits": VISITS, "level": target, "rated": rated,
                   "protocol": PROTOCOL, "stockfish": sf_name, "tablebases": tb_on,
                   "sf_ms": SF_MS, "sf_ms_move": round(move_ms(target), 1), "games": PER_LEVEL, "w": w, "d": d, "l": l, "score": pct,
                   "elo_gap": elo_gap(pct), "est": est, "lo": lo, "hi": hi,
                   "secs_per_move": secs_total / max(moves_total, 1),
                   # a pondering cell searched VISITS on top of this much carry, so it is flagged
                   "ponder": bool(PONDER and VISITS > 0),
                   "carried_per_move": round(carried_total / max(moves_total, 1), 1),
                   "search": SEALED if VISITS > 0 else "none", "opts": OPTS,
                   "at": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(RESULTS, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + chr(10))
        print("", flush=True)

    print("=" * 62)
    print(f"  {'opponent':>18} {'score':>7} {'W-D-L':>10} {'MayFly Elo':>12} "
          f"{'95% interval':>18}")
    for target, rated, pct, est, lo, hi, w, d, l in results:
        name = rung_name(target)
        cell = f"{est:.0f}" if est is not None else "n/a"
        iv = f"{lo:.0f} to {hi:.0f}" if est is not None else "not an Elo scale"
        print(f"  {name:>18} {100*pct:>6.1f}% {f'{w}-{d}-{l}':>10} {cell:>12} {iv:>18}")

    # weight each level by how informative it was, since a shutout bounds the answer but does not locate it
    useful = [(t, p, e) for t, r, p, e, lo, hi, w, d, l in results
              if r and e is not None and 0.05 < p < 0.95]
    print("")
    if useful:
        wts = [1.0 - abs(p - 0.5) * 2.0 + 1e-6 for _, p, _ in useful]
        est = sum(e * k for (_, _, e), k in zip(useful, wts)) / sum(wts)
        print(f"  {VERSION} at {VISITS} visits a move: about {est:.0f} Elo")
        print(f"  from {len(useful)} informative level(s); levels it shut out or was shut out by")
        print(f"  bound the answer but do not locate it, so they are excluded from the average.")
    else:
        lows = [t for t, r, p, _, _, _, _, _, _ in results if r and p >= 0.95]
        highs = [t for t, r, p, _, _, _, _, _, _ in results if r and p <= 0.05]
        if lows and not highs:
            print(f"  Beat every level tested. MayFly is ABOVE {max(lows)}; test higher.")
        elif highs and not lows:
            print(f"  Lost every level tested. MayFly is BELOW {min(highs)}; test lower.")
        else:
            # a Skill Level row still locates the engine even though it cannot rate it, so report
            # the gap and refuse only the Elo

            skill = [(t, p) for t, r, p, _, _, _, _, _, _ in results if not r and 0.02 < p < 0.98]
            if skill:
                for t, p in skill:
                    print(f"  Against Stockfish Skill Level {t}: scored {100*p:.1f}%, a gap of "
                          f"{elo_gap(p):+.0f} Elo.")
                print(f"  Skill Level is not a calibrated scale, so that gap cannot be turned")
                print(f"  into a rating here. Stockfish's rated floor is UCI_Elo 1320, which is")
                print(f"  STRONGER than Skill 0, so the calibrated ladder cannot reach this low.")
            else:
                print("  No level produced an informative score. Widen or narrow the ladder.")
    print("")
    print(f"  This number is for {VISITS} visits a move and nothing else. At 0 visits it is the")
    print(f"  bare policy head, and at 6,400 it is a different engine.")
