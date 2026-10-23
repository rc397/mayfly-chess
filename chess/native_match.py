r"""A version against itself at a fixed time a move, the numpy pick against the compiled pick.

The compiled pick (native/pick.c) gives the numpy pick's answer bit for bit (harness.py checks it), so
at a given visit count it cannot change a move. It can change how many visits fit in the time, and
this measures whether that is worth games. Both sides use the same network and sealed search in one
process. A is the version as sealed (native_pick off for v4), B is the same search with native_pick
on, so B's score is the value of the speed alone.

Each move is the search given an allocation of tc seconds under the version's own sealed time rules
(tm_extend and tm_stop), as the bot plays, without pondering. Each side keeps its tree between moves.
Visits, evaluations and seconds per searched move are recorded for both sides.

The same 50 openings as search_match (seed 23), each from both sides, give 100 games a time control.
A game ends by the rules (threefold or the fifty-move rule, as in search_match), or
  * decided, when both engines agree over their last 4 moves (one side's q above 0.95, the other's
    below -0.95) and Stockfish at SF_MS agrees by 500 centipawns,
  * dead drawn, after ply 120 with both sides' last 8 |q| under 0.05 and Stockfish within 50,
  * at MAX_PLIES, where Stockfish decides at +-300 centipawns, as in search_match.
Stockfish is asked only when one of the first two fires, and at most once every 8 plies.

  python chess/native_match.py
  env: TCS=0.1,0.2,0.5,1,3  GAMES=100  OUT=<data>/native_match  VERSION=mayfly_v4.pt
       SF_MS=200  MAX_PLIES=200  OPENING_SEED=23  SF_PATH=<stockfish>
Resumable. Every finished game goes to OUT/games_<tc>.jsonl and a rerun replays it. One summary row
per finished control goes to OUT/native_match.jsonl, and OUT/progress.txt gets a line per game.
"""
import gc
import json
import math
import os
import sys
import threading
import time

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from search_match import elo, openings  # noqa: E402
import paths  # noqa: E402

E = os.environ.get
TCS = [float(x) for x in E("TCS", "0.1,0.2,0.5,1,3").split(",") if x.strip()]
GAMES = int(E("GAMES", 100))
OUT = E("OUT", paths.under("native_match"))
VERSION = E("VERSION", "mayfly_v4.pt")
SF_MS = int(E("SF_MS", 200))
MAX_PLIES = int(E("MAX_PLIES", 200))
SEED = int(E("OPENING_SEED", 23))
FIELDS = ("moves", "searched", "visits", "evals", "spent")


def wilson(score, k):
    """The 95% interval on a score, Wilson as in search_match (Wald collapses at a shutout)."""
    z2 = 1.96 ** 2
    mid = (score + z2 / (2 * k)) / (1 + z2 / k)
    half = 1.96 * math.sqrt(score * (1 - score) / k + z2 / (4 * k * k)) / (1 + z2 / k)
    return max(0.001, mid - half), min(0.999, mid + half)


def per_move(t):
    n = max(t["searched"], 1)
    return {"visits_per_move": round(t["visits"] / n, 1), "evals_per_move": round(t["evals"] / n, 1),
            "secs_per_move": round(t["spent"] / n, 3),
            "visits_per_sec": round(t["visits"] / t["spent"], 1) if t["spent"] > 0 else None}


def main():
    import mcts as M
    from mcts import MCTS, pinned_search
    from play_web import load_version

    native = M.NATIVE_PICK
    if native is None:
        raise SystemExit("the compiled pick did not load here, so there is nothing to compare")
    seal = pinned_search(VERSION)
    if not seal:
        raise SystemExit(f"{VERSION} has no sealed search")
    os.makedirs(OUT, exist_ok=True)
    book = openings(GAMES // 2, seed=SEED)
    if len(book) < GAMES // 2:
        raise SystemExit(f"only {len(book)} openings found, {GAMES // 2} needed")
    model, label, _ = load_version(VERSION)
    dev = str(next(model.parameters()).device)

    sf_path = E("SF_PATH") or open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)

    # As in search_match. python-chess keeps Stockfish on a non-daemon thread, so an uncaught error
    # would leave the interpreter waiting on it forever, and it is killed as shutdown begins.
    def _kill_engine(e=eng):
        try:
            e.transport.kill()
        except Exception:
            pass
    threading._register_atexit(_kill_engine)
    eng.configure({"Threads": 2, "Hash": 256})

    def sf_cp(board):
        """Stockfish's score for White in centipawns, a mate as +-10000."""
        return eng.analyse(board, chess.engine.Limit(time=SF_MS / 1000.0))["score"].white().score(
            mate_score=10000)

    def build(is_b):
        t = MCTS(model, dev, batch=48, cache_size=200000).apply(seal)
        # The only difference between the sides. v4's seal pins native_pick off, so A is v4 as sealed
        # and B is v4 with the compiled pick switched on.
        t.native_pick = bool(is_b)
        return t

    def search(t, board, tc):
        return t.search(board, visits=10_000_000, seconds=tc, reuse=True)

    over = lambda b: b.is_game_over() or b.is_repetition(3) or b.is_fifty_moves()  # noqa: E731

    def play(fen, b_is_white, tc):
        board = chess.Board(fen)
        trees = {chess.WHITE: build(b_is_white), chess.BLACK: build(not b_is_white)}
        is_b = {chess.WHITE: b_is_white, chess.BLACK: not b_is_white}
        rec = {s: dict.fromkeys(FIELDS, 0) for s in ("a", "b")}
        qs = {chess.WHITE: [], chess.BLACK: []}
        plies, asked_at, how, ws = 0, -99, None, None
        while not over(board) and plies < MAX_PLIES:
            side = board.turn
            t = trees[side]
            t0 = time.perf_counter()
            mv, info = search(t, board, tc)
            el = time.perf_counter() - t0
            if mv is None or mv not in board.legal_moves:
                break
            r = rec["b" if is_b[side] else "a"]
            r["moves"] += 1
            if "proof" not in info:                # a search, not a proven move (mate in one, tablebase)
                carried = int(info.get("carried", 0))
                # v4 discounts a carried root to ponder_discount of its visits before searching on
                kept = int(round(carried * t.ponder_discount)) if carried else 0
                r["searched"] += 1
                r["visits"] += int(info["visits"]) - kept
                r["evals"] += int(info.get("evals", 0))
                r["spent"] += el
            qs[side].append(float(info.get("q", 0.0)))
            board.push(mv)
            plies += 1
            if plies - asked_at < 8:
                continue
            w4, b4 = qs[chess.WHITE][-4:], qs[chess.BLACK][-4:]
            dec = None
            if len(w4) == 4 and len(b4) == 4:
                if min(w4) > 0.95 and max(b4) < -0.95:
                    dec = 1.0
                elif max(w4) < -0.95 and min(b4) > 0.95:
                    dec = 0.0
            if dec is not None:
                asked_at = plies
                cp = sf_cp(board)
                if cp is not None and ((dec == 1.0 and cp > 500) or (dec == 0.0 and cp < -500)):
                    ws, how = dec, "decided"
                    break
            elif plies > 120:
                w8, b8 = qs[chess.WHITE][-8:], qs[chess.BLACK][-8:]
                if len(w8) == 8 and len(b8) == 8 and max(abs(x) for x in w8 + b8) < 0.05:
                    asked_at = plies
                    cp = sf_cp(board)
                    if cp is not None and abs(cp) < 50:
                        ws, how = 0.5, "dead drawn"
                        break
        if how is None:
            if over(board):
                res = board.result() if board.is_game_over() else "1/2-1/2"
                ws = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
                how = "played out"
            else:
                cp = sf_cp(board)
                ws = 0.5 if cp is None else (1.0 if cp > 300 else (0.0 if cp < -300 else 0.5))
                how = "adjudicated"
        trees.clear()
        gc.collect()
        return (ws if b_is_white else 1.0 - ws), plies, how, rec, board

    def prog(line):
        with open(os.path.join(OUT, "progress.txt"), "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%m-%d %H:%M:%S ") + line + "\n")

    prog(f"{label} against itself: A numpy pick, B compiled pick; {len(book)} openings x 2 colours; "
         f"controls {TCS}; on {dev}")
    # warm the GPU and both picks before anything is timed
    for b_side in (False, True):
        search(build(b_side), chess.Board(), 0.2)

    for tc in TCS:
        tag = ("%g" % tc).replace(".", "p")
        glog = os.path.join(OUT, f"games_{tag}.jsonl")
        spec = json.dumps({"version": VERSION, "tc": tc, "games": GAMES, "seed": SEED,
                           "max_plies": MAX_PLIES, "a": "numpy pick", "b": "compiled pick"}, sort_keys=True)
        played = {}
        if os.path.exists(glog):
            for line in open(glog, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                    # blank, or the torn last line of a killed run
                if r.get("spec") != spec:
                    raise SystemExit(f"{glog} was written by a different experiment, move it aside")
                played[int(r["k"])] = r
        tot = {s: dict.fromkeys(FIELDS, 0) for s in ("a", "b")}
        score, w, d, l, k, t0 = 0.0, 0, 0, 0, 0, time.time()
        for fen in book:
            for b_white in (True, False):
                k += 1
                if k in played:
                    r = played[k]
                else:
                    sc, plies, how, rec, board = play(fen, b_white, tc)
                    r = {"k": k, "spec": spec, "fen": fen, "b_white": b_white, "score": sc,
                         "plies": plies, "how": how, "a": rec["a"], "b": rec["b"],
                         "moves": " ".join(m.uci() for m in board.move_stack)}
                    with open(glog, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(r) + "\n")
                sc = float(r["score"])
                score += sc
                w += sc == 1.0
                d += sc == 0.5
                l += sc == 0.0
                for s in ("a", "b"):
                    for f in FIELDS:
                        tot[s][f] += r[s][f]
                pa, pb = per_move(tot["a"]), per_move(tot["b"])
                prog(f"{tc:g}s game {k:>3}/{GAMES}: B {'White' if r['b_white'] else 'Black'} "
                     f"{('win' if sc == 1 else 'draw' if sc == 0.5 else 'loss'):>4} in {r['plies']:>3} "
                     f"plies ({r['how']}); B {score:.1f}/{k} (+{w} ={d} -{l}); visits a move "
                     f"A {pa['visits_per_move']} B {pb['visits_per_move']}")
        pct = score / k
        lo, hi = wilson(pct, k)
        row = {"when": time.strftime("%Y-%m-%d %H:%M"), "version": VERSION, "tc": tc, "games": k,
               "b_score": score, "wins": w, "draws": d, "losses": l, "pct": round(100 * pct, 1),
               "elo": round(elo(pct)), "elo_lo": round(elo(lo)), "elo_hi": round(elo(hi)),
               "a": per_move(tot["a"]), "b": per_move(tot["b"]),
               "minutes": round((time.time() - t0) / 60, 1)}
        with open(os.path.join(OUT, "native_match.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        prog(f"DONE {tc:g}s: B (compiled) {score:.1f}/{k} = {100 * pct:.1f}%, {elo(pct):+.0f} Elo "
             f"[{elo(lo):+.0f}, {elo(hi):+.0f}]; visits a move A {row['a']['visits_per_move']} "
             f"B {row['b']['visits_per_move']}")
    eng.quit()
    prog("ALL DONE")


if __name__ == "__main__":
    main()
