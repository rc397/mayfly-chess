"""Measure MayFly's game accuracy at one search budget against a fixed opponent.

Every version and budget plays the same opponent, full-strength Stockfish (two threads, 256 MB hash, SF_SECONDS
a move), from the ladder's twelve openings with colours alternating. Stockfish's own search scores each position
MayFly's move made, so every MayFly move is judged at Stockfish's full depth for free. A game's accuracy uses the
Lichess formula (game_accuracy.py), and the result is the mean over games.

    python chess/accuracy_match.py
    env VERSION (mayfly_v5.pt), VISITS (400, 0 is the bare network), GAMES (12), SF_SECONDS (1.0),
        MAX_PLIES (200), OUT (chess/accuracy_results.jsonl), LOGDIR

Each finished game is a line in LOGDIR/<version>_<visits>.jsonl with its PGN beside it, so a re-run resumes.
The summary row goes to OUT once every game is in, tagged with PROTOCOL.
"""
import json
import math
import os
import threading
import time

import chess
import chess.engine
import chess.pgn

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
PROTOCOL = "fixed-sf-full-1s"          # the cards read only rows measured this way
VERSION = E("VERSION", "mayfly_v5.pt")
VISITS = int(E("VISITS", 400))
GAMES = int(E("GAMES", 12))
SF_SECONDS = float(E("SF_SECONDS", 1.0))
MAX_PLIES = int(E("MAX_PLIES", 200))
OUT = E("OUT", os.path.join(HERE, "accuracy_results.jsonl"))
LOGDIR = E("LOGDIR", paths.under("abq", "accuracy"))


def white_cp(score):
    """A python-chess PovScore as White's centipawns, clamped to +-1,000 as Lichess does. A mate is +-1,000."""
    return max(-1000, min(1000, score.white().score(mate_score=1000)))


def terminal_cp(board):
    if board.is_checkmate():
        return -1000 if board.turn == chess.WHITE else 1000
    return 0


def main():
    from game_accuracy import game_accuracy, win_percent
    from elo_vs_sf import BOOK, game_over, game_result, _kill
    from mcts import MCTS, pinned_search, policy_move
    from play_web import load_version

    model, label, _ = load_version(VERSION)
    dev = str(next(model.parameters()).device)
    sealed = pinned_search(VERSION)
    if VISITS > 0 and not sealed:
        raise SystemExit(f"{VERSION} has no sealed search; refusing to measure it on today's")
    eng = chess.engine.SimpleEngine.popen_uci(open(os.path.join(HERE, "stockfish_path.txt")).read().strip())
    threading._register_atexit(lambda: _kill(eng))
    eng.configure({"Threads": 2, "Hash": 256})
    sf_name = eng.id.get("name", "Stockfish")
    os.makedirs(LOGDIR, exist_ok=True)
    stem = f"{os.path.splitext(VERSION)[0]}_{VISITS}"
    log_path, pgn_path = os.path.join(LOGDIR, stem + ".jsonl"), os.path.join(LOGDIR, stem + ".pgn")
    key = f"{VERSION}|{VISITS}|{SF_SECONDS:g}|{sf_name}|{MAX_PLIES}|{PROTOCOL}"
    done = {}
    if os.path.exists(log_path):
        for line in open(log_path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("key") == key:
                done[int(r["g"])] = r
    print(f"{label} at {VISITS} visits against {sf_name} full strength, {SF_SECONDS:g} s a move, {GAMES} games"
          + (f" ({len(done)} already played)" if done else ""), flush=True)

    for g in range(GAMES):
        if g in done:
            continue
        name, fen = BOOK[g % len(BOOK)]
        mf_white = (g // len(BOOK) + g) % 2 == 0          # the colours alternate, and swap on a second pass
        board = chess.Board(fen)
        start_white = board.turn == chess.WHITE
        tree = MCTS(model, dev, batch=48, cache_size=20000).apply(sealed) if VISITS > 0 else None
        cps = []                                            # White's centipawns at the start and after every ply
        if (board.turn == chess.WHITE) == mf_white:
            cps.append(white_cp(eng.analyse(board, chess.engine.Limit(time=SF_SECONDS), game=g)["score"]))
        plies = 0
        while not game_over(board) and plies < MAX_PLIES:
            if (board.turn == chess.WHITE) == mf_white:
                if tree is not None:
                    mv, _ = tree.search(board, visits=VISITS, reuse=True)
                else:
                    mv = policy_move(model, board, dev, sealed.get("policy_head", "cls"),
                                     castle=sealed.get("castle_prior", "king"))
                if mv is None or mv not in board.legal_moves:
                    break
                board.push(mv)
            else:
                res = eng.play(board, chess.engine.Limit(time=SF_SECONDS), info=chess.engine.INFO_SCORE, game=g)
                cp = white_cp(res.info["score"]) if "score" in res.info else (cps[-1] if cps else 0)
                if len(cps) == plies:
                    cps.append(cp)                          # the position MayFly's move made, at full depth
                board.push(res.move)
                cps.append(cp)                              # Stockfish's best move keeps that value
            plies += 1
        # score the last position when the game ends on MayFly's move or at the ply cap
        if len(cps) < plies + 1:
            cps.append(terminal_cp(board) if game_over(board) else
                       white_cp(eng.analyse(board, chess.engine.Limit(time=SF_SECONDS), game=g)["score"]))
        wins = [win_percent(c) for c in cps[:plies + 1]]
        acc = game_accuracy(start_white, wins, mf_white)
        result = game_result(board) if game_over(board) else "1/2-1/2"
        ws = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
        s = ws if mf_white else 1.0 - ws
        row = {"key": key, "g": g, "opening": name, "white": mf_white, "s": s, "plies": plies,
               "accuracy": None if acc is None else round(acc[0], 3), "moves": None if acc is None else acc[1]}
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        game = chess.pgn.Game.from_board(board)
        game.headers.update({"Event": f"MayFly accuracy: {sf_name} full strength, {SF_SECONDS:g} s a move",
                             "White": f"MayFly {os.path.splitext(VERSION)[0]} ({VISITS} visits)" if mf_white else sf_name,
                             "Black": sf_name if mf_white else f"MayFly {os.path.splitext(VERSION)[0]} ({VISITS} visits)",
                             "Result": result, "Opening": name, "Date": time.strftime("%Y.%m.%d")})
        with open(pgn_path, "a", encoding="utf-8") as fh:
            fh.write(str(game) + "\n\n")
        done[g] = row
        print(f"  {g + 1:>2} {name:<18} as {'White' if mf_white else 'Black':<5} "
              f"{('win' if s == 1 else 'draw' if s == 0.5 else 'loss'):>4} {plies:>3} plies, "
              f"accuracy {row['accuracy']}", flush=True)
    eng.quit()

    accs = [r["accuracy"] for r in done.values() if r.get("accuracy") is not None]
    if len(done) < GAMES or not accs:
        return
    mean = sum(accs) / len(accs)
    sd = math.sqrt(sum((a - mean) ** 2 for a in accs) / max(1, len(accs) - 1))
    w = sum(r["s"] == 1 for r in done.values())
    d = sum(r["s"] == 0.5 for r in done.values())
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "base": VERSION, "network": label, "visits": VISITS,
           "games": len(accs), "accuracy": round(mean, 2), "se": round(sd / math.sqrt(len(accs)), 2),
           "protocol": PROTOCOL, "opponent": f"{sf_name} full strength, {SF_SECONDS:g} s a move",
           "w": w, "d": d, "l": len(done) - w - d}
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")
    print(f"{label} at {VISITS} visits: accuracy {mean:.1f}% (+-{1.96 * out['se']:.1f}) over {len(accs)} games, "
          f"{w}-{d}-{out['l']} against {out['opponent']}", flush=True)


if __name__ == "__main__":
    main()
