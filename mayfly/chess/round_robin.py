r"""MayFly against MayFly, one pairing of a round robin of every version at every budget.

The full round robin covers each version at no search, 400, 1,600 and 6,400 visits, 100 games a pairing. Games
run to MAX_PLIES, and at the cap Stockfish adjudicates, a draw within ADJ_CP of level and a win for the side ahead
beyond it.

No tablebases by default, so the result does not depend on which machine has Syzygy tables. No position is
scored by the tables and no player's search reads them (use_tb off on every tree, overriding the seal). TB=1
puts both back.

The round robin can be split between machines. With CLAIM_DIR set, a pairing is played only by the machine
whose claim is in that folder first, a file named for the pairing (either way round) holding the owner's name,
created exclusively. With another owner's claim this run plays nothing. To claim without playing, run
python chess/round_robin.py --claim, which prints CLAIM OK or CLAIM TAKEN.

  python chess/round_robin.py [--claim]
      env A, B (players as version:visits, e.g. mayfly_v5.pt:6400, visits 0 is the bare network)
          GAMES (100)  MAX_PLIES (400)  ADJ_CP (300)  ADJ_SECONDS (2.0)  OUT_DIR (<data>/abq/round_robin)
          TB (0)  CLAIM_DIR (unset means no claims)  CLAIM_OWNER (this machine's name)
          BOOK_FILE (round_robin_book.json beside this file)

There are fifty openings, each played once from each side, so a pairing of 100 is 100 different games. Every
version's search is deterministic (open_temp sealed at 0), so one opening with the same colours would be one game.
The fifty are master main lines judged level by Stockfish (round_robin_book.py), and the book's fingerprint is part
of the protocol, so games from another book never count together.

Each side uses its own version's sealed search at its own visits, keeps its tree from move to move, and never
ponders. One process plays a pairing at a time (a lock beside its log). Resumable by game, since each finished
game is a line in OUT_DIR/<pair>.jsonl with a PGN beside it. When every game is in, one row goes to
OUT_DIR/results.jsonl with the pairing's W-D-L from A's side and its Elo difference. round_robin_elo.py fits the
whole table from the game logs.
"""
import hashlib
import json
import math
import os
import socket
import sys
import threading
import time

import chess
import chess.engine
import chess.pgn

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
A, B = E("A", "mayfly_v5.pt:6400"), E("B", "mayfly_v5.pt:1600")
GAMES = int(E("GAMES", 100))
MAX_PLIES = int(E("MAX_PLIES", 400))
ADJ_CP = int(E("ADJ_CP", 300))
ADJ_SECONDS = float(E("ADJ_SECONDS", 2.0))
OUT_DIR = E("OUT_DIR", paths.under("abq", "round_robin"))
USE_TB = E("TB", "0") == "1"
CLAIM_DIR = E("CLAIM_DIR", "")
CLAIM_OWNER = E("CLAIM_OWNER", "") or socket.gethostname()
BOOK_FILE = E("BOOK_FILE", os.path.join(HERE, "round_robin_book.json"))


def load_book(path=BOOK_FILE):
    """([(name, fen)], fingerprint), the openings in order and the first six hex digits of their FENs' md5."""
    with open(path, encoding="utf-8") as fh:
        openings = json.load(fh)["openings"]
    fens = [o["fen"] for o in openings]
    return [(o["name"], o["fen"]) for o in openings], hashlib.md5("\n".join(fens).encode()).hexdigest()[:6]


BOOK, BOOK_TAG = load_book()
PROTOCOL = f"rr-{MAX_PLIES}-{'tb' if USE_TB else 'notb'}-sf{ADJ_CP}-b{len(BOOK)}.{BOOK_TAG}"


def player(spec):
    version, _, visits = spec.partition(":")
    return version, int(visits or 0)


def short(spec):
    version, visits = player(spec)
    return f"{os.path.splitext(version)[0].replace('mayfly_', '')}_{visits}"


def claim():
    """(ours, owner), whether this machine may play the pairing, and whose claim it is.

    The file is named for the pairing either way round, so A against B and B against A are one claim. It is created
    exclusively, so of two machines reaching it together the second finds the first one's name in it. An owner that
    finds its own claim is resuming, so a pairing is finished where it was started."""
    if not CLAIM_DIR:
        return True, CLAIM_OWNER
    os.makedirs(CLAIM_DIR, exist_ok=True)
    path = os.path.join(CLAIM_DIR, "__".join(sorted([short(A), short(B)])) + ".claim")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        with open(path, encoding="utf-8") as fh:
            owner = (fh.readline().strip() or "an unnamed claim")
        return owner == CLAIM_OWNER, owner
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(f"{CLAIM_OWNER}\n{time.strftime('%Y-%m-%d %H:%M')}\n")
    return True, CLAIM_OWNER


def hold(path):
    """The pairing's lock, held while this process plays it, or None if another process holds it.

    A second copy of one pairing then stops at once instead of playing every game twice. The operating system
    releases the lock when the process ends, however it ends."""
    fh = open(path, "a+")
    fh.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def main():
    pair = f"{short(A)}__{short(B)}"
    ours, owner = claim()
    if "--claim" in sys.argv[1:]:
        print(f"CLAIM {'OK' if ours else 'TAKEN'} {owner} {pair}", flush=True)
        return
    if not ours:
        print(f"{pair} is claimed by {owner}: not played here", flush=True)
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    lock = hold(os.path.join(OUT_DIR, pair + ".lock"))
    if lock is None:
        print(f"{pair} is being played by another process already: not played twice", flush=True)
        return

    from elo_vs_sf import game_over, _kill
    from mcts import MCTS, pinned_search, policy_move, tablebase, TB_PIECES
    from play_web import load_version

    players = [player(A), player(B)]
    models, sealed = {}, {}
    for version, visits in players:
        if version not in models:
            models[version] = load_version(version)
            sealed[version] = pinned_search(version)
        if visits > 0 and not sealed[version]:
            raise SystemExit(f"{version} has no sealed search; refusing to measure it on today's")
    tb = tablebase() if USE_TB else None
    eng = chess.engine.SimpleEngine.popen_uci(open(os.path.join(HERE, "stockfish_path.txt")).read().strip())
    threading._register_atexit(lambda: _kill(eng))
    eng.configure({"Threads": 2, "Hash": 256})
    sf_name = eng.id.get("name", "Stockfish")

    log_path, pgn_path = os.path.join(OUT_DIR, pair + ".jsonl"), os.path.join(OUT_DIR, pair + ".pgn")
    key = f"{A}|{B}|{GAMES}|{MAX_PLIES}|{PROTOCOL}|{sf_name}|tb{TB_PIECES if tb else 0}"
    done = {}
    if os.path.exists(log_path):
        for line in open(log_path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("key") == key:
                done[int(r["g"])] = r
    label = {0: f"MayFly {short(A)}", 1: f"MayFly {short(B)}"}
    print(f"{label[0]} against {label[1]}, {GAMES} games, to {MAX_PLIES} plies, "
          + (f"tablebases at {TB_PIECES} pieces" if tb else "no tablebases")
          + f", {sf_name} adjudicating at {ADJ_CP} cp, {len(BOOK)} openings (book {BOOK_TAG})"
          + (f" ({len(done)} already played)" if done else ""), flush=True)
    if GAMES > 2 * len(BOOK):
        print(f"  NOTE: {GAMES} games from {len(BOOK)} openings: past game {2 * len(BOOK)} the games repeat", flush=True)

    for g in range(GAMES):
        if g in done:
            continue
        name, fen = BOOK[(g // 2) % len(BOOK)]
        a_white = g % 2 == 0
        board = chess.Board(fen)
        sides = {}
        for idx, (version, visits) in enumerate(players):
            model = models[version][0]
            dev = str(next(model.parameters()).device)
            search = dict(sealed[version], **({} if USE_TB else {"use_tb": False}))
            tree = MCTS(model, dev, batch=48, cache_size=20000).apply(search) if visits > 0 else None
            sides[idx] = (version, visits, model, dev, tree)
        plies, result, how = 0, None, "rules"
        while True:
            if game_over(board):
                result = board.result(claim_draw=False) if board.is_game_over() else "1/2-1/2"
                break
            pieces = chess.popcount(board.occupied)
            if tb is not None and pieces <= TB_PIECES and not board.castling_rights:
                try:
                    wdl = tb.probe_wdl(board)
                except Exception:                       # a position the tables do not hold
                    wdl = None
                if wdl is not None:
                    stm_white = board.turn == chess.WHITE
                    if wdl == 2:
                        result = "1-0" if stm_white else "0-1"
                    elif wdl == -2:
                        result = "0-1" if stm_white else "1-0"
                    else:
                        result = "1/2-1/2"              # a draw, or a win or loss the fifty-move rule takes away
                    how = "tablebase"
                    break
            if plies >= MAX_PLIES:
                info = eng.analyse(board, chess.engine.Limit(time=ADJ_SECONDS))
                cp = info["score"].white().score(mate_score=100000)
                result = "1-0" if cp >= ADJ_CP else "0-1" if cp <= -ADJ_CP else "1/2-1/2"
                how = f"adjudicated at {MAX_PLIES} plies, {cp:+d} cp"
                break
            idx = 0 if (board.turn == chess.WHITE) == a_white else 1
            version, visits, model, dev, tree = sides[idx]
            if tree is not None:
                mv, _ = tree.search(board, visits=visits, reuse=True)
            else:
                mv = policy_move(model, board, dev, sealed[version].get("policy_head", "cls"),
                                 castle=sealed[version].get("castle_prior", "king"))
            if mv is None or mv not in board.legal_moves:
                result, how = ("0-1" if board.turn == chess.WHITE else "1-0"), "no legal move returned"
                break
            board.push(mv)
            plies += 1
        ws = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
        s_a = ws if a_white else 1.0 - ws
        row = {"key": key, "g": g, "opening": name, "a_white": a_white, "result": result, "s_a": s_a,
               "plies": plies, "how": how}
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        game = chess.pgn.Game.from_board(board)
        game.headers.update({"Event": f"MayFly round robin: {label[0]} against {label[1]}",
                             "White": label[0] if a_white else label[1], "Black": label[1] if a_white else label[0],
                             "Result": result, "Opening": name, "Termination": how, "Date": time.strftime("%Y.%m.%d")})
        with open(pgn_path, "a", encoding="utf-8") as fh:
            fh.write(str(game) + "\n\n")
        done[g] = row
        score = sum(r["s_a"] for r in done.values())
        print(f"  {g + 1:>3} {name:<18} {label[0]} as {'White' if a_white else 'Black':<5} {result:<7} {plies:>3} plies"
              f" ({how})  running {score:.1f}/{len(done)}", flush=True)
    eng.quit()

    if len(done) < GAMES:
        return
    w = sum(r["s_a"] == 1.0 for r in done.values())
    d = sum(r["s_a"] == 0.5 for r in done.values())
    n = len(done)
    s = (w + d / 2) / n
    clipped = min(max(s, 0.5 / n), 1 - 0.5 / n)
    elo = 400 * math.log10(clipped / (1 - clipped))
    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "a": A, "b": B, "games": n, "w": w, "d": d, "l": n - w - d,
           "score_a": round(s, 4), "elo_a_minus_b": round(elo, 1), "protocol": PROTOCOL, "adjudicator": sf_name,
           "tablebase_pieces": TB_PIECES if tb else 0}
    with open(os.path.join(OUT_DIR, "results.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")
    print(f"{label[0]} {w}-{d}-{n - w - d} against {label[1]}: {100 * s:.1f}%, {elo:+.0f} Elo", flush=True)


if __name__ == "__main__":
    main()
