"""Release gate on a short clock, MayFly against Stockfish with both sides on the same game clock.

The other gate criterion is the ladder at 6,400 visits (ladder_climb.py). Here MayFly runs as the
real engine through uci.py, so its own time manager spends the clock, with NET_OVERHEAD=0.02 since
the game is local. No pondering. Stockfish uses the ladder's settings (Threads 2, Hash 256, Skill
Level under 100, UCI_Elo from 100) and manages its own clock. The trainer stands aside while the gate
plays (aside.py), so the search has the GPU to itself.

Each side's time is its measured wall time per move, the increment is added after each move, and a
side whose clock runs out loses (a draw if the other side cannot mate). The twelve openings from
elo_vs_sf.py are played from both sides. Like ladder_climb.py it climbs one level at a time and stops
at the first level scored under STOP. Games are logged to <OUT>/clock_gate_<version>_<tc>.games.jsonl
so a re-run resumes, and each finished level adds a row to <OUT>/clock_gate_<version>.jsonl.

    python chess/clock_gate.py
    env VERSION=mayfly_v4.pt, VERSION_DIR (a candidate's folder), TC=60+1, LEVELS=5,...,20,
        GAMES=24, STOP=0.5, OUT, MAX_PLIES=240
"""
import json
import os
import sys
import threading
import time

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from elo_vs_sf import BOOK, game_over  # noqa: E402
import paths  # noqa: E402

E = os.environ.get
VERSION = E("VERSION", "mayfly_v4.pt")
TC = E("TC", "60+1")
BASE, INC = (float(x) for x in TC.split("+"))
LEVELS = [int(x) for x in E("LEVELS", ",".join(str(k) for k in range(5, 21))).split(",")]
GAMES = int(E("GAMES", 24))
STOP = float(E("STOP", 0.5))
OUT = E("OUT", paths.under("abq"))
MAX_PLIES = int(E("MAX_PLIES", 240))
NAME = VERSION.replace(".pt", "")
GAMES_PATH = os.path.join(OUT, f"clock_gate_{NAME}_{TC.replace('+', 'p')}.games.jsonl")
CELLS_PATH = os.path.join(OUT, f"clock_gate_{NAME}.jsonl")
# The gate holds the main run's aside flag while it plays, so the trainer waits between steps and
# resumes within seconds of the gate ending or dying. The flag lives in the main run's checkpoint
# folder, which a candidate's snapshot folder does not know, hence MAIN_CKPT_DIR. GATE_ASIDE=0 plays
# beside training instead.
GATE_ASIDE = E("GATE_ASIDE", "1") == "1"
ASIDE_PATH = E("ASIDE_FILE", os.path.join(E("MAIN_CKPT_DIR", paths.under("ckpt")), "gate_aside.flag"))


# python-chess keeps an engine on a non-daemon thread until the engine exits, so an uncaught
# exception would leave the interpreter waiting forever. Every engine opened here is killed as
# shutdown begins.
def _kill(engine):
    try:
        engine.transport.kill()
    except Exception:
        pass


def played(level):
    """The games already on file for this level, as {game index: row}."""
    got = {}
    if os.path.exists(GAMES_PATH):
        for line in open(GAMES_PATH, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue                    # blank, or the torn last line of a killed run
            if r["level"] == level and r["tc"] == TC:
                got[r["g"]] = r
    return got


def mayfly():
    env = dict(os.environ, VERSION=VERSION, NET_OVERHEAD="0.02", PYTHONUNBUFFERED="1")
    env.pop("LOG", None)
    eng = chess.engine.SimpleEngine.popen_uci([sys.executable, os.path.join(HERE, "uci.py")],
                                              env=env, cwd=HERE, timeout=180)
    # Load the network before any clock runs. uci.py answers isready at once, but go waits for the
    # warm-up, so one throwaway move on one visit returns only once the network is loaded and warm.

    threading._register_atexit(lambda e=eng: _kill(e))
    eng.ping()
    eng.play(chess.Board(), chess.engine.Limit(nodes=1), game=("warm", 0))
    return eng


def stockfish(level):
    sf = chess.engine.SimpleEngine.popen_uci(open(os.path.join(HERE, "stockfish_path.txt")).read().strip())
    if level >= 100:
        sf.configure({"Threads": 2, "Hash": 256, "UCI_LimitStrength": True, "UCI_Elo": level})
    else:
        sf.configure({"Threads": 2, "Hash": 256, "Skill Level": level})
    threading._register_atexit(lambda e=sf: _kill(e))
    return sf


def one_game(mf, sf, fen, mf_white, g):
    board = chess.Board(fen)
    clock = {chess.WHITE: BASE, chess.BLACK: BASE}
    spent, moves, flag = 0.0, 0, None
    plies = 0
    while not game_over(board) and plies < MAX_PLIES:
        side = board.turn
        mf_turn = (side == chess.WHITE) == mf_white
        lim = chess.engine.Limit(white_clock=max(0.01, clock[chess.WHITE]),
                                 black_clock=max(0.01, clock[chess.BLACK]),
                                 white_inc=INC, black_inc=INC)
        t0 = time.perf_counter()
        res = (mf if mf_turn else sf).play(board, lim, game=("gate", g))
        dt = time.perf_counter() - t0
        clock[side] -= dt
        if mf_turn:
            spent += dt
            moves += 1
        if clock[side] < 0.0:
            flag = side
            break
        clock[side] += INC
        if res.move is None or res.move not in board.legal_moves:
            flag = side                 # no legal answer is a loss for that side, like a flag
            break
        board.push(res.move)
        plies += 1
    if flag is not None:
        other = not flag
        # a flag loses unless the other side has no way to mate at all
        ws = 0.5 if board.has_insufficient_material(other) else (1.0 if other == chess.WHITE else 0.0)
        how = "flag " + ("MayFly" if (flag == chess.WHITE) == mf_white else "Stockfish")
    elif game_over(board):
        ws = {"1-0": 1.0, "0-1": 0.0}.get(board.result(claim_draw=True), 0.5)
        how = "played out"
    else:
        ws, how = 0.5, "ply cap"
    s = ws if mf_white else 1.0 - ws
    left = clock[chess.WHITE] if mf_white else clock[chess.BLACK]
    return {"s": s, "plies": plies, "how": how, "secs_per_move": spent / max(moves, 1),
            "mf_clock_left": round(left, 2)}


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{VERSION} on a {TC} clock against Stockfish, levels {LEVELS}, {GAMES} games a level, "
          f"stopping under {100 * STOP:.0f}%", flush=True)
    mf = mayfly()
    try:
        for level in LEVELS:
            got = played(level)
            if len(got) < GAMES:
                sf = stockfish(level)
                try:
                    for g in range(GAMES):
                        if g in got:
                            continue
                        name, fen = BOOK[(g // 2) % len(BOOK)]
                        mf_white = g % 2 == 0
                        r = one_game(mf, sf, fen, mf_white, (level, g))
                        r.update(level=level, tc=TC, g=g, opening=name,
                                 colour="white" if mf_white else "black")
                        with open(GAMES_PATH, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(r) + "\n")
                        got[g] = r
                        print(f"  Skill {level} game {g + 1:>2} {name:<16} as {r['colour']:<5} "
                              f"{('win' if r['s'] == 1 else 'draw' if r['s'] == 0.5 else 'loss'):>4} "
                              f"({r['how']}), {r['plies']} plies, {r['secs_per_move']:.2f}s a move, "
                              f"{r['mf_clock_left']:.1f}s left", flush=True)
                finally:
                    sf.quit()
            rows = [got[g] for g in range(GAMES)]
            score = sum(r["s"] for r in rows) / GAMES
            cell = {"version": VERSION, "tc": TC, "level": level, "games": GAMES,
                    "score": round(score, 4), "w": sum(r["s"] == 1 for r in rows),
                    "d": sum(r["s"] == 0.5 for r in rows), "l": sum(r["s"] == 0 for r in rows),
                    "mayfly_flags": sum(r["how"] == "flag MayFly" for r in rows),
                    "secs_per_move": round(sum(r["secs_per_move"] for r in rows) / GAMES, 3),
                    "when": time.strftime("%Y-%m-%d %H:%M")}
            with open(CELLS_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(cell) + "\n")
            print(f"  Skill {level} at {TC}: {100 * score:.1f}%  {cell['w']}W {cell['d']}D {cell['l']}L, "
                  f"{cell['mayfly_flags']} lost on time", flush=True)
            if score < STOP:
                print(f"  under {100 * STOP:.0f}% at Skill {level}: this is where it stops", flush=True)
                break
        else:
            print("  every level on the list was beaten", flush=True)
    finally:
        mf.quit()


if __name__ == "__main__":
    if GATE_ASIDE:
        import aside
        print(f"the trainer stands aside while this runs ({ASIDE_PATH})", flush=True)
        with aside.hold(ASIDE_PATH):
            main()
    else:
        main()
