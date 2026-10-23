r"""Opening positions from master games, and how often a version's move keeps the balance.

The positions come from real games between a side's 4th and 12th moves, kept when Stockfish calls them within
LOPSIDED centipawns of level. A version's move keeps the balance when it gives away no more than HOLD centipawns
against Stockfish's best. Both moves are priced the same way, Stockfish at NODES nodes, one thread, a fresh hash,
with the search held to that one move at the root.

  python chess/opening_suite.py build [N]       writes opening_suite.json (N positions, default 300, WORKERS=6)
  python chess/opening_suite.py score [VISITS]  env VERSION=mayfly_v5.pt [BASE=mayfly_v5.pt]
                                                VISITS 0 is the bare network
  env: SF=<stockfish binary>  PGN=<folder of .pgn files, default chess/twic or <data>/twic>

A score run appends one row to opening_results.jsonl with the share of moves that kept the balance, its
interval and the mean loss in centipawns. Every judged move is kept in opening_evals.json beside the suite, so a
move two versions or budgets share is judged once. BASE defaults to VERSION, a version's own sealed search.
"""
import glob
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import chess
import chess.engine
import chess.pgn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

E = os.environ.get
SUITE = E("SUITE", os.path.join(HERE, "opening_suite.json"))
EVALS = E("EVALS", os.path.join(HERE, "opening_evals.json"))
RESULTS = os.path.join(HERE, "opening_results.jsonl")
PGN_DIR = E("PGN", os.path.join(HERE, "twic") if os.path.isdir(os.path.join(HERE, "twic"))
            else paths.under("twic"))
NODES = int(E("SF_NODES", 1_000_000))
HOLD = int(E("HOLD_CP", 30))
LOPSIDED = int(E("LOPSIDED_CP", 150))
PER_FILE = int(E("PER_FILE", 12))
SF_PATHS = (E("SF", ""),)          # the Stockfish binary, set SF


def sf_path():
    return next(p for p in SF_PATHS if p and os.path.exists(p))


def stockfish():
    path = sf_path()
    sf = chess.engine.SimpleEngine.popen_uci(path)
    # One thread and a hash cleared before every search (each analyse below is its own game), so at a fixed
    # node count every price repeats exactly on any machine.
    sf.configure({"Threads": 1, "Hash": 64})
    return sf, path


def price(sf, board, move):
    """Centipawns for the side to move after `move`, searched alone at the root at NODES."""
    info = sf.analyse(board, chess.engine.Limit(nodes=NODES), root_moves=[move], game=object())
    return info["score"].relative.score(mate_score=100000)


def candidates(n, seed=12):
    """Up to n positions from real games, PER_FILE from each file, a side's 4th to 12th move."""
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
    rng.shuffle(files)
    out, seen = [], set()
    for path in files:
        took = 0
        with open(path, encoding="utf-8", errors="replace") as fh:
            while took < PER_FILE and len(out) < n:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                mv = list(g.mainline_moves())
                if len(mv) < 30 or rng.random() < 0.7:          # a spread of games, not the first in each file
                    continue
                b = g.board()
                for m in mv[:rng.randint(6, 23)]:
                    b.push(m)
                key = " ".join(b.fen().split()[:4])
                if key in seen or b.is_game_over():
                    continue
                seen.add(key)
                out.append(b.fen())
                took += 1
        if len(out) >= n:
            break
    return out


def _label(fen):
    sf, _ = stockfish()
    try:
        board = chess.Board(fen)
        best = sf.analyse(board, chess.engine.Limit(nodes=NODES), game=object())["pv"][0]
        cp = price(sf, board, best)
        return {"fen": fen, "best": best.uci(), "best_cp": cp} if abs(cp) <= LOPSIDED else None
    finally:
        sf.quit()


def build(want):
    fens = candidates(int(want * 1.6))
    with mp.Pool(int(E("WORKERS", 6))) as pool:
        got = pool.map(_label, fens)
    keep = [p for p in got if p][:want]
    json.dump({"built": time.strftime("%Y-%m-%d %H:%M"), "nodes": NODES, "lopsided_cp": LOPSIDED,
               "stockfish": os.path.basename(sf_path()), "scanned": len(fens), "positions": keep},
              open(SUITE, "w", encoding="utf-8"), indent=0)
    print(f"{len(keep)} positions kept of {len(fens)} scanned (|eval| <= {LOPSIDED} cp at {NODES:,} nodes) -> {SUITE}")


def _load_evals():
    try:
        return json.load(open(EVALS, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def score(visits):
    from mcts import MCTS, pinned_search, policy_move
    from play_web import load_version
    version = E("VERSION", "mayfly_v5.pt")
    base_name = E("BASE", version)
    base = pinned_search(base_name)
    assert base, f"no sealed search for {base_name}"
    suite = json.load(open(SUITE, encoding="utf-8"))["positions"]
    model, label, _ = load_version(version)
    dev = str(next(model.parameters()).device)
    t0, moves = time.time(), []
    for i, p in enumerate(suite):
        board = chess.Board(p["fen"])
        if visits <= 0:
            mv = policy_move(model, board, dev, base.get("policy_head", "cls"), castle=base.get("castle_prior", "king"))
        else:
            mv, _ = MCTS(model, dev, batch=48, cache_size=20000).apply(base).search(board, visits=visits, reuse=False)
        moves.append(mv)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} moves played", flush=True)
    t_play = time.time() - t0
    evals = _load_evals()
    sf, path = stockfish()
    losses = []
    try:
        for p, mv in zip(suite, moves):
            if mv is None:
                losses.append(10000)
                continue
            if mv.uci() == p["best"]:
                losses.append(0)
                continue
            key = f"{p['fen']}|{mv.uci()}"
            if key not in evals:
                evals[key] = price(sf, chess.Board(p["fen"]), mv)
            losses.append(max(0, p["best_cp"] - evals[key]))
    finally:
        sf.quit()
    merged = _load_evals()
    merged.update(evals)                                    # another run may have judged moves meanwhile
    tmp = EVALS + ".tmp"
    json.dump(merged, open(tmp, "w", encoding="utf-8"))
    os.replace(tmp, EVALS)
    n = len(losses)
    held = sum(1 for x in losses if x <= HOLD)
    rate = held / n
    se = math.sqrt(max(rate * (1 - rate), 1e-9) / n)
    mean_loss = sum(min(x, 1000) for x in losses) / n
    print(f"{label}, {visits} visits, base {base_name}: {held}/{n} kept the balance (within {HOLD} cp), "
          f"{100 * rate:.1f}% (95% +-{196 * se:.1f}), mean loss {mean_loss:.0f} cp; played in {t_play / 60:.1f} min, "
          f"judged in {(time.time() - t0 - t_play) / 60:.1f} min")
    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M"), "network": label, "base": base_name,
                             "visits": visits, "held": held, "n": n, "rate": round(rate, 4),
                             "mean_loss_cp": round(mean_loss, 1), "hold_cp": HOLD, "nodes": NODES,
                             "stockfish": os.path.basename(path),
                             # every move and its loss, so a stricter line than HOLD can be read off later
                             "moves": [None if mv is None else mv.uci() for mv in moves],
                             "losses": losses}) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    else:
        score(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
