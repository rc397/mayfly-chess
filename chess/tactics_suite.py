r"""tactics_suite builds only-move positions from master games and scores how often a search finds the move.

The suite is built the way Lichess puzzles are, from real game positions where exactly one move wins.
Stockfish at a fixed depth with two lines scores each candidate, and a position is kept when the best
move is worth at least WIN centipawns (or mates) and the second best is at least GAP worse. The answer is
then unique, so each position is a clean yes or no.

  python chess/tactics_suite.py build [N]                   writes tactics_suite.json
  python chess/tactics_suite.py score [VISITS] [opts]       env VERSION=mayfly_v3.pt BASE=...
  python chess/tactics_suite.py score 400 "proofs=1"        against the same baseline
  python chess/tactics_suite.py score 0                     the bare network, as the ladder plays it

A score run prints the solve rate with its interval and appends one row to tactics_results.jsonl.
The version cards read the rows with BASE set to the version itself and no opts. Seeds and the file are
fixed, so two runs of one configuration see the same positions.
"""
import os, sys, json, glob, random, math, time
import chess, chess.pgn, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
SUITE = E("SUITE", os.path.join(HERE, "tactics_suite.json"))
PGN_DIR = E("PGN", os.path.join(HERE, "twic") if os.path.isdir(os.path.join(HERE, "twic"))
            else paths.under("twic"))
DEPTH = int(E("SF_DEPTH", 18))
WIN = int(E("WIN_CP", 200))
GAP = int(E("GAP_CP", 200))


def _cp(score):
    return score.pov(score.turn).score(mate_score=100000)


def candidates(n, seed=11):
    """Positions from real games, a few from each, spread over the middlegame and endgame."""
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
                if len(mv) < 24:
                    continue
                for cut in rng.sample(range(16, len(mv)), min(3, len(mv) - 16)):
                    b = g.board()
                    for m in mv[:cut]:
                        b.push(m)
                    f = b.fen()
                    if not b.is_game_over() and f not in seen:
                        seen.add(f)
                        out.append(f)
        if len(out) >= n:
            break
    return out[:n]


def _save(keep, scanned):
    """Written as it goes, and atomically. About one position in twenty is kept at depth 18, so a
    full build is hours of Stockfish and should survive an interruption."""
    tmp = SUITE + ".part"
    json.dump({"depth": DEPTH, "win_cp": WIN, "gap_cp": GAP, "scanned": scanned,
               "positions": keep}, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, SUITE)


def build(want):
    keep, scanned = [], 0
    if os.path.exists(SUITE):
        try:
            was = json.load(open(SUITE, encoding="utf-8"))
            if (was.get("depth"), was.get("win_cp"), was.get("gap_cp")) == (DEPTH, WIN, GAP):
                keep, scanned = was.get("positions", []), int(was.get("scanned", 0))
                print(f"  resuming: {len(keep)} kept, {scanned} candidates already scored",
                      flush=True)
        except (OSError, ValueError):
            pass
    if len(keep) >= want:
        print(f"{len(keep)} positions already in {SUITE}")
        return
    sf = chess.engine.SimpleEngine.popen_uci(open(os.path.join(HERE, "stockfish_path.txt")).read().strip())
    sf.configure({"Threads": int(E("SF_THREADS", 8)), "Hash": 1024})
    t0, looked = time.time(), 0
    # the candidate list is seeded and built in order, so a longer one starts with the same positions
    # a shorter one had and `scanned` resumes into it
    for fen in candidates(want * 20)[scanned:]:
        scanned += 1
        b = chess.Board(fen)
        if b.legal_moves.count() < 2:
            continue
        looked += 1
        info = sf.analyse(b, chess.engine.Limit(depth=DEPTH), multipv=2)
        if len(info) < 2:
            continue
        best, second = _cp(info[0]["score"]), _cp(info[1]["score"])
        if best >= WIN and best - second >= GAP:
            keep.append({"fen": fen, "best": info[0]["pv"][0].uci(), "cp": best,
                         "second": second, "mate": info[0]["score"].is_mate()})
            if len(keep) % 25 == 0:
                _save(keep, scanned)
                print(f"  {len(keep)} kept of {looked} looked at this run, "
                      f"{time.time()-t0:.0f}s", flush=True)
        if len(keep) >= want:
            break
    sf.quit()
    _save(keep, scanned)
    mates = sum(p["mate"] for p in keep)
    print(f"{len(keep)} only-move positions ({mates} mates) from {scanned} candidates, "
          f"Stockfish depth {DEPTH}, win {WIN} cp, gap {GAP} cp -> {SUITE}")


def score(visits, spec):
    from mcts import MCTS, pinned_search, policy_move
    from play_web import load_version
    from search_match import parse
    opts = parse(spec)
    base_name = E("BASE", "")
    base = pinned_search(base_name) if base_name else {}
    suite = json.load(open(SUITE, encoding="utf-8"))["positions"]
    model, label, _ = load_version(E("VERSION", "mayfly_v3.pt"))
    dev = str(next(model.parameters()).device)
    hit, t0 = 0, time.time()
    for i, p in enumerate(suite):
        if visits <= 0:
            # No search. The bare network's top legal move, from the head and castling the seal names,
            # as the ladder's 0-visit cells play it.
            mv = policy_move(model, chess.Board(p["fen"]), dev, base.get("policy_head", "cls"),
                             castle=base.get("castle_prior", "king"))
            hit += mv is not None and mv.uci() == p["best"]
            continue
        t = MCTS(model, dev, batch=48, cache_size=20000).apply(base)
        for k, v in opts.items():
            if not hasattr(t, k):
                raise SystemExit(f"MCTS has no option {k!r}")
            setattr(t, k, v)
        mv, _ = t.search(chess.Board(p["fen"]), visits=visits, reuse=False)
        hit += mv is not None and mv.uci() == p["best"]
        if (i + 1) % 100 == 0:
            print(f"  {i+1}: {hit} solved", flush=True)
    n = len(suite)
    rate = hit / n
    se = math.sqrt(max(rate * (1 - rate), 1e-9) / n)
    print(f"{label}, {visits} visits, base {base_name or 'defaults'} + {opts}: "
          f"{hit}/{n} solved, {100*rate:.1f}% (95% +-{196*se:.1f}), {(time.time()-t0)/60:.1f} min")
    with open(os.path.join(HERE, "tactics_results.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M"), "network": label,
                             "base": base_name or "defaults", "opts": opts, "visits": visits,
                             "solved": hit, "n": n, "rate": round(rate, 4)}) + chr(10))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 500)
    else:
        score(int(sys.argv[2]) if len(sys.argv) > 2 else 400, sys.argv[3] if len(sys.argv) > 3 else "")
