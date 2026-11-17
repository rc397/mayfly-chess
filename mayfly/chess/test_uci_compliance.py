r"""test_uci_compliance -- MayFly's UCI through python-chess, the way a GUI or a rating list drives it.

Starts the engine with the launcher (mayfly_uci.sh beside this file, or --engine), checks every
command a GUI sends (uci, isready, setoption for each option, ucinewgame, position startpos and fen
with moves, go movetime, nodes, depth, a clock and infinite, stop, ponder and ponderhit, quit), that
every bestmove is legal and that every info line parses, then plays games of MayFly against itself
on a clock with no crash, hang, flag or illegal move. Loads the real network, so it is a script and
not a selftest unit. It runs from a checkout or from an unpacked release bundle.

  python3 chess/test_uci_compliance.py              the checks, then 6 games (about half an hour)
  python3 chess/test_uci_compliance.py --quick      the checks only (about two minutes)
  python3 chess/test_uci_compliance.py --bench 5    only the speed: visits a second at go movetime 5000
  options: --engine CMD  --games N  --only 3,6  --skip-protocol  --out DIR (logs and PGNs, default a temp folder)
"""
import argparse, concurrent.futures, logging, os, sys, tempfile, time
import chess, chess.engine, chess.pgn

HERE = os.path.dirname(os.path.abspath(__file__))
# castling rights for both sides and an en passant capture on the first move (1.e4 f5 2.e5 d5)
EP_FEN = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
OPTIONS = ("Version", "Visits", "Ponder", "SyzygyPath", "Move Overhead")

FAILS = []


def check(ok, what, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {what}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(what)
    return ok


class Warnings(logging.Handler):
    """python-chess's complaints about engine output: unparsable info, unexpected lines."""
    def __init__(self):
        super().__init__(logging.WARNING)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


WARN = Warnings()
logging.getLogger("chess.engine").addHandler(WARN)


def start(cmd, out, name, **env):
    log = os.path.join(out, f"uci_{name}.log")
    if os.path.exists(log):
        os.remove(log)
    full = dict(os.environ, LOG=log, PONDER_MEM_GB=os.environ.get("PONDER_MEM_GB", "2"), **env)
    err = open(os.path.join(out, f"uci_{name}.err"), "w")
    eng = chess.engine.SimpleEngine.popen_uci(cmd, env=full, stderr=err, timeout=60)
    return eng, log


def info_ok(info, board):
    """depth, nodes, a score and a pv that starts with a legal move."""
    pv = info.get("pv") or []
    return ("depth" in info and "nodes" in info and "score" in info and pv and pv[0] in board.legal_moves)


def timed(eng, board, limit, **kw):
    t0 = time.monotonic()
    res = eng.play(board, limit, info=chess.engine.INFO_ALL, **kw)
    return res, time.monotonic() - t0


# ------------------------------------------------------------------ the protocol, one command at a time
def protocol(cmd, out):
    t0 = time.monotonic()
    eng, log = start(cmd, out, "protocol")
    check(True, "uci: handshake", f"uciok after {time.monotonic() - t0:.1f}s")
    check("MayFly" in eng.id.get("name", ""), "uci: id name", eng.id.get("name", ""))
    check(bool(eng.id.get("author")), "uci: id author", eng.id.get("author", ""))
    missing = [o for o in OPTIONS if o not in eng.options]
    check(not missing, "uci: options listed", ", ".join(eng.options) + (f"; missing {missing}" if missing else ""))
    eng.ping()
    check(True, "isready: readyok")

    # the first search waits for the network (a position past move one skips the opening fallback)
    board = chess.Board()
    for u in ("e2e4", "e7e5"):
        board.push_uci(u)
    t0 = time.monotonic()
    res, _ = timed(eng, board, chess.engine.Limit(nodes=32), game="g1")
    check(res.move in board.legal_moves and info_ok(res.info, board), "first search (ucinewgame, position startpos moves, go nodes)",
          f"{res.move} after {time.monotonic() - t0:.1f}s including the load, info {res.info.get('nodes')} nodes")

    # setoption, every option
    eng.configure({"Visits": 48})
    res, took = timed(eng, board, chess.engine.Limit(depth=3))
    check(res.move in board.legal_moves and res.info.get("nodes", 0) <= 48 + 400, "setoption Visits, go depth (plays Visits)",
          f"{res.move}, {res.info.get('nodes')} nodes, {took:.2f}s")
    eng.configure({"Move Overhead": 20})
    eng.configure({"SyzygyPath": ""})
    eng.configure({"Version": eng.options["Version"].default})
    eng.configure({"Version": "no_such_version.pt"})
    eng.ping()
    res, _ = timed(eng, board, chess.engine.Limit(nodes=16))
    check(res.move in board.legal_moves, "setoption Move Overhead, SyzygyPath, Version (and a bad Version refused)",
          f"still answers: {res.move}")
    # a SyzygyPath with no tables in it: a four piece ending is searched as before
    eng.configure({"SyzygyPath": tempfile.mkdtemp(prefix="mayfly_no_tables_")})
    kb = chess.Board("8/8/4k3/8/8/2KR4/8/7r w - - 0 60")
    res, _ = timed(eng, kb, chess.engine.Limit(nodes=200))
    check(res.move in kb.legal_moves and info_ok(res.info, kb), "setoption SyzygyPath (a folder with no tables), 4 pieces",
          f"{res.move}")
    eng.configure({"SyzygyPath": ""})
    eng.configure({"Visits": eng.options["Visits"].default})

    # position fen with moves, castling rights and en passant possible
    fb = chess.Board(EP_FEN)
    res, _ = timed(eng, fb, chess.engine.Limit(nodes=64), game="g2")
    check(res.move in fb.legal_moves and info_ok(res.info, fb), "ucinewgame, position fen, go nodes", f"{res.move}")
    fb.push(chess.Move.from_uci("e5d6"))         # en passant
    fb.push(chess.Move.from_uci("c7d6"))
    res, _ = timed(eng, fb, chess.engine.Limit(nodes=64))
    check(res.move in fb.legal_moves and info_ok(res.info, fb), "position fen ... moves (after an en passant capture)", f"{res.move}")

    # go movetime
    for ms in (300, 1000):
        res, took = timed(eng, board, chess.engine.Limit(time=ms / 1000))
        check(res.move in board.legal_moves and took <= ms / 1000 + 0.35, f"go movetime {ms}",
              f"{res.move} in {took:.2f}s, {res.info.get('nodes')} nodes")

    # go wtime btime winc binc, white and black to move
    bb = board.copy()
    bb.push_uci("g1f3")
    for b, clock in ((board, 20.0), (bb, 20.0)):
        lim = chess.engine.Limit(white_clock=clock, black_clock=clock + 3, white_inc=0.5, black_inc=0.5)
        res, took = timed(eng, b, lim)
        check(res.move in b.legal_moves and took < clock / 5, f"go wtime btime winc binc ({'white' if b.turn else 'black'} to move)",
              f"{res.move} in {took:.2f}s, {res.info.get('nodes')} nodes")
    lim = chess.engine.Limit(white_clock=10.0, black_clock=10.0)
    res, took = timed(eng, board, lim)
    check(res.move in board.legal_moves and took < 2.0, "go wtime btime (no increment)", f"{res.move} in {took:.2f}s")

    # go infinite, then stop: no bestmove before the stop
    with eng.analysis(board) as an:
        time.sleep(2.0)
        quiet = an.would_block()
        an.stop()
        best = an.wait()
    check(quiet, "go infinite: no bestmove before stop")
    check(best.move in board.legal_moves and info_ok(an.info, board), "stop: bestmove and info",
          f"{best.move}, {an.info.get('nodes')} nodes, score {an.info.get('score')}")

    # stop during a long fixed search
    with eng.analysis(board, chess.engine.Limit(nodes=10 ** 7)) as an:
        time.sleep(1.0)
        t0 = time.monotonic()
        an.stop()
        best = an.wait()
    check(best.move in board.legal_moves and time.monotonic() - t0 < 2.0, "stop during go nodes 10000000",
          f"{best.move} {time.monotonic() - t0:.2f}s after stop")

    # ponder: a hit and a miss, on a clock
    eng.configure({"Visits": eng.options["Visits"].default})
    pb = chess.Board()
    lim = lambda: chess.engine.Limit(white_clock=30, black_clock=30, white_inc=0.5, black_inc=0.5)
    r1 = eng.play(pb, lim(), ponder=True, game="g3", info=chess.engine.INFO_ALL)
    check(r1.ponder is not None, "go: bestmove ... ponder", f"{r1.move} ponder {r1.ponder}")
    pb.push(r1.move)
    time.sleep(2.0)
    pb.push(r1.ponder)
    r2, took = timed(eng, pb, lim(), ponder=True, game="g3")          # ponderhit (same game, same options)
    check(r2.move in pb.legal_moves, "go ponder, ponderhit", f"{r2.move} in {took:.2f}s")
    pb.push(r2.move)
    time.sleep(1.0)
    pb.push(next(m for m in pb.legal_moves if m != r2.ponder))
    r3, took = timed(eng, pb, lim(), ponder=True, game="g3")          # stop, then a new search
    check(r3.move in pb.legal_moves, "go ponder, stop (a miss), new search", f"{r3.move} in {took:.2f}s")

    t0 = time.monotonic()
    eng.quit()
    code = eng.transport.get_returncode()
    check(code == 0 and time.monotonic() - t0 < 15, "quit: the process exits", f"exit code {code}, {time.monotonic() - t0:.1f}s")
    text = open(log, encoding="utf-8").read() if os.path.exists(log) else ""
    check("# ponderhit" in text and "# ponder miss" in text, "the log saw the ponderhit and the miss")


# ------------------------------------------------------------------ games
GAMES = [
    # name, start fen, white limit, black limit, ponder
    ("1+1", None, "60+1", "60+1", False),
    ("1+1 ponder", None, "60+1", "60+1", True),
    ("1+1 from a fen, castling and en passant", EP_FEN, "60+1", "60+1", False),
    ("2+0 sudden death", None, "120+0", "120+0", False),
    ("400 nodes a move", None, "n400", "n400", False),
    ("1+1 against itself at 1 node a move", None, "60+1", "n1", False),
]


def play_game(cmd, out, gi, name, fen, wspec, bspec, ponder):
    engs = {}
    for side, spec in ((chess.WHITE, wspec), (chess.BLACK, bspec)):
        engs[side] = start(cmd, out, f"game{gi}_{'w' if side else 'b'}")[0]
    clock = {s: float(spec.split("+")[0]) for s, spec in ((chess.WHITE, wspec), (chess.BLACK, bspec)) if "+" in spec}
    inc = {s: float(spec.split("+")[1]) for s, spec in ((chess.WHITE, wspec), (chess.BLACK, bspec)) if "+" in spec}
    spec = {chess.WHITE: wspec, chess.BLACK: bspec}
    board = chess.Board(fen) if fen else chess.Board()
    problem, low = None, dict(clock)
    pool = concurrent.futures.ThreadPoolExecutor(1)
    started = time.monotonic()
    while not board.is_game_over(claim_draw=True) and board.ply() < 400:
        side = board.turn
        if spec[side].startswith("n"):
            lim = chess.engine.Limit(nodes=int(spec[side][1:]))
            wait = 120.0
        else:
            lim = chess.engine.Limit(white_clock=clock.get(chess.WHITE), black_clock=clock.get(chess.BLACK),
                                     white_inc=inc.get(chess.WHITE), black_inc=inc.get(chess.BLACK))
            wait = clock[side] + 30.0
        t0 = time.monotonic()
        fut = pool.submit(engs[side].play, board, lim, ponder=ponder and side in clock, game=f"g{gi}",
                          info=chess.engine.INFO_ALL)
        try:
            res = fut.result(timeout=wait)
        except concurrent.futures.TimeoutError:
            problem = f"no bestmove after {wait:.0f}s at ply {board.ply()}"
            break
        except chess.engine.EngineError as exc:
            problem = f"engine error at ply {board.ply()}: {exc}"
            break
        took = time.monotonic() - t0
        if res.move is None or res.move not in board.legal_moves:
            problem = f"illegal bestmove {res.move} at ply {board.ply()}"
            break
        if not info_ok(res.info, board):
            problem = f"info line missing or unparsed at ply {board.ply()}: {res.info}"
            break
        if side in clock:
            clock[side] -= took
            low[side] = min(low[side], clock[side])
            if clock[side] < 0:
                problem = f"{'white' if side else 'black'} lost on time at ply {board.ply()} ({clock[side]:.2f}s)"
                break
            clock[side] += inc[side]
        board.push(res.move)
    for e in engs.values():
        try:
            e.quit()
        except Exception as exc:
            problem = problem or f"quit failed: {exc}"
    pool.shutdown(wait=False)
    game = chess.pgn.Game.from_board(board)
    game.headers.update(Event=f"MayFly UCI test {gi}: {name}", White=f"MayFly {wspec}", Black=f"MayFly {bspec}")
    if fen:
        game.headers.update(FEN=fen, SetUp="1")
    over = board.outcome(claim_draw=True)
    result = over.result() if over else ("*" if problem else "1/2-1/2")
    why = over.termination.name.lower() if over else ("ply cap" if not problem else "stopped")
    game.headers["Result"] = result
    with open(os.path.join(out, f"game{gi}.pgn"), "w", encoding="utf-8") as fh:
        print(game, file=fh)
    secs = time.monotonic() - started
    lows = ", ".join(f"{'white' if s else 'black'} {v:.1f}s" for s, v in low.items())
    check(problem is None, f"game {gi} ({name})",
          f"{result} by {why} in {board.ply()} plies, {secs / 60:.1f} min" + (f", lowest clock {lows}" if lows else "")
          + (f"; {problem}" if problem else ""))
    return result


BENCH = [
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N1P/PP1P1PP1/RNBQR1K1 b - - 0 9",
    "r2q1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R2Q1RK1 w - - 0 10",
    "2r3k1/pp3pp1/4p2p/3pP3/3P4/P4N1P/1P3PP1/2R3K1 w - - 0 25",
    "8/5pk1/6p1/3R4/5P2/6KP/r7/8 b - - 0 45",
]


def bench(cmd, out, secs):
    """Visits a second at `go movetime`, on five positions from the opening to an ending."""
    eng, _ = start(cmd, out, "bench")
    eng.play(chess.Board(BENCH[0]), chess.engine.Limit(nodes=64), game="w")   # waits for the load
    rates = []
    for i, fen in enumerate(BENCH):
        b = chess.Board(fen)
        res, took = timed(eng, b, chess.engine.Limit(time=secs), game=f"b{i}")
        rates.append(res.info.get("nodes", 0) / took)
        print(f"bench {i + 1}: {res.info.get('nodes', 0):7d} visits in {took:.2f}s, {rates[-1]:6.0f} a second, "
              f"{res.move}", flush=True)
    eng.quit()
    print(f"bench: {sum(rates) / len(rates):.0f} visits a second on average", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.path.join(HERE, "mayfly_uci.sh"))
    ap.add_argument("--games", type=int, default=len(GAMES))
    ap.add_argument("--only", default="", help="games to play, as numbers: 3,6")
    ap.add_argument("--skip-protocol", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bench", type=float, default=0, help="only measure speed, this many seconds a position")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or tempfile.mkdtemp(prefix="mayfly_uci_test_")
    os.makedirs(out, exist_ok=True)
    cmd = [a.engine]
    print(f"engine: {a.engine}\nlogs and games: {out}", flush=True)
    if a.bench:
        return bench(cmd, out, a.bench)
    if not a.skip_protocol:
        protocol(cmd, out)
    if not a.quick:
        pick = {int(x) for x in a.only.split(",") if x.strip()}
        for gi, g in enumerate(GAMES[:a.games], 1):
            if not pick or gi in pick:
                play_game(cmd, out, gi, *g)
    overruns = 0
    for f in sorted(os.listdir(out)):
        if f.endswith(".log"):
            overruns += sum(" OVERRUN" in line for line in open(os.path.join(out, f), encoding="utf-8"))
    print(f"searches past 1.25x their allowance (OVERRUN in the logs): {overruns}")
    check(not WARN.lines, "python-chess parsed every engine line", "; ".join(WARN.lines[:5]))
    print(f"\n{'ALL PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
