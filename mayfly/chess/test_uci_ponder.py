r"""test_uci_ponder drives uci.py through python-chess the way lichess-bot does, with pondering on.

It checks a hit (`ponderhit`, and the search starts with the visits it already has), a miss (`stop`, a
throwaway bestmove, then a normal search), a premove (the predicted reply arrives the moment pondering
starts, where the stopping thread could undo the ponderhit) and a ponder that fills a tiny memory ceiling
and stops by itself. It loads a real network, so it is a script and not a selftest unit.

  python chess/test_uci_ponder.py            env: VERSION=mayfly_v3.pt
"""
import os, sys, time, tempfile, chess, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__))
MIDDLEGAME = "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N1P/PP1P1PP1/RNBQR1K1 b - - 0 9"


def engine(name, **extra):
    log = os.path.join(tempfile.gettempdir(), f"uci_ponder_{name}.log")
    if os.path.exists(log):
        os.remove(log)
    env = dict(os.environ, VERSION=os.environ.get("VERSION", "mayfly_v3.pt"), LOG=log, **extra)
    eng = chess.engine.SimpleEngine.popen_uci([sys.executable, "uci.py"], cwd=HERE, env=env,
                                              stderr=open(os.devnull, "w"))
    return eng, log


lim = lambda: chess.engine.Limit(white_clock=40, black_clock=40, white_inc=1, black_inc=1)

# ---- a hit, a miss and a premove
eng, log = engine("main")
board = chess.Board()
r = eng.play(board, lim(), ponder=True)
assert r.ponder is not None, "no reply named to ponder on"
board.push(r.move); time.sleep(4); board.push(r.ponder)
r2 = eng.play(board, lim(), ponder=True)                      # a hit
board.push(r2.move); time.sleep(4)
board.push(next(m for m in board.legal_moves if m != r2.ponder))
r3 = eng.play(board, lim(), ponder=True)                      # a miss
assert r3.ponder is not None, "no reply named to ponder on after the miss"
board.push(r3.move); board.push(r3.ponder)                    # a premove, no thinking time at all
t0 = time.time()
r4 = eng.play(board, lim(), ponder=True)
took = time.time() - t0
eng.quit()
text = open(log, encoding="utf-8").read()
assert text.count("# ponderhit") >= 2 and "# ponder miss" in text, "a hit or the miss never happened"
assert took < 8, f"a ponderhit that arrived with the ponder took {took:.1f}s to answer"
print("\n".join(l for l in text.splitlines() if "carried" in l))
print(f"premove answered in {took:.1f}s")

# ---- a 5 MB ceiling is about 590 nodes, so the ponder has to stop by itself and the game go on
eng, log = engine("ceiling", PONDER_MEM_GB="0.005")
board = chess.Board(MIDDLEGAME)
r = eng.play(board, lim(), ponder=True)
board.push(r.move)
deadline = time.time() + 120
while time.time() < deadline and "memory ceiling" not in open(log, encoding="utf-8").read():
    time.sleep(0.5)
board.push(r.ponder)
r2 = eng.play(board, lim(), ponder=True)                      # a hit on a full tree
eng.quit()
text = open(log, encoding="utf-8").read()
stopped = [l for l in text.splitlines() if "memory ceiling" in l]
assert stopped, "the ponder never reached a 5 MB ceiling in two minutes"
sizes = [int(l.split(" tree ")[1].split()[0]) for l in text.splitlines() if " tree " in l]
assert sizes and max(sizes) <= int(0.005e9 / 8500), f"a tree grew past the ceiling: {sizes}"
print(stopped[0].split("# ")[-1])
print(f"trees after each move: {sizes} nodes, ceiling {int(0.005e9 / 8500)}")
print("pass: a hit, a miss, a premove and a full ceiling, all answered")
