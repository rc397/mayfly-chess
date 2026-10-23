"""Where a search visit's time goes, with the compiled pick. Only writes one results file.

Runs v4's sealed search on v4's weights, 9 positions at 1,600 visits, then today's defaults at 400 visits.
For each it reports the plain rate, then exclusive phase timers. Then cProfile over three v4 searches,
then the network on its own (fp32 weights under autocast, then FAST_EVAL with a CUDA graph for
reference, on a separate copy of the network loaded last).

  env: P_OUT=<data>/profile_results.txt  P_CODE=chess/  P_SUITE=<P_CODE>/blunder_suite.json
       P_VB=1600  P_VS=400  P_NP=9  P_LIMIT=900"""
import collections, cProfile, faulthandler, io, json, os, pstats, statistics, subprocess, sys, time, traceback

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402
OUT = E("P_OUT", paths.under("profile_results.txt"))
CODE = E("P_CODE", HERE)
SUITE = E("P_SUITE", os.path.join(CODE, "blunder_suite.json"))
VB, VS, NP = int(E("P_VB", 1600)), int(E("P_VS", 400)), int(E("P_NP", 9))
sys.path.insert(0, CODE)
import chess, numpy as np, torch
import mcts as M
from mcts import MCTS, Node, pinned_search
from conversion_suite import load

perf = time.perf_counter
_FH = None


def _watchdog(limit=float(os.environ.get("P_LIMIT", 900))):
    time.sleep(limit)
    try:
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%H:%M:%S") + " WATCHDOG: stopped after %d s\n" % limit)
    finally:
        os._exit(3)


import threading
threading.Thread(target=_watchdog, daemon=True).start()


def log(s=""):
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(s + "\n")


FENS = [p["fen"] for p in json.load(open(SUITE))[:8]]
FENS.append(chess.STARTING_FEN)
FENS = FENS[-NP:]

# ---------------------------------------------------------------- exclusive timers
EXCL, INCL, CNT = collections.defaultdict(float), collections.defaultdict(float), collections.Counter()
STACK, ON, DEPTH = [], [False], []
ORIG = []


def timed(owner, name, label=None):
    f = getattr(owner, name)
    label = label or name

    def w(*a, **k):
        if not ON[0]:
            return f(*a, **k)
        fr = [perf(), 0.0]
        STACK.append(fr)
        try:
            r = f(*a, **k)
            if label == "_descend" and r is not None:
                DEPTH.append(len(r[0]))
            return r
        finally:
            el = perf() - fr[0]
            STACK.pop()
            INCL[label] += el
            EXCL[label] += el - fr[1]
            CNT[label] += 1
            if STACK:
                STACK[-1][1] += el
    ORIG.append((owner, name, f))
    setattr(owner, name, w)


for nm in ("search", "_descend", "_pick_fast", "_pick", "_gather", "_evaluate", "_prepare", "_dispatch",
           "_collect", "_expand", "_backup", "_unwind", "_choose", "_tb_value", "_repeats", "_cache_key"):
    timed(MCTS, nm)
timed(Node, "__init__", "Node.__init__")
timed(chess.Board, "copy", "Board.copy")
timed(chess.Board, "push", "Board.push")
timed(M, "encode_keys")
timed(M, "policy_index")


def unpatch():
    for owner, name, f in reversed(ORIG):
        setattr(owner, name, f)


def run(model, opts, visits, fens):
    done = secs = 0.0
    st = collections.Counter()
    t = None
    for f in fens:
        t = MCTS(model, device="cuda", visits=visits)
        if opts is not None:
            t.apply(opts)
        t.open_temp = 0.0
        t0 = perf()
        mv, info = t.search(chess.Board(f), visits=visits, reuse=False)
        secs += perf() - t0
        done += info["visits"]
        for k in ("evals", "batches", "cache_hits", "collisions"):
            st[k] += t.stats.get(k, 0)
    return done, secs, st, t


def table(title, done, secs, st):
    log(f"== {title}: {done:.0f} visits in {secs:.2f} s = {done / secs:.1f} visits/s (timers on)")
    log(f"   evals {st['evals']}, batches {st['batches']} ({st['evals'] / max(st['batches'], 1):.1f} a batch), "
        f"cache hits {st['cache_hits']}, collisions {st['collisions']}, mean descent depth "
        f"{(sum(DEPTH) / len(DEPTH)) if DEPTH else 0:.2f} over {len(DEPTH)} leaf walks")
    tot = sum(EXCL.values())
    log(f"   {'part':<16}{'calls':>9}{'excl s':>9}{'share':>8}{'us/visit':>10}{'incl s':>9}")
    for k, v in sorted(EXCL.items(), key=lambda kv: -kv[1]):
        log(f"   {k:<16}{CNT[k]:>9}{v:>9.3f}{100 * v / tot:>7.1f}%{1e6 * v / done:>10.1f}{INCL[k]:>9.3f}")
    log(f"   (sum of exclusive times {tot:.2f} s against {secs:.2f} s of searching)")
    log()


def reset():
    EXCL.clear(); INCL.clear(); CNT.clear(); DEPTH.clear()


_FH = open(OUT + ".fault", "w"); faulthandler.enable(_FH)
log(time.strftime("%m-%d %H:%M") + " start, native pick loaded: " + str(M.NATIVE_PICK is not None))
log("torch " + torch.__version__ + ", " + torch.cuda.get_device_name(0))
try:
    log("nvidia-smi: " + subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
except Exception as e:
    log("nvidia-smi failed " + repr(e))
log()

m = load("mayfly_v4.pt")
V4 = pinned_search("mayfly_v4.pt")
w = MCTS(m, device="cuda", visits=64); w.apply(V4); w.open_temp = 0.0
w.search(chess.Board(FENS[0]), visits=64, reuse=False)          # warm the GPU

# ---------------------------------------------------------------- v4 at 1,600
done, secs, st, _ = run(m, V4, VB, FENS)
log(f"v4 seal, 1,600 visits, timers off: {done / secs:.1f} visits/s ({secs:.2f} s)")
reset(); ON[0] = True
done, secs, st, last_v4 = run(m, V4, VB, FENS)
ON[0] = False
table("v4 seal, 1,600 visits", done, secs, st)

# ---------------------------------------------------------------- the defaults at 400
done, secs, st, _ = run(m, None, VS, FENS)
log(f"v5 defaults, 400 visits, timers off: {done / secs:.1f} visits/s ({secs:.2f} s)")
reset(); ON[0] = True
done, secs, st, _ = run(m, None, VS, FENS)
ON[0] = False
table("v5 defaults, 400 visits", done, secs, st)

# ---------------------------------------------------------------- cProfile, v4
unpatch()
pr = cProfile.Profile()
pr.enable()
done, secs, st, _ = run(m, V4, VB, FENS[:3])
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(40)
log(f"== cProfile, v4 seal, 3 positions at 1,600 ({done:.0f} visits in {secs:.2f} s under the profiler)")
log("\n".join(l for l in s.getvalue().splitlines() if l.strip())[:9000])
log()

# ---------------------------------------------------------------- the network alone
nodes, seen, q = [], set(), [last_v4.root]
while q and len(nodes) < 128:
    nd = q.pop(0)
    if nd is None or nd.key in seen:
        continue
    seen.add(nd.key)
    if nd.term is None:
        nodes.append(Node(nd.board.copy(stack=False), last_v4.draw_plies))
    if nd.kids:
        q.extend(k for k in nd.kids if k is not None)
log(f"== the network alone: {len(nodes)} positions from the last v4 tree, medians of 30 passes")


def net_alone(t, ns, reps=30):
    prep = t._prepare(ns)
    for _ in range(3):
        t._collect(t._dispatch(prep))
    a, b, c, p = [], [], [], []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = perf(); h = t._dispatch(prep); t1 = perf()
        torch.cuda.synchronize(); t2 = perf()
        t._collect(h); t3 = perf()
        a.append(t1 - t0); b.append(t2 - t1); c.append(t3 - t2)
    for _ in range(10):
        t0 = perf(); t._prepare(ns); p.append(perf() - t0)
    md = statistics.median
    tot = md([x + y + z for x, y, z in zip(a, b, c)])
    return (f"prepare {1e3 * md(p):6.2f} ms, dispatch {1e3 * md(a):6.2f}, wait {1e3 * md(b):6.2f}, "
            f"collect {1e3 * md(c):5.2f}; dispatch to answer {1e3 * tot:6.2f} ms = "
            f"{1e3 * tot / len(ns):5.3f} ms a position")


tb = MCTS(m, device="cuda", visits=64); tb.apply(V4)
for k in (16, 32, 48, 96, 128):
    if k <= len(nodes):
        tb.batch = k
        log(f"   bot mode (fp32 weights, autocast), batch {k:>3}: " + net_alone(tb, nodes[:k]))

del last_v4, tb, w
import gc; gc.collect(); m.cpu(); torch.cuda.empty_cache()
log(time.strftime("%H:%M:%S") + " loading the second copy")
try:
    m2 = load("mayfly_v4.pt")
    log(f"   (second copy for FAST_EVAL is a separate object: {m2 is not m})")
    if m2 is not m:
        tf = MCTS(m2, device="cuda", visits=64); tf.apply(V4)
        tf.fast_eval = True; tf.cuda_graph = True
        for k in (48, 96, 128):
            if k <= len(nodes):
                tf.batch = k
                log(f"   FAST_EVAL + CUDA graph,          batch {k:>3}: " + net_alone(tf, nodes[:k]))
except Exception:
    log("FAST_EVAL part failed: " + traceback.format_exc()[-1500:])
log()
log(time.strftime("%m-%d %H:%M") + " ALL DONE")
