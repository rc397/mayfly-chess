"""Label self-play positions with deep Stockfish action values for training (route 1).

Each position's top MULTIPV moves (0 means every legal move) are scored by one Stockfish per core, with
a budget per position of POSITION_SECONDS, POSITION_NODES or POSITION_DEPTH. MultiPV shares one search
tree across the moves, so a 15 s budget per position searches far deeper than ChessBench's 50 ms per
move. Labelling the top moves rather than all of them gives the moves that decide a position a deeper
search (measured, the top five reached about seven plies deeper than every move at the same budget).
Rows are written as 40-byte action-value records on ChessBench's win-probability curve.

With HARD=1 a second job relabels MayFly's own mistakes deeper (see hard_positions.py).

    python chess/deep_label.py [POSITIONS_FILE]
    WATCH=1 keeps it running beside the generator, picking up new positions
    env WORKERS=26, POSITION_SECONDS=15, MULTIPV=10, OUT (shard folder), PREFIX=dp,
        POSITION_NODES=0 (a node budget instead), POSITION_DEPTH=0 (a depth for every line instead),
        DEPTH_NODE_CAP=2000000000, STOCKFISH (else label_stockfish_path.txt, else stockfish_path.txt),
        WORKERS_FILE, IN (positions file), STATE (place file),
        HARD=1, HARD_NODES=24000000, HARD_SHARE=0.3, HARD_CHUNK, CHECK_NODES,
        HARD_IN, CRIT_IN, HARD_STATE, CRIT_STATE (beside STATE), HARD_OUT=OUT
"""
import math, os, sys, json, time, queue as _queue, multiprocessing as mp
# One BLAS thread, set before numpy loads. OpenBLAS reserves memory for every thread it might use,
# about 0.78 GB of committed memory per worker on a 28-thread machine, and this file never multiplies
# a matrix. The workers inherit the variable.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import chess, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
IN = E("IN", os.path.join(paths.SELFPLAY, "positions.txt"))
OUT = E("OUT", paths.SHARDS)
PREFIX = E("PREFIX", "dp")
WORKERS = int(E("WORKERS", 26))
POSITION_SECONDS = float(E("POSITION_SECONDS", 15))
MULTIPV = int(E("MULTIPV", 10))         # how many of a position's best moves are labelled, 0 is every legal move
STATE = E("STATE", os.path.join(paths.SELFPLAY, ".labelled.json"))
# Engine errors are logged here, each worker's first 20 a batch with the error and the position. Repeated
# positions are blanked before the workers see them and also counted as failed, but are not logged.
ERRORS = E("ERRORS", os.path.join(os.path.dirname(STATE), "label_errors.log"))
SHARD = int(E("SHARD", 4_000_000))
# How long the parent waits with no worker reporting before it decides an engine has wedged. Workers
# report at least once a minute.
STALL_MINUTES = float(E("STALL_MINUTES", 15))
# A node budget instead of a clock. A time limit on a busy machine quietly gives shallower labels, while
# a node budget labels fewer positions but never worse ones. About 6M nodes matches 15 s on a 24-worker
# machine. 0, the default, keeps the clock.
POSITION_NODES = int(E("POSITION_NODES", 0))
# A depth instead of a budget. POSITION_DEPTH > 0 searches every line to that depth whatever it costs, with
# DEPTH_NODE_CAP only as a backstop. It replaces the clock and POSITION_NODES for the main job. Ten lines to
# depth 30 took 44M to 137M nodes on hard positions, a few minutes on one core.
POSITION_DEPTH = int(E("POSITION_DEPTH", 0))
DEPTH_NODE_CAP = int(E("DEPTH_NODE_CAP", 2_000_000_000))
# Windows 11 keeps background work like this on the efficiency cores of a hybrid CPU. HIGH_QOS=1 has every worker
# and its Stockfish opt out of power throttling (SetProcessInformation with ProcessPowerThrottling, per process,
# no system setting changes), which gave several times the throughput in one measurement. Off by default.
HIGH_QOS = E("HIGH_QOS", "0") == "1"
# The second job relabels MayFly's own mistakes deeper (hard_positions.py says which). After each main batch,
# positions where MayFly's self-play move is rated well below the best go to HARD_IN, and only-move positions to
# CRIT_IN. They are labelled at HARD_NODES, with MayFly's own move valued too, into HARD_PREFIX shards that the
# trainer does not read by default. HARD_SHARE of the labelling time goes to them while the main queue has work,
# all of it when the main queue is empty, and crit lines only when both are empty. Off unless HARD=1.
# With HARD_LABEL=0 the finder still writes HARD_IN and CRIT_IN, but the batches are left for kaggle_label.py.
HARD = E("HARD", "0") == "1"
HARD_LABEL = E("HARD_LABEL", "1") == "1"
_DIR = os.path.dirname(STATE)
HARD_IN = E("HARD_IN", os.path.join(_DIR, "hard_queue.txt"))
CRIT_IN = E("CRIT_IN", os.path.join(_DIR, "crit_queue.txt"))
HARD_STATE = E("HARD_STATE", os.path.join(_DIR, ".hard_labelled.json"))
CRIT_STATE = E("CRIT_STATE", os.path.join(_DIR, ".crit_labelled.json"))
HARD_OUT = E("HARD_OUT", OUT)
HARD_PREFIX = E("HARD_PREFIX", "dh")
HARD_NODES = int(E("HARD_NODES", 24_000_000))
HARD_SHARE = float(E("HARD_SHARE", 0.3))
# For a chain of labelling stages at increasing depth. With HANDOFF set, every finished main batch is appended to
# that file for the next stage, MayFly's mistakes first (lines tagged deep or check, as "FEN|uci|kind|best"), then
# the rest of the batch as plain FENs.
HANDOFF = E("HANDOFF", "")
BASE_NODES = POSITION_NODES or 6_000_000          # 15 s is about 6M nodes on a busy 24-worker machine
# a hard batch lasts about as long as a main one, four times the search on a quarter of the positions
HARD_CHUNK = int(E("HARD_CHUNK", max(WORKERS, int(int(E("CHUNK", 20000)) * BASE_NODES / HARD_NODES))))
# a check measures one move alone, at a quarter of the usual budget
CHECK_NODES = int(E("CHECK_NODES", max(1, BASE_NODES // 4)))
MAIN_JOB = {"name": "main", "out": OUT, "prefix": PREFIX, "state": STATE, "nodes": POSITION_NODES,
            "seconds": POSITION_SECONDS, "depth": POSITION_DEPTH}
HARD_JOB = {"name": "hard", "out": HARD_OUT, "prefix": HARD_PREFIX, "state": HARD_STATE, "nodes": HARD_NODES,
            "seconds": POSITION_SECONDS, "src": HARD_IN, "depth": 0}
CRIT_JOB = dict(HARD_JOB, name="crit", state=CRIT_STATE, src=CRIT_IN)


def hand_off(todo, found):
    """Append one finished main batch to HANDOFF, the mistakes first (lines that arrived tagged deep or check, then
    the ones this batch's finder tagged), then every other position as a plain FEN. Returns (mistakes, others)."""
    tagged = [l for l in todo if l and ("|deep|" in l or "|check|" in l)]
    mine = [l for l in (found or []) if "|deep|" in l or "|check|" in l]
    skip = {l.split("|")[0] for l in tagged + mine}
    plain, seen = [], set()
    for l in todo:
        fen = l.split("|")[0].strip() if l else ""
        if fen and fen not in skip and fen not in seen:
            seen.add(fen)
            plain.append(fen)
    out = tagged + mine + plain
    if out:
        with open(HANDOFF, "a", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return len(tagged) + len(mine), len(plain)


def high_qos(pid):
    """Ask Windows not to power-throttle process `pid`. Only under HIGH_QOS=1 on Windows, best effort."""
    if not HIGH_QOS or os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes as wt

        class State(ctypes.Structure):
            _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG), ("StateMask", wt.ULONG)]
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wt.HANDLE
        h = k32.OpenProcess(0x0200 | 0x1000, False, int(pid))     # set information, query limited information
        if h:
            st = State(1, 1, 0)                                     # version 1, execution speed controlled, never throttled
            k32.SetProcessInformation(h, 4, ctypes.byref(st), ctypes.sizeof(st))    # 4 is ProcessPowerThrottling
            k32.CloseHandle(h)
    except Exception:
        pass


def stockfish_path():
    """The labeller's Stockfish, from STOCKFISH, else label_stockfish_path.txt beside this file, else
    stockfish_path.txt. The labelling engine can be upgraded without changing the one the ladders measure
    against, which must stay the same from version to version."""
    if E("STOCKFISH"):
        return E("STOCKFISH")
    for name in ("label_stockfish_path.txt", "stockfish_path.txt"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            return open(p).read().strip()
    raise SystemExit("no Stockfish named: set STOCKFISH or write label_stockfish_path.txt")


K_CHESSBENCH = 0.00368208     # ChessBench's win-probability curve, which the av corpus uses


def win_prob(score, turn):
    """A Stockfish score as the side to move's win probability on ChessBench's curve.

    1 / (1 + exp(-0.00368208 * cp)), so these rows mean the same as the ChessBench rows they are mixed
    with. Not pack.py's 400-Elo value curve, which differs a lot (+300 cp is 0.849 there, 0.752 here).
    A mate is stored as certain, as ChessBench does."""
    s = score.pov(turn)
    if s.is_mate():
        return 1.0 if s.mate() > 0 else 0.0
    return 1.0 / (1.0 + math.exp(-K_CHESSBENCH * s.score()))


def worker(wid, lines, seconds, out_q, nodes=None, depth=0):
    """One Stockfish on one core, labelling its stripe of the positions.

    `nodes`, a job's own budget, replaces POSITION_NODES. A hard line (hard_positions.py, "FEN|uci|kind|best")
    also gets MayFly's own move valued. A deep or crit line labels its top moves at the job's budget, plus that
    move at a quarter of it if the top moves leave it out. A check line first measures that move alone at
    CHECK_NODES and is labelled deep only if the move is GAP below the first label's best, otherwise it is
    consumed with nothing written."""
    import chess.engine
    from encode import board_to_ids, clock_and_ep   # not chess_net, which imports torch
    from moves import move_to_idx
    from pack_av import DT_AV, encode_av
    from hard_positions import GAP, parse
    sf = stockfish_path()
    budget = POSITION_NODES if nodes is None else int(nodes)
    limit = chess.engine.Limit(nodes=budget) if budget > 0 else chess.engine.Limit(time=seconds)
    alone = chess.engine.Limit(nodes=max(1, budget // 4)) if budget > 0 else chess.engine.Limit(time=seconds / 4)
    if depth > 0:                       # every line to that depth, the node cap only a backstop
        limit = chess.engine.Limit(depth=depth, nodes=DEPTH_NODE_CAP)
        alone = chess.engine.Limit(depth=depth, nodes=DEPTH_NODE_CAP)
    check = chess.engine.Limit(nodes=CHECK_NODES)
    eng = chess.engine.SimpleEngine.popen_uci(sf)
    eng.configure({"Threads": 1, "Hash": 128})
    high_qos(os.getpid())
    high_qos(eng.transport.get_pid())
    rows, done, failed = [], 0, 0
    logged = 0                                      # errors written to ERRORS by this worker this batch
    last_put = time.time()
    for line in lines:
        fen, uci, kind, best = parse(line) if "|" in line else (line, "", "", None)
        try:
            b = chess.Board(fen)
        except Exception:
            failed += 1
            continue
        legal = list(b.legal_moves)
        if not legal or b.is_game_over():
            failed += 1                      # consumed with nothing to label, counted rather than lost
            continue
        own = None
        if uci:
            try:
                own = chess.Move.from_uci(uci)
            except ValueError:
                own = None
            if own not in legal:
                own = None
        infos, extra = [], None              # extra is MayFly's own move and its score, when the top moves leave it out
        try:
            cleared = False                  # a check whose move turned out fine is consumed with nothing written
            if kind == "check" and own is not None:
                got = eng.analyse(b, check, root_moves=[own])
                got = got[0] if isinstance(got, list) else got
                cleared = "score" in got and best is not None and best - win_prob(got["score"], b.turn) < GAP
            if not cleared:
                infos = eng.analyse(b, limit,
                                    multipv=min(MULTIPV, len(legal)) if MULTIPV > 0 else len(legal))
                if isinstance(infos, dict):
                    infos = [infos]
                if kind and own is not None and own not in {i["pv"][0] for i in infos if i.get("pv")}:
                    got = eng.analyse(b, alone, root_moves=[own])
                    got = got[0] if isinstance(got, list) else got
                    if "score" in got:
                        extra = (own, got["score"])
        except Exception as e:
            # restart a Stockfish that died, or every position left in this stripe would fail
            failed += 1
            logged += 1
            if logged <= 20:
                try:
                    with open(ERRORS, "a", encoding="utf-8") as fh:
                        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} worker {wid}: {type(e).__name__}: "
                                 f"{str(e)[:200]!r} | {fen}" + "\n")
                except OSError:
                    pass
            try:
                eng.quit()
            except Exception:
                pass
            try:
                eng = chess.engine.SimpleEngine.popen_uci(sf)
                eng.configure({"Threads": 1, "Hash": 128})
                high_qos(eng.transport.get_pid())
            except Exception:
                pass
            continue
        ids, cast, flip = board_to_ids(b)
        hm, ep = clock_and_ep(b, flip)
        bits = (int(cast[0]) | (int(cast[1]) << 1) | (int(cast[2]) << 2) | (int(cast[3]) << 3))
        scored = [(i["pv"][0], i["score"]) for i in infos if i.get("pv") and "score" in i]
        if extra is not None:
            scored.append(extra)
        for mv, score in scored:
            row = np.zeros(1, DT_AV)[0]
            encode_av(ids, [bits & 1, (bits >> 1) & 1, (bits >> 2) & 1, (bits >> 3) & 1],
                      move_to_idx(mv, flip), win_prob(score, b.turn),
                      (1 if mv.promotion else 0) | 6, row, hm=hm, ep=ep)
            rows.append(row.tobytes())
        done += 1
        # report every few hundred rows or every minute, so a quiet worker means a wedged engine
        if len(rows) >= 500 or time.time() - last_put > 60:
            out_q.put((wid, done, failed, b"".join(rows)))
            rows, done, failed = [], 0, 0
            last_put = time.time()
    eng.quit()
    out_q.put((wid, done, failed, b"".join(rows)))
    out_q.put((wid, None, None, None))


def label_batch(todo, start, job=None, keep=None):
    """Label a list of FENs across WORKERS processes. Returns (positions, action values, failed, seconds).

    `job` (MAIN_JOB by default) names the shards, the place file and the budget. `keep`, a list, also collects
    the rows written, which the main job hands to hard_positions.find."""
    from pack_av import ShardWriter, REC_AV
    job = job or MAIN_JOB
    sw = ShardWriter(job["out"], job["prefix"], SHARD)
    q = mp.Queue(maxsize=WORKERS * 2)
    procs = []
    for k in range(WORKERS):
        p = mp.Process(target=worker, args=(k, todo[k::WORKERS], job["seconds"], q, job["nodes"],
                                                    job.get("depth", 0)), daemon=True)
        p.start()
        procs.append(p)
    note_workers([p.pid for p in procs])

    live, done, failed, pairs = WORKERS, 0, 0, 0
    per = [0] * WORKERS            # positions each worker has reported, in its own stripe's order
    t0 = last = last_progress = time.time()
    while live:
        try:
            wid, d, f, blob = q.get(timeout=60)
            last_progress = time.time()
        except _queue.Empty:
            # workers that died without sending their sentinel must not strand this loop
            alive = sum(1 for p in procs if p.is_alive())
            if alive == 0:
                print(f"  every worker is gone with {live} still unaccounted for, "
                      f"ending this batch early", flush=True)
                break
            # alive is not working, a worker blocked inside a hung Stockfish never reports
            idle = time.time() - last_progress
            if idle > STALL_MINUTES * 60:
                print(f"  nothing reported in {idle/60:.0f} minutes with {alive} worker(s) still "
                      f"alive: an engine has wedged. Ending the batch.", flush=True)
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                break
            continue
        if d is None:
            live -= 1
            continue
        done += d
        failed += f
        per[wid] += d + f
        if blob:
            sw.write_raw(blob)
            pairs += len(blob) // REC_AV
            if keep is not None:
                keep.append(blob)
        if time.time() - last > 120:
            last = time.time()
            el = time.time() - t0
            print(f"  {done:,}/{len(todo):,} positions, {pairs:,} action values, "
                  f"{done/max(el,1)*86400:,.0f} a day, {failed} failed, {live} workers",
                  flush=True)
            if start is not None:
                save_state(start + done + failed, job["state"])
    sw.close()
    for p in procs:
        p.join(timeout=10)
    # A stripe cut short is finished here, not skipped, since workers take interleaved stripes and what a
    # cut-short batch consumed is not a single place in the file. The rest of every unfinished stripe goes
    # to a fresh pool, less the first unreported position, which may be what wedged it and counts as failed.
    # If nothing moved at all (no engine started, say), the place stays where it was.
    cut = [k for k in range(WORKERS) if per[k] < len(todo[k::WORKERS])]
    if start is None or (cut and not (done or failed)):
        return done, pairs, failed, time.time() - t0
    if cut:
        rest = [x for k in cut for x in todo[k::WORKERS][per[k] + 1:]]
        print(f"  {len(cut)} stripe(s) cut short, labelling their other {len(rest):,} "
              f"positions with a fresh pool", flush=True)
        d2, p2, f2, _ = label_batch(rest, None, job, keep) if rest else (0, 0, 0, 0)
        done, pairs, failed = done + d2, pairs + p2, failed + f2 + len(cut)
    save_state(start + len(todo), job["state"])
    return done, pairs, failed, time.time() - t0


def save_state(done, path=None):
    """Save the place in the file atomically, through a temporary file, so a kill cannot leave it empty.
    `path` is a job's own place file."""
    path = path or STATE
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"done": done, "at": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=1)
    os.replace(tmp, path)


WORKERS_FILE = E("WORKERS_FILE", os.path.join(HERE, ".pid_deep_label_workers.txt"))


def note_workers(pids):
    """Record this batch's workers and their labeller for reap_orphans. Not the pid file, which the
    supervisor rewrites with the new labeller's id before it can be read."""
    try:
        with open(WORKERS_FILE, "w") as fh:
            json.dump({"parent": os.getpid(), "workers": pids}, fh)
    except OSError:
        pass


def orphans_of(last, procs, me):
    """Which of `procs` [(pid, command line)] are the last labeller's workers, named in its note and
    still naming it as their parent, which a recycled process id cannot do."""
    try:
        parent, mine = int(last["parent"]), {int(w) for w in last["workers"]}
    except (KeyError, TypeError, ValueError):
        return []
    if parent == me:
        return []
    tag = f"parent_pid={parent},"
    return sorted(pid for pid, cmd in procs
                  if pid in mine and pid != me and "multiprocessing.spawn" in (cmd or "")
                  and tag in (cmd or ""))


def reap_orphans():
    """Kill worker processes left behind by a previous labeller that did not exit cleanly.

    On Windows a `taskkill /F` on the parent leaves its multiprocessing workers running, still using
    every core with the old settings. Only this labeller's own workers are reaped, those recorded by
    note_workers whose command line still names that parent, so other multiprocessing programs (such
    as lichess-bot) are left alone. The worker's Stockfish goes with it (/T). This runs at startup,
    since a `finally` does not run when the process is killed.
    """
    import subprocess as _sp
    try:
        last = json.load(open(WORKERS_FILE))
    except Exception:
        return                                  # nothing written down, so nothing known to be ours
    ps = ("Get-CimInstance Win32_Process -Filter \"Name LIKE 'py%'\" | "
          "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress")
    try:
        out = _sp.run(["powershell", "-NoProfile", "-Command", ps],
                      capture_output=True, text=True, timeout=90).stdout
        rows = json.loads(out) if out.strip() else []
        rows = rows if isinstance(rows, list) else [rows]
        procs = [(int(r["ProcessId"]), r.get("CommandLine")) for r in rows]
    except Exception as e:
        print(f"  could not look for orphaned workers: {type(e).__name__}: {e}", flush=True)
        return
    killed = orphans_of(last, procs, os.getpid())
    for pid in killed:
        _sp.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30)
    if killed:
        print(f"  reaped {len(killed)} orphaned worker(s) of labeller {last.get('parent')}",
              flush=True)


if __name__ == "__main__":
    reap_orphans()
    from pidfile import claim
    claim("deep_label")
    mp.freeze_support()
    src = sys.argv[1] if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) else IN
    # WATCH keeps it running beside a generator, picking up positions as they are appended
    watch = E("WATCH", "0") == "1"
    poll = int(E("POLL", 300))
    budget = (f"depth {POSITION_DEPTH} on every line" if POSITION_DEPTH > 0 else
              f"{POSITION_NODES:,} nodes a position" if POSITION_NODES > 0 else f"{POSITION_SECONDS}s a position")
    print(f"{WORKERS} workers, {budget}, top {MULTIPV or 'every move'}"
          + (f", watching {os.path.basename(src)} every {poll}s" if watch else ""), flush=True)

    if HARD and not HARD_LABEL:
        print(f"hard positions found after each batch and written to {HARD_IN} and {CRIT_IN} for Kaggle; "
              f"none labelled here", flush=True)
    elif HARD:
        print(f"hard positions on: {HARD_NODES:,} nodes a position, {100 * HARD_SHARE:.0f}% of the time while the "
              f"main queue has work, batches of {HARD_CHUNK:,}, checks at {CHECK_NODES:,} nodes", flush=True)

    def pending(path, state, cap):
        """(lines, place, todo, repeats) for one queue, the next `cap` lines past its place."""
        try:
            fens = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
        except FileNotFoundError:
            fens = []
        try:
            start = json.load(open(state)).get("done", 0)
        except FileNotFoundError:
            start = 0
        except ValueError as e:
            # an unreadable place is not line zero, so stop rather than relabel everything
            raise SystemExit(f"{state} is unreadable ({e}); not relabelling from line 0")
        todo = fens[start:start + cap]
        # One label per position. Self-play games repeat the opening positions, so a position already
        # earlier in the file (move number aside) is blanked, counted as failed and never labelled twice.
        # A hard line is keyed on its position.
        key = lambda f: f.split("|", 1)[0].rsplit(" ", 1)[0]
        known = {key(f) for f in fens[:start]} if todo else set()
        for i, f in enumerate(todo):
            if key(f) in known:
                todo[i] = ""
            known.add(key(f))
        # repeats are counted in "failed", so they are reported separately to avoid reading them as engine errors
        return fens, start, todo, sum(1 for f in todo if not f)

    def queue_hard(lines):
        """Append the finder's lines to the hard and crit queues, on disk before the next batch reads them."""
        for path, part in ((HARD_IN, [l for l in lines if "|crit|" not in l]),
                           (CRIT_IN, [l for l in lines if "|crit|" in l])):
            if part:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(part) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())

    total_pos = total_pairs = 0
    spent = {"main": 0.0, "hard": 0.0}     # this run's labelling time, for HARD_SHARE
    t_start = time.time()
    while True:
        fens, start, todo, repeats = pending(src, STATE, int(E("CHUNK", 20000)))
        job = None
        if HARD and HARD_LABEL:
            hard = pending(HARD_IN, HARD_STATE, HARD_CHUNK)
            behind = spent["hard"] < HARD_SHARE * (spent["hard"] + spent["main"])
            if hard[2] and (not todo or behind):
                job = ("hard", HARD_JOB) + hard
            elif not todo:
                crit = pending(CRIT_IN, CRIT_STATE, HARD_CHUNK)
                if crit[2]:
                    job = ("crit", CRIT_JOB) + crit
        if job is None and todo:
            job = ("main", MAIN_JOB, fens, start, todo, repeats)
        if job is None:
            if not watch:
                print("nothing to label")
                break
            # print while idle, since the supervisor reads a long silence as a wedged engine

            print(f"  idle: all {len(fens):,} positions labelled"
                  + (", and every hard one" if HARD and HARD_LABEL else "") + f", looking again in {poll}s", flush=True)
            time.sleep(poll)
            continue
        name, spec, jfens, jstart, jtodo, jrep = job
        print("")
        if name == "main":
            print(f"{len(jfens):,} positions available, {jstart:,} done, labelling {len(jtodo):,} "
                  f"({jrep:,} of them repeats, skipped and counted failed; "
                  f"{len(jtodo)*POSITION_SECONDS/WORKERS/3600:.1f}h at this rate)", flush=True)
        else:
            print(f"{name}: {len(jfens):,} queued, {jstart:,} done, labelling {len(jtodo):,} at "
                  f"{HARD_NODES:,} nodes ({jrep:,} repeats skipped)", flush=True)
        keep = [] if (HARD and name == "main") else None
        found = []
        d, pr, fl, el = label_batch(jtodo, jstart, spec, keep)
        if name != "crit":                 # crit runs only when nothing else waits, so it takes no share
            spent[name] += el
        total_pos += d
        total_pairs += pr
        print(f"  {'' if name == 'main' else name + ' '}batch done: {d:,} positions, {pr:,} action values, "
              f"{fl} failed, {el/3600:.2f}h  |  session total {total_pos:,} positions, "
              f"{total_pairs:,} action values in {(time.time()-t_start)/3600:.2f}h", flush=True)
        if keep is not None:
            from hard_positions import find
            from pack_av import DT_AV as _DT_AV
            try:
                lines, counts = find(jfens, jstart, jstart + len(jtodo), np.frombuffer(b"".join(keep), _DT_AV))
                found = lines
                queue_hard(lines)
                print(f"  hard positions: {counts['deep']:,} mistakes, {counts['check']:,} to check, "
                      f"{counts['crit']:,} only-move, of {counts['scanned']:,} labelled", flush=True)
            except Exception as e:            # a finder fault must never stop the labeller itself
                print(f"  hard positions not found this batch: {type(e).__name__}: {e}", flush=True)
        if HANDOFF and name == "main":
            try:
                n_m, n_p = hand_off(jtodo, found)
                print(f"  handed on: {n_m:,} mistakes first, then {n_p:,} positions -> {os.path.basename(HANDOFF)}",
                      flush=True)
            except Exception as e:            # a hand-off fault must never stop the labeller itself
                print(f"  not handed on this batch: {type(e).__name__}: {e}", flush=True)
        if not watch:
            break
    print("")
    print(f"{total_pos:,} positions, {total_pairs:,} action values, "
          f"{(time.time()-t_start)/3600:.2f}h")
