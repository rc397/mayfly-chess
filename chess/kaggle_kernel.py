# Deep labels for a batch of hard positions on one Kaggle session. kaggle_label.py writes a copy of this file with
# its batch in LINES (or a dataset file named in LINES_FILE) and its id in BATCH, and pushes it as a private notebook.
# colab_worker.py runs the same file on Colab.
#
# It downloads Stockfish 19 and runs one single-thread engine per core the session really has (cgroup quota and
# affinity respected), labelling hard_positions.py lines ("FEN|uci|kind|best") as deep_label.py's second job does.
# A check line first measures MayFly's own move alone, and if that move is within GAP of the best the position is
# labelled as a normal one. Normal positions get the MULTIPV best moves to at least MIN_DEPTH, and mistakes the
# MISTAKE_MULTIPV best to at least MISTAKE_DEPTH, plus MayFly's own move when the top moves leave it out. Each search
# runs its node budget first, then a second search to the minimum depth, which starts where the first stopped since
# the hash still holds it. NODE_CAP_X only stops a search no session could finish, which is then marked "capped".
#
# It writes what Stockfish said, not training rows. results.jsonl has a line per position with its index and each
# move's UCI, score (centipawns or mate, side to move) and depth, and kaggle_label.py turns them into rows with the
# repo's encoder. Kaggle stops a notebook at about nine hours, so at TIME_BUDGET the engines stop, the results written
# so far stay, the summary is saved, and unreached positions go back on the queue. Time left over after the batch is
# spent deepening the shallowest positions one ply at a time, each pass writing another record ("deepened", its
# target depth), and the deepest record of each position is kept.
import json
import multiprocessing as mp
import os
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request


def cpus():
    """The cores this session really has, the scheduler's affinity capped by a cgroup CPU quota if one is set."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, n)


def ram_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


BATCH = "__BATCH__"
NODES = 24_000_000
CHECK_NODES = 1_500_000
GAP = 0.05
MULTIPV = 10        # the ten best moves
MIN_DEPTH = 35
MISTAKE_MULTIPV = 5    # MayFly's mistakes get the top five at depth 40 at least
MISTAKE_DEPTH = 40
NODE_CAP_X = 512       # 12.3B nodes at 24M, about a whole session of one core
HASH = 512
WORKERS = cpus()
TIME_BUDGET = 8 * 3600 + 30 * 60
K_CHESSBENCH = 0.00368208
URL = "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz"
LINES = r"""__LINES__""".splitlines()
# A big batch comes as a file, since Kaggle refuses a notebook of over about a megabyte. kaggle_label.py puts the
# batch in its batches dataset as <batch>.txt, attaches the dataset and names the file here instead of filling LINES.
# Kaggle may unpack an uploaded .gz on its side, so the name is looked for with and without .gz, and gzip is detected
# by its first two bytes.
LINES_FILE = "__LINES_FILE__"
INPUT = "/kaggle/input"
if not LINES_FILE.startswith("__"):
    import glob
    import gzip
    _names = [LINES_FILE, LINES_FILE[:-3] if LINES_FILE.endswith(".gz") else LINES_FILE + ".gz"]
    _found = [p for n in _names for p in sorted(glob.glob(os.path.join(INPUT, "**", n), recursive=True))
              if os.path.isfile(p)]
    if not _found:
        raise SystemExit(f"{LINES_FILE} is not under {INPUT}: the dataset was not attached")
    with open(_found[0], "rb") as _fh:
        _raw = _fh.read()
    LINES = (gzip.decompress(_raw) if _raw[:2] == b"\x1f\x8b" else _raw).decode("utf-8").splitlines()
OUT = "/kaggle/working"
SF = None


def find_stockfish():
    urllib.request.urlretrieve(URL, "/tmp/sf.tgz")
    with tarfile.open("/tmp/sf.tgz") as tf:
        tf.extractall("/tmp/sf")
    for root, _, files in os.walk("/tmp/sf"):
        for f in sorted(files):
            p = os.path.join(root, f)
            if "stockfish" in f.lower() and not f.endswith((".nnue", ".md", ".txt", ".tar", ".gz")):
                os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
                try:
                    r = subprocess.run([p], input="uci\nquit\n", capture_output=True, text=True, timeout=60)
                    if "uciok" in r.stdout:
                        return p
                except Exception:
                    pass
    raise SystemExit("no working Stockfish in the download")


def win_prob(score):
    """deep_label.win_prob on a score already from the side to move, ChessBench's curve with a mate as certain."""
    if score.is_mate():
        return 1.0 if score.mate() > 0 else 0.0
    import math
    return 1.0 / (1.0 + math.exp(-K_CHESSBENCH * score.score()))


def pack(move, score, depth):
    if score.is_mate():
        return [move.uci(), None, score.mate(), depth]
    return [move.uci(), score.score(), None, depth]


def hash_mb():
    """HASH MB an engine, or less when the session's memory would not hold one per worker with half of it to spare."""
    ram = ram_mb()
    return min(HASH, max(64, ram // (2 * WORKERS))) if ram else HASH


def machine():
    """What the session runs on, for the summary."""
    info = {"cores": WORKERS, "ram_gb": round(ram_mb() / 1024, 1), "hash_mb": hash_mb()}
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            info["gpu"] = r.stdout.strip().splitlines()[0][:80]
    except (OSError, subprocess.SubprocessError):
        pass
    tpu = [k for k in os.environ if "TPU" in k.upper()]
    if tpu:
        info["tpu_env"] = sorted(tpu)[:6]
    return info


_eng = None


def _engine():
    global _eng
    import chess.engine
    if _eng is None:
        _eng = chess.engine.SimpleEngine.popen_uci(SF)
        _eng.configure({"Threads": 1, "Hash": hash_mb()})
    return _eng


def search(eng, b, k, nodes, root_moves=None, min_depth=None):
    """MultiPV k at `nodes`, then a second search to `min_depth` if any line is short of it.

    The second search is capped at NODE_CAP_X times the budget, and the hash lets it start where the first
    stopped. min_depth defaults to MIN_DEPTH. Returns (the lines, capped)."""
    min_depth = MIN_DEPTH if min_depth is None else min_depth
    import chess.engine
    kw = {"multipv": k}
    if root_moves:
        kw["root_moves"] = root_moves
    infos = eng.analyse(b, chess.engine.Limit(nodes=nodes), **kw)
    infos = infos if isinstance(infos, list) else [infos]
    if min(i.get("depth", 0) for i in infos) >= min_depth:
        return infos, False
    infos = eng.analyse(b, chess.engine.Limit(depth=min_depth, nodes=nodes * NODE_CAP_X), **kw)
    infos = infos if isinstance(infos, list) else [infos]
    return infos, min(i.get("depth", 0) for i in infos) < min_depth


def label(i):
    """One queue line as (index, record or None). A record is {"i", "moves"} or {"i", "cleared"}."""
    global _eng
    import chess
    import chess.engine
    parts = LINES[i].split("|")
    fen = parts[0].strip()
    uci = parts[1].strip() if len(parts) > 1 else ""
    kind = parts[2].strip() if len(parts) > 2 else ""
    try:
        best = float(parts[3]) if len(parts) > 3 and parts[3] else None
    except ValueError:
        best = None
    try:
        b = chess.Board(fen)
        legal = list(b.legal_moves)
        if not legal or b.is_game_over():
            return i, {"i": i, "skipped": True}
        own = chess.Move.from_uci(uci) if uci else None
        if own not in legal:
            own = None
        eng = _engine()
        mistake = kind in ("deep", "check")
        if kind == "check" and own is not None:
            got = eng.analyse(b, chess.engine.Limit(nodes=CHECK_NODES), root_moves=[own])
            if "score" in got and best is not None and best - win_prob(got["score"].pov(b.turn)) < GAP:
                mistake = False             # the move is fine, so label it as a normal position
        k = MISTAKE_MULTIPV if mistake else MULTIPV
        md = MISTAKE_DEPTH if mistake else MIN_DEPTH
        infos, capped = search(eng, b, min(k, len(legal)), NODES, min_depth=md)
        moves, seen = [], set()
        for info in infos:
            if info.get("pv") and "score" in info:
                mv = info["pv"][0]
                seen.add(mv)
                moves.append(pack(mv, info["score"].pov(b.turn), info.get("depth")))
        if kind and own is not None and own not in seen:
            got, cap_own = search(eng, b, 1, max(1, NODES // 4), root_moves=[own], min_depth=md)
            capped = capped or cap_own
            if got and "score" in got[0]:
                moves.append(pack(own, got[0]["score"].pov(b.turn), got[0].get("depth")))
        rec = {"i": i, "moves": moves}
        if mistake:
            rec["mistake"] = True
        if capped:
            rec["capped"] = True
        return i, rec
    except Exception as e:
        try:
            if _eng is not None:
                _eng.quit()
        except Exception:
            pass
        _eng = None
        return i, {"i": i, "error": f"{type(e).__name__}: {str(e)[:120]}"}


def depth_of(rec):
    """A record's depth, the shallowest of its moves."""
    d = [m[3] for m in rec.get("moves") or [] if m[3] is not None]
    return min(d) if d else 0


def deepen(task):
    """Search a labelled position again from scratch to `target`, returning (index, record).

    The same moves label() gave it, the ten best plus MayFly's own when they leave it out, each to the target
    depth (NODE_CAP_X times the budget at most). The record carries "deepened" set to target, and "capped" when
    it fell short."""
    global _eng
    import chess
    import chess.engine
    i, target = task
    parts = LINES[i].split("|")
    uci = parts[1].strip() if len(parts) > 1 else ""
    kind = parts[2].strip() if len(parts) > 2 else ""
    try:
        b = chess.Board(parts[0].strip())
        legal = list(b.legal_moves)
        own = chess.Move.from_uci(uci) if uci else None
        if own not in legal:
            own = None
        eng = _engine()
        infos = eng.analyse(b, chess.engine.Limit(depth=target, nodes=NODES * NODE_CAP_X),
                            multipv=min(MULTIPV, len(legal)))
        infos = infos if isinstance(infos, list) else [infos]
        moves, seen = [], set()
        for info in infos:
            if info.get("pv") and "score" in info:
                seen.add(info["pv"][0])
                moves.append(pack(info["pv"][0], info["score"].pov(b.turn), info.get("depth")))
        if kind and own is not None and own not in seen:
            got = eng.analyse(b, chess.engine.Limit(depth=target, nodes=max(1, NODES // 4) * NODE_CAP_X),
                              multipv=1, root_moves=[own])
            got = got if isinstance(got, list) else [got]
            if got and "score" in got[0]:
                moves.append(pack(own, got[0]["score"].pov(b.turn), got[0].get("depth")))
        rec = {"i": i, "moves": moves, "deepened": target}
        if depth_of(rec) < target:
            rec["capped"] = True
        return i, rec
    except Exception as e:
        try:
            if _eng is not None:
                _eng.quit()
        except Exception:
            pass
        _eng = None
        return i, {"i": i, "error": f"{type(e).__name__}: {str(e)[:120]}", "deepened": target}


def work(t0):
    """The batch, then the time left on its shallowest positions. Returns (positions done, counts)."""
    done, counts, depth = 0, {}, {}
    with open(os.path.join(OUT, "results.jsonl"), "w", encoding="utf-8") as fh:
        pool = mp.Pool(WORKERS)
        try:
            late = False
            for i, rec in pool.imap_unordered(label, range(len(LINES))):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done += 1
                k = "moves" if "moves" in rec else ("cleared" if rec.get("cleared") else
                                                    ("error" if "error" in rec else "skipped"))
                counts[k] = counts.get(k, 0) + 1
                if rec.get("capped"):
                    counts["capped"] = counts.get("capped", 0) + 1
                elif rec.get("moves"):
                    depth[i] = depth_of(rec)
                if done % 100 == 0:
                    print(f"  {done}/{len(LINES)} in {(time.time() - t0) / 3600:.2f}h {counts}", flush=True)
                if time.time() - t0 > TIME_BUDGET:
                    print("  time budget reached, stopping", flush=True)
                    late = True
                    break
            rounds = deeper = 0
            while not late and depth:
                rounds += 1
                tasks = [(i, depth[i] + 1) for i in sorted(depth, key=lambda j: (depth[j], j))]
                print(f"  deepening round {rounds} at {(time.time() - t0) / 3600:.2f}h: {len(tasks)} positions, "
                      f"shallowest now {tasks[0][1] - 1}", flush=True)
                for i, rec in pool.imap_unordered(deepen, tasks):
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    d = depth_of(rec)
                    if "error" in rec or rec.get("capped") or d <= depth.get(i, 0):
                        depth.pop(i, None)    # no deeper this way, so it keeps the record it has

                    else:
                        depth[i] = d
                        deeper += 1
                        if deeper % 100 == 0:
                            print(f"  {deeper} deepened in {(time.time() - t0) / 3600:.2f}h", flush=True)
                    if time.time() - t0 > TIME_BUDGET:
                        print("  time budget reached while deepening, stopping", flush=True)
                        late = True
                        break
            if rounds:
                counts["deepened"] = deeper
                counts["deepening rounds"] = rounds
        finally:
            pool.terminate()
    return done, counts


def main():
    global SF
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chess"], check=True)
    SF = find_stockfish()
    m = machine()
    print(f"batch {BATCH}: {len(LINES)} lines, stockfish {SF}, {WORKERS} engines, machine {m}, "
          f"setup {time.time() - t0:.0f}s", flush=True)
    done, counts = work(t0)
    summary = {"batch": BATCH, "lines": len(LINES), "done": done, "counts": counts,
               "hours": round((time.time() - t0) / 3600, 3), "machine": m}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"))
    print("summary", summary, flush=True)


if __name__ == "__main__":
    main()
