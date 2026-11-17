"""Run the corpus downloads one at a time, in order, resumably.

Parallel transfers saturate the link and break the long Lichess stream, so the jobs run in turn.
    1  chessbench   waits for the already running ChessBench converters to reach TARGET
    2  lichess      lz_parallel.py, which resumes at its recorded line
    3  syzygy       about 150 GB of six-piece tablebases
Each job is resumable on its own. The queue's state file records which have finished, and a restart
picks up at the first that had not.

    python chess/dlqueue.py            run the queue
    python chess/dlqueue.py status     what is done and what is next
    env TARGET=10e9, POLL=120, STATE (queue state file), SKIP_WAIT=0
"""
import os, sys, json, glob, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
SHARDS = E("SHARDS", paths.SHARDS)
TARGET = float(E("TARGET", 10e9))
POLL = int(E("POLL", 120))
STATE = E("STATE", paths.under(".dlqueue.json"))
PY = sys.executable


def load():
    try:
        return json.load(open(STATE))
    except Exception:
        return {"done": [], "started": None}


def save(s):
    tmp = STATE + ".tmp"
    json.dump(s, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def av_pairs():
    n = 0
    for p in glob.glob(os.path.join(SHARDS, ".av*.state.json")):
        try:
            n += json.load(open(p)).get("pairs", 0)
        except Exception:
            pass
    return n


def chessbench_running():
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         # the name filter matters, or the powershell process running this query matches itself
         "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
         "$_.CommandLine -like '*chessbench.py*' }).Count"],
        capture_output=True, text=True).stdout.strip()
    try:
        return int(out or 0)
    except ValueError:
        return 0


def wait_for_chessbench():
    """Done when the target is reached, or when every converter has exited on its own."""
    while True:
        n, live = av_pairs(), chessbench_running()
        if n >= TARGET:
            # the converters overshoot slightly and then exit, so give them a little time to finish,
            # but do not wait forever on a wedged one
            print(f"  chessbench reached {n:,.0f} action values, "
                  f"letting {live} converter(s) finish", flush=True)
            for _ in range(20):
                if chessbench_running() == 0:
                    break
                time.sleep(60)
            return True
        if live == 0:
            print(f"  no converters left and {n:,.0f} packed of {TARGET:,.0f}; "
                  f"treating chessbench as finished", flush=True)
            return True
        print(f"  waiting: {n:,.0f} packed ({100*n/TARGET:.1f}%), {live} converter(s) alive",
              flush=True)
        time.sleep(POLL)


def run(script, env=None, log=None):
    e = dict(os.environ)
    e.update(env or {})
    out = open(os.path.join(HERE, log), "a", buffering=1) if log else None
    print(f"  running {script}" + (f", logging to {log}" if log else ""), flush=True)
    r = subprocess.run([PY, os.path.join(HERE, script)], cwd=HERE, env=e,
                       stdout=out, stderr=subprocess.STDOUT)
    if out:
        out.close()
    print(f"  {script} exited {r.returncode}", flush=True)
    return r.returncode == 0


JOBS = [
    ("chessbench", lambda: wait_for_chessbench()),
    # lz_parallel fans out across the cores and waits for all its workers
    ("lichess", lambda: run("lz_parallel.py", {"WORKERS": E("LZ_WORKERS", "26")},
                            "lz_parallel.log")),
    ("syzygy", lambda: run("syzygy.py", {"MAX_MBPS": E("SYZYGY_MBPS", "0")}, "syzygy.log")),
]


if __name__ == "__main__":
    st = load()
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(f"{av_pairs():,.0f} action values packed, target {TARGET:,.0f}")
        for name, _ in JOBS:
            mark = "done" if name in st["done"] else "pending"
            print(f"  {name:<12} {mark}")
        raise SystemExit(0)

    print(f"queue: {' -> '.join(n for n, _ in JOBS)}")
    print(f"already done: {', '.join(st['done']) or 'nothing'}\n", flush=True)
    for name, fn in JOBS:
        if name in st["done"]:
            print(f"[skip] {name}", flush=True)
            continue
        print(f"[{name}] starting {time.strftime('%H:%M:%S')}", flush=True)
        st["started"] = name
        save(st)
        ok = fn()
        if not ok:
            print(f"[{name}] did not finish cleanly, stopping the queue here so the next job "
                  f"does not start on top of it", flush=True)
            save(st)
            raise SystemExit(1)
        st["done"].append(name)
        st["started"] = None
        save(st)
        print(f"[{name}] done {time.strftime('%H:%M:%S')}\n", flush=True)
    print("queue empty")
