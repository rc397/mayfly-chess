"""Start training once the corpus is ready.

The trigger is a condition, not a timer, since a machine that sleeps would leave the corpus behind the
clock. Training starts when the whole download queue (ChessBench, Lichess, Syzygy) is finished, every
converter has exited, and the earliest start time has passed. Waiting for the full corpus keeps every
frozen version trained on the same data mix. The launch settings are saved to autostart.json, which
other scripts read as the main run's environment. A reboot ends the wait, so restart it by hand.

    python chess/autostart.py                  wait, then launch
    python chess/autostart.py now              launch immediately
    python chess/autostart.py status           show what it is waiting for
    env NOT_BEFORE_HOURS=6, TARGET=10e9, POLL=300, SETTLE=600, TAG=mayfly
"""
import os, sys, json, glob, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
SHARDS = E("SHARDS", paths.SHARDS)
TARGET = float(E("TARGET", 10e9))
POLL = int(E("POLL", 300))
SETTLE = int(E("SETTLE", 600))
NOT_BEFORE_HOURS = float(E("NOT_BEFORE_HOURS", 6))
TAG = E("TAG", "mayfly")
STATE = os.path.join(HERE, "autostart.json")

# The production run, with measured choices. 139M parameters at d1024 by 16 layers, batch 128
# because throughput saturates there, bf16 because it beat fp32 and fp8, and 1.2e-4 because 3e-4
# diverged at this width.
TRAIN_ENV = {k: E(k, v) for k, v in {
    "D": "1024", "LAYERS": "16", "NHEAD": "16", "FF_MULT": "2.0",
    "AV_BSZ": "96", "POL_BSZ": "32",
    "LR": "1.2e-4", "WARMUP": "12000",
    # Leela's labels are shallow, but its policy target is a visit distribution from its own
    # search, which is the shape a tree wants for priors.
    "SOURCES": "lz,lc", "PRECISION": "bf16",
    # every validation rewrites a 1.72 GB resume checkpoint, so validating every 2,000 steps
    # would write tens of terabytes to an SSD over a long run
    "VAL_EVERY": "10000", "VAL_BATCHES": "40",
    "VERSION_HOURS": "120",
    "RESUME": "1",
}.items()}
TRAIN_ENV["TAG"] = TAG


QUEUE_STATE = E("QUEUE_STATE", paths.under(".dlqueue.json"))
QUEUE_JOBS = ("chessbench", "lichess", "syzygy")


def queue_done():
    """Every download finished, according to the queue's own ledger."""
    try:
        done = set(json.load(open(QUEUE_STATE)).get("done", []))
    except Exception:
        return False, set()
    return set(QUEUE_JOBS) <= done, done


def av_pairs():
    n = 0
    for p in glob.glob(os.path.join(SHARDS, ".av*.state.json")):
        try:
            n += json.load(open(p)).get("pairs", 0)
        except Exception:
            pass
    return n


def running(pattern):
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
         f"$_.CommandLine -like '*{pattern}*' }}).Count"],
        capture_output=True, text=True).stdout.strip()
    try:
        return int(out or 0)
    except ValueError:
        return 0


def launch():
    env = dict(os.environ)
    env.update(TRAIN_ENV)
    log = os.path.join(HERE, f"train_{TAG}.log")
    cmd = ("Start-Process -FilePath python -ArgumentList 'train_av.py' "
           f"-WorkingDirectory '{HERE}' -RedirectStandardOutput '{log}' "
           f"-RedirectStandardError '{os.path.join(HERE, f'train_{TAG}.err')}' -WindowStyle Hidden")
    setters = "; ".join(f"$env:{k}='{v}'" for k, v in TRAIN_ENV.items())
    subprocess.run(["powershell", "-NoProfile", "-Command", f"{setters}; {cmd}"],
                   cwd=HERE, env=env)
    json.dump({"launched": time.strftime("%Y-%m-%d %H:%M:%S"), "corpus": av_pairs(),
               "env": TRAIN_ENV}, open(STATE, "w"), indent=1)
    print(f"  launched, logging to {os.path.basename(log)}", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "wait"
    deadline = time.time() + NOT_BEFORE_HOURS * 3600

    if arg == "status":
        n = av_pairs()
        print(f"corpus {n:,.0f} of {TARGET:,.0f} ({100*n/TARGET:.1f}%)")
        ok, done = queue_done()
        print(f"downloads done:     {', '.join(sorted(done)) or 'none'}"
              f" ({'all three' if ok else 'still waiting'})")
        print(f"converters running: {running('chessbench.py')}")
        print(f"trainer running:    {running('train_av.py')}")
        if os.path.exists(STATE):
            print(f"already launched: {json.load(open(STATE))['launched']}")
        raise SystemExit(0)

    if arg == "now":
        print("starting immediately at your instruction", flush=True)
        launch()
        raise SystemExit(0)

    print(f"waiting for the whole queue: {', '.join(QUEUE_JOBS)}")
    print(f"earliest start {time.strftime('%H:%M:%S', time.localtime(deadline))} "
          f"({NOT_BEFORE_HOURS:.1f}h from now)", flush=True)
    while True:
        n = av_pairs()
        ok, done = queue_done()
        busy = (running("chessbench.py") + running("fetch_lichess_evals.py")
                + running("syzygy.py") + running("dlqueue.py"))
        early = time.time() < deadline
        left = max(deadline - time.time(), 0) / 3600
        print(f"  {time.strftime('%H:%M:%S')}  corpus {n:,.0f} ({100*n/TARGET:.1f}%)  "
              f"done [{', '.join(sorted(done)) or 'none'}]  {busy} job(s) running  "
              + (f"{left:.1f}h before the floor lifts" if early else "floor passed"),
              flush=True)
        if ok and busy == 0 and not early:
            break
        time.sleep(POLL)

    if SETTLE:
        print(f"\n  corpus ready, settling {SETTLE//60} minutes before touching the disk",
              flush=True)
        time.sleep(SETTLE)
    if running("train_av.py"):
        print("  a trainer is already running, leaving it alone", flush=True)
        raise SystemExit(0)
    print(f"  starting {TAG} at {time.strftime('%H:%M:%S')}", flush=True)
    launch()
