r"""Convert the Lichess evaluation archive with many fetch_lichess_evals.py processes at once.

Worker k takes every line whose number modulo N is k, so nothing needs to know the file's length in
advance. Skipping lines is far cheaper than converting them. Every worker reads the same local
archive, which is fine when it fits in the OS file cache and a bad idea when it does not.

  python chess/lz_parallel.py [WORKERS]
  env: WORKERS=26  SHARDS=<data>/shards
"""
import os, sys, time, subprocess, glob, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else int(E("WORKERS", 26))
SHARDS = E("SHARDS", paths.SHARDS)
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def kept_total():
    n = 0
    for p in glob.glob(os.path.join(SHARDS, ".lz*.progress.json")):
        try:
            n += json.load(open(p)).get("kept", 0)
        except Exception:
            pass
    return n


def records_on_disk():
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(SHARDS, "lz_*.bin"))) // 112


if __name__ == "__main__":
    assert WORKERS <= len(LETTERS), f"at most {len(LETTERS)} workers"
    try:
        start_line = json.load(open(os.path.join(SHARDS, ".lz.progress.json")))["seen"]
    except Exception:
        start_line = 0
    print(f"{WORKERS} workers, interleaved, resuming from line {start_line:,}")
    print(f"{records_on_disk():,} Lichess records already on disk\n", flush=True)

    procs = []
    t0 = time.time()
    for k in range(WORKERS):
        tag = LETTERS[k]
        log = open(os.path.join(HERE, f"lz_{tag}.log"), "a", buffering=1)
        env = dict(os.environ)
        env.update({"GUARD": "0", "LOCAL": "1"})
        procs.append((tag, subprocess.Popen(
            [sys.executable, os.path.join(HERE, "fetch_lichess_evals.py"),
             "--stripe", str(WORKERS), "--id", str(k), "--tag", tag],
            cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    print(f"launched {len(procs)} workers", flush=True)

    last = records_on_disk()
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(60)
        now = records_on_disk()
        alive = sum(1 for _, p, _ in procs if p.poll() is None)
        el = time.time() - t0
        print(f"  {time.strftime('%H:%M:%S')}  {now:,} records "
              f"(+{now-last:,} a minute)  {alive}/{WORKERS} alive  {el/3600:.2f}h", flush=True)
        last = now

    codes = [(t, p.returncode) for t, p, _ in procs]
    for _, _, log in procs:
        log.close()
    bad = [t for t, c in codes if c != 0]
    print(f"\n{records_on_disk():,} Lichess records after {(time.time()-t0)/3600:.2f}h")
    if bad:
        print(f"workers that did not exit cleanly: {', '.join(bad)}")
    raise SystemExit(1 if bad else 0)
