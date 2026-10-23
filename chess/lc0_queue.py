r"""
Work through the Leela archives one at a time, recording what is finished.

The full source is terabytes, so only a few archives sit on disk at once. Each is fetched,
optionally converted to packed records (CONVERT=1), deleted, and recorded done. The manifest in
fetch.py means a restart costs at most the archive in flight. A downloader thread runs ahead of the
converter through a small bounded queue, and free space is checked before each fetch so a full
drive stalls the queue instead of the machine. lc0_pool.py is the faster, multi-process version.

  python chess/lc0_queue.py [MAX_ARCHIVES]
  env: RUNS=test80  MIN_FREE_GB=300  CONVERT=0  RAW=<data>/lc0_raw  OUT=<data>/shards
"""
import os, sys, re, glob, time, shutil, threading, queue
import requests
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fetch import fetch, load_manifest, mark_done
import paths

E = os.environ.get
BASE = "https://storage.lczero.org/files/training_data"
RUNS = E("RUNS", "test80").split(",")
RAW = E("RAW", paths.under("lc0_raw"))
OUT = E("OUT", paths.under("shards"))
MIN_FREE_GB = int(E("MIN_FREE_GB", 300))
AHEAD = int(E("AHEAD", 2))   # archives the downloader may run ahead of the converter
CONVERT = E("CONVERT", "0") == "1"
MAX_N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000


def listing(run):
    """Archive urls for one run, newest last, read from the directory index."""
    url = f"{BASE}/{run}/"
    try:
        html = requests.get(url, timeout=120).text
    except Exception as e:
        print(f"cannot list {run}: {e}")
        return []
    names = sorted(set(re.findall(r'href="([^"]+\.tar)"', html)))
    return [url + n if not n.startswith("http") else n for n in names]


def free_gb(path):
    try:
        return shutil.disk_usage(os.path.dirname(path) or path).free / 1e9
    except Exception:
        return 1e9


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    urls = []
    for r in RUNS:
        u = listing(r)
        print(f"{r}: {len(u)} archives")
        urls += u
    urls = urls[:MAX_N]
    if not urls:
        raise SystemExit("no archives found")
    print(f"\n{len(urls)} archives queued, about {len(urls)*1.8/1000:.2f} TB")
    print(f"conversion is {'ON' if CONVERT else 'OFF, downloading only'}")
    print(f"pausing whenever free space drops below {MIN_FREE_GB} GB\n", flush=True)

    done = load_manifest(RAW)
    converter = None
    if CONVERT:
        from lc0_convert import records_in, read_record
        from fetch_lichess_evals import Sharder
        converter = Sharder(OUT, E("PREFIX", "lc"), int(E("SHARD", 4_000_000)))

    todo = [u for u in urls if u.rsplit("/", 1)[-1] not in done]
    n_done = len(urls) - len(todo)
    print(f"{n_done} already fetched, {len(todo)} to go")
    print("", flush=True)

    # Archives downloaded by a run that died before converting them are already in the manifest,
    # so they would be skipped forever. A .tar still on disk means not converted, so do those first.
    leftovers = sorted(glob.glob(os.path.join(RAW, "*.tar")))
    if leftovers:
        print(f"{len(leftovers)} archive(s) downloaded but not converted, handling first",
              flush=True)

    q = queue.Queue(maxsize=AHEAD)
    stop = threading.Event()

    def downloader():
        for p in leftovers:
            q.put(p)
        for url in todo:
            if stop.is_set():
                return
            name = url.rsplit("/", 1)[-1]
            while free_gb(RAW) < MIN_FREE_GB and not stop.is_set():
                print(f"  only {free_gb(RAW):.0f} GB free, waiting", flush=True)
                time.sleep(300)
            t0 = time.time()
            p = fetch(url, os.path.join(RAW, name), quiet=True)
            if p:
                mb = os.path.getsize(p) / 1e6
                print(f"  fetched {name} {mb:.0f} MB in {time.time()-t0:.0f}s "
                      f"({mb*8/max(time.time()-t0,1):.1f} Mbps)", flush=True)
                q.put(p)
            else:
                print(f"  {name}: failed, skipping", flush=True)
        q.put(None)

    t = threading.Thread(target=downloader, daemon=True)
    t.start()

    n_new = 0
    while True:
        p = q.get()
        if p is None:
            break
        n_new += 1
        if converter is None:
            continue
        t0, kept = time.time(), 0
        for buf in records_in(p):
            r = read_record(buf)
            if r is None:
                continue
            ids, c, idx, cps, cp, feat, rule50, plies = r
            converter.write((ids, c, idx, cps, cp, feat, 0, False, rule50, 0, True, False, plies))
            kept += 1
        os.remove(p)
        print(f"  converted {kept:,} positions in {time.time()-t0:.0f}s, source removed",
              flush=True)
    if converter is not None:
        converter.close()
    print(f"\n{n_new} fetched this session, {n_done} already had")
