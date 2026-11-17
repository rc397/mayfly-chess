r"""Download Leela training archives and convert them with several lc0_convert processes at once.

One thread downloads, and WORKERS processes convert, since conversion in Python is several times
slower than a modest download. A cycle costs about max(download, convert / WORKERS), so past the
point where the network is the limit more workers do not help. Processes rather than threads,
because read_record holds the GIL.

Each worker writes under its own shard tag (p0, p1, ...), so workers never race on a shard index
and a restart resumes each worker's partial shard. Single letter tags belong to the Lichess
producer. The prefix defaults to lz, which the trainer reads.

  python chess/lc0_pool.py [MAX_ARCHIVES]
  env: RUNS=test91  WORKERS=3  PREFIX=lz  SHARD=4000000  AHEAD=2  ORDER=old|new
       MIN_FREE_GB=300  RAW=<data>/lc0_raw  OUT=<data>/shards
"""
import os, sys, re, time, glob, json, queue, shutil, threading, subprocess
import requests
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fetch import fetch, load_manifest
import paths

E = os.environ.get
BASE = "https://storage.lczero.org/files/training_data"
RUNS = E("RUNS", "test91").split(",")
RAW = E("RAW", paths.under("lc0_raw"))
OUT = E("OUT", paths.under("shards"))
PREFIX = E("PREFIX", "lz")
SHARD = E("SHARD", "4000000")
WORKERS = int(E("WORKERS", 3))
MIN_FREE_GB = int(E("MIN_FREE_GB", 300))
AHEAD = int(E("AHEAD", 2))
MAX_N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
CONVERTER = os.path.join(HERE, "lc0_convert.py")

STATE = {"fetched": 0, "converted": 0, "failed": 0, "kept": 0, "fetch_s": 0.0, "conv_s": 0.0}
LOCK = threading.Lock()
STOP = threading.Event()


def listing(run):
    """Archive urls for one run, newest last, read from the directory index."""
    url = f"{BASE}/{run}/"
    try:
        html = requests.get(url, timeout=120).text
    except Exception as e:
        print(f"cannot list {run}: {e}", flush=True)
        return []
    names = sorted(set(re.findall(r'href="([^"]+\.tar)"', html)))
    return [n if n.startswith("http") else url + n for n in names]


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return 1e9


def worker_tags(n):
    """A private shard tag per worker, with what each already has on disk.

    An empty tag is a first run, and a tag with shards is a restart that resumes its own shard.
    """
    out = []
    for i in range(n):
        t = f"p{i}"
        have = sorted(glob.glob(os.path.join(OUT, f"{PREFIX}_{t}[0-9][0-9][0-9].bin")))
        out.append((t, len(have), sum(os.path.getsize(p) for p in have)))
    return out


# Mark and rewind. A conversion killed partway leaves partial records in the worker's shard, and
# converting the archive again would write them twice. Each worker records its shard sizes before an
# archive, and an unfinished conversion is taken back to that mark before the archive runs again.
# What is taken back goes to a quarantine folder, never deleted.
MARKS = os.path.join(OUT, ".lc0_pool_marks")
QUARANTINE = os.path.join(os.path.dirname(OUT.rstrip("/\\")), "_quarantine_lc0_rewind")


def _tag_shards(tag):
    return sorted(glob.glob(os.path.join(OUT, f"{PREFIX}_{tag}[0-9][0-9][0-9].bin")))


def _mark(tag, path):
    os.makedirs(MARKS, exist_ok=True)
    with open(os.path.join(MARKS, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"archive": path, "sizes": {p: os.path.getsize(p) for p in _tag_shards(tag)}}, f)


def _unmark(tag):
    try:
        os.remove(os.path.join(MARKS, f"{tag}.json"))
    except FileNotFoundError:
        pass


def _rewind(tag, wid):
    """Take an unfinished conversion's output back to its mark, into quarantine."""
    mp = os.path.join(MARKS, f"{tag}.json")
    if not os.path.exists(mp):
        return
    with open(mp, encoding="utf-8") as f:
        m = json.load(f)
    os.makedirs(QUARANTINE, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for s in _tag_shards(tag):
        base = os.path.basename(s)
        if s not in m["sizes"]:
            shutil.move(s, os.path.join(QUARANTINE, f"{base}.{stamp}"))
            print(f"  [{wid}] rewind: {base} was started by an unfinished conversion, quarantined",
                  flush=True)
        elif os.path.getsize(s) > m["sizes"][s]:
            keep = m["sizes"][s]
            tail = os.path.join(QUARANTINE, f"{base}.tail.{stamp}")
            with open(s, "r+b") as f:
                f.seek(keep)
                with open(tail, "wb") as q:
                    shutil.copyfileobj(f, q)
                try:
                    f.truncate(keep)
                except OSError:
                    os.remove(tail)        # only this moment's copy of records still in the shard
                    raise
            print(f"  [{wid}] rewind: {base} taken back to {keep:,} bytes, the tail quarantined",
                  flush=True)
    _unmark(tag)


def _try_rewind(tag, wid, name):
    """_rewind, or False when a shard cannot be taken back.

    Windows refuses to shrink or move a file another process has memory-mapped, and the trainer
    maps every lz shard. Then the mark and the archive stay on disk, unconverted rather than
    written twice, and the worker carries on."""
    try:
        _rewind(tag, wid)
        return True
    except (OSError, ValueError) as e:
        print(f"  [{wid}] cannot rewind {tag} ({type(e).__name__}: {e}); {name} left on disk "
              f"unconverted. Stop the trainer's hold on {PREFIX}_{tag} shards to clear it.",
              flush=True)
        return False


def convert(path, tag, wid):
    """Run one archive through lc0_convert in its own process. Returns (kept, seconds, ok)."""
    name = os.path.basename(path)
    if not _try_rewind(tag, wid, name):   # anything an earlier, interrupted conversion left behind
        return 0, 0.0, False
    _mark(tag, path)
    try:
        kept, el, ok = _convert(path, tag, wid)
    except BaseException:
        _try_rewind(tag, wid, name)
        raise
    if ok:
        _unmark(tag)
    else:
        _try_rewind(tag, wid, name)
    return kept, el, ok


def _convert(path, tag, wid):
    env = dict(os.environ, OUT=OUT, PREFIX=PREFIX, SHARD=str(SHARD), TAG=tag)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, CONVERTER, path], env=env, cwd=HERE,
                           capture_output=True, text=True)
    except Exception as e:
        print(f"  [{wid}] {os.path.basename(path)}: cannot start converter: "
              f"{type(e).__name__}: {e}", flush=True)
        return 0, time.time() - t0, False
    el = time.time() - t0
    name = os.path.basename(path)
    if r.returncode != 0:
        # A failed verification assert in lc0_convert lands here.
        tail = [ln for ln in (r.stderr or r.stdout or "").strip().splitlines() if ln.strip()]
        print(f"  [{wid}] {name}: converter exited {r.returncode}: "
              f"{tail[-1][:200] if tail else 'no output'}", flush=True)
        return 0, el, False
    m = re.search(r"([\d,]+) written from ([\d,]+) records", r.stdout or "")
    kept = int(m.group(1).replace(",", "")) if m else 0
    if kept == 0:
        # Keep the source. An archive that yields nothing is corrupt or a format change.
        print(f"  [{wid}] {name}: kept nothing, source left on disk for inspection", flush=True)
        return 0, el, False
    return kept, el, True


def worker(wid, tag, q):
    while True:
        p = q.get()
        try:
            if p is None or STOP.is_set():
                return
            kept, el, ok = convert(p, tag, wid)
            if ok:
                try:
                    os.remove(p)
                except OSError as e:
                    print(f"  [{wid}] cannot remove {os.path.basename(p)}: {e}", flush=True)
            with LOCK:
                STATE["converted"] += int(ok)
                STATE["failed"] += int(not ok)
                STATE["kept"] += kept
                STATE["conv_s"] += el
                n, tot = STATE["converted"], STATE["kept"]
            if ok:
                print(f"  [{wid}] {os.path.basename(p)} -> {kept:,} positions in {el:.0f}s "
                      f"({kept/max(el,1):.0f}/s)   {n} converted, {tot/1e6:.1f}M positions, "
                      f"queue {q.qsize()}", flush=True)
        finally:
            q.task_done()


def downloader(todo, leftovers, q):
    """One thread, one connection.

    The queue depth it prints shows the bottleneck. Full means the converters are behind, empty
    means the network is the limit."""
    for p in leftovers:
        if STOP.is_set():
            break
        q.put(p)
    for url in todo:
        if STOP.is_set():
            break
        name = url.rsplit("/", 1)[-1]
        while free_gb(RAW) < MIN_FREE_GB and not STOP.is_set():
            print(f"  only {free_gb(RAW):.0f} GB free, waiting", flush=True)
            STOP.wait(300)
        if STOP.is_set():
            break
        t0 = time.time()
        try:
            p = fetch(url, os.path.join(RAW, name), quiet=True)
        except Exception as e:
            print(f"  {name}: {type(e).__name__}: {e}", flush=True)
            p = None
        el = time.time() - t0
        if not p:
            print(f"  {name}: failed, skipping", flush=True)
            continue
        mb = os.path.getsize(p) / 1e6
        with LOCK:
            STATE["fetched"] += 1
            STATE["fetch_s"] += el
            f = STATE["fetched"]
        print(f"  fetched {name} {mb:.0f} MB in {el:.0f}s ({mb*8/max(el,1):.1f} Mbps)"
              f"   {f} fetched, queue {q.qsize()}/{q.maxsize}", flush=True)
        q.put(p)
    for _ in range(WORKERS):
        q.put(None)


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    urls = []
    for r in RUNS:
        u = listing(r)
        print(f"{r}: {len(u)} archives")
        urls += u
    # The listing is chronological and a Leela run gets stronger as it goes, so the newest archives
    # carry the best policy targets. ORDER=old takes the first MAX_N, ORDER=new the last MAX_N,
    # newest first. The manifest skips anything already fetched.
    if E("ORDER", "old").lower() == "new":
        urls = urls[::-1][:MAX_N]
    else:
        urls = urls[:MAX_N]
    if not urls:
        raise SystemExit("no archives found")

    done = load_manifest(RAW)
    todo = [u for u in urls if u.rsplit("/", 1)[-1] not in done]
    n_done = len(urls) - len(todo)
    # Archives fetched by an earlier run that died before converting them. The manifest records
    # downloads and a .tar on disk means not yet converted, so these go first.
    leftovers = sorted(glob.glob(os.path.join(RAW, "*.tar")))

    print(f"\n{len(urls)} archives listed, {n_done} already fetched, {len(todo)} to go")
    if leftovers:
        print(f"{len(leftovers)} archive(s) on disk but not converted, those go first")
    print(f"{WORKERS} converters behind one downloader, {free_gb(RAW):.0f} GB free")
    for t, n, sz in worker_tags(WORKERS):
        state = "new" if not n else f"resuming, {n} shard(s), {sz/1e9:.1f} GB"
        print(f"  worker tag {PREFIX}_{t}NNN.bin   {state}")
    print(f"pausing whenever free space drops below {MIN_FREE_GB} GB\n", flush=True)

    q = queue.Queue(maxsize=AHEAD + WORKERS)
    threads = [threading.Thread(target=worker, args=(i, f"p{i}", q), daemon=True)
               for i in range(WORKERS)]
    threads.append(threading.Thread(target=downloader, args=(todo, leftovers, q), daemon=True))
    t0 = time.time()
    for t in threads:
        t.start()
    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=1.0)
    except KeyboardInterrupt:
        STOP.set()
        print("\nstopping: the archive in each converter is finished first", flush=True)
        for t in threads:
            t.join(timeout=1800)

    el = max(time.time() - t0, 1.0)
    print(f"\n{STATE['fetched']} fetched, {STATE['converted']} converted, "
          f"{STATE['failed']} failed, {STATE['kept']:,} positions in {el/3600:.1f}h")
    if STATE["converted"]:
        # Both totals are wall-clock seconds summed over threads.
        print(f"  {STATE['conv_s']/max(STATE['fetch_s'],1):.2f} converter-seconds per "
              f"downloader-second (under {WORKERS}.0 means the network is now the floor)")
        print(f"  {el/STATE['converted']/60:.1f} minutes an archive end to end")
