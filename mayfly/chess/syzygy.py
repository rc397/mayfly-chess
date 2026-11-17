r"""syzygy downloads and verifies the Syzygy endgame tablebases up to six pieces.

The action-value teacher is weakest in long endgames, and a tablebase gives the exact result for any
position with six pieces or fewer.

  3-4-5-wdl and 3-4-5-dtz   290 files, under a gigabyte, covering most endgames that occur. Fetched first.
  6-wdl                     365 files of win, draw or loss, which is what a search probes.
  6-dtz                     365 files of distance to zeroing, needed at the root to convert a win
                            without running into the fifty move rule.

The -nr (no fifty move rule) sets are skipped, and the 18 TB seven piece set is out of scope. The
download is verified by probing positions with known results, since a truncated table can still have a
plausible size.

  python chess/syzygy.py            fetch everything missing, then verify
  python chess/syzygy.py verify     just check what is already on disk
  env: DIR (default paths.under("syzygy"))  MAX_MBPS=0  WORKERS=6  MIN_FREE_GB=200
       SETS=3-4-5-wdl,3-4-5-dtz,6-wdl,6-dtz
"""
import os, sys, re, time, shutil, threading, queue
import requests
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from fetch import fetch, load_manifest
import paths

E = os.environ.get
BASE = E("BASE", "https://tablebase.lichess.ovh/tables/standard/")
DIR = E("DIR", paths.under("syzygy"))
SETS = E("SETS", "3-4-5-wdl,3-4-5-dtz,6-wdl,6-dtz").split(",")
MIN_FREE_GB = int(E("MIN_FREE_GB", 200))
# about a thousand separate files, and the mirror limits each connection, so download in parallel
WORKERS = int(E("WORKERS", 6))


def listing(name):
    """Every table file in one directory of the mirror, smallest first."""
    url = f"{BASE}{name}/"
    try:
        html = requests.get(url, timeout=180).text
    except Exception as e:
        print(f"  cannot list {name}: {type(e).__name__}")
        return []
    return [url + f for f in sorted(set(re.findall(r'href="([^"]+\.rtb[wz])"', html)))]


def free_gb(p):
    try:
        return shutil.disk_usage(os.path.dirname(p) or p).free / 1e9
    except Exception:
        return 1e9


def verify(d):
    """Ask the tables about positions whose result is not a matter of opinion."""
    import chess
    import chess.syzygy
    cases = [
        ("8/8/8/4k3/8/8/8/3QK3 w - - 0 1", 2, "king and queen against a bare king"),
        ("8/8/8/4k3/8/8/8/3RK3 w - - 0 1", 2, "king and rook against a bare king"),
        ("8/8/8/4k3/8/8/8/3BK3 w - - 0 1", 0, "king and bishop, a draw"),
        ("8/8/8/3bk3/8/8/4P3/4K3 w - - 0 1", None, "four pieces, whatever it is"),
        # two bishops on the same colour cannot force mate, so the check below asserts they differ
        ("8/8/4k3/8/8/3B4/8/2B1K3 w - - 0 1", 2, "two bishops on opposite colours, a win"),
        ("8/8/1k6/8/8/1K1R4/7r/8 w - - 0 1", None, "six pieces, rook against rook"),
    ]
    two_b = chess.Board("8/8/4k3/8/8/3B4/8/2B1K3 w - - 0 1")
    sq = list(two_b.pieces(chess.BISHOP, chess.WHITE))
    assert len(sq) == 2 and (chess.square_rank(sq[0]) + chess.square_file(sq[0])) % 2 !=         (chess.square_rank(sq[1]) + chess.square_file(sq[1])) % 2,         "the two bishops must be on opposite colours or that position is a draw"

    n = ok = 0
    with chess.syzygy.open_tablebase(d) as tb:
        for fen, want, name in cases:
            b = chess.Board(fen)
            try:
                got = tb.probe_wdl(b)
            except (KeyError, chess.syzygy.MissingTableError):
                print(f"  MISSING  {name}")
                continue
            except Exception as e:
                print(f"  FAILED   {name}: {type(e).__name__}")
                n += 1
                continue
            n += 1
            good = (want is None) or (got == want)
            ok += good
            print(f"  {'ok  ' if good else 'WRONG'}  {name}: probe says {got}"
                  + ("" if want is None else f", expected {want}"))
    print(f"\n  {ok} of {n} probes answered correctly")
    return n > 0 and ok == n


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        raise SystemExit(0 if verify(DIR) else 1)

    os.makedirs(DIR, exist_ok=True)
    urls = []
    for name in SETS:
        u = listing(name)
        print(f"{name}: {len(u)} files")
        urls += u
    if not urls:
        raise SystemExit("nothing listed, the mirror layout may have changed")

    done = load_manifest(DIR)
    todo = [u for u in urls if u.rsplit("/", 1)[-1] not in done]
    print(f"\n{len(urls)} files, {len(urls) - len(todo)} already here, {len(todo)} to fetch")
    print(f"into {DIR}, pausing below {MIN_FREE_GB} GB free\n", flush=True)

    got = bytes_in = 0
    lock = threading.Lock()
    work = queue.Queue()
    for u in todo:
        work.put(u)
    t0 = time.time()

    def worker():
        # global, not nonlocal, because this is defined at module level and the counters are module globals
        global got, bytes_in
        while True:
            try:
                url = work.get_nowait()
            except queue.Empty:
                return
            name = url.rsplit("/", 1)[-1]
            while free_gb(DIR) < MIN_FREE_GB:
                print(f"  only {free_gb(DIR):.0f} GB free, waiting", flush=True)
                time.sleep(300)
            t1 = time.time()
            got_path = fetch(url, os.path.join(DIR, name), quiet=True)
            if not got_path:
                print(f"  {name}: FAILED", flush=True)
                continue
            mb = os.path.getsize(got_path) / 1e6
            el = max(time.time() - t1, 0.001)
            with lock:
                got += 1
                bytes_in += mb
                n, gb, hrs = got, bytes_in / 1000, (time.time() - t0) / 3600
            if n % 20 == 0:
                print(f"  [{n}/{len(todo)}] {name} {mb:,.0f} MB at {mb*8/el:.0f} Mbps | "
                      f"{gb:.1f} GB in {hrs:.2f}h, overall "
                      f"{gb*8000/max(hrs*3600, 1):.0f} Mbps", flush=True)

    print(f"{WORKERS} downloads at a time", flush=True)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"{got} files, {bytes_in/1000:.1f} GB in {(time.time()-t0)/3600:.2f}h")
    print("verifying by probing:")
    raise SystemExit(0 if verify(DIR) else 1)
