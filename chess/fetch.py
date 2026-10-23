"""Resumable, crash-safe downloads for very large corpora.

Each file downloads into a .part file, and an interrupted download continues from its current length
with an HTTP Range request (Lichess honours ranges, storage.lczero.org advertises them but does not).
A file is renamed into place only once its length matches Content-Length, and a manifest on disk
records what completed, so a restart skips finished files without touching the network.

    python chess/fetch.py <url> <dest>              one file
    python chess/fetch.py --manifest <file> <dir>   a list of urls, one per line
    env RETRIES=8, CHUNK=8388608, MAX_MBPS=0 (0 is no limit)
"""
import os, sys, json, time, threading
import requests

E = os.environ.get
RETRIES = int(E("RETRIES", 8))
CHUNK = int(E("CHUNK", 8 << 20))
# A bandwidth cap, since parallel downloads can saturate a shared connection. Zero means no limit.
MAX_MBPS = float(E("MAX_MBPS", 0))


def _manifest_path(d):
    return os.path.join(d, ".fetched.json")


def load_manifest(d):
    p = _manifest_path(d)
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p))
    except Exception:
        return {}


# Several files can download at once, and updating the manifest is a read, a change and a write, so
# without a lock two workers finishing together could lose one entry.
_MANIFEST_LOCK = threading.Lock()


def mark_done(d, name, size):
    with _MANIFEST_LOCK:
        return _mark_done(d, name, size)


def _mark_done(d, name, size):
    p = _manifest_path(d)
    m = load_manifest(d)
    m[name] = {"size": size, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = p + ".tmp"
    json.dump(m, open(tmp, "w"), indent=1)
    os.replace(tmp, p)                      # atomic, so the manifest is never half written


def remote_size(url, timeout=60):
    try:
        h = requests.head(url, timeout=timeout, allow_redirects=True)
        return int(h.headers.get("Content-Length", 0))
    except Exception:
        return 0


def fetch(url, dest, expect=None, quiet=False):
    """Download one file, resuming a .part if present. Returns the final path or None."""
    d = os.path.dirname(dest) or "."
    os.makedirs(d, exist_ok=True)
    name = os.path.basename(dest)
    man = load_manifest(d)
    if name in man and os.path.exists(dest) and os.path.getsize(dest) == man[name]["size"]:
        if not quiet:
            print(f"  {name}: already complete", flush=True)
        return dest
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if expect in (None, size):
            mark_done(d, name, size)
            return dest

    part = dest + ".part"
    total = expect or remote_size(url)
    for attempt in range(RETRIES):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if total and have == total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if have and r.status_code == 200:
                    # the server ignored the range and restarted, so throw away the partial
                    have = 0
                    open(part, "wb").close()
                elif r.status_code not in (200, 206):
                    raise IOError(f"HTTP {r.status_code}")
                t0, start = time.time(), have
                t_cap, cap_from = time.time(), have
                with open(part, "ab") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        have += len(chunk)
                        if MAX_MBPS:
                            owed = ((have - cap_from) * 8 / (MAX_MBPS * 1e6)
                                    - (time.time() - t_cap))
                            if owed > 0:
                                time.sleep(min(owed, 2.0))
                        if not quiet and total and time.time() - t0 > 30:
                            mbps = (have - start) * 8 / (time.time() - t0) / 1e6
                            print(f"  {name}: {100*have/total:5.1f}%  {mbps:.0f} Mbps", flush=True)
                            t0, start = time.time(), have
            if total and have < total:
                raise IOError(f"short read {have} of {total}")
            break
        except Exception as e:
            wait = min(2 ** attempt, 60)
            if not quiet:
                print(f"  {name}: {type(e).__name__}, resuming from "
                      f"{have/1e6:.0f} MB in {wait}s", flush=True)
            time.sleep(wait)
    else:
        return None

    final = os.path.getsize(part)
    if total and final != total:
        return None
    os.replace(part, dest)                  # atomic, only a complete file gets the real name
    mark_done(d, name, final)
    return dest


def fetch_all(urls, out_dir, on_done=None):
    """Fetch a list in order, skipping what the manifest already has.

    on_done(path) runs after each file, so a pipeline can convert and delete each archive before
    the next one arrives.

    """
    os.makedirs(out_dir, exist_ok=True)
    man = load_manifest(out_dir)
    done = ok = 0
    for i, url in enumerate(urls, 1):
        name = url.rsplit("/", 1)[-1]
        if name in man:
            done += 1
            continue
        dest = os.path.join(out_dir, name)
        print(f"[{i}/{len(urls)}] {name}", flush=True)
        got = fetch(url, dest)
        if got:
            ok += 1
            if on_done:
                try:
                    on_done(got)
                except Exception as e:
                    print(f"  handler failed: {type(e).__name__}: {e}", flush=True)
        else:
            print(f"  {name}: FAILED after {RETRIES} attempts, moving on", flush=True)
    print(f"\n{ok} fetched, {done} already had, {len(urls)-ok-done} failed")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "--manifest":
        urls = [l.strip() for l in open(a[1]) if l.strip() and not l.startswith("#")]
        fetch_all(urls, a[2])
    elif len(a) >= 2:
        p = fetch(a[0], a[1])
        print(f"\n{'wrote ' + p if p else 'FAILED'}")
    else:
        print(__doc__)
