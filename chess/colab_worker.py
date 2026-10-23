"""Deep-label batches on a Google Colab runtime, the Colab side of kaggle_label.py.

Only the host CPUs are used. colab_label.ipynb fetches this file from the batches dataset and runs it.
kaggle_label.py keeps a few batches waiting in that dataset, listed in colab.json with the time each
may take and a sha1 of its lines, next to the kaggle_kernel.py its Kaggle notebooks run.

The loop takes the oldest batch in colab.json that has no result yet, fills kaggle_kernel.py with that
batch, its time and local paths (as kaggle_label.py does for a notebook), and runs it with one engine
per core. The result goes to an uploader thread while the next batch starts. Each upload is a new
version of the results dataset, holding every result whose batch colab.json still lists, plus
status.json. kaggle_label.py reads results in and drops their batches from colab.json.

Colab ends a free runtime after at most twelve hours, so no batch starts with less than MIN_BATCH_S +
SAVE_S left of SESSION_H, and a late batch is cut short so its result is saved in time. A runtime that
dies early loses only its current batch, which stays listed for the next one.

The notebook copies a Kaggle token from Colab's secrets to ~/.kaggle/access_token, where the Kaggle CLI
reads it. Nothing here reads or prints it. KAGGLE_USER names the account that owns the datasets.

    python colab_worker.py      env MAYFLY_ROOT=/content/mayfly, MAYFLY_START (runtime start, epoch seconds)
"""
import hashlib
import importlib.util
import itertools
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time

E = os.environ.get
USER = E("KAGGLE_USER", "")
BATCHES = E("MAYFLY_BATCHES", f"{USER}/mayfly-batches")
RESULTS = E("MAYFLY_RESULTS", f"{USER}/mayfly-colab-results")
ROOT = E("MAYFLY_ROOT", "/content/mayfly")
START = float(E("MAYFLY_START") or time.time())
SESSION_H = float(E("MAYFLY_SESSION_H", 11.5))
SAVE_S = float(E("MAYFLY_SAVE_S", 15 * 60))
MIN_BATCH_S = float(E("MAYFLY_MIN_BATCH_S", 20 * 60))
WAIT_S = float(E("MAYFLY_WAIT_S", 180))          # the first wait while no batch is waiting, doubling to 15 minutes
STOCKFISH = E("MAYFLY_STOCKFISH")                # an engine already on disk instead of kaggle_kernel.py's download
WORKERS = E("MAYFLY_WORKERS")                    # engines instead of one a core (a test on another machine)
# Lines filled in in kaggle_kernel.py, each checked before a batch runs. The engine path goes in at the top so the
# pool's processes have it however they start (forked, or spawned and re-reading the file).

FILL_BUDGET = "TIME_BUDGET = 8 * 3600 + 30 * 60"
FILL_OUT = '\nOUT = "/kaggle/working"\n'
FILL_SF = "\nSF = None\n"
FILL_FIND = "    SF = find_stockfish()"
FILL_WORKERS = "\nWORKERS = cpus()\n"
RESULT = re.compile(r"(b\d{5})\.(summary\.json|results\.jsonl)")
LISTED = re.compile(r"\b(b\d{5})\.summary\.json\b")
MISSING = re.compile(r"\b40[34]\b|not found|forbidden", re.I)
_ids = itertools.count(1)


def say(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def kaggle(*args, timeout=900):
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return 1, f"kaggle {args[0]} {args[1]} timed out after {timeout}s"
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def fetch(dataset, tag):
    """Unpack the dataset's latest version into a new folder. Returns (folder, "") or (None, what Kaggle said)."""
    d = os.path.join(ROOT, "in", f"{tag}{next(_ids):04d}")
    os.makedirs(d, exist_ok=True)
    rc, out = kaggle("datasets", "download", dataset, "-p", d, "--unzip", "-o", "-q")
    return (d, "") if rc == 0 else (None, out.strip()[-300:])


def manifest(folder):
    """colab.json from a fetched folder, or None."""
    try:
        man = json.load(open(os.path.join(folder, "colab.json"), encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return man if isinstance(man, dict) and isinstance(man.get("batches"), list) else None


def arrived():
    """The batches with a result in the results dataset (none while it does not exist yet), or None when Kaggle
    could not say."""
    rc, out = kaggle("datasets", "files", RESULTS, "--page-size", "200", timeout=300)
    if rc != 0:
        return set() if MISSING.search(out) else None
    return set(LISTED.findall(out))


def stockfish(kernel):
    """kaggle_kernel.py's own Stockfish 19 download, once a session."""
    spec = importlib.util.spec_from_file_location("mayfly_kernel", kernel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.find_stockfish()


def run_batch(job, folder, sf, seconds):
    """Run one batch through kaggle_kernel.py. Returns the folder holding its summary.json and results.jsonl, or None."""
    b = job["batch"]
    try:
        raw = open(os.path.join(folder, job["file"]), "rb").read()
        src = open(os.path.join(folder, "kaggle_kernel.py"), encoding="utf-8").read()
    except OSError as e:
        say(f"{b}: {e}")
        return None
    if hashlib.sha1(raw).hexdigest() != job.get("sha1"):
        say(f"{b}: the file is not the one colab.json names; skipped")
        return None
    work = os.path.join(ROOT, "work", f"{b}_{next(_ids):04d}")
    os.makedirs(work, exist_ok=True)
    fills = [(FILL_BUDGET, f"TIME_BUDGET = {int(seconds)}"), (FILL_OUT, f"\nOUT = {work!r}\n"),
             (FILL_SF, f"\nSF = {sf!r}\n"), (FILL_FIND, "    SF = SF or find_stockfish()")]
    if WORKERS:
        fills.append((FILL_WORKERS, f"\nWORKERS = {int(WORKERS)}\n"))
    for old, _ in fills + [("__BATCH__", ""), ("__LINES__", "")]:
        if old not in src:
            raise SystemExit(f"kaggle_kernel.py has no {old.strip()!r} to fill in: this worker is out of date")
    for old, new in fills:
        src = src.replace(old, new)
    lines = raw.decode("utf-8").splitlines()
    src = src.replace("__BATCH__", b).replace("__LINES__", "\n".join(lines))
    script = os.path.join(work, "label.py")
    with open(script, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    say(f"{b}: {len(lines):,} lines, {seconds / 3600:.2f}h")
    p = subprocess.Popen([sys.executable, script], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        print("    " + line.rstrip(), flush=True)
    rc = p.wait()
    try:
        summary = json.load(open(os.path.join(work, "summary.json"), encoding="utf-8"))
    except (OSError, ValueError):
        say(f"{b}: no summary (exit {rc}); it stays waiting")
        return None
    summary.update(sha1=job["sha1"], where="colab")
    with open(os.path.join(work, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh)
    say(f"{b}: done in {summary.get('hours')}h, {summary.get('counts')}")
    return work


class Uploader(threading.Thread):
    """Each finished batch into the results dataset, one version at a time, while the next batch runs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.todo = queue.Queue()
        self.lock = threading.Lock()
        self.done = {}                           # batch to its work folder, for every batch finished in this run

    def add(self, b, work):
        with self.lock:
            self.done[b] = work
        self.todo.put(b)

    def finish(self):
        self.todo.put(None)
        self.join()

    def run(self):
        while True:
            b = self.todo.get()
            if b is None:
                return
            for tries in range(6):
                if self.publish(b):
                    break
                time.sleep(60 * (tries + 1))
            else:
                say(f"{b}: not uploaded after six tries; the next upload takes it, or it stays waiting")

    def publish(self, b):
        """Publish a new version of the results dataset. True when Kaggle took it."""
        folder, err = fetch(BATCHES, "up")
        man = manifest(folder) if folder else None
        if man is None:
            say(f"upload of {b}: colab.json not readable ({err or 'no colab.json'}); trying again")
            return False
        wanted = {j["batch"] for j in man["batches"]}
        have, err = fetch(RESULTS, "res")
        if have is None and not MISSING.search(err):
            say(f"upload of {b}: the results dataset not readable ({err}); trying again")
            return False
        stage = os.path.join(ROOT, "up", f"{b}_{next(_ids):04d}")
        os.makedirs(stage, exist_ok=True)
        for f in sorted(os.listdir(have)) if have else []:
            m = RESULT.fullmatch(f)
            if m and m.group(1) in wanted:
                shutil.copy(os.path.join(have, f), os.path.join(stage, f))
        with self.lock:
            mine = dict(self.done)
        for bb, work in mine.items():
            if bb in wanted:
                shutil.copy(os.path.join(work, "summary.json"), os.path.join(stage, bb + ".summary.json"))
                shutil.copy(os.path.join(work, "results.jsonl"), os.path.join(stage, bb + ".results.jsonl"))
        kept = sorted({RESULT.fullmatch(f).group(1) for f in os.listdir(stage) if RESULT.fullmatch(f)})
        with open(os.path.join(stage, "status.json"), "w", encoding="utf-8") as fh:
            json.dump({"updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "results": kept,
                       "session_hours": round((time.time() - START) / 3600, 2), "done_this_session": sorted(mine)},
                      fh, indent=1)
        with open(os.path.join(stage, "dataset-metadata.json"), "w", encoding="utf-8") as fh:
            json.dump({"title": "mayfly colab results", "id": RESULTS, "licenses": [{"name": "CC0-1.0"}]}, fh, indent=1)
        rc, out = kaggle("datasets", "version", "-p", stage, "-m", f"{b} from colab")
        if rc != 0 and MISSING.search(out):
            rc, out = kaggle("datasets", "create", "-p", stage)
        if rc != 0:
            say(f"upload of {b} failed: {out.strip()[-200:]}")
            return False
        say(f"{b} uploaded: the results dataset now holds {', '.join(kept) or 'only status.json'}")
        return True


def main():
    os.makedirs(ROOT, exist_ok=True)
    folder, err = fetch(BATCHES, "boot")
    if folder is None or not os.path.exists(os.path.join(folder, "kaggle_kernel.py")):
        raise SystemExit(f"could not fetch {BATCHES}: {err or 'no kaggle_kernel.py in it'}")
    sf = STOCKFISH or stockfish(os.path.join(folder, "kaggle_kernel.py"))
    end = START + SESSION_H * 3600
    say(f"engine {sf}; batches start for the next {(end - MIN_BATCH_S - SAVE_S - time.time()) / 3600:.1f}h")
    up = Uploader()
    up.start()
    idle = misses = 0
    try:
        while True:
            left = end - time.time()
            if left < MIN_BATCH_S + SAVE_S:
                say(f"{left / 60:.0f} minutes of the session left: no more batches")
                break
            folder, err = fetch(BATCHES, "in")
            man = manifest(folder) if folder else None
            have = arrived() if man is not None else None
            todo = [j for j in man["batches"] if j["batch"] not in have and j["batch"] not in up.done] if (
                man is not None and have is not None) else []
            if not todo:
                wait = min(WAIT_S * 2 ** idle, 900)
                idle += 1
                why = "no batch waiting" if man is not None and have is not None else (
                    f"the batches or the results not readable ({err or 'listing failed'})")
                say(f"{why}; looking again in {wait / 60:.0f} minutes")
                time.sleep(wait)
                continue
            idle = 0
            job = todo[0]
            work = run_batch(job, folder, sf, min(float(job.get("seconds", 3600)), left - SAVE_S))
            if work is None:
                misses += 1
                if misses >= 3:
                    say("three batches in a row failed: stopping")
                    break
                time.sleep(60)
                continue
            misses = 0
            up.add(job["batch"], work)
    finally:
        say("waiting for the last upload")
        up.finish()
        say(f"stopped: {len(up.done)} batch(es) done this run")


if __name__ == "__main__":
    main()
