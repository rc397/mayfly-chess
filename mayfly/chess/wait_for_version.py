r"""wait_for_version holds the A/B queue until a version is frozen.

Put it in the queue ahead of tests that need the new version. It returns once versions.json has a sealed row
for the target (a row with a search block, the one pinned_search reads) and the weights file exists and has
stopped changing, so nothing after it loads a checkpoint that is still being written.

  python chess/wait_for_version.py
  env: TARGET=mayfly_v4.pt  VERSION_DIR (default paths.VERSIONS)  POLL=120  DEADLINE_HOURS=14
       exit 0 when frozen, 1 at the deadline
"""
import os, sys, json, time

import paths

E = os.environ.get
TARGET = E("TARGET", "mayfly_v4.pt")
VDIR = E("VERSION_DIR", paths.under("versions"))
POLL = int(E("POLL", 120))
DEADLINE = time.time() + float(E("DEADLINE_HOURS", 14)) * 3600


def sealed():
    try:
        with open(os.path.join(VDIR, "versions.json"), encoding="utf-8-sig") as fh:
            rows = json.load(fh)
    except (OSError, ValueError):
        return None
    for r in rows:
        if r.get("file") == TARGET and r.get("search"):
            return r
    return None


def weights():
    """(size, mtime) of the checkpoint, or None if it is not there yet."""
    try:
        st = os.stat(os.path.join(VDIR, TARGET))
        return st.st_size, st.st_mtime
    except OSError:
        return None


print(f"{time.strftime('%H:%M')} waiting for {TARGET} to be sealed in {VDIR}", flush=True)
last = None
while time.time() < DEADLINE:
    row, w = sealed(), weights()
    if row and w and w == last and time.time() - w[1] > 60:
        print(f"{time.strftime('%H:%M')} {TARGET} is frozen: sealed at {row.get('at')}, "
              f"{len(row['search'])} search settings, {w[0] / 1e6:.0f} MB, unchanged for a poll",
              flush=True)
        sys.exit(0)
    last = w
    time.sleep(POLL)
print(f"{time.strftime('%H:%M')} DEADLINE: {TARGET} was not frozen in time", flush=True)
sys.exit(1)
