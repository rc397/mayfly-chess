r"""Let a self-play machine label some of its own positions with deep_label.py.

When a machine hands new self-play positions to the central labeller, it first tops up its own
queue (queue.txt) to target.txt unlabelled positions and passes on only the rest. A busy machine
takes little and a switched-off one takes nothing, and the queue holds about an hour of work, so
nothing waits long. Without enabled.txt nothing is kept back.

deep_label writes small shards into rows/, and every shard but the newest is finished. Each
finished shard is sent on under a name that cannot repeat (<machine>_<generation>_<file>, with the
generation made once per rows folder), and append_rows.py appends it once into dp_<machine>NNNN.bin.
Files never leave rows/, so the writer's numbering never restarts.

There can be two folders, a heavy labeller (LOCAL_LABEL_DIR) and a cheap shallow one
(CHEAP_LABEL_DIR). Every function takes the folder it works on, the heavy one by default.
"""
import glob
import json
import os
import uuid

import paths

BASE = os.environ.get("LOCAL_LABEL_DIR", paths.under("selfplay", "local_label"))
QUEUE = os.path.join(BASE, "queue.txt")
STATE = os.path.join(BASE, ".labelled.json")
ROWS = os.path.join(BASE, "rows")
SENT = os.path.join(BASE, ".rows_sent.json")      # every name already appended downstream
CHEAP = os.environ.get("CHEAP_LABEL_DIR", paths.under("selfplay", "cheap_label"))


def enabled(base=None):
    return os.path.exists(os.path.join(base or BASE, "enabled.txt"))


def target(base=None):
    """How many positions this machine keeps queued for itself, about an hour of its own labelling."""
    try:
        return int(open(os.path.join(base or BASE, "target.txt")).read().strip())
    except (OSError, ValueError):
        return 0


def backlog(base=None):
    """Positions queued here and not yet labelled."""
    base = base or BASE
    try:
        with open(os.path.join(base, "queue.txt"), "rb") as fh:
            queued = sum(1 for _ in fh)
    except FileNotFoundError:
        queued = 0
    try:
        done = int(json.load(open(os.path.join(base, ".labelled.json"))).get("done", 0))
    except (OSError, ValueError):
        done = 0
    return max(0, queued - done)


def take(data, base=None):
    """How many leading bytes of data (whole lines) to keep here, enough to top the queue up to the target."""
    if not enabled(base):
        return 0
    room = target(base) - backlog(base)
    pos = 0
    while room > 0:
        nl = data.find(b"\n", pos)
        if nl < 0:
            break
        pos, room = nl + 1, room - 1
    return pos


def keep(data, base=None):
    """Add whole lines to this machine's queue, on disk before anyone records them as taken."""
    base = base or BASE
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "queue.txt"), "ab") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def generation(base=None):
    """A tag made once per rows folder, so a folder started again never reuses a name already sent."""
    rows = os.path.join(base or BASE, "rows")
    os.makedirs(rows, exist_ok=True)
    p = os.path.join(rows, "generation.txt")
    if not os.path.exists(p):
        with open(p, "w") as fh:
            fh.write(uuid.uuid4().hex[:8])
    return open(p).read().strip()


def finished(machine, base=None):
    """[(name to send it under, path)] for every shard that will not be written again.

    Covers the first labels (dp) and the deeper second labels of MayFly's mistakes (dh, deep_label.py
    HARD=1), each series less its newest file, which is still being written."""
    g = generation(base)
    rows = os.path.join(base or BASE, "rows")
    out = []
    for prefix in ("dp", "dh"):
        files = sorted(glob.glob(os.path.join(rows, f"{prefix}_*.bin")))
        out += [(f"{machine}_{g}_{os.path.basename(p)}", p) for p in files[:-1]]
    return out


def sent(base=None):
    try:
        return set(json.load(open(os.path.join(base or BASE, ".rows_sent.json"))))
    except (OSError, ValueError):
        return set()


def mark_sent(name, base=None):
    base = base or BASE
    path = os.path.join(base, ".rows_sent.json")
    s = sent(base) | {name}
    os.makedirs(base, exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(sorted(s), fh)
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--rows"]:
        # This machine's finished shards, name and path one a line, for another machine to fetch.
        for name, p in finished(sys.argv[2] if len(sys.argv) > 2 else "sp"):
            print(name, p)
    else:
        print(f"enabled {enabled()}, target {target():,}, backlog {backlog():,}, "
              f"finished shards {len(finished('x'))}")
