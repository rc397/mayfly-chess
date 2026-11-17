"""Deep labels for MayFly's hard positions from Kaggle's free CPU sessions, with Colab and spare TPU cores as extra slots.

Kaggle gives a free CPU session four cores for up to twelve hours. queue.txt holds hard_positions.py lines
("FEN|uci|kind|best"), MayFly's own mistakes, checks and only-move positions, gathered from deep_label.py's
HANDOFF file (KAGGLE_SOURCES) and from files dropped into incoming/. Each notebook gets a batch costed to finish
inside its session (a deep or crit line costs 1, a check 0.08), with the session's budget written in, so it wraps
up before Kaggle stops it. kaggle_kernel.py labels the batch and returns what Stockfish said, and this file turns
that into rows with the repo's encoder and appends them to dh_kg shards, which the trainer does not read by default.

Each run is one pass and is safe to repeat, for example every five minutes from a scheduler. For each slot
(<KAGGLE_USER>/mayfly-label-00 to -09) a finished notebook whose summary names its batch is read in and its
unreached lines go back on the queue, a failed one gives its whole batch back, and a free slot gets the next batch.
slots.txt says how many slots to keep busy (0 stops new pushes and lets running notebooks finish).

accel.txt names accelerator slots to run as well (gpu without the file, none when empty). A GPU session's host
labels like a CPU session while its weekly quota lasts, with its budget cut to the quota left. A TPU session's host
is much larger, but Kaggle stops TPU sessions that never use the TPU itself, so the TPU slot is off by default. A
slot dropped from accel.txt is tended until its batch comes back. A batch over GZ_LINES lines is too big to embed in
a notebook, so it goes as <batch>.txt in the private batches dataset, which the notebook attaches (LINES_FILE).

A Colab runtime cannot be pushed to, so colab.txt says how many batches to keep waiting for it in the batches
dataset, listed in colab.json with their time and a sha1 of their lines, beside kaggle_kernel.py and
colab_worker.py. colab_worker.py puts each result into a second private dataset, and each pass reads in those that
arrived. A batch stays waiting until its result arrives, so nothing times out. external() hands batches to the spare
cores of a TPU training session (kaggle_train_ctl.py).

Every Kaggle call points KAGGLE_API_TOKEN at ~/.kaggle/access_token, so a stale token in the environment cannot win.
Nothing here reads or prints the token.

    python chess/kaggle_label.py            env KAGGLE_DIR (work folder), OUT (shard folder), KAGGLE_USER
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import chess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from encode import board_to_ids, clock_and_ep        # noqa: E402
from moves import move_to_idx                         # noqa: E402
from pack_av import DT_AV, ShardWriter, encode_av     # noqa: E402
import paths                                          # noqa: E402

E = os.environ.get
BASE = E("KAGGLE_DIR", paths.under("kaggle"))
QUEUE = os.path.join(BASE, "queue.txt")
STATE = os.path.join(BASE, "state.json")
SLOTS = os.path.join(BASE, "slots.txt")
LOG = os.path.join(BASE, "kaggle_label.log")
LOCK = os.path.join(BASE, ".lock")
OUT = E("OUT", paths.SHARDS)
USER = E("KAGGLE_USER", "")
TOKEN = os.path.expanduser(os.path.join("~", ".kaggle", "access_token"))
MAX_SLOTS = 10
# A CPU batch sized to finish. At depth 26 a CPU session labelled about 113 deep positions an hour. Kaggle ends a CPU
# or GPU session at twelve hours and a TPU session at nine, so CPU sessions run 11.5 hours and accelerator slots 8.5.
# A batch of 1,150 finishes in about ten hours and leaves the rest for kaggle_kernel.py's deepening.
BATCH_COST = float(E("KAGGLE_BATCH_COST", 1150))
CPU_SESSION_H = float(E("KAGGLE_CPU_SESSION_H", 11.5))
COST = {"deep": 1.0, "crit": 1.0, "check": 0.08}
K_CHESSBENCH = 0.00368208                              # deep_label.win_prob's curve
# Accelerator slots, sized below their measured pace (GPU host about 128 cost an hour, TPU host about 5,450) so the
# tail of each session goes on deepening.
ACCEL = {"gpu": {"acc": "NvidiaTeslaT4", "quota": "GPU", "cost_per_hour": 100.0},
         "tpu": {"acc": "TpuV5E8", "quota": "TPU", "cost_per_hour": 4800.0}}
GZ_LINES = 8000                                        # a batch longer than this goes as a dataset file
DATASET = f"{USER}/mayfly-batches"
DS_DIR = os.path.join(BASE, "dataset")
DS_WAIT = 900                                          # seconds to wait for Kaggle to make a new version ready
ACCEL_FILE = os.path.join(BASE, "accel.txt")
MIN_QUOTA_H = 1.0
SESSION_H = 8.5                                        # kaggle_kernel.py's TIME_BUDGET
KERNEL_BUDGET = "TIME_BUDGET = 8 * 3600 + 30 * 60"
# the Colab slot, a quarter of the TPU host's cores, so a quarter of its pace
COLAB_FILE = os.path.join(BASE, "colab.txt")
COLAB_RESULTS = f"{USER}/mayfly-colab-results"
COLAB_COST_PER_HOUR = float(E("KAGGLE_COLAB_COST_PER_HOUR", 1200))
COLAB_BATCH_H = float(E("KAGGLE_COLAB_BATCH_H", 1.0))
COLAB_CODE = ("kaggle_kernel.py", "colab_worker.py")
DS_KEEP = {"dataset-metadata.json", "colab.json", *COLAB_CODE}  # files a dataset version always keeps


def say(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def kaggle(*args, timeout=600):
    env = dict(os.environ, KAGGLE_API_TOKEN=TOKEN, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def name(slot):
    """A slot's name: 00 to 09 for the CPU slots, an accelerator slot's own name (gpu) otherwise."""
    return f"{slot:02d}" if isinstance(slot, int) else str(slot)


def slug(slot):
    return f"{USER}/mayfly-label-{name(slot)}"


def quota(resource):
    """Hours left of this week's GPU or TPU quota, from `kaggle quota`, or None when it cannot be read."""
    rc, out = kaggle("quota", timeout=120)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == resource:
            try:
                return float(parts[2].rstrip("h"))
            except ValueError:
                return None
    return None


def status(slot):
    """running, queued, complete, error, cancelled, or missing (never pushed) / unknown."""
    rc, out = kaggle("kernels", "status", slug(slot), timeout=120)
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?([A-Za-z_]+)', out)
    if m:
        s = m.group(1).lower()
        return {"cancel_requested": "running", "cancel_acknowledged": "cancelled"}.get(s, s)
    return "missing" if ("404" in out or "not found" in out.lower()) else "unknown"


def load_state():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except (OSError, ValueError):
        return {"place": 0, "slots": {}, "next_batch": 0}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)
    os.replace(tmp, STATE)


def complete_lines(path):
    """A file's lines up to its last newline, so a line another process is still writing is left for the next pass."""
    try:
        data = open(path, "rb").read()
    except FileNotFoundError:
        return []
    cut = data.rfind(b"\n")
    return data[:cut + 1].decode("utf-8", "replace").splitlines() if cut >= 0 else []


def queue_lines():
    return complete_lines(QUEUE)


# Where the hard positions come from. The local labeller writes every batch it finishes to handoff_queue.txt
# (deep_label.py HANDOFF), MayFly's mistakes first as "FEN|uci|deep|best" lines and the rest as plain FENs. Sources
# are read by line count, in order. Other labelling machines can drop files named by source and line range into
# incoming/, which are merged once each. kaggle_kernel.py gives a mistake line the top five at depth 40 and any
# other the top ten at depth 35.
SOURCES = [p for p in E("KAGGLE_SOURCES", os.path.join(paths.SELFPLAY, "handoff_queue.txt")).split(";") if p]
INCOMING = os.path.join(BASE, "incoming")


def gather(st):
    """Append new hard lines from the sources and from incoming/ to the end of the queue. Returns how many."""
    added = 0
    offsets = st.setdefault("sources", {})
    for p in SOURCES:
        lines = complete_lines(p)
        start = offsets.get(p)
        if start is None and not p.endswith("_queue.txt"):
            # a source of all positions, first seen, starts at its end, since earlier positions already have labels
            start = len(lines)
        if start is None:
            # a queue source, first seen, starts after whatever the local hard job already labelled from it

            done_file = os.path.join(os.path.dirname(p), "." + os.path.basename(p).replace("_queue.txt", "_labelled.json"))
            try:
                start = int(json.load(open(done_file)).get("done", 0))
            except (OSError, ValueError):
                start = 0
        if len(lines) > start:
            requeue(lines[start:])
            added += len(lines) - start
        offsets[p] = max(start, len(lines))
    os.makedirs(os.path.join(INCOMING, "done"), exist_ok=True)
    ledger_path = os.path.join(INCOMING, "merged.txt")
    merged = set(open(ledger_path, encoding="utf-8").read().split()) if os.path.exists(ledger_path) else set()
    for f in sorted(os.listdir(INCOMING)):
        p = os.path.join(INCOMING, f)
        if not f.endswith(".txt") or f == "merged.txt" or not os.path.isfile(p):
            continue
        if f not in merged:
            lines = complete_lines(p)
            requeue(lines)
            added += len(lines)
            with open(ledger_path, "a", encoding="utf-8") as fh:
                fh.write(f + "\n")
        os.replace(p, os.path.join(INCOMING, "done", f))
    if added:
        say(f"gathered {added:,} new lines into the queue")
    return added


def requeue(lines):
    """Lines a notebook did not reach go to the end of the queue, written before anyone records them as handled."""
    if lines:
        with open(QUEUE, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def win_prob(cp, mate):
    if mate is not None:
        return 1.0 if mate > 0 else 0.0
    return 1.0 / (1.0 + math.exp(-K_CHESSBENCH * cp))


def rows_for(line, moves):
    """deep_label's rows for one position from the notebook's [uci, cp, mate, depth] list."""
    b = chess.Board(line.split("|", 1)[0].strip())
    ids, cast, flip = board_to_ids(b)
    hm, ep = clock_and_ep(b, flip)
    bits = int(cast[0]) | (int(cast[1]) << 1) | (int(cast[2]) << 2) | (int(cast[3]) << 3)
    out = []
    for uci, cp, mate, _depth in moves:
        mv = chess.Move.from_uci(uci)
        if mv not in b.legal_moves:
            continue
        row = np.zeros(1, DT_AV)[0]
        encode_av(ids, [bits & 1, (bits >> 1) & 1, (bits >> 2) & 1, (bits >> 3) & 1], move_to_idx(mv, flip),
                  win_prob(cp, mate), (1 if mv.promotion else 0) | 6, row, hm=hm, ep=ep)
        out.append(row.tobytes())
    return out


def ingest(slot, job, lines):
    """Read a finished notebook in. True when its batch is accounted for (rows written, the rest requeued)."""
    d = os.path.join(BASE, "out", f"{name(slot)}_{job['batch']}")
    os.makedirs(d, exist_ok=True)
    rc, out = kaggle("kernels", "output", slug(slot), "-p", d, "-o")
    return read_in(name(slot), job, lines, d, out)


def read_in(key, job, lines, d, note=""):
    """Turn a batch's summary.json and results.jsonl in folder d into rows, for a notebook's output and a Colab
    result alike. True when its batch is accounted for (rows written, the rest requeued)."""

    try:
        summary = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
    except (OSError, ValueError):
        say(f"slot {key}: no summary yet ({note.strip()[-160:]})")
        return False
    if summary.get("batch") != job["batch"]:
        say(f"slot {key}: output is batch {summary.get('batch')}, waiting for {job['batch']}")
        return False
    body = batch_body(job, lines)
    seen, raw, counts = set(), [], {}
    try:
        recs = [json.loads(l) for l in open(os.path.join(d, "results.jsonl"), encoding="utf-8") if l.strip()]
    except (OSError, ValueError):
        recs = []
    # A notebook with time left searches its shallowest positions again and writes another record for each
    # (kaggle_kernel.py, "deepened"), so only the record whose shallowest move went deepest becomes rows.
    best = {}
    for r in recs:
        i = int(r.get("i", -1))
        if not 0 <= i < len(body):
            continue
        if "error" in r:
            continue                          # requeued below with the unreached, unless another record has it
        seen.add(i)
        if r.get("moves"):
            d = min([m[3] for m in r["moves"] if m[3] is not None] or [0])
            if i not in best or d > best[i][0]:
                best[i] = (d, r)
    for i, (d, r) in sorted(best.items()):
        raw += rows_for(body[i], r["moves"])
        if r.get("deepened"):
            counts["deepened"] = counts.get("deepened", 0) + 1
    counts["labelled"] = len(best)
    counts["cleared or skipped"] = len(seen - set(best))
    if raw:
        sw = ShardWriter(OUT, "dh", 4_000_000, tag="kg")
        sw.write_raw(b"".join(raw))
        sw.close()
    back = [body[i] for i in range(len(body)) if i not in seen]
    requeue(back)
    say(f"slot {key}: batch {job['batch']} read in: {len(raw):,} rows from {counts}, "
        f"{len(back):,} lines back on the queue, {summary.get('hours')}h on "
        + ("Colab" if key == "colab" else "Kaggle"))
    return True


# Kinds passed over when making batches. Crit lines, where MayFly found the only good move, are skipped by default so
# the mistakes clear faster, and they stay in queue.txt in case they are wanted later.
SKIP = {k.strip() for k in E("KAGGLE_SKIP_KINDS", "crit").split(",") if k.strip()}
BATCHES = os.path.join(BASE, "batches")


def make_batch(st, lines, budget=None):
    """(lo, hi, body), the next queue lines worth about BATCH_COST (or `budget`) of a session, skipping SKIP kinds.

    None when nothing is left to send. hi is the new place and body is what the notebook gets."""
    budget = BATCH_COST if budget is None else budget
    lo = hi = st["place"]
    cost, body = 0.0, []
    while hi < len(lines) and cost < budget:
        parts = lines[hi].split("|")
        kind = parts[2].strip() if len(parts) > 2 else "deep"
        hi += 1
        if kind in SKIP:
            continue
        cost += COST.get(kind, 1.0)
        body.append(lines[hi - 1])
    if not body:
        st["place"] = hi                      # nothing but skipped kinds, so pass over them
        return None
    return lo, hi, body


def batch_body(job, lines):
    """The lines a batch's notebook was given, its saved body, else the queue's own slice (for old batches whose
    bodies were not saved, which skipped nothing)."""
    p = os.path.join(BATCHES, job["batch"] + ".txt")
    if os.path.exists(p):
        return complete_lines(p)
    lo, hi = job["lines"]
    return lines[lo:hi]


def live_batches(st):
    """Batches still out, whose files the private dataset keeps, the unfinished slots' and those waiting for Colab."""
    return {j["batch"] for j in st["slots"].values() if not j.get("done")} | set(st.get("colab", {}))


def ds_ready():
    """The dataset's state from `kaggle datasets status`, ready, pending, missing, or what Kaggle said."""
    rc, out = kaggle("datasets", "status", DATASET, timeout=120)
    low = out.lower()
    if "404" in low or "not found" in low or "403" in low:
        return "missing"                      # Kaggle answers 403 for a dataset that does not exist
    for s in ("ready", "pending", "processing", "error"):
        if s in low:
            return s
    return out.strip()[-120:] or "unknown"


def ds_publish(files, keep, note, need=None):
    """Write `files` ({name: text}) into the private dataset's folder and publish it as a new version.

    The batches in `keep` that are still out stay beside them. Waits until Kaggle lists the names in `need` (all of
    `files` when not given). Returns True when it does, False when the version went up but is not listed yet, None
    when the upload failed. Plain text, because Kaggle unpacks an uploaded .gz itself. Files of batches already read
    in move to old/ (the canonical body stays in batches/), which is never uploaded. DS_KEEP files always stay."""
    os.makedirs(os.path.join(DS_DIR, "old"), exist_ok=True)
    meta = os.path.join(DS_DIR, "dataset-metadata.json")
    if not os.path.exists(meta):
        with open(meta, "w", encoding="utf-8") as fh:
            json.dump({"title": "mayfly batches", "id": DATASET, "licenses": [{"name": "CC0-1.0"}]}, fh, indent=1)
    for f, text in files.items():
        with open(os.path.join(DS_DIR, f), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    live = set(keep) | {f.split(".", 1)[0] for f in files}
    for f in os.listdir(DS_DIR):
        stem = f.split(".", 1)[0]
        if f not in DS_KEEP and os.path.isfile(os.path.join(DS_DIR, f)) and stem not in live:
            os.replace(os.path.join(DS_DIR, f), os.path.join(DS_DIR, "old", f))
    if ds_ready() == "missing":
        rc, out = kaggle("datasets", "create", "-p", DS_DIR, timeout=900)
        if rc != 0 and "already" in out.lower():
            rc, out = kaggle("datasets", "version", "-p", DS_DIR, "-m", note, timeout=900)
    else:
        rc, out = kaggle("datasets", "version", "-p", DS_DIR, "-m", note, timeout=900)
        if rc != 0 and re.search(r"403|404", out):
            rc, out = kaggle("datasets", "create", "-p", DS_DIR, timeout=900)
    if rc != 0:
        say(f"dataset {DATASET}: upload of {note} FAILED: {out.strip()[-200:]}")
        return None
    need = list(files) if need is None else list(need)
    t0 = time.time()
    while time.time() - t0 < DS_WAIT:
        s = ds_ready()
        # ready means the file is listed, since straight after an upload Kaggle still calls the last version ready
        if s == "ready":
            listing = kaggle("datasets", "files", DATASET, "--page-size", "200", timeout=120)[1] if need else ""
            if all(n in listing for n in need):
                say(f"dataset {DATASET}: {note} ready after {time.time() - t0:.0f}s")
                return True
        if s == "error":
            break
        time.sleep(20)
    say(f"dataset {DATASET}: {note} not listed after {time.time() - t0:.0f}s ({ds_ready()})")
    return False


def push(slot, batch_id, body, acc=None, seconds=None, keep=()):
    """Push one batch as the slot's notebook, a CPU session, or `acc` (an accelerator id), with its budget cut to
    `seconds`. A batch over GZ_LINES goes as a dataset file (ds_publish, beside the batches in `keep`)."""
    d = os.path.join(BASE, "push", name(slot))
    os.makedirs(d, exist_ok=True)
    src = open(os.path.join(HERE, "kaggle_kernel.py"), encoding="utf-8").read()
    if seconds:
        if KERNEL_BUDGET not in src:
            say(f"slot {name(slot)}: kaggle_kernel.py has no '{KERNEL_BUDGET}' to shorten; not pushed")
            return False
        src = src.replace(KERNEL_BUDGET, f"TIME_BUDGET = {int(seconds)}")
    sources = []
    if len(body) > GZ_LINES:
        if '"__LINES_FILE__"' not in src:
            say(f"slot {name(slot)}: kaggle_kernel.py cannot read a batch file; not pushed")
            return False
        if not ds_publish({batch_id + ".txt": "\n".join(body) + "\n"}, keep, f"{batch_id} ({len(body):,} lines)"):
            push.last = "dataset not ready"
            return False
        src = src.replace('"__LINES_FILE__"', f'"{batch_id}.txt"').replace('r"""__LINES__"""', 'r""""""')
        sources = [DATASET]
    else:
        src = src.replace("__LINES__", "\n".join(body))
    src = src.replace("__BATCH__", batch_id)
    with open(os.path.join(d, "label.py"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    meta = {"id": slug(slot), "title": f"mayfly label {name(slot)}", "code_file": "label.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": bool(acc and acc.startswith("Nvidia")),
            "enable_tpu": bool(acc and acc.startswith("Tpu")), "enable_internet": True, "dataset_sources": sources,
            "competition_sources": [], "kernel_sources": []}
    if acc:
        meta["machine_shape"] = acc
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    rc, out = kaggle("kernels", "push", "-p", d, *(("--accelerator", acc) if acc else ()))
    push.last = out
    ok = rc == 0 and "successfully pushed" in out
    say(f"slot {name(slot)}: push of batch {batch_id} ({len(body):,} lines, {len(src) / 1e6:.1f} MB)"
        + (f" on {acc}" if acc else "") + (f", {int(seconds) / 3600:.2f}h" if seconds else "")
        + (" ok" if ok else f" FAILED: {out.strip()[-200:]}"))
    return ok


push.last = ""
TOO_BIG = re.compile(r"(?i)too large|too big|entity|payload|413|size limit|exceed")


def tend(slot, st, lines):
    """Check a slot's notebook. True while it is busy (running, queued, or finished but not readable yet).

    A finished one is read in and a failed one gives its batch back, and either way the slot is then free (False)."""
    key = name(slot)
    job = st["slots"].get(key)
    if not job or job.get("done"):
        return False
    s = status(slot)
    if s in ("running", "queued"):
        return True
    if s == "complete":
        if not ingest(slot, job, lines):
            return True                       # finished but not readable yet, so try again next pass
        job["done"] = True
    elif s in ("error", "cancelled"):
        back = batch_body(job, lines)
        requeue(back)
        job["done"] = True
        say(f"slot {key}: batch {job['batch']} ended {s}, its {len(back):,} lines back on the queue")
    else:
        say(f"slot {key}: status {s}, leaving it")
        return True
    save_state(st)
    return False


def send(slot, st, budget=None, acc=None, seconds=None):
    """Send the next batch to a free slot. True when pushed, False when refused (the cap, an outage), None when the
    queue has nothing to send."""
    lines = queue_lines()                     # requeued lines are part of it now
    budget = BATCH_COST if budget is None else budget
    for halvings in range(4):
        span = make_batch(st, lines, budget)
        if span is None:
            save_state(st)                    # the place may have moved past skipped kinds
            return None
        lo, hi, body = span
        batch_id = f"b{st['next_batch']:05d}"
        os.makedirs(BATCHES, exist_ok=True)
        with open(os.path.join(BATCHES, batch_id + ".txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(body) + "\n")  # before the push, so a read-in always finds what was sent
        if push(slot, batch_id, body, acc, seconds, live_batches(st)):
            break
        if halvings == 3 or not TOO_BIG.search(push.last or ""):
            return False
        budget /= 2                           # refused as too large, so retry from the same place with half the batch
    st["slots"][name(slot)] = {"batch": batch_id, "lines": [lo, hi], "pushed": time.strftime("%Y-%m-%d %H:%M")}
    st["place"] = hi
    st["next_batch"] += 1
    save_state(st)
    return True


def accel(st, lines):
    """The accelerator slots accel.txt names, tended like the CPU slots and refilled while their week's quota lasts.
    A slot dropped from accel.txt is still tended until its batch comes back, and gets no new push. Returns how many
    are busy."""
    try:
        names = open(ACCEL_FILE, encoding="utf-8").read().replace(",", " ").split()
    except OSError:
        names = ["gpu"]
    busy = 0
    waits = st.setdefault("accel_wait", {})
    dropped = [n for n, job in st["slots"].items() if n not in names and ACCEL.get(n.rstrip("0123456789"))
               and not job.get("done")]
    for n in dropped:
        if tend(n, st, lines):
            busy += 1
    for n in names:
        cfg = ACCEL.get(n.rstrip("0123456789"))
        if cfg is None:
            continue
        if tend(n, st, lines):
            busy += 1
            continue
        if time.time() < waits.get(n, 0):
            continue
        left = quota(cfg["quota"])
        if left is None or left < MIN_QUOTA_H:
            waits[n] = time.time() + 3600     # look again in an hour, the quota resets weekly
            save_state(st)
            continue
        hours = min(SESSION_H, left - 0.25)
        got = send(n, st, budget=cfg["cost_per_hour"] * hours, acc=cfg["acc"], seconds=hours * 3600)
        if got:
            busy += 1
        elif got is False:
            waits[n] = time.time() + 1800     # refused, so try again in half an hour
            save_state(st)
    return busy


def colab_want():
    """How many batches colab.txt wants kept waiting for Colab (none without the file)."""
    try:
        return max(0, min(20, int(open(COLAB_FILE).read().strip())))
    except (OSError, ValueError):
        return 0


def colab_back(st, jobs, b, lines, why):
    """Give back a batch waiting for Colab, its lines to the end of the queue and its entry removed."""
    back = batch_body(jobs[b], lines)
    requeue(back)
    del jobs[b]
    save_state(st)
    say(f"slot colab: batch {b} {why}, its {len(back):,} lines back on the queue")


def colab_read(st, jobs, lines):
    """Read in the results that have arrived for batches still waiting.

    They are listed in the results dataset, which is downloaded whole (colab_worker.py keeps it to the results not
    yet read in), and each one's two files are moved out."""
    rc, out = kaggle("datasets", "files", COLAB_RESULTS, "--page-size", "200", timeout=120)
    arrived = sorted(set(re.findall(r"\b(b\d{5})\.summary\.json\b", out)) & set(jobs)) if rc == 0 else []
    if not arrived:
        return
    dl = os.path.join(BASE, "out", "colab_dl")
    os.makedirs(dl, exist_ok=True)
    rc, out = kaggle("datasets", "download", COLAB_RESULTS, "-p", dl, "--unzip", "-o", "-q", timeout=900)
    if rc != 0:
        say(f"slot colab: results for {', '.join(arrived)} not downloaded: {out.strip()[-200:]}")
        return
    for b in arrived:
        d = os.path.join(BASE, "out", f"colab_{b}")
        os.makedirs(d, exist_ok=True)
        pairs = ((b + ".summary.json", "summary.json"), (b + ".results.jsonl", "results.jsonl"))
        if not all(os.path.exists(os.path.join(dl, f)) for f, _ in pairs):
            say(f"slot colab: batch {b} listed but not in the download; next pass")
            continue
        for f, to in pairs:
            os.replace(os.path.join(dl, f), os.path.join(d, to))
        try:
            sha = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8")).get("sha1")
        except (OSError, ValueError):
            sha = None
        if sha != jobs[b]["sha1"]:
            # labelled from other lines under this id, so never made into rows
            colab_back(st, jobs, b, lines, f"came back labelled from other lines (sha1 {str(sha)[:12]})")
        elif read_in("colab", jobs[b], lines, d, out):
            del jobs[b]
            save_state(st)


def colab(st):
    """The Colab slot (see the top). Results are read in, batches past colab.txt's number handed back, the rest topped
    up to it, and colab.json with the code published whenever any of it changed. Returns how many batches wait."""
    want = colab_want()
    jobs = st.setdefault("colab", {})
    if jobs:
        colab_read(st, jobs, queue_lines())
    for b in sorted(jobs)[want:]:
        colab_back(st, jobs, b, queue_lines(), f"handed back (colab.txt wants {want})")
    place, next_batch = st["place"], st["next_batch"]
    lines = queue_lines()                     # requeued lines are part of it now
    new = {}
    while len(jobs) + len(new) < want:
        span = make_batch(st, lines, COLAB_COST_PER_HOUR * COLAB_BATCH_H)
        if span is None:
            break
        lo, hi, body = span
        b = f"b{st['next_batch']:05d}"
        text = "\n".join(body) + "\n"
        os.makedirs(BATCHES, exist_ok=True)
        with open(os.path.join(BATCHES, b + ".txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        new[b] = {"batch": b, "lines": [lo, hi], "pushed": time.strftime("%Y-%m-%d %H:%M"),
                  "seconds": int(COLAB_BATCH_H * 3600), "n": len(body),
                  "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest()}
        st["place"] = hi
        st["next_batch"] += 1
    out = dict(jobs, **new)
    batches = [{"batch": b, "file": b + ".txt", "lines": j["n"], "seconds": j["seconds"], "sha1": j["sha1"]}
               for b, j in sorted(out.items())]
    code = {f: open(os.path.join(HERE, f), encoding="utf-8").read() for f in COLAB_CODE}
    sig = hashlib.sha1(json.dumps([batches, code]).encode("utf-8")).hexdigest()
    if sig == st.get("colab_sig") or (not out and "colab_sig" not in st):
        save_state(st)                        # the place may have moved past skipped kinds
        return len(jobs)
    files = {b + ".txt": open(os.path.join(BATCHES, b + ".txt"), encoding="utf-8").read() for b in out}
    files["colab.json"] = json.dumps({"made": time.strftime("%Y-%m-%d %H:%M"), "results": COLAB_RESULTS,
                                      "batches": batches}, indent=1) + "\n"
    files.update(code)
    ok = ds_publish(files, live_batches(st) | set(out), "colab " + (", ".join(new) or "update"),
                    need=[b + ".txt" for b in new])
    if ok is None:                            # nothing went up, so the new batches were never out
        st["place"], st["next_batch"] = place, next_batch
        return len(jobs)
    jobs.update(new)
    if ok:
        st["colab_sig"] = sig                 # else not listed yet, and the same files go up again next pass
    save_state(st)
    if new:
        say(f"slot colab: {', '.join(new)} waiting for Colab ({sum(j['n'] for j in new.values()):,} lines), "
            f"{len(jobs)} in all")
    return len(jobs)


EXT_DIR = os.path.join(BASE, "trainlab")


def external(st):
    """A batch for the spare host cores of a TPU training session (kaggle_train_ctl.py).

    The training controller asks with request.json ({"budget": positions}). This pass makes the batch from the queue
    as for a notebook, records it as slot "trainlab", and writes the batch file and ready.json. The controller ships it
    with the session and, when the session is home, writes done.json ({"batch", "ok", "dir"}). This pass then reads
    the results in from that folder like a notebook's output, or gives the whole batch back when the session failed.
    Every state change stays inside this labeller's own pass. Returns 1 while a batch is out there, else 0."""
    os.makedirs(EXT_DIR, exist_ok=True)
    req, ready, done = (os.path.join(EXT_DIR, f) for f in ("request.json", "ready.json", "done.json"))
    lines = queue_lines()
    job = st["slots"].get("trainlab")
    if job and not job.get("done"):
        if not os.path.exists(done):
            return 1
        d = json.load(open(done, encoding="utf-8"))
        if d.get("batch") != job["batch"]:
            return 1
        if d.get("ok"):
            if not read_in("trainlab", job, lines, d["dir"]):
                return 1                      # not readable yet, try next pass
        else:
            back = batch_body(job, lines)
            requeue(back)
            say(f"slot trainlab: batch {job['batch']} came back without results, its {len(back):,} lines back on "
                f"the queue")
        job["done"] = True
        save_state(st)
        for p in (done, ready):
            try:
                os.remove(p)
            except OSError:
                pass
        return 0
    if not os.path.exists(req) or os.path.exists(ready):
        return 0
    try:
        budget = float(json.load(open(req, encoding="utf-8"))["budget"])
    except (OSError, ValueError, KeyError):
        return 0
    span = make_batch(st, lines, budget)
    if span is None:
        save_state(st)
        return 0
    lo, hi, body = span
    batch_id = f"b{st['next_batch']:05d}"
    os.makedirs(BATCHES, exist_ok=True)
    for p in (os.path.join(BATCHES, batch_id + ".txt"), os.path.join(EXT_DIR, batch_id + ".txt")):
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(body) + "\n")
    st["slots"]["trainlab"] = {"batch": batch_id, "lines": [lo, hi], "pushed": time.strftime("%Y-%m-%d %H:%M"),
                               "external": True}
    st["place"] = hi
    st["next_batch"] += 1
    save_state(st)
    with open(ready, "w", encoding="utf-8") as fh:
        json.dump({"batch": batch_id, "file": os.path.join(EXT_DIR, batch_id + ".txt"), "positions": len(body)}, fh)
    os.remove(req)
    say(f"slot trainlab: batch {batch_id} ({len(body):,} lines) ready for a TPU training session's spare cores")
    return 1


def main():
    os.makedirs(BASE, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        if time.time() - os.path.getmtime(LOCK) < 3600:
            print("another kaggle_label pass is running", flush=True)
            return 0
        os.utime(LOCK, None)                  # a lock older than an hour was left by a pass that died
    try:
        try:
            want = max(0, min(MAX_SLOTS, int(open(SLOTS).read().strip())))
        except (OSError, ValueError):
            want = 0                          # without slots.txt nothing new is pushed
        st = load_state()
        gather(st)
        save_state(st)
        lines = queue_lines()
        busy = 0
        for slot in range(MAX_SLOTS):
            if tend(slot, st, lines):
                busy += 1
                continue
            if slot >= want:
                continue
            got = send(slot, st, seconds=CPU_SESSION_H * 3600)
            if got:
                busy += 1
            elif got is False:
                break                         # a refused push (the cap, an outage) ends this pass

        on_accel = accel(st, queue_lines())
        try:
            on_colab = colab(st)
        except Exception as e:                # a Colab failure is reported and the pass still ends cleanly
            on_colab = len(st.get("colab", {}))
            say(f"slot colab: failed this pass: {type(e).__name__}: {str(e)[:200]}")
        try:
            on_ext = external(st)
        except Exception as e:                # a failure here is reported and the pass still ends cleanly
            on_ext = 0
            say(f"slot trainlab: failed this pass: {type(e).__name__}: {str(e)[:200]}")
        left = len(queue_lines()) - st["place"]
        say(f"pass done: {busy} notebook(s) busy, {want} wanted, {on_accel} on accelerators, "
            f"{on_colab} waiting for Colab, {on_ext} on a TPU training session, {left:,} queue lines not yet sent")
        return 0
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
