"""Collect volunteers' runs of the public Kaggle notebook and check them before they're used.

Looks at every public copy of rc7530/mayfly-selfplay-volunteer (or any notebook using the public dataset), downloads
each finished version of it once, and keeps it only if it passes the checks below. Every version is fetched, not just
the newest, since volunteers often press Save Version several times on one copy. Kept runs go to
VOL_OUT/<user>/<run>, rejected ones to VOL_OUT/_rejected with the reason. Labelling runs are turned into deep-label
rows (dh shards tagged "vl").
"""
import gzip
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

E = os.environ.get
PARENT = "rc7530/mayfly-selfplay-volunteer"
DATASET = "rc7530/mayfly-selfplay-public"
OWN = {"rc7530/mayfly-selfplay-volunteer", "rc7530/mayfly-selfplay-volunteer-test"}
VOL_OUT = E("VOL_OUT", paths.under("selfplay", "volunteer"))
LEDGER = os.path.join(VOL_OUT, "ledger.json")
VERSION = E("VOL_VERSION", "mayfly_v6.pt")
MIN_VISITS = int(E("VOL_MIN_VISITS", 800))
SAMPLE, MAX_BAD = 300, 0.02
MAX_VERSIONS = int(E("VOL_MAX_VERSIONS", 40))     # versions looked at per copy
FILES = ["summary.json", "targets.jsonl.gz", "positions.txt.gz", "fast.jsonl.gz", "results.jsonl", "chunk.txt"]


def kaggle(*args, timeout=1800):
    tok = os.path.expanduser(os.path.join("~", ".kaggle", "access_token"))
    env = dict(os.environ, KAGGLE_API_TOKEN=tok, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def client():
    """A Kaggle API client. The command line tool can only fetch a notebook's newest output, the API any version's."""
    os.environ["KAGGLE_API_TOKEN"] = os.path.expanduser(os.path.join("~", ".kaggle", "access_token"))
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api.build_kaggle_client()


def fetch_version(k, ref, v, dest):
    """Download version v's output files into dest. False if it has no summary yet (still running, or no such version)."""
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    owner, slug = ref.split("/")
    os.makedirs(dest, exist_ok=True)
    for name in FILES:
        req = ApiDownloadKernelOutputRequest()
        req.owner_slug, req.kernel_slug, req.file_path, req.version_number = owner, slug, name, v
        try:
            r = k.kernels.kernels_api_client.download_kernel_output(req)
        except Exception:
            if name == "summary.json":
                return False
            continue
        with open(os.path.join(dest, name), "wb") as fh:
            fh.write(r.content)
    return True


def copies():
    refs = set()
    for flag, val in (("--parent", PARENT), ("--dataset", DATASET)):
        rc, out = kaggle("kernels", "list", flag, val, "--page-size", "200")
        refs |= set(re.findall(r"^([\w-]+/[\w-]+)\s", out, re.M))
    return sorted(r for r in refs if r not in OWN and not r.startswith("ref"))


def best_records(d):
    """The chunk a labelling run was given, and the deepest result it wrote for each position."""
    chunk = [l for l in open(os.path.join(d, "chunk.txt"), encoding="utf-8").read().splitlines() if l.strip()]
    best = {}
    for line in open(os.path.join(d, "results.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        i = int(r.get("i", -1))
        if not 0 <= i < len(chunk) or "error" in r or not r.get("moves"):
            continue
        dep = min([m[3] for m in r["moves"] if m[3] is not None] or [0])
        if i not in best or dep > best[i][0]:
            best[i] = (dep, r)
    return chunk, best


def check_labels(d):
    try:
        chunk, best = best_records(d)
    except (OSError, ValueError) as e:
        return f"labels unreadable ({type(e).__name__})"
    if not best:
        return "no labelled positions"
    bad = deep = 0
    sample = random.Random(7).sample(sorted(best), min(SAMPLE, len(best)))
    for i in sample:
        dep, r = best[i]
        try:
            b = chess.Board(chunk[i].split("|", 1)[0].strip())
            ok = b.is_valid() and all(chess.Move.from_uci(m[0]) in b.legal_moves
                                      and (m[2] is not None or abs(m[1]) <= 32000) for m in r["moves"])
        except Exception:
            ok = False
        bad += not ok
        deep += dep >= 35
    if bad / len(sample) > MAX_BAD:
        return f"{100 * bad / len(sample):.1f}% of sampled labels malformed"
    # a few positions can't reach depth 35 in one session, but most should
    if deep / len(sample) < 0.5:
        return f"only {100 * deep / len(sample):.0f}% of sampled positions reached depth 35"
    return None


def ingest_labels(d):
    """Write a kept labelling run as dh rows. Returns (positions, rows)."""
    import kaggle_label as kl
    chunk, best = best_records(d)
    raw = []
    for i, (dep, r) in sorted(best.items()):
        raw += kl.rows_for(chunk[i], r["moves"])
    if raw:
        sw = kl.ShardWriter(kl.OUT, "dh", 4_000_000, tag="vl")
        sw.write_raw(b"".join(raw))
        sw.close()
    return len(best), len(raw)


def check(d):
    """None if the run looks good, otherwise the reason it doesn't."""
    try:
        summ = json.load(open(os.path.join(d, "summary.json")))
    except (OSError, ValueError):
        return "no summary.json"
    if summ.get("kind") == "label":
        return check_labels(d)
    if summ.get("accel") not in ("tpu", "gpu"):
        return f"no accelerator ({summ.get('accel')})"
    if summ.get("version") != VERSION:
        return f"played {summ.get('version')}, not {VERSION}"
    try:
        with gzip.open(os.path.join(d, "targets.jsonl.gz"), "rt", encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh]
    except (OSError, ValueError) as e:
        return f"targets unreadable ({type(e).__name__})"
    if not rows:
        return "no target rows"
    bad = 0
    for r in random.Random(7).sample(rows, min(SAMPLE, len(rows))):
        try:
            b = chess.Board(r["fen"])
            n = r["n"]
            ok = (b.is_valid() and sum(n.values()) >= MIN_VISITS // 2
                  and all(chess.Move.from_uci(m) in b.legal_moves for m in n)
                  and all(-1.0001 <= v <= 1.0001 for v in r["qa"].values())
                  and (r.get("played") is None or chess.Move.from_uci(r["played"]) in b.legal_moves)
                  and r.get("z") in (None, 0.0, 0.5, 1.0))
        except Exception:
            ok = False
        bad += not ok
    frac = bad / min(SAMPLE, len(rows))
    if frac > MAX_BAD:
        return f"{100 * frac:.1f}% of sampled rows malformed"
    return None


def main():
    os.makedirs(VOL_OUT, exist_ok=True)
    seen = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {}
    with client() as k:
        for ref in copies():
            for v in range(1, MAX_VERSIONS + 1):
                if f"{ref}/{v}" in seen:
                    continue
                tmp = os.path.join(VOL_OUT, "_incoming", f"{ref.replace('/', '__')}_v{v}")
                shutil.rmtree(tmp, ignore_errors=True)
                if fetch_version(k, ref, v, tmp):
                    take(ref, v, tmp, seen)
                else:
                    shutil.rmtree(tmp, ignore_errors=True)
    json.dump(seen, open(LEDGER, "w"), indent=1)


def take(ref, v, tmp, seen):
    """Check one downloaded version, file it as kept or rejected, and turn kept labels into rows."""
    try:
        run = json.load(open(os.path.join(tmp, "summary.json"))).get("run", f"v{v}")
    except (OSError, ValueError):
        run = f"v{v}"
    run = f"{run}_v{v}"            # unique even if two versions started in the same minute
    key = f"{ref}/{v}"
    why = check(tmp)
    user = ref.split("/")[0]
    dest = os.path.join(VOL_OUT, user, run) if why is None else os.path.join(VOL_OUT, "_rejected", user, run)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(tmp, dest)
    if why is None and os.path.exists(os.path.join(dest, "chunk.txt")):
        n_pos, n_rows = ingest_labels(dest)
        sp = os.path.join(dest, "summary.json")
        summ = json.load(open(sp))
        summ["labelled"] = n_pos      # the leaderboard reads this
        json.dump(summ, open(sp, "w"), indent=1)
        print(f"  {n_pos:,} positions labelled, {n_rows:,} rows written", flush=True)
    seen[key] = {"at": time.strftime("%Y-%m-%d %H:%M"), "kept": why is None, "why": why}
    print(f"{key}: {'kept' if why is None else 'REJECTED: ' + why}", flush=True)
    json.dump(seen, open(LEDGER, "w"), indent=1)      # saved after each one, so a crash loses nothing


if __name__ == "__main__":
    main()
