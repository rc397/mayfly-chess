r"""loader_fast_check checks that LOADER_PIN and LEGAL_WORKER change nothing the trainer is fed.

Every training loader train_av builds for a run is made twice on one seed, as today and with both settings on, and
N batches of each are compared tensor by tensor (dtype, shape, every byte), with the shard reads behind them. The
legality worker is checked against pack.legal_mask on rows from every policy source and from the frozen validation
set (an array saved under an older record). Then each side's batch statistics, which agree when the bytes do.

  python loader_fast_check.py ENV_JSON [N] [VAL_NPY]     ENV_JSON: a run's env (main_env.json, autostart.json)
"""
import hashlib
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import pack  # noqa: E402
from data_av import AVLoader  # noqa: E402
from data_packed import PackedLoader  # noqa: E402
from loader_fast import LegalWorker  # noqa: E402
from moves import N_MOVES  # noqa: E402

env = json.load(open(sys.argv[1], encoding="utf-8-sig"))
env = env.get("env", env)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
VAL_NPY = sys.argv[3] if len(sys.argv) > 3 else None
S, AV, POL = env["SHARDS"], int(env.get("AV_BSZ", 96)), int(env.get("POL_BSZ", 32))
share = lambda k, n: int(round(n * float(env.get(k, 0.0))))
rows = {k: share(k + "_SHARE", AV) for k in ("DP", "LV", "FS", "DH", "ST")}
MLH = env.get("MLH", "0") == "1"
mlh_rows = share("MLH_SHARE", POL) if MLH else 0
sv_rows = share("SV_SHARE", POL)
LEGAL = env.get("MASK_LEGAL", "0") == "1"
since = time.mktime(time.strptime(env.get("MLH_SINCE", "2026-09-24 19:10"), "%Y-%m-%d %H:%M"))
sources = None if env.get("SOURCES", "lz,lc").lower() == "all" else env.get("SOURCES", "lz,lc").split(",")
SEED = 12345
dev = "cuda"

# the training loaders exactly as train_av's main builds them, on fixed seeds and no refresh
specs = {"av": (AVLoader, dict(batch=AV - sum(rows.values()), seed=SEED))}
thin = {"DP": int(env.get("DP_THIN", 32)), "LV": int(env.get("LV_THIN", 8)), "FS": int(env.get("FS_THIN", 8)),
        "DH": int(env.get("DH_THIN", 4)), "ST": int(env.get("ST_THIN", 4))}
off = {"DP": 2, "LV": 4, "FS": 5, "DH": 6, "ST": 7}
for k, n in rows.items():
    if n:
        kw = dict(batch=n, seed=SEED + off[k], prefix=k.lower(), thin=thin[k])
        if k == "DP" and env.get("DP_DOWN_KEYS") and float(env.get("DP_DOWN_WEIGHT", 1.0)) < 1.0:
            kw.update(down_keys=env["DP_DOWN_KEYS"], down_weight=float(env["DP_DOWN_WEIGHT"]))
        specs[k.lower()] = (AVLoader, kw)
if sv_rows:
    specs["sv"] = (PackedLoader, dict(batch=sv_rows, seed=SEED + 7, sources=("sv",), legal=LEGAL, mleft=MLH))
specs["pol"] = (PackedLoader, dict(batch=POL - mlh_rows - sv_rows, seed=SEED + 3, sources=sources, legal=LEGAL,
                                   mleft=MLH))
if MLH:
    specs["mlh"] = (PackedLoader, dict(batch=mlh_rows, seed=SEED + 5, sources=("lz",), legal=LEGAL, mleft=True,
                                       need_flag=8, min_mtime=since))

# every shard read, so the two sides can be shown to read the same rows of the same shards
reads, rlock = [], threading.Lock()
_get = pack.ShardReader.__getitem__


def _logged(self, k):
    out = _get(self, k)
    with rlock:
        reads.append((os.path.basename(self.path), k.start if isinstance(k, slice) else k, len(out)
                      if isinstance(k, slice) else 1))
    return out


pack.ShardReader.__getitem__ = _logged


def stop(ld):
    ld.close()
    while ld.t.is_alive():
        try:
            ld.q.get_nowait()
        except Exception:
            time.sleep(0.02)


def stats(name, b):
    """A batch's statistics: what the step sees from it."""
    if len(b) == 6:                                   # action values: ids, cast, hm, ep, mv, q
        return {"q": b[5].double().mean().item(), "mv": b[4].double().mean().item(),
                "hm_unknown": (b[2] == 101).double().mean().item(), "ep": (b[3] > 0).double().mean().item(),
                "pieces": (b[0] > 0).double().sum(1).mean().item()}
    pol = b[4].double()
    ent = -(pol * torch.log(pol.clamp_min(1e-30))).sum(1).mean().item()
    s = {"value": b[5].double().mean().item(), "policy_entropy": ent, "aux": b[11].double().mean().item(),
         "pieces": (b[0] > 0).double().sum(1).mean().item()}
    if LEGAL:
        s["legal_per_row"] = b[12].double().sum(1).mean().item()
    if MLH:
        s["mleft"] = b[-2].double().mean().item()
        s["mleft_labelled"] = b[-1].double().mean().item()
    return s


def run(name, cls, kw, fast):
    reads.clear()
    ld = cls(S, split="train", device=dev, refresh=0, **kw,
             **({"pin": True} if fast else {}), **({"legal_worker": True} if fast and cls is PackedLoader else {}))
    digests, tot = [], {}
    for _ in range(N):
        b = ld.next()
        torch.cuda.synchronize()
        h = hashlib.sha1()
        for t in b:
            c = t.cpu().contiguous()
            h.update(str((c.dtype, tuple(c.shape))).encode())
            h.update(c.numpy().tobytes())
        digests.append(h.hexdigest())
        for k, v in stats(name, b).items():
            tot[k] = tot.get(k, 0.0) + v / N
    stop(ld)
    with rlock:
        log = list(reads)
    return digests, tot, log, ld


ok = True
for name, (cls, kw) in specs.items():
    da, sa, ra, la = run(name, cls, kw, False)
    db, sb, rb, lb = run(name, cls, kw, True)
    same = sum(x == y for x, y in zip(da, db))
    m = min(len(ra), len(rb))
    shards = {}
    for s, _, n in ra[:m]:
        shards[s.split("_")[0]] = shards.get(s.split("_")[0], 0) + n
    print(f"{name:4s} batch {kw['batch']:3d}: {same}/{N} batches identical byte for byte; shard reads "
          f"{'identical' if ra[:m] == rb[:m] else 'DIFFERENT'} over the first {m} (of {len(ra)}/{len(rb)}), "
          f"{len({s for s, _, _ in ra[:m]})} shards, rows by source {shards}; fast side "
          f"{'pinned' if lb.ship is not None else 'NOT pinned'}"
          + (f", masks from a worker ({'alive' if not lb.mask_fn.dead else 'FELL BACK'})"
             if getattr(lb, "mask_fn", None) is not None else ""), flush=True)
    print(f"      today {json.dumps({k: round(v, 6) for k, v in sa.items()})}")
    print(f"      fast  {json.dumps({k: round(v, 6) for k, v in sb.items()})}", flush=True)
    ok &= same == N and ra[:m] == rb[:m]

# the worker against legal_mask on rows from each policy source and the frozen validation set
w = LegalWorker("check")
sets = []
for src in (["lz"], ["lc"], ["sv"]):
    try:
        ld = PackedLoader(S, batch=64, split="train", device="cpu", seed=99, sources=src, refresh=0)
    except SystemExit:
        continue
    stop(ld)
    ld._refill()
    sets.append((src[0], ld.pool[:20000]))
if VAL_NPY and os.path.exists(VAL_NPY):
    v = np.load(VAL_NPY)
    sets.append((f"validation {os.path.basename(VAL_NPY)} ({len(v.dtype.names)} fields)", v[:20000]))
for label, r in sets:
    t0 = time.perf_counter()
    a = np.concatenate([pack.legal_mask(r[i:i + 64], N_MOVES) for i in range(0, len(r), 64)])
    t1 = time.perf_counter()
    b = np.concatenate([w(r[i:i + 64], N_MOVES) for i in range(0, len(r), 64)])
    t2 = time.perf_counter()
    eq = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    ok &= eq and not w.dead
    print(f"legal masks, {label}: {len(r):,} rows, worker {'identical' if eq else 'DIFFERENT'} to legal_mask "
          f"({a.sum(1).mean():.2f} legal moves a row; {1e6 * (t1 - t0) / len(r):.0f} us a row here, "
          f"{1e6 * (t2 - t1) / len(r):.0f} us through the worker)", flush=True)
w.close()
print("ALL IDENTICAL" if ok else "MISMATCH", flush=True)
