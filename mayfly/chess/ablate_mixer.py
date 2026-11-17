"""Ablate the tropical mixer fixes one at a time on a fixed training budget.

The trained mixer averages over 7 to 27 of the 65 squares instead of selecting from them.
LEARN_TS gives the square stage its own learned temperature, which makes it select. MIX_NORM
rescales the mixer output to unit RMS with a learned gain. Each arm runs over several seeds so the
seed spread can be read next to the effect. Every arm gets the same budget, so the result is a
screen, not a converged comparison.

    python chess/ablate_mixer.py [ITERS]
    SEEDS=0,1,2 sets the seeds.
"""
import os, sys, json, time, subprocess
import statistics as stat

HERE = os.path.dirname(os.path.abspath(__file__))
ITERS = sys.argv[1] if len(sys.argv) > 1 else "8000"
BSZ, D, L = "128", "256", "4"

# the tuned tropical configuration from the sweep, held fixed across every arm
BASE = {"ARCH": "trop", "LR": "3e-3", "T": "0.3", "H": "4", "SC": "3.0",
        "VAL_EVERY": "1000", "VAL_BATCHES": "20", "RESUME": "0",
        "CRAWL_WINDOW": "999", "CRAWL_THRESH": "0.0", "MAX_HOURS": "1.0"}

ARMS = [
    ("base",    {}),
    ("ts",      {"LEARN_TS": "1"}),
    ("norm",    {"MIX_NORM": "1"}),
    ("ts_norm", {"LEARN_TS": "1", "MIX_NORM": "1"}),
]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
OUT = os.path.join(HERE, "ablate_mixer.json")


def run(name, extra, seed):
    tag = f"ab_{name}_s{seed}"
    hist_p = os.path.join(HERE, f"hist_long_{tag}.json")
    if os.path.exists(hist_p):
        try:
            h = json.load(open(hist_p))
            if h and h[-1]["it"] >= int(ITERS):
                print(f"  {name:10s} seed {seed} already done", flush=True)
                return h
        except Exception:
            pass
    env = dict(os.environ)
    env.update(BASE)
    env.update(extra)
    env["TAG"] = tag
    env["SEED"] = str(seed)
    t0 = time.time()
    with open(os.path.join(HERE, f"cell_{tag}.log"), "w", encoding="utf-8") as f:
        subprocess.run([sys.executable, os.path.join(HERE, "train_chess_long.py"),
                        ITERS, BSZ, D, L], cwd=HERE, env=env, stdout=f,
                       stderr=subprocess.STDOUT)
    try:
        h = json.load(open(hist_p))
    except Exception:
        print(f"  {name:10s} seed {seed} FAILED, see cell_{tag}.log", flush=True)
        return None
    last = h[-1]
    w = (last.get("mix") or {}).get("width")
    print(f"  {name:10s} seed {seed}  policy {last['policy']:.4f}  top1 {last['top1']:.4f}  "
          f"squares {w}  ({time.time() - t0:.0f}s)", flush=True)
    return h


def summarise(raw):
    agg = {}
    for name, runs in raw.items():
        a = {"n": len(runs)}
        for k in ("policy", "top1"):
            vals = [r[k] for r in runs]
            a[f"{k}_mean"] = stat.mean(vals)
            a[f"{k}_sd"] = stat.pstdev(vals) if len(vals) > 1 else 0.0
        w = [r["mix"]["width"] for r in runs if r.get("mix")]
        a["width"] = [stat.mean(c) for c in zip(*w)] if w else None
        agg[name] = a
    return agg


if __name__ == "__main__":
    print(f"mixer ablation, {ITERS} iterations, {len(ARMS)} arms, seeds {SEEDS}", flush=True)
    print("", flush=True)
    raw = {}
    for name, extra in ARMS:
        for seed in SEEDS:
            h = run(name, extra, seed)
            if not h:
                continue
            last = h[-1]
            raw.setdefault(name, []).append(
                {"seed": seed, "policy": last["policy"], "top1": last["top1"],
                 "top5": last["top5"], "bishop": last["bishop"], "value": last["value"],
                 "mix": last.get("mix"), "seconds": last["t"], "extra": extra})
            json.dump({"runs": raw, "summary": summarise(raw)}, open(OUT, "w"), indent=1)

    if not raw:
        raise SystemExit("every arm failed")
    agg = summarise(raw)
    json.dump({"runs": raw, "summary": agg}, open(OUT, "w"), indent=1)

    print("", flush=True)
    print("=" * 80)
    base = agg.get("base")
    order = sorted(agg, key=lambda k: agg[k]["policy_mean"])
    print(f"{'arm':10s} {'policy loss':>18s} {'vs base':>9s} {'top-1':>18s} {'vs base':>9s} {'n':>3s}")
    print("-" * 80)
    for k in order:
        a = agg[k]
        dp = f"{a['policy_mean'] - base['policy_mean']:+.4f}" if base else "n/a"
        dt = f"{a['top1_mean'] - base['top1_mean']:+.4f}" if base else "n/a"
        print(f"{k:10s} {a['policy_mean']:9.4f} +-{a['policy_sd']:.4f} {dp:>9s} "
              f"{a['top1_mean']:9.4f} +-{a['top1_sd']:.4f} {dt:>9s} {a['n']:3d}")
    print("")
    print("A difference smaller than the seed spread beside it is not a result.")
    print(f"wrote {os.path.basename(OUT)}")
