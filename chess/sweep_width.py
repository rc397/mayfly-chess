"""
sweep_width measures what hard selection costs in the tropical mixer.

In the ablation, lower loss went with more blending, but those arms changed mixer amplitude as well as
width. Here amplitude is fixed (MIX_NORM on, so the mixer always writes unit RMS into the residual) and only
the number of squares the mixer may read changes. Selectivity is calibrated on real scores and then frozen.
A flat loss across widths means selection is free, a loss that rises as width falls gives its price, and a
loss that falls means selection helps.

  python chess/sweep_width.py [ITERS]
  env: SEEDS=0,1   WIDTHS=1.5,3,6,12,24,48
"""
import os, sys, json, time, subprocess
import statistics as stat

HERE = os.path.dirname(os.path.abspath(__file__))
ITERS = sys.argv[1] if len(sys.argv) > 1 else "8000"
BSZ, D, L = "128", "256", "4"

BASE = {"ARCH": "trop", "LR": "3e-3", "T": "0.3", "H": "4", "SC": "3.0",
        "VAL_EVERY": "1000", "VAL_BATCHES": "20", "RESUME": "0",
        "CRAWL_WINDOW": "999", "CRAWL_THRESH": "0.0", "MAX_HOURS": "1.0",
        "MIX_NORM": "1", "LEARN_TS": "1", "TS_FIX": "1"}

WIDTHS = [float(x) for x in os.environ.get("WIDTHS", "1.5,3,6,12,24,48").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
OUT = os.path.join(HERE, "sweep_width.json")


def run(width, seed):
    tag = f"w{width:g}_s{seed}"
    hist_p = os.path.join(HERE, f"hist_long_{tag}.json")
    if os.path.exists(hist_p):
        try:
            h = json.load(open(hist_p))
            if h and h[-1]["it"] >= int(ITERS):
                print(f"  width {width:5g} seed {seed} already done", flush=True)
                return h
        except Exception:
            pass
    env = dict(os.environ)
    env.update(BASE)
    env["TAG"] = tag
    env["SEED"] = str(seed)
    env["TS_TARGET"] = str(width)
    t0 = time.time()
    with open(os.path.join(HERE, f"cell_{tag}.log"), "w", encoding="utf-8") as f:
        subprocess.run([sys.executable, os.path.join(HERE, "train_chess_long.py"),
                        ITERS, BSZ, D, L], cwd=HERE, env=env, stdout=f,
                       stderr=subprocess.STDOUT)
    try:
        h = json.load(open(hist_p))
    except Exception:
        print(f"  width {width:5g} seed {seed} FAILED, see cell_{tag}.log", flush=True)
        return None
    last = h[-1]
    got = (last.get("mix") or {}).get("width")
    print(f"  width {width:5g} seed {seed}  policy {last['policy']:.4f}  "
          f"top1 {last['top1']:.4f}  actually read {got}  ({time.time() - t0:.0f}s)", flush=True)
    return h


if __name__ == "__main__":
    print(f"selectivity sweep at fixed mixer amplitude, {ITERS} iterations", flush=True)
    print(f"target widths {WIDTHS}, seeds {SEEDS}", flush=True)
    print("", flush=True)
    raw = {}
    for width in WIDTHS:
        for seed in SEEDS:
            h = run(width, seed)
            if not h:
                continue
            last = h[-1]
            raw.setdefault(f"{width:g}", []).append(
                {"seed": seed, "target": width, "policy": last["policy"], "top1": last["top1"],
                 "top5": last["top5"], "bishop": last["bishop"], "value": last["value"],
                 "mix": last.get("mix"), "seconds": last["t"]})
            json.dump(raw, open(OUT, "w"), indent=1)

    if not raw:
        raise SystemExit("every cell failed")
    print("", flush=True)
    print("=" * 74)
    print(f"{'target':>8s} {'read':>8s} {'policy loss':>18s} {'top-1':>18s} {'n':>3s}")
    print("-" * 74)
    rows = []
    for key in sorted(raw, key=float):
        runs = raw[key]
        pol = [r["policy"] for r in runs]
        t1 = [r["top1"] for r in runs]
        got = [stat.mean(r["mix"]["width"]) for r in runs if r.get("mix")]
        rows.append((float(key), stat.mean(got) if got else float("nan"), stat.mean(pol)))
        print(f"{float(key):8g} {stat.mean(got) if got else float('nan'):8.1f} "
              f"{stat.mean(pol):9.4f} +-{stat.pstdev(pol) if len(pol) > 1 else 0:.4f} "
              f"{stat.mean(t1):9.4f} +-{stat.pstdev(t1) if len(t1) > 1 else 0:.4f} {len(runs):3d}")
    if len(rows) > 2:
        lo, hi = rows[0], rows[-1]
        print("")
        print(f"price of selection: {hi[2] - lo[2]:+.4f} nats going from {hi[0]:g} squares "
              f"down to {lo[0]:g}")
    print(f"wrote {os.path.basename(OUT)}")
