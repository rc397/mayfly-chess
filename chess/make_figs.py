"""
Render the tropical (max-plus) versus control experiment results into PNGs in chess/figs/.

Reads whatever JSON is on disk and skips what is missing, so it can run at any point.

  python chess/make_figs.py

Inputs, each optional:
  sweep_trop.json      coordinate-descent sweep, tropical
  sweep_ctrl.json      coordinate-descent sweep, control
  hist_long_trop.json  convergence history, tropical
  hist_long_ctrl.json  convergence history, control
  strength.json        head-to-head and Stockfish ladder
"""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

TROP = "#2a78d6"
CTRL = "#eb6834"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)


def axis_of(sweep, name):
    if not sweep:
        return None
    for a in sweep["axes"]:
        if a["name"] == name:
            return a
    return None


st = load("sweep_trop.json")
sc = load("sweep_ctrl.json")
ht = load("hist_long_trop.json")
hc = load("hist_long_ctrl.json")
strength = load("strength.json")
made = []


# ---------------------------------------------------------------- 1. learning rate
if st or sc:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for sweep, colour, marker, style, label in (
            (st, TROP, "o", "-", "tropical (max-plus)"),
            (sc, CTRL, "D", "--", "control (transformer)")):
        a = axis_of(sweep, "LR")
        if not a:
            continue
        xs = [float(k) for k in a["results"]]
        ys = [a["results"][k] for k in a["results"]]
        pairs = sorted(zip(xs, ys))
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        ax.plot(xs, ys, style, color=colour, marker=marker, ms=6, lw=2, label=label)
        bx = float(a["chosen"])
        by = a["results"][a["chosen"]]
        ax.scatter([bx], [by], s=170, facecolors="none", edgecolors=colour, lw=2, zorder=5)
        ax.annotate(f"{by:.3f}", (bx, by), textcoords="offset points", xytext=(0, -18),
                    ha="center", color=colour, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("policy loss (nats)")
    ax.set_title("Learning rate sweep, 3000 step probe. Circled point is the chosen value.", loc="left")
    ax.legend(loc="upper left")
    tidy(ax)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig1_learning_rate.png")
    fig.savefig(p); plt.close(fig); made.append(p)


# ---------------------------------------------------------------- 2. every other axis
panels = []
for sweep, colour, label in ((st, TROP, "tropical"), (sc, CTRL, "control")):
    if not sweep:
        continue
    for a in sweep["axes"]:
        if a["name"] == "LR":
            continue
        panels.append((a, colour, label))

if panels:
    n = len(panels)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.9 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n:
            ax.axis("off")
            continue
        a, colour, label = panels[i]
        keys = list(a["results"])
        try:
            keys = sorted(keys, key=float)
        except ValueError:
            pass
        ys = [a["results"][k] for k in keys]
        bars = ax.bar(range(len(keys)), ys, color=colour, width=0.6, alpha=0.35)
        pick = keys.index(a["chosen"])
        bars[pick].set_alpha(1.0)
        lo, hi = min(ys), max(ys)
        pad = max((hi - lo) * 0.35, 0.02)
        ax.set_ylim(lo - pad, hi + pad * 0.6)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys)
        ax.set_title(f"{label}: {a['name']}", loc="left", color=colour)
        ax.set_ylabel("policy loss")
        for j, (k, y) in enumerate(zip(keys, ys)):
            ax.text(j, y, f"{y:.3f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
        tidy(ax)
    fig.suptitle("Coordinate-descent sweep, remaining axes. Solid bar is the value carried forward.",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(FIGS, "fig2_sweep_axes.png")
    fig.savefig(p); plt.close(fig); made.append(p)


# ---------------------------------------------------------------- 3. convergence
if ht or hc:
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for hist, colour, style, label in ((ht, TROP, "-", "tropical (max-plus)"),
                                       (hc, CTRL, "--", "control (transformer)")):
        if not hist:
            continue
        it = [r["it"] for r in hist]
        pol = [r["policy"] for r in hist]
        wall = [r["t"] / 60.0 for r in hist]
        axs[0].plot(it, pol, style, color=colour, lw=2, label=label)
        axs[1].plot(wall, pol, style, color=colour, lw=2, label=label)
        b = min(range(len(pol)), key=lambda i: pol[i])
        axs[0].scatter([it[b]], [pol[b]], s=40, color=colour, zorder=5)
        axs[0].annotate(f"{pol[b]:.4f}", (it[b], pol[b]), textcoords="offset points",
                        xytext=(6, 6), color=colour, fontsize=9)
    for ax, xlab, title in ((axs[0], "iteration", "By training step"),
                            (axs[1], "wall clock (minutes)", "By wall clock")):
        ax.set_xlabel(xlab)
        ax.set_ylabel("policy loss (nats)")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper right")
        tidy(ax)
    lows = [min(r["policy"] for r in h) for h in (ht, hc) if h]
    if lows:
        for ax in axs:
            ax.set_ylim(min(lows) - 0.15, min(min(lows) + 3.2, 7.0))
    fig.suptitle("Convergence at each architecture's own tuned configuration.",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(FIGS, "fig3_convergence.png")
    fig.savefig(p); plt.close(fig); made.append(p)


# ---------------------------------------------------------------- 4. the other heads
if ht or hc:
    keys = [("bishop", "bishop pair loss (counting probe)", True),
            ("value", "value head loss", False),
            ("top1", "top-1 move accuracy", False),
            ("top5", "top-5 move accuracy", False)]
    have = [k for k, _, _ in keys
            if any(h and k in h[0] for h in (ht, hc))]
    keys = [k for k in keys if k[0] in have]
    if keys:
        fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0), squeeze=False)
        for i, (key, title, logy) in enumerate(keys[:4]):
            ax = axes[i // 2][i % 2]
            for hist, colour, style, label in ((ht, TROP, "-", "tropical"),
                                               (hc, CTRL, "--", "control")):
                if not hist or key not in hist[0]:
                    continue
                ax.plot([r["it"] for r in hist], [max(r[key], 1e-4) for r in hist],
                        style, color=colour, lw=2, label=label)
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("iteration")
            ax.set_title(title, loc="left")
            ax.legend(loc="best")
            tidy(ax)
        for j in range(len(keys), 4):
            axes[j // 2][j % 2].axis("off")
        fig.suptitle("Auxiliary heads. The bishop pair head is pure counting.",
                     x=0.01, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = os.path.join(FIGS, "fig4_heads.png")
        fig.savefig(p); plt.close(fig); made.append(p)


# ---------------------------------------------------------------- 5. playing strength
if strength and "vs_sf" in strength:
    import math
    n = strength.get("games", 40)
    err = lambda p: 1.96 * math.sqrt(max(p * (1 - p), 1e-6) / n)   # approximate, draws ignored
    skills = sorted(strength["vs_sf"], key=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    w = 0.34
    xs = list(range(len(skills)))
    for off, key, colour, label in ((-w / 2, "trop", TROP, "tropical"),
                                    (w / 2, "ctrl", CTRL, "control")):
        ys = [strength["vs_sf"][s][key] for s in skills]
        ax.bar([x + off for x in xs], ys, width=w, color=colour, label=label,
               yerr=[err(y) for y in ys], capsize=4,
               error_kw={"ecolor": MUTED, "lw": 1})
        for x, y in zip(xs, ys):
            ax.text(x + off, y + err(y) + 0.004, f"{y*100:.1f}%", ha="center",
                    fontsize=8.5, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"skill {s}" for s in skills])
    ax.set_ylabel("score (1 = win, 0.5 = draw)")
    ax.set_title(f"Against Stockfish, {n} games each. Bars are 95 percent intervals.", loc="left")
    ax.legend(loc="upper right")
    tidy(ax)

    ax = axes[1]
    s = strength.get("h2h_score_trop", 0.5)
    ax.barh([0], [s], color=TROP, height=0.42, label="tropical")
    ax.barh([0], [1 - s], left=[s], color=CTRL, height=0.42, label="control")
    e = err(s)
    ax.errorbar([s], [0], xerr=[[e], [e]], fmt="none", ecolor=INK, lw=1.6, capsize=6)
    ax.axvline(0.5, color=INK, lw=1, ls=":")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("share of points to tropical")
    ax.set_title(f"Head to head, {n} games, colours alternated", loc="left")
    ax.set_ylim(-0.55, 0.55)
    ax.text(s, 0.26, f"{s*100:.1f}%", ha="center", fontsize=10)
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    tidy(ax)
    ax.grid(False)

    fig.suptitle("Playing strength, policy head only, no search.", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(FIGS, "fig5_strength.png")
    fig.savefig(p); plt.close(fig); made.append(p)


# ---------------------------------------------------------------- 6. the cost of the borrowed lr
a_t = axis_of(st, "LR")
if a_t and "3e-4" in a_t["results"] and a_t["chosen"] in a_t["results"]:
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    borrowed = a_t["results"]["3e-4"]
    tuned = a_t["results"][a_t["chosen"]]
    bars = ax.bar([0, 1], [borrowed, tuned], color=[MUTED, TROP], width=0.55)
    bars[0].set_alpha(0.45)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["3e-4\nborrowed from the control", f"{a_t['chosen']}\nits own optimum"])
    ax.set_ylabel("policy loss (nats)")
    ax.set_ylim(0, borrowed * 1.18)
    for x, y in ((0, borrowed), (1, tuned)):
        ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=10)
    ax.annotate(f"{borrowed - tuned:.2f} nats of handicap", xy=(0.5, (borrowed + tuned) / 2),
                ha="center", color=INK, fontsize=10)
    ax.set_title("What the inherited learning rate cost the tropical model", loc="left")
    tidy(ax)
    fig.tight_layout()
    p = os.path.join(FIGS, "fig6_lr_handicap.png")
    fig.savefig(p); plt.close(fig); made.append(p)



# ---------------------------------------------------------------- 7. test-time training
ttt, ttt_nm = load("hist_ttt.json"), load("hist_ttt_nomem.json")
if ttt:
    BUCK = ("0-20", "20-40", "40-70", "70-")
    ARMS = (("nomem", CTRL, "--", "memory off"),
            ("mem", TROP, "-", "memory on"),
            ("scram", MUTED, ":", "memory fed another game"))
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 3.9))

    for key, colour, style, label in ARMS:
        xs = [r["it"] for r in ttt if key in r]
        ys = [r[key]["loss"] for r in ttt if key in r]
        if xs:
            axs[0].plot(xs, ys, style, color=colour, lw=2, label=label)
    if ttt_nm:
        axs[0].plot([r["it"] for r in ttt_nm if "nomem" in r],
                    [r["nomem"]["loss"] for r in ttt_nm if "nomem" in r],
                    "-.", color="#1baf7a", lw=2, label="trained without memory")
    axs[0].set_xlabel("iteration"); axs[0].set_ylabel("move prediction loss (nats)")
    axs[0].set_title("The memory and its scrambled control sit on top of each other", loc="left")
    axs[0].legend(loc="upper right"); tidy(axs[0])

    last = ttt[-1]
    w, xs = 0.26, list(range(len(BUCK)))
    for off, (key, colour, _, label) in zip((-w, 0, w), ARMS):
        if key not in last:
            continue
        ys = [last[key]["acc"][b] * 100 for b in BUCK]
        axs[1].bar([x + off for x in xs], ys, width=w, color=colour, label=label)
    axs[1].set_xticks(xs); axs[1].set_xticklabels(["plies " + b for b in BUCK])
    axs[1].set_ylabel("move prediction accuracy (%)")
    axs[1].set_title("Accuracy falls with ply for every arm alike", loc="left")
    axs[1].legend(loc="upper right"); tidy(axs[1])

    fig.suptitle("Test-time training: a fast weight updated on the opponent's real moves.",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(FIGS, "fig7_ttt.png")
    fig.savefig(p); plt.close(fig); made.append(p)


# ---------------------------------------------------------------- 8. mixer ablation
ab = load("ablate_mixer.json")
if ab:
    NAMES = {"base": "nothing", "ts": "learned temp", "norm": "mixer rescaled",
             "ts_norm": "both", "head": "head clip", "ts_head": "temp + clip"}
    keys = [k for k in ("base", "ts", "norm", "ts_norm", "head", "ts_head") if k in ab]
    fig, axs = plt.subplots(1, 4, figsize=(14.6, 4.2))
    xs = list(range(len(keys)))
    lab = [NAMES.get(k, k) for k in keys]

    for ax, field, title, lower_better in ((axs[0], "policy", "Policy loss (nats)", True),
                                           (axs[1], "top1", "Top-1 move agreement", False)):
        ys = [ab[k][field] for k in keys]
        pick = min(ys) if lower_better else max(ys)
        ax.bar(xs, ys, color=[TROP if y == pick else "#9dc0e8" for y in ys], width=0.6)
        pad = (max(ys) - min(ys)) * 0.45 + 1e-4
        ax.set_ylim(min(ys) - pad, max(ys) + pad * 0.7)
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, loc="left")

    for ax, field, title in ((axs[2], "width", "Squares actually read"),
                             (axs[3], "share", "Mixer share of the residual")):
        for li in range(4):
            vals = [ab[k]["mix"][field][li] if ab[k].get("mix") else float("nan") for k in keys]
            ax.plot(xs, vals, "-o", ms=5, lw=1.8, label=f"layer {li + 1}",
                    color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"][li])
        ax.set_title(title, loc="left")
        ax.legend(loc="best", fontsize=8)

    for ax in axs:
        ax.set_xticks(xs)
        ax.set_xticklabels(lab, fontsize=8, rotation=30, ha="right")
        tidy(ax)
    fig.suptitle("Which mixer fix does anything. Selection improves the top move and worsens the full distribution.",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(FIGS, "fig8_ablation.png")
    fig.savefig(p); plt.close(fig); made.append(p)


if not made:
    print("nothing to plot yet")
    sys.exit(0)
for p in made:
    print(f"wrote {p}  ({os.path.getsize(p)/1024:.0f} KB)")
