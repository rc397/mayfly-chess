"""Plot the training run, losses, top-1 and top-3 agreement, value errors and bare-network strength.

The losses share a log axis, so equal vertical distances mean equal proportional improvement. The x
axis is hours of training, which is what version freezes are scheduled against. Panels share the x
axis, and version freezes and other events are marked across all of them.

    python chess/curve.py            writes curve_mayfly.png
    env TAG=mayfly, OUT=curve_mayfly.png, XMAX (hours, blank for all)
"""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
TAG = E("TAG", "mayfly")
# the history lives with the checkpoints, so read the folder the run was launched with
def _ckpt_dir():
    try:
        env = json.load(open(os.path.join(HERE, "autostart.json"), encoding="utf-8-sig"))["env"]
        if env.get("CKPT_DIR"):
            return env["CKPT_DIR"]
    except Exception:
        pass
    return E("CKPT_DIR", HERE)


HIST = os.path.join(_ckpt_dir(), f"hist_{TAG}.json")
VERSIONS = E("VERSION_DIR", paths.under("versions"))
OUT = os.path.join(HERE, E("OUT", f"curve_{TAG}.png"))

h = json.load(open(HIST))
h = [r for r in h if r.get("it", 0) > 0]
if not h:
    raise SystemExit("no evaluations recorded yet")

# "t" is seconds since the process started, so it resets on every restart. A restart carries the
# hours at the checkpoint it resumed from, not at the last evaluation, or training done between the
# two would be dropped. The resume iteration is backed out of the new process's pace, and the old
# process's hours at that iteration are read off its own evaluations.
def _segments(rows):
    segs, prev_t = [], -1.0
    for r in rows:
        if not segs or r["t"] < prev_t:
            segs.append([])
        segs[-1].append(r)
        prev_t = r["t"]
    return segs


def _pace(seg):
    """Iterations an hour over one process's evaluations, or None with fewer than two."""
    dt = (seg[-1]["t"] - seg[0]["t"]) / 3600
    return (seg[-1]["it"] - seg[0]["it"]) / dt if len(seg) > 1 and dt > 0 else None


def _hours_at(seg, it, pace):
    its = [r["it"] for r in seg]
    ts = [r["t"] / 3600 for r in seg]
    if it <= its[0]:
        return ts[0] - (its[0] - it) / pace
    if it >= its[-1]:
        return ts[-1] + (it - its[-1]) / pace
    for k in range(1, len(its)):
        if it <= its[k]:
            f = (it - its[k - 1]) / max(its[k] - its[k - 1], 1)
            return ts[k - 1] + f * (ts[k] - ts[k - 1])
    return ts[-1]


hours, acc = [], 0.0
_segs = _segments(h)
for k, seg in enumerate(_segs):
    if k:
        before = _segs[k - 1]
        p_old = _pace(before) or _pace(seg)
        p_new = _pace(seg) or p_old
        if p_old and p_new:
            resumed_at = seg[0]["it"] - (seg[0]["t"] / 3600) * p_new
            acc += max(0.0, _hours_at(before, resumed_at, p_old))
        else:                          # one evaluation either side, so no pace to measure
            acc += before[-1]["t"] / 3600
    hours += [acc + r["t"] / 3600 for r in seg]

def smooth(v, w=9):
    """Centred rolling mean, shrinking the window at the ends rather than dropping them.

    Every top-1 point is one validation batch of 320, with a standard error of about 2.4 points,
    so the trend is hard to see without smoothing.
    """
    out = []
    for i in range(len(v)):
        a, b = max(0, i - w // 2), min(len(v), i + w // 2 + 1)
        out.append(sum(v[a:b]) / (b - a))
    return out


q = [r["q"] for r in h]
pol = [r["policy"] for r in h]
top1 = [100 * r["top1"] for r in h]
# where the objective changed, if it has, the unmasked series are retired
cut = next((i for i, r in enumerate(h) if "policy_legal" in r), len(h))

# One strip per quantity in its own units, sharing the x axis. Losses on a log scale, top-1 and top-3
# as percentages, bare-network strength from strength_probe.py (puzzles, games against Stockfish), move
# quality, and the win-probability error of the value heads.
fig, (ax, axt, axs, axq, axm) = plt.subplots(
    5, 1, figsize=(11.5, 13.0), sharex=True,
    gridspec_kw={"height_ratios": [3.0, 1.9, 1.7, 1.3, 1.5], "hspace": 0.12})


def points(key, scale=1.0):
    xs = [hours[i] for i, r in enumerate(h) if key in r]
    ys = [scale * r[key] for r in h if key in r]
    return xs, ys


# the losses
ax.set_yscale("log")
keep = []
if cut < len(h):
    l, = ax.plot(hours[:cut], pol[:cut], color="#9fb8d0", lw=1.2,
                 label="policy, all moves (retired at v4)")
    keep.append(l)
for key, colour, label in (("policy_legal", "#1f4e79", "policy (legal moves)"),
                           ("att_policy_legal", "#d98b1f", "policy, ATTENTION head"),
                           ("q", "#c0504d", "action value"),
                           ("att_q", "#e07b7b", "action value, ATTENTION head"),
                           # held-out Lichess deep evaluations, Stockfish at about 25M nodes
                           ("lv_q", "#7a3e9d", "action value vs Lichess deep labels")):
    xs, ys = points(key)
    if xs:
        l, = ax.plot(xs, ys, color=colour, lw=1.8, label=label)
        keep.append(l)
# dh_q is the action-value loss on held-out Kaggle deep labels (top ten moves at depth 35), logged from
# the first evaluation after DH_SHARE went on. A dotted line in every panel marks the switch.
xs, ys = points("dh_q")
if xs:
    l, = ax.plot(xs, ys, color="#0f7c8c", lw=1.8, marker="o", ms=4.5, label="action value vs Kaggle deep labels")
    keep.append(l)
    for a in (ax, axt, axm):
        a.axvline(xs[0], color="#0f7c8c", ls=":", lw=1.1, alpha=0.8)
    ax.annotate(f"deep labels in training: {ys[-1]:.4f}", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(-205, 24), fontsize=8, color="#0f7c8c")
ax.set_ylabel("loss, log scale")
# label enough ticks on the log axis to read values off it
ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.4, 0.5, 0.7, 1, 1.5, 2, 3, 4, 6]))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
ax.grid(True, which="both", alpha=0.16)
ax.legend(handles=keep, loc="upper right", fontsize=8, framealpha=0.92, ncol=2)

# top-1 and top-3 agreement
keep = []
if cut < len(h):
    l, = axt.plot(hours[:cut], smooth(top1[:cut]), color="#a9c79a", lw=1.3,
                  label="top-1, all moves (retired at v4)")
    keep.append(l)
last = None
for key, colour, label in (("top1_legal", "#2f6b1d", "top-1 (legal moves)"),
                           ("att_top1_legal", "#b86e00", "top-1, ATTENTION head")):
    xs, ys = points(key, 100.0)
    if xs:
        ys = smooth(ys, 5)
        l, = axt.plot(xs, ys, color=colour, lw=2.1, label=label)
        keep.append(l)
        last = (xs[-1], ys[-1], colour)
if last:
    axt.annotate(f"{last[1]:.1f}%", (last[0], last[1]), textcoords="offset points",
                 xytext=(-36, 5), fontsize=9, color=last[2], fontweight="bold")
# Top-3 dashed beside top-1, from the trainer's validation. Older versions get one marker each, all
# scored on the same validation positions by top3_versions.py.
_t3 = []
for key, colour, label in (("top3_legal", "#2f6b1d", "top-3 (legal moves)"),
                           ("att_top3_legal", "#b86e00", "top-3, ATTENTION head")):
    xs, ys = points(key, 100.0)
    if xs:
        ys = smooth(ys, 5)
        l, = axt.plot(xs, ys, color=colour, lw=1.6, ls="--", label=label)
        keep.append(l)
        _t3 += ys
try:
    t3 = json.load(open(os.path.join(os.path.dirname(HIST), "top3_versions.json")))
except (OSError, ValueError):
    t3 = {}
_marks = False
for name, r in t3.items():
    if "top3_legal" not in r:
        continue
    # the live checkpoint does not carry its hours, so it goes at the newest point on the curve
    x = float(hours[-1]) if (name == "live" or not r.get("hours")) else float(r["hours"])
    axt.plot([x], [100 * r["top3_legal"]], marker="s", ms=6, mfc="white", mec="#2f6b1d", mew=1.6, ls="")
    axt.plot([x], [100 * r["top1_legal"]], marker="o", ms=5, color="#2f6b1d", ls="")
    if "att_top3_legal" in r:
        axt.plot([x], [100 * r["att_top3_legal"]], marker="s", ms=6, mfc="white", mec="#b86e00", mew=1.6, ls="")
    axt.annotate(f"{name} top-3 {100 * r['top3_legal']:.0f}%", (x, 100 * r["top3_legal"]),
                 textcoords="offset points", xytext=(4, 5), fontsize=7, color="#2f6b1d")
    _t3.append(100 * r["top3_legal"])
    _marks = True
if _marks:
    keep.append(matplotlib.lines.Line2D([], [], marker="s", mfc="white", mec="#2f6b1d", mew=1.6, ls="",
                                        label="top-3 of each version (same positions)"))
    keep.append(matplotlib.lines.Line2D([], [], marker="o", color="#2f6b1d", ls="",
                                        label="top-1 of each version (same positions)"))
axt.set_ylabel("top-1 / top-3 (%)")
axt.grid(True, alpha=0.16)
_t1 = top1[:cut] + [100 * r.get("top1_legal", 0) for r in h if "top1_legal" in r]
axt.set_ylim(0, min(100, max(max(_t1 + _t3) * 1.2, 5)))
axt.legend(handles=keep, loc="lower right", fontsize=7, framealpha=0.92, ncol=2)

# win-probability error of the value heads
keep = []
for key, colour, label in (("mae", "#c0504d", "action value, one number"),
                           ("qdist_mae", "#7a4fa3", "action value, 64 buckets"),
                           ("vdist_mae", "#8c6d3f", "position value, 64 buckets")):
    xs, ys = points(key)
    if xs:
        l, = axm.plot(xs, ys, color=colour, lw=1.7, label=label)
        keep.append(l)
axm.set_ylabel("win-probability error")
# the moves-left head's mean error in plies on held-out Leela positions, on a second axis
xs, ys = points("mleft_mae")
if xs:
    axl = axm.twinx()
    l, = axl.plot(xs, smooth(ys, 3), color="#2a7f8f", lw=1.6, ls="-.", label="moves left, error in plies (right)")
    axl.set_ylabel("moves-left error (plies)", color="#2a7f8f")
    axl.tick_params(axis="y", colors="#2a7f8f")
    keep.append(l)
axm.set_xlabel("hours of training")
axm.grid(True, alpha=0.16)
if keep:
    axm.legend(handles=keep, loc="upper right", fontsize=8, framealpha=0.92)

# strength of the bare network (strength_probe.py), one point per measured checkpoint
SP = os.path.join(os.path.dirname(HIST), f"strength_{TAG}.jsonl")
try:
    _sp = [json.loads(l) for l in open(SP, encoding="utf-8") if l.strip()]
except OSError:
    _sp = []
sp_rows = list({(r["label"], r["iter"]): r for r in _sp}.values())     # a re-measured checkpoint keeps its last row
_by_it = sorted(zip([r["it"] for r in h], hours))


def hours_at_iter(it):
    """Where an iteration falls on this chart's own hours axis."""
    its = [a for a, _ in _by_it]
    hs = [b for _, b in _by_it]
    if it <= its[0]:
        return hs[0] * it / max(its[0], 1)
    if it >= its[-1]:
        k = max(0, len(its) - 20)
        pace = (its[-1] - its[k]) / max(hs[-1] - hs[k], 1e-9)
        return hs[-1] + (it - its[-1]) / pace
    j = next(j for j in range(1, len(its)) if its[j] >= it)
    f = (it - its[j - 1]) / max(its[j] - its[j - 1], 1)
    return hs[j - 1] + f * (hs[j] - hs[j - 1])


def sp_x(r):
    # versions sit on their own vertical marks, everything else is placed by iteration
    if r["label"][1:].isdigit() and r.get("hours"):
        return float(r["hours"])
    return hours_at_iter(r["iter"])


axs2 = axs.twinx()
if sp_rows:
    raw = sorted((r for r in sp_rows if not r.get("cooled")), key=sp_x)
    cool = [r for r in sp_rows if r.get("cooled")]
    handles = []
    for rows_, style, mfc, name in ((raw, "o-", "#355c7d", "puzzle rating (left)"),
                                    (cool, "o", "white", "cooled copy (hollow)")):
        pz = [r for r in rows_ if r.get("puzzles") and not r["puzzles"].get("bound")]
        if pz:
            e = axs.errorbar([sp_x(r) for r in pz], [r["puzzles"]["rating"] for r in pz],
                             yerr=[1.96 * (r["puzzles"]["se"] or 0) for r in pz], fmt=style, color="#355c7d",
                             mfc=mfc, ms=5, lw=1.5, capsize=2, label=name)
            handles.append(e)
    for nodes, ls in (("1000", "--"), ("4000", ":")):
        g = [r for r in raw if r.get("games", {}).get(nodes)]
        if g:
            l, = axs2.plot([sp_x(r) for r in g], [100 * r["games"][nodes]["score"] for r in g], ls=ls, marker=".",
                           color="#c06c84", lw=1.4, label=f"score vs Stockfish {int(nodes):,} nodes (right)")
            handles.append(l)
        gc = [r for r in cool if r.get("games", {}).get(nodes)]
        if gc:
            axs2.plot([sp_x(r) for r in gc], [100 * r["games"][nodes]["score"] for r in gc], ls="", marker="o",
                      mfc="white", color="#c06c84", ms=4)
    for r in sp_rows:
        if r.get("puzzles") and not r["puzzles"].get("bound") and (r["label"][1:].isdigit() or r["label"] == "live"):
            axs.annotate(r["label"] + (" cooled" if r.get("cooled") else ""), (sp_x(r), r["puzzles"]["rating"]),
                         textcoords="offset points", xytext=(4, -10), fontsize=7, color="#355c7d")
    axs.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.92)

    # how much the network's own move gives away, on held-out positions with every move scored
    rq = [r for r in raw if r.get("regret", {}).get("dp")]
    keep_q = []
    if rq:
        xs = [sp_x(r) for r in rq]
        l1, = axq.plot(xs, [100 * r["regret"]["dp"]["regret"] for r in rq], "o-", color="#6c5b7b", ms=4, lw=1.5,
                       label="win % given away per move")
        l2, = axq.plot(xs, [100 * r["regret"]["dp"]["blunders"] for r in rq], "s--", color="#f67280", ms=3,
                       lw=1.3, label="blunders, % of moves (15+ points)")
        keep_q += [l1, l2]
    rc = [r for r in cool if r.get("regret", {}).get("dp")]
    if rc:
        axq.plot([sp_x(r) for r in rc], [100 * r["regret"]["dp"]["regret"] for r in rc], "o", color="#6c5b7b",
                 mfc="white", ms=5)
        axq.plot([sp_x(r) for r in rc], [100 * r["regret"]["dp"]["blunders"] for r in rc], "s", color="#f67280",
                 mfc="white", ms=4)
    if keep_q:
        axq.legend(handles=keep_q, loc="upper right", fontsize=7, framealpha=0.92)
else:
    axs.text(0.5, 0.5, "no strength_probe.py rows yet", transform=axs.transAxes, ha="center", va="center",
             fontsize=9, color="#777")
axs.set_ylabel("puzzle rating\n(no search)")
axs2.set_ylabel("score (%)", color="#c06c84")
axs2.tick_params(axis="y", colors="#c06c84")
axs.grid(True, alpha=0.16)
axq.set_ylabel("move quality (%)")
axq.grid(True, alpha=0.16)

# one set of vertical marks for the whole figure, labelled once inside the top panel
try:
    vs = json.load(open(os.path.join(VERSIONS, "versions.json")))
except Exception:
    vs = []
try:
    events = json.load(open(os.path.join(os.path.dirname(HIST), f"events_{TAG}.json"),
                            encoding="utf-8"))
except Exception:
    events = []
mi = [i for i, r in enumerate(h) if "policy_legal" in r]
ai = [i for i, r in enumerate(h) if "att_policy_legal" in r]
di = [i for i, r in enumerate(h) if "qdist" in r]
marks = [(v["hours"], f"v{v['version']}", "#777", ":") for v in vs]
if mi:
    marks.append((hours[mi[0]], "v4 recipe", "#b0413e", "--"))
if ai:
    marks.append((hours[ai[0]], "attention heads", "#d98b1f", "-."))
if di:
    marks.append((hours[di[0]], "distributional heads", "#7a4fa3", "-."))
marks += [(ev["h"], ev["label"].split(",")[0], "#7a4fa3", "--") for ev in events]
# labels take turns at three heights so close marks do not overprint, and long labels are cut
for k, (x, label, colour, style) in enumerate(sorted(marks)):
    for a in (ax, axt, axs, axq, axm):
        a.axvline(x, color=colour, lw=1.0, ls=style, alpha=0.65)
    ax.annotate(label[:26], (x, (0.97, 0.66, 0.36)[k % 3]), xycoords=("data", "axes fraction"),
                rotation=90, textcoords="offset points", xytext=(3, 0), fontsize=7.2,
                color=colour, ha="left", va="top")

# the version numbers as one small table where the top-1 panel is empty, rather than three boxes
# scattered across the curves
if vs:
    # From v4 the policy is trained masked to legal moves, so its unmasked numbers mean nothing.
    # Each version shows the measure it was trained under, marked when it is the legal one.

    def _pol(v):
        return v.get("policy_legal", v.get("policy", 0))

    def _top1(v):
        return v.get("top1_legal", v.get("top1", 0))

    rows = chr(10).join(f"v{v['version']:<2} {v.get('hours', 0):>4.0f}h  "
                        f"policy {_pol(v):6.3f}  action {v.get('q', 0):.3f}  "
                        f"top-1 {100*_top1(v):4.1f}%" + ("  legal" if "top1_legal" in v else "")
                        for v in sorted(vs, key=lambda r: r.get("version", 0)))
    axt.text(0.012, 0.96, rows, transform=axt.transAxes, va="top", ha="left", fontsize=7.2,
             family="monospace", color="#333",
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#ccc", alpha=0.93))

xmax = E("XMAX")
if xmax:
    ax.set_xlim(0, float(xmax))
if mi:
    ttl = (f"policy (legal) {h[-1]['policy_legal']:.4f}, action value {q[-1]:.4f}, "
           f"top-1 legal {100*h[-1]['top1_legal']:.1f}%")
else:
    ttl = f"policy {pol[-1]:.4f}, action value {q[-1]:.4f}, top-1 {top1[-1]:.1f}%"
if ai:
    ttl += f"  |  attention: policy {h[-1]['att_policy_legal']:.4f}, top-1 {100*h[-1]['att_top1_legal']:.1f}%"
ax.set_title(f"MayFly {TAG}: {hours[-1]:.1f}h, {h[-1]['it']:,} steps, " + ttl, fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
if vs:
    print(f"  {'version':>8} {'hours':>7} {'iteration':>11} {'policy':>8} {'action':>8} "
          f"{'top-1':>7}")
    for v in sorted(vs, key=lambda r: r.get("version", 0)):
        print(f"  {'v' + str(v['version']):>8} {v.get('hours', 0):>7.2f} "
              f"{v.get('iter', 0):>11,} {_pol(v):>8.4f} {v.get('q', 0):>8.4f} "
              f"{100*_top1(v):>6.1f}%" + ("  legal" if "top1_legal" in v else ""))
print(f"  {len(h)} evaluations over {hours[-1]:.2f}h")
print(f"  policy       {pol[0]:.4f} -> {pol[cut-1]:.4f}"
      + (f"  (all moves, to {hours[cut-1]:.1f}h; retired at v4)" if cut < len(h) else ""))
print(f"  action value {q[0]:.4f} -> {q[-1]:.4f}  (best {min(q):.4f})")
print(f"  top-1        {top1[0]:.1f}% -> {top1[cut-1]:.1f}%"
      + ("  (all moves; retired at v4)" if cut < len(h) else ""))
if mi:
    f0, f1 = h[mi[0]], h[-1]
    print(f"  since the v4 config at {hours[mi[0]]:.1f}h, {len(mi)} evaluations:")
    print(f"    policy (legal)  {f0['policy_legal']:.4f} -> {f1['policy_legal']:.4f}")
    print(f"    top-1 (legal)   {100*f0['top1_legal']:.1f}% -> {100*f1['top1_legal']:.1f}%  "
          f"(5-point mean "
          f"{smooth([100 * r['top1_legal'] for r in h if 'top1_legal' in r], 5)[-1]:.1f}%)")
    print(f"    action value    {f0['q']:.4f} -> {f1['q']:.4f}")
if ai:
    g0, g1 = h[ai[0]], h[-1]
    print(f"  attention heads since {hours[ai[0]]:.1f}h, {len(ai)} evaluations:")
    print(f"    policy (legal)  {g0['att_policy_legal']:.4f} -> {g1['att_policy_legal']:.4f}"
          f"   (CLS head now {g1['policy_legal']:.4f})")
    print(f"    top-1 (legal)   {100*g0['att_top1_legal']:.1f}% -> {100*g1['att_top1_legal']:.1f}%"
          f"   (CLS head now {100*g1['top1_legal']:.1f}%)")
    print(f"    action value    {g0['att_q']:.4f} -> {g1['att_q']:.4f}   (CLS head now {g1['q']:.4f})")
if vs:
    print(f"  versions marked: {', '.join('v' + str(v['version']) for v in vs)}")
if sp_rows:
    print(f"  strength, no search ({os.path.basename(SP)}):")
    for r in sorted(sp_rows, key=sp_x):
        pz, g, rg = r.get("puzzles") or {}, r.get("games") or {}, (r.get("regret") or {}).get("dp") or {}
        cells = "  ".join(f"{n}n {100 * g[n]['score']:.0f}%" for n in sorted((k for k in g if k.isdigit()), key=int))
        print(f"    {r['label'] + (' (cooled)' if r.get('cooled') else ''):<22} {r['iter']:>11,}  "
              f"puzzles {pz.get('rating', '-'):>5}  regret {100 * rg.get('regret', 0):.2f}%  "
              f"blunders {100 * rg.get('blunders', 0):.2f}%  {cells}")
