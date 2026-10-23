"""
train_chess_long runs an architecture until its loss stops improving, not to a fixed step count.

At a shared step count one architecture may be settled while the other is still improving fast, so
each runs until its own loss crawls. Same data, same order, matched parameters.

  - warmup then a constant learning rate, so convergence belongs to the model and not the schedule
  - stop when the best loss improves by less than CRAWL_THRESH over CRAWL_WINDOW evaluations
  - resume from the last checkpoint after a crash or reboot
  - a JSON history so the convergence rate can be compared, not just the endpoint

  python chess/train_chess_long.py [ITERS] [BSZ] [D] [LAYERS]
  env: ARCH=trop|ctrl TAG=... LR=3e-4 VAL_EVERY=1000 CRAWL_WINDOW=10 CRAWL_THRESH=0.05 MAX_HOURS=7
"""
import os, sys, glob, json, time, math, random
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import chess
from chess_net import ChessNet, board_to_ids, move_to_idx, N_MOVES
from mayfly_chess import MayflyChessNet, matched_dff_h

E = os.environ.get
# Corpus folders, comma separated. "rich" is Stockfish self-play at 100 to 200ms and "rich5" is
# tournament positions labelled at 5ms. List both to train on the union.
RICH = [os.path.join(HERE, d) for d in os.environ.get("DATA", "rich").split(",")]
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
BSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 128
D = int(sys.argv[3]) if len(sys.argv) > 3 else 256
LAYERS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
ARCH = E("ARCH", "trop")
TAG = E("TAG", ARCH)
H = int(E("H", 8))
T_TROP = float(E("T", 0.66))          # tropical temperature
SC_TROP = float(E("SC", 1.5))         # tropical init scale
NHEAD = int(E("NHEAD", 0)) or max(4, D // 64)   # control attention heads
FF_MULT = float(E("FF_MULT", 4.0))    # control feed-forward multiplier
DROPOUT = float(E("DROPOUT", 0.1))    # control dropout (TransformerEncoderLayer default)
LR = float(E("LR", 3e-4))
# Fixed seed for the initialisation, so an ablation compares arms and not seeds. The data order is
# also fixed across arms.
SEED = int(E("SEED", 0))
# Without these the mixer read 7 to 27 of 65 squares, averaging the board rather than selecting.
# LEARN_TS gives the square-stage contraction its own learned temperature, calibrated on real scores
# at init to TS_TARGET squares. MIX_NORM rescales the centred mixer output, which otherwise wrote only
# 14 to 26 percent of the residual.
LEARN_TS = E("LEARN_TS", "0") == "1"
MIX_NORM = E("MIX_NORM", "0") == "1"
TS_TARGET = float(E("TS_TARGET", 4.0))
# TS_FIX calibrates selectivity to TS_TARGET and then freezes it, so the number of squares read is an
# independent variable.
TS_FIX = E("TS_FIX", "0") == "1"
# A max-plus trunk sends its gradient to one input per unit, so the trunk's gradients are sparse and
# spiky and one global clip would drag the heads down too. Zero means share the trunk's value.
LR_HEAD = float(E("LR_HEAD", 0))
CLIP = float(E("CLIP", 1.0))
CLIP_HEAD = float(E("CLIP_HEAD", 0))
VAL_EVERY = int(E("VAL_EVERY", 1000))
VAL_BATCHES = int(E("VAL_BATCHES", 20))
# Stop when the loss crawls, not after a fixed budget.
CRAWL_WINDOW = int(E("CRAWL_WINDOW", 10))     # evaluations to look back over
CRAWL_THRESH = float(E("CRAWL_THRESH", 0.05)) # nats of improvement over that window that count as still moving
MAX_HOURS = float(E("MAX_HOURS", 7.0))        # wall-clock safety cap
RESUME = E("RESUME", "1") == "1"
WARMUP = max(100, ITERS // 50)
device = "cuda" if torch.cuda.is_available() else "cpu"
TEMP = 100.0
HOLDOUT = 20

CKPT = os.path.join(HERE, f"ckpt_long_{TAG}.pt")
LAST = os.path.join(HERE, f"ckpt_long_{TAG}_last.pt")
HIST = os.path.join(HERE, f"hist_long_{TAG}.json")


def encode(rec):
    board = chess.Board(rec["fen"])
    ids, cast, flip = board_to_ids(board)
    tk = rec["topk"]
    idxs = [move_to_idx(chess.Move.from_uci(u), flip) for u, _ in tk]
    cps = np.array([c for _, c in tk], dtype=np.float32)
    w = np.exp((cps - cps.max()) / TEMP); w = (w / w.sum()).tolist()
    wp = 1.0 / (1.0 + 10 ** (-rec["cp"] / 400.0))
    f = rec["feat"]
    sw = (lambda a: [a[1], a[0]]) if flip else (lambda a: a)
    mat, ks, bp, pw = sw(f["material"]), sw(f["king_safety"]), sw(f["bishop_pair"]), sw(f["pawns"])
    pawns = [pw[0][0], pw[0][1], pw[0][2], pw[1][0], pw[1][1], pw[1][2]]
    return {"ids": ids, "cast": cast, "p_idx": idxs, "p_w": w, "value": wp,
            "material": [mat[0] / 39.0, mat[1] / 39.0], "king": [ks[0] / 8.0, ks[1] / 8.0],
            "pawns": [x / 8.0 for x in pawns], "mob": f["stm_mobility"] / 50.0,
            "bishop": [float(bp[0]), float(bp[1])]}


def stream(want_val, seed):
    rng = random.Random(seed)
    while True:
        files = sorted(f for d in RICH for f in glob.glob(os.path.join(d, "*.jsonl")))
        if not files:
            raise SystemExit("no rich/*.jsonl found")
        rng.shuffle(files)
        for fp in files:
            try:
                for i, line in enumerate(open(fp, encoding="utf-8")):
                    if (i % HOLDOUT == 0) != want_val:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield encode(json.loads(line))
                    except Exception:
                        continue
            except Exception:
                continue


def make_batch(gen, bsz):
    recs = [next(gen) for _ in range(bsz)]
    ids = torch.tensor([r["ids"] for r in recs], dtype=torch.long, device=device)
    cast = torch.tensor([r["cast"] for r in recs], dtype=torch.float, device=device)
    P = torch.zeros(bsz, N_MOVES, device=device)
    for i, r in enumerate(recs):
        for idx, wt in zip(r["p_idx"], r["p_w"]):
            P[i, idx] += wt
    g = lambda k: torch.tensor([r[k] for r in recs], dtype=torch.float, device=device)
    return ids, cast, P, g("value"), g("material"), g("king"), g("pawns"), g("mob"), g("bishop")


def losses(out, P, value, material, king, pawns, mob, bishop):
    lp = -(P * F.log_softmax(out["policy"], dim=-1)).sum(dim=-1).mean()
    lv = F.binary_cross_entropy_with_logits(out["value"], value)
    la = (F.mse_loss(out["material"], material) + F.mse_loss(out["king"], king)
          + F.mse_loss(out["pawns"], pawns) + F.mse_loss(out["mob"], mob)
          + F.binary_cross_entropy_with_logits(out["bishop"], bishop))
    return lp + 0.5 * lv + 0.2 * la, lp, lv, la


@torch.no_grad()
def validate(model, n):
    model.eval()
    vgen = stream(True, 99)
    tot = {"policy": 0.0, "value": 0.0, "bishop": 0.0, "top1": 0.0, "top5": 0.0}
    for _ in range(n):
        b = make_batch(vgen, BSZ)
        out = model(b[0], b[1])
        tot["policy"] += -(b[2] * F.log_softmax(out["policy"], -1)).sum(-1).mean().item()
        tot["value"] += F.binary_cross_entropy_with_logits(out["value"], b[3]).item()
        tot["bishop"] += F.binary_cross_entropy_with_logits(out["bishop"], b[8]).item()
        gold = b[2].argmax(-1)
        tot["top1"] += (out["policy"].argmax(-1) == gold).float().mean().item()
        tot["top5"] += (out["policy"].topk(5, -1).indices == gold[:, None]).any(-1).float().mean().item()
    model.train()
    return {k: v / n for k, v in tot.items()}


def warm(it):
    """Warmup, then constant. A decay tied to a step budget can stop progress and look like convergence."""
    return (it + 1) / WARMUP if it < WARMUP else 1.0


def lr_at(it):
    return LR * warm(it)


torch.manual_seed(SEED)
if ARCH == "trop":
    # matched_dff_h assumes a control with a 4x feed-forward. Against the swept 2x control that gives
    # tropical 40 percent more parameters, so DFF_H overrides it.
    dff_h = int(E("DFF_H", 0)) or matched_dff_h(D, H)
    model = MayflyChessNet(d=D, layers=LAYERS, dff_h=dff_h, H=H, T=T_TROP, sc=SC_TROP,
                           learn_ts=LEARN_TS, mix_norm=MIX_NORM).to(device)
    label = (f"MAYFLY tropical (H={H} dff_h={dff_h} T={T_TROP} sc={SC_TROP}"
             f"{' learn_ts' if LEARN_TS else ''}{' mix_norm' if MIX_NORM else ''})")
else:
    model = ChessNet(d=D, nhead=NHEAD, layers=LAYERS, dim_ff=int(FF_MULT * D)).to(device)
    for m in model.modules():                       # dropout is a swept hyperparameter, not a default
        if isinstance(m, torch.nn.Dropout):
            m.p = DROPOUT
    label = f"CONTROL transformer (nhead={NHEAD} ff={FF_MULT}x drop={DROPOUT})"

# Everything needed to rebuild this model. Head count and temperature change no tensor shape, so a
# loader that guessed them would silently build the wrong model.
CFG = {"arch": ARCH, "d": D, "layers": LAYERS, "H": H, "T": T_TROP, "sc": SC_TROP,
       "learn_ts": LEARN_TS, "mix_norm": MIX_NORM, "ts_target": TS_TARGET,
       "nhead": NHEAD, "ff_mult": FF_MULT, "dropout": DROPOUT,
       "lr": LR, "lr_head": LR_HEAD or LR, "clip": CLIP, "clip_head": CLIP_HEAD or CLIP}

# The heads are ordinary linear maps and the trunk is not. With one learning rate and one shared clip
# the trunk's spiky gradients throttled the heads, so they get their own groups.
HEAD_ROOTS = {"policy", "value", "h_material", "h_king", "h_pawns", "h_mob", "h_bishop", "ln"}
named = list(model.named_parameters())
is_head = lambda n: n.split(".")[0] in HEAD_ROOTS
# No weight decay on scalars and vectors. Besides norm gains and biases this matters for the learned
# mixer temperature, which is stored as a log, so decay would pull the temperature toward one and back
# to blending. Applied to both architectures.
groups = []
for tag, base in (("trunk", LR), ("head", LR_HEAD or LR)):
    want_head = tag == "head"
    for decay, wd in (("decay", 0.05), ("nodecay", 0.0)):
        ps = [p for n, p in named
              if is_head(n) == want_head and ((p.ndim > 1) == (decay == "decay"))]
        if ps:
            groups.append({"params": ps, "base_lr": base, "weight_decay": wd, "tag": f"{tag}_{decay}"})
opt = torch.optim.AdamW(groups, lr=LR, weight_decay=0.05, betas=(0.9, 0.95))
head_p = [p for n, p in named if is_head(n)]
trunk_p = [p for n, p in named if not is_head(n)]
print("param groups: " + ", ".join(f"{g['tag']} {sum(p.numel() for p in g['params'])/1e6:.3f}M "
                                   f"lr {g['base_lr']:.1e} wd {g['weight_decay']}"
                                   for g in groups), flush=True)
start_it, best_val, best_it, hist, gain = 0, 1e9, 0, [], None

if RESUME and os.path.exists(LAST):
    ck = torch.load(LAST, map_location=device)
    model.load_state_dict(ck["model"])
    try:
        opt.load_state_dict(ck["optim"])
    except ValueError as e:
        # older checkpoints have one parameter group and cannot load into two. Fresh moments cost
        # a few hundred steps, better than refusing to resume.
        print(f"optimiser state not loadable ({e}); continuing with fresh moments", flush=True)
    for g, src in zip(opt.param_groups, groups):
        g["base_lr"] = src["base_lr"]    # param_groups survive a load, the extra keys may not
        g["weight_decay"] = src["weight_decay"]
        g["tag"] = src["tag"]
    start_it, best_val, best_it = ck["iter"] + 1, ck["best_val"], ck["best_it"]
    hist = ck.get("hist", [])
    print(f"RESUMED from iter {ck['iter']} | best {best_val:.4f} at {best_it}", flush=True)
elif ARCH == "trop" and LEARN_TS:
    # calibrate on real scores rather than guessing a temperature
    cal = make_batch(stream(False, 4321), BSZ)
    chosen = model.calibrate(cal[0], cal[1], target_width=TS_TARGET)
    if TS_FIX:
        for blk in model.blocks:
            blk.log_ts.requires_grad_(False)
    print("calibrated mixer to ~%.1f squares per layer%s, Ts = %s"
          % (TS_TARGET, " and FROZEN" if TS_FIX else "",
             " ".join(f"{t:.4f}" for t in chosen)), flush=True)

print(f"{label} | {model.num_params()/1e6:.3f}M params | d{D} L{LAYERS} bsz{BSZ} | "
      f"cap {ITERS} iters / {MAX_HOURS}h | crawl < {CRAWL_THRESH} per "
      f"{CRAWL_WINDOW * VAL_EVERY} iters | constant lr {LR} | {device}", flush=True)

tgen = stream(False, 1234)
for _ in range(start_it % 1000):            # cheap partial-stream advance after a resume
    next(tgen)
model.train()
t0 = time.time()
stopped = None

for it in range(start_it, ITERS + 1):
    if it % VAL_EVERY == 0:
        v = validate(model, VAL_BATCHES)
        mix = None
        if ARCH == "trop":
            # selectivity is logged beside the loss on every evaluation
            mb = make_batch(stream(True, 7), 64)
            st = model.mixer_stats(mb[0], mb[1])
            mix = {"width": [round(s["width"], 2) for s in st],
                   "top1": [round(s["top1"], 3) for s in st],
                   "share": [round(s["mix_share"], 3) for s in st],
                   "ts": [round(s["ts"], 4) for s in st]}
        hist.append({"it": it, **v, "lr": lr_at(it), "t": round(time.time() - t0, 1),
                     **({"mix": mix} if mix else {})})
        json.dump(hist, open(HIST, "w"), indent=1)
        # how much the best has improved over the last CRAWL_WINDOW evaluations
        pol = [h["policy"] for h in hist]
        gain = (min(pol[:-CRAWL_WINDOW]) - min(pol)) if len(pol) > CRAWL_WINDOW else None
        if v["policy"] < best_val - 1e-4:
            best_val, best_it = v["policy"], it
            torch.save({"model": model.state_dict(), "arch": ARCH, "val": v, "iter": it,
                        "cfg": CFG}, CKPT)
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "iter": it,
                    "best_val": best_val, "best_it": best_it, "hist": hist}, LAST)
        print(f"  it {it:7d}  policy {v['policy']:.4f}  top1 {v['top1']:.4f}  top5 {v['top5']:.4f}  "
              f"value {v['value']:.4f}  bishop {v['bishop']:.4f}  lr {lr_at(it):.2e}  "
              f"gain/{CRAWL_WINDOW}ev {gain if gain is None else round(gain, 4)}  "
              f"{time.time()-t0:6.0f}s", flush=True)
        if mix:
            print(f"           squares read {' '.join(f'{w:.1f}' for w in mix['width'])}   "
                  f"top1 {' '.join(f'{t:.2f}' for t in mix['top1'])}   "
                  f"mixer share {' '.join(f'{s:.2f}' for s in mix['share'])}", flush=True)
        if gain is not None and gain < CRAWL_THRESH:
            stopped = it
            print(f"CRAWLING: best improved only {gain:.4f} nats over the last "
                  f"{CRAWL_WINDOW * VAL_EVERY} iters (threshold {CRAWL_THRESH}). "
                  f"best {best_val:.4f} at {best_it}", flush=True)
            break
        if (time.time() - t0) / 3600.0 > MAX_HOURS:
            stopped = it
            print(f"TIME CAP: {MAX_HOURS}h reached, still improving {gain}. "
                  f"best {best_val:.4f} at {best_it}", flush=True)
            break
    if it == ITERS:
        break
    w = warm(it)
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * w
    b = make_batch(tgen, BSZ)
    out = model(b[0], b[1])
    loss, lp, lv, la = losses(out, *b[2:])
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trunk_p, CLIP)
    torch.nn.utils.clip_grad_norm_(head_p, CLIP_HEAD or CLIP)
    opt.step()

print(f"\nDONE {label}")
print(f"  params      {model.num_params()/1e6:.3f}M")
print(f"  best policy {best_val:.4f} at iter {best_it}")
print(f"  stopped at  {stopped if stopped else ITERS}")
print(f"  wall clock  {time.time()-t0:.0f}s")
print(f"  history     {os.path.basename(HIST)}", flush=True)
