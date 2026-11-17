r"""
train_engine trains ChessNet on the packed shards in bf16, resumably, while new data keeps arriving.

  loader      packed records take 1.10ms per batch against 16.90ms for JSON, so the GPU is the bottleneck
  precision   bf16 ran at 768 positions/s against 334 for fp32. FP8 (PRECISION=fp8) was slower at this
              size because of torchao's scaling overhead
  split       held out by position, not by line, so a repeated board always lands on the same side
  live data   the loader re-maps its shards on a timer and picks up positions as they are written

  python chess/train_engine.py [ITERS]
  env: SOURCES=lz,lc     which corpora to train on, comma separated, or 'all'
                         lz = Lichess deep Stockfish, gm = tournament labels,
                         sp = self-play, lc = Leela
       D=1024 LAYERS=16 BSZ=128 LR=1.2e-4 PRECISION=bf16 TAG=engine
       VAL_EVERY=2000 MAX_HOURS=720 RESUME=1  SHARDS (default paths.under("shards"))
"""
import os, sys, json, time, math
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet, N_MOVES
from data_packed import PackedLoader
from precision import apply_precision, autocast_ctx
import paths

E = os.environ.get
SHARDS = E("SHARDS", paths.under("shards"))
# gm and sp are excluded by default. They were labelled at 5ms (about 1,900 nodes) against 25M
# nodes for Lichess and 468 Leela visits, so mixing them in mostly adds label noise.
SOURCES = None if E("SOURCES", "lz,lc").lower() == "all" else E("SOURCES", "lz,lc").split(",")
D, LAYERS = int(E("D", 1024)), int(E("LAYERS", 16))
NHEAD, FF_MULT = int(E("NHEAD", 16)), float(E("FF_MULT", 2.0))
BSZ = int(E("BSZ", 128))
# 3e-4 from the small runs diverged at this size. The learning rate should fall roughly with width,
# and the warmup should be long enough for the optimiser moments to settle.
LR = float(E("LR", 1.2e-4))
TAG = E("TAG", "engine")
VAL_EVERY = int(E("VAL_EVERY", 2000))
VAL_BATCHES = int(E("VAL_BATCHES", 40))
MAX_HOURS = float(E("MAX_HOURS", 720))
RESUME = E("RESUME", "1") == "1"
SEED = int(E("SEED", 0))
ITERS = int(float(sys.argv[1])) if len(sys.argv) > 1 and sys.argv[1].replace(".", "").isdigit() \
    else int(float(E("ITERS", 4_000_000)))
WARMUP = int(E("WARMUP", 12000))
device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = os.path.join(HERE, f"ckpt_{TAG}.pt")
LAST = os.path.join(HERE, f"ckpt_{TAG}_last.pt")
HIST = os.path.join(HERE, f"hist_{TAG}.json")


def lr_at(it):
    """Warmup then constant. A decay tied to a step budget can stop progress and look like convergence."""
    return LR * min(1.0, (it + 1) / WARMUP)


def losses(out, pol, val, mat, king, pawns, mob, bish, aux):
    """aux is 1 where the source really had the positional features and 0 where they were filled
    with zeros. Leela's format carries no king safety or pawn structure, so training those heads on
    its records teaches that they are always zero on half the corpus."""
    lp = -(pol * F.log_softmax(out["policy"].float(), -1)).sum(-1).mean()
    lv = F.binary_cross_entropy_with_logits(out["value"].float(), val)
    w = aux.view(-1, 1)
    n = aux.sum().clamp(min=1.0)
    msew = lambda a, b, k: (((a.float() - b) ** 2).mean(-1, keepdim=True) * k).sum() / n
    la = (msew(out["material"], mat, w) + msew(out["king"], king, w)
          + msew(out["pawns"], pawns, w) + msew(out["mob"].unsqueeze(-1), mob.unsqueeze(-1), w)
          + (F.binary_cross_entropy_with_logits(out["bishop"].float(), bish, reduction="none")
             .mean(-1, keepdim=True) * w).sum() / n)
    return lp + 0.5 * lv + 0.2 * la, lp, lv, la


@torch.no_grad()
def validate(model, used, va, n):
    model.eval()
    tot = {"policy": 0.0, "value": 0.0, "top1": 0.0, "top5": 0.0}
    for _ in range(n):
        b = va.next()
        with autocast_ctx(used):
            out = model(b[0], b[1], b[2], b[3])
        lp = -(b[4] * F.log_softmax(out["policy"].float(), -1)).sum(-1).mean()
        tot["policy"] += float(lp)
        tot["value"] += float(F.binary_cross_entropy_with_logits(out["value"].float(), b[5]))
        tgt = b[4].argmax(-1)
        tot["top1"] += float((out["policy"].argmax(-1) == tgt).float().mean())
        tot["top5"] += float((out["policy"].topk(5, -1).indices == tgt[:, None]).any(-1)
                             .float().mean())
    model.train()
    return {k: v / n for k, v in tot.items()}


if __name__ == "__main__":
    torch.manual_seed(SEED)
    tr = PackedLoader(SHARDS, batch=BSZ, split="train", device=device, seed=SEED,
                      sources=SOURCES, refresh=600)
    va = PackedLoader(SHARDS, batch=BSZ, split="val", device=device, seed=SEED + 1,
                      sources=SOURCES, refresh=600)
    print(tr.describe(), flush=True)

    model = ChessNet(d=D, nhead=NHEAD, layers=LAYERS, dim_ff=int(FF_MULT * D)).to(device)
    model, used = apply_precision(model)
    # no weight decay on scalars and vectors, so norm gains and biases are not pulled to zero
    named = list(model.named_parameters())
    decay = [p for _, p in named if p.ndim > 1]
    nodecay = [p for _, p in named if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.95))
    print(f"{model.num_params()/1e6:.1f}M params | d{D} x {LAYERS} | batch {BSZ} | "
          f"{used} | {device}", flush=True)

    start, hist, best = 0, [], 1e9
    if RESUME and os.path.exists(LAST):
        ck = torch.load(LAST, map_location=device)
        model.load_state_dict(ck["model"])
        try:
            opt.load_state_dict(ck["optim"])
        except (ValueError, KeyError) as e:
            print(f"optimiser state not loadable ({type(e).__name__}); fresh moments", flush=True)
        start, hist, best = ck["iter"] + 1, ck.get("hist", []), ck.get("best", 1e9)
        print(f"RESUMED at iter {start:,}, best policy {best:.4f}", flush=True)

    t0 = time.time()
    gnorm, skipped = 1.0, 0
    model.train()
    for it in range(start, ITERS + 1):
        if it % VAL_EVERY == 0:
            v = validate(model, used, va, VAL_BATCHES)
            el = time.time() - t0
            rate = (it - start) * BSZ / max(el, 1)
            hist.append({"it": it, **v, "t": round(el, 1), "pos_s": round(rate),
                         "corpus": len(tr), "grad": round(gnorm, 3),
                         "skipped": skipped})
            json.dump(hist, open(HIST, "w"), indent=1)
            if v["policy"] < best:
                best = v["policy"]
                torch.save({"model": model.state_dict(), "iter": it, "val": v,
                            "cfg": {"d": D, "layers": LAYERS, "nhead": NHEAD,
                                    "ff_mult": FF_MULT, "sources": SOURCES}}, CKPT)
            torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "iter": it,
                        "hist": hist, "best": best}, LAST)
            print(f"  it {it:8,}  policy {v['policy']:.4f}  top1 {v['top1']:.4f}  "
                  f"top5 {v['top5']:.4f}  value {v['value']:.4f}  "
                  f"{rate:.0f} pos/s  grad {gnorm:.2f}  corpus {len(tr):,}  "
                  f"{el/3600:.2f}h", flush=True)
            if el / 3600 > MAX_HOURS:
                print("time cap reached", flush=True)
                break
        if it == ITERS:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        b = tr.next()
        with autocast_ctx(used):
            out = model(b[0], b[1], b[2], b[3])
        loss, lp, lv, la = losses(out, *b[4:])
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        # A divergence shows in the gradient norm long before the loss. Skipping a spike costs one batch.
        if not math.isfinite(gn) or gn > 50.0:
            skipped += 1
            opt.zero_grad()
            if skipped % 20 == 1:
                print(f"  skipped a batch at it {it:,}, grad norm {gn:.1f} "
                      f"({skipped} so far)", flush=True)
            continue
        gnorm = 0.98 * gnorm + 0.02 * gn
        opt.step()

    print(f"\nDONE {TAG}  best policy {best:.4f}  {(time.time()-t0)/3600:.2f}h", flush=True)
    tr.close(); va.close()
