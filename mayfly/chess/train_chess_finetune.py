"""
train_chess_finetune fine-tunes ckpt_chess_net.pt with extra weight on the deep-labelled repertoire positions.

A fraction REP_FRAC of every batch comes from rich/rich_rep_*.jsonl (written by annotate_fens.py) and the
rest from the general rich files. The learning rate is lower than a from-scratch run. Saves
ckpt_chess_net_rep.pt so the base model stays intact for comparison.

  python chess/train_chess_finetune.py [ITERS=20000] [BSZ=128] [REP_FRAC=0.85] [LR=1.5e-4]
Needs a GPU, and annotate_fens.py run first.
"""
import os, sys, glob, json, time, random, math
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import chess
from chess_net import ChessNet, board_to_ids, move_to_idx, N_MOVES

RICH = os.path.join(HERE, "rich")
ITERS    = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
BSZ      = int(sys.argv[2]) if len(sys.argv) > 2 else 128
REP_FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85
LR       = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
from contextlib import nullcontext
bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
TEMP = 100.0

def encode(rec):                                   # identical record encoding to train_chess_net
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
            "material": [mat[0]/39.0, mat[1]/39.0], "king": [ks[0]/8.0, ks[1]/8.0],
            "pawns": [x/8.0 for x in pawns], "mob": f["stm_mobility"]/50.0,
            "bishop": [float(bp[0]), float(bp[1])]}

def stream(rep):
    while True:
        allf = sorted(glob.glob(os.path.join(RICH, "*.jsonl")))
        files = [f for f in allf if os.path.basename(f).startswith("rich_rep_")] if rep \
                else [f for f in allf if not os.path.basename(f).startswith("rich_rep_")]
        random.shuffle(files)
        if not files:
            yield None; continue
        for fp in files:
            try:
                for line in open(fp, encoding="utf-8"):
                    line = line.strip()
                    if line:
                        try:
                            yield encode(json.loads(line))
                        except Exception:
                            continue
            except Exception:
                continue

rep_gen, gen_gen = stream(rep=True), stream(rep=False)

def next_rec():
    if random.random() < REP_FRAC:
        r = next(rep_gen)
        if r is not None:
            return r
    r = next(gen_gen)
    return r if r is not None else next(rep_gen)

def make_batch(bsz):
    recs = [next_rec() for _ in range(bsz)]
    ids = torch.tensor([r["ids"] for r in recs], dtype=torch.long, device=device)
    cast = torch.tensor([r["cast"] for r in recs], dtype=torch.float, device=device)
    P = torch.zeros(bsz, N_MOVES, device=device)
    for i, r in enumerate(recs):
        for idx, wt in zip(r["p_idx"], r["p_w"]):
            P[i, idx] += wt
    g = lambda k: torch.tensor([r[k] for r in recs], dtype=torch.float, device=device)
    return ids, cast, P, g("value"), g("material"), g("king"), g("pawns"), g("mob"), g("bishop")

ck = torch.load(os.path.join(HERE, "ckpt_chess_net.pt"), map_location=device)
cfg = ck["cfg"]; D, LAYERS = cfg["d"], cfg["layers"]
model = ChessNet(d=D, nhead=max(4, D // 64), layers=LAYERS, dim_ff=4 * D).to(device)
model.load_state_dict(ck["model"])
print(f"fine-tune ChessNet {model.num_params()/1e6:.1f}M (d{D} L{LAYERS}) | REP_FRAC {REP_FRAC} on {device}", flush=True)
PEAK_LR, MIN_LR, WARMUP = LR, LR / 10, 300      # 5e-5 under-fit
try:
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.05, betas=(0.9, 0.95), fused=(device == "cuda"))
except Exception:
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=0.05, betas=(0.9, 0.95))
def lr_at(it):
    if it < WARMUP:
        return PEAK_LR * (it + 1) / WARMUP
    r = (it - WARMUP) / max(1, ITERS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (PEAK_LR - MIN_LR)

model.train(); t0 = time.time()
for it in range(ITERS):
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    ids, cast, P, value, material, king, pawns, mob, bishop = make_batch(BSZ)
    with ctx:
        out = model(ids, cast)
        loss_p = -(P * F.log_softmax(out["policy"], dim=-1)).sum(dim=-1).mean()
        loss_v = F.binary_cross_entropy_with_logits(out["value"], value)
        loss_aux = (F.mse_loss(out["material"], material) + F.mse_loss(out["king"], king)
                    + F.mse_loss(out["pawns"], pawns) + F.mse_loss(out["mob"], mob)
                    + F.binary_cross_entropy_with_logits(out["bishop"], bishop))
        loss = loss_p + 0.5 * loss_v + 0.2 * loss_aux
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if it % 200 == 0:
        with torch.no_grad():
            top1 = (out["policy"].argmax(-1) == P.argmax(-1)).float().mean().item()
        print(f"it{it:6d} loss {loss.item():.3f} (p {loss_p.item():.3f} v {loss_v.item():.3f} "
              f"aux {loss_aux.item():.3f}) top1 {top1:.3f} {time.time()-t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "cfg": {"d": D, "layers": LAYERS}},
                   os.path.join(HERE, "ckpt_chess_net_rep.pt"))
torch.save({"model": model.state_dict(), "cfg": {"d": D, "layers": LAYERS}}, os.path.join(HERE, "ckpt_chess_net_rep.pt"))
print("DONE -> chess/ckpt_chess_net_rep.pt (engine, repertoire-focused)", flush=True)
