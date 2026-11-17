"""
train_human_net trains a small policy-only style net by behaviour cloning on human moves.

It reads chess/human/*.jsonl (from parse_gm.py and parse_lichess_human.py) and trains a standalone ChessNet
(d512 by 8 layers, about 28M parameters) with cross-entropy to the move the human played, seen from the side
to move. The engine net gives strength and this net gives style. They stay separate and can be blended at
play time, the engine proposing moves and this net re-ranking them.

  python chess/train_human_net.py [ITERS=40000] [BSZ=256] [D=512] [LAYERS=8]
Saves chess/ckpt_human_net.pt in the same cfg format as the engine checkpoint.
"""
import os, sys, glob, json, time, random, math
import torch, torch.nn.functional as F, chess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet, board_to_ids, move_to_idx, N_MOVES

HUMAN = os.path.join(HERE, "human")
ITERS  = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
BSZ    = int(sys.argv[2]) if len(sys.argv) > 2 else 256
D      = int(sys.argv[3]) if len(sys.argv) > 3 else 512
LAYERS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
device = "cuda" if torch.cuda.is_available() else "cpu"
from contextlib import nullcontext
bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()

def encode(rec):
    board = chess.Board(rec["fen"])
    ids, cast, flip = board_to_ids(board)
    return ids, cast, move_to_idx(chess.Move.from_uci(rec["move"]), flip)

def stream():
    while True:
        files = sorted(glob.glob(os.path.join(HUMAN, "*.jsonl"))); random.shuffle(files)
        if not files:
            raise SystemExit("no human/*.jsonl -- run parse_gm.py / parse_lichess_human.py first")
        for fp in files:
            try:
                for line in open(fp, encoding="utf-8"):
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
    ids = torch.tensor([r[0] for r in recs], dtype=torch.long, device=device)
    cast = torch.tensor([r[1] for r in recs], dtype=torch.float, device=device)
    y = torch.tensor([r[2] for r in recs], dtype=torch.long, device=device)
    return ids, cast, y

model = ChessNet(d=D, nhead=max(4, D // 64), layers=LAYERS, dim_ff=4 * D).to(device)
print(f"HumanNet (style) {model.num_params()/1e6:.1f}M (d{D} L{LAYERS}) on {device}", flush=True)
PEAK, MIN, WARMUP = 3e-4, 3e-5, 1000
try:
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK, weight_decay=0.05, betas=(0.9, 0.95), fused=(device == "cuda"))
except Exception:
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK, weight_decay=0.05, betas=(0.9, 0.95))

def lr_at(it):
    if it < WARMUP:
        return PEAK * (it + 1) / WARMUP
    r = (it - WARMUP) / max(1, ITERS - WARMUP)
    return MIN + 0.5 * (1 + math.cos(math.pi * r)) * (PEAK - MIN)

gen = stream(); model.train(); t0 = time.time()
for it in range(ITERS):
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    ids, cast, y = make_batch(gen, BSZ)
    with ctx:
        logits = model(ids, cast)["policy"]
        loss = F.cross_entropy(logits, y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if it % 200 == 0:
        with torch.no_grad():
            top1 = (logits.argmax(-1) == y).float().mean().item()
        print(f"it{it:6d} loss {loss.item():.3f} top1 {top1:.3f} {time.time()-t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "cfg": {"d": D, "layers": LAYERS}}, os.path.join(HERE, "ckpt_human_net.pt"))
torch.save({"model": model.state_dict(), "cfg": {"d": D, "layers": LAYERS}}, os.path.join(HERE, "ckpt_human_net.pt"))
print("DONE -> chess/ckpt_human_net.pt", flush=True)
