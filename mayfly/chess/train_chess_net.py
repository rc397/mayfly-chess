"""
train_chess_net trains the multi-task ChessNet on the JSONL that gen_chess_rich.py writes to rich/*.jsonl.

Records are streamed and encoded on the fly. The loss has three parts.
  policy  soft cross-entropy to Stockfish's action values (a softmax over the top-K centipawns)
  value   BCE to the win probability from the centipawn score
  aux     MSE or BCE on the positional features
Saves chess/ckpt_chess_net.pt.

  python chess/train_chess_net.py [ITERS=20000] [BSZ=256] [D=384] [LAYERS=6]
Needs a GPU for a real run. Small arguments make a smoke test.
"""
import os, sys, glob, json, time, random
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import chess
from chess_net import ChessNet, board_to_ids, move_to_idx, N_MOVES

RICH = os.path.join(HERE, "rich")
ITERS  = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
BSZ    = int(sys.argv[2]) if len(sys.argv) > 2 else 256
D      = int(sys.argv[3]) if len(sys.argv) > 3 else 384
LAYERS = int(sys.argv[4]) if len(sys.argv) > 4 else 6
device = "cuda" if torch.cuda.is_available() else "cpu"
TEMP = 100.0   # cp softmax temperature for the policy (action-value) soft target

def encode(rec):
    board = chess.Board(rec["fen"])
    ids, cast, flip = board_to_ids(board)
    tk = rec["topk"]
    idxs = [move_to_idx(chess.Move.from_uci(u), flip) for u, _ in tk]
    cps = np.array([c for _, c in tk], dtype=np.float32)
    w = np.exp((cps - cps.max()) / TEMP); w = (w / w.sum()).tolist()
    wp = 1.0 / (1.0 + 10 ** (-rec["cp"] / 400.0))
    f = rec["feat"]
    sw = (lambda a: [a[1], a[0]]) if flip else (lambda a: a)         # index 0 is the side to move
    mat, ks, bp, pw = sw(f["material"]), sw(f["king_safety"]), sw(f["bishop_pair"]), sw(f["pawns"])
    pawns = [pw[0][0], pw[0][1], pw[0][2], pw[1][0], pw[1][1], pw[1][2]]
    return {"ids": ids, "cast": cast, "p_idx": idxs, "p_w": w, "value": wp,
            "material": [mat[0]/39.0, mat[1]/39.0], "king": [ks[0]/8.0, ks[1]/8.0],
            "pawns": [x/8.0 for x in pawns], "mob": f["stm_mobility"]/50.0,
            "bishop": [float(bp[0]), float(bp[1])]}

def stream():
    while True:
        files = sorted(glob.glob(os.path.join(RICH, "*.jsonl"))); random.shuffle(files)
        if not files:
            raise SystemExit("no rich/*.jsonl yet -- let gen_chess_rich.py run first")
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

model = ChessNet(d=D, nhead=max(4, D // 64), layers=LAYERS, dim_ff=4 * D).to(device)
print(f"ChessNet {model.num_params()/1e6:.1f}M (d{D} L{LAYERS}) on {device}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05, betas=(0.9, 0.95))
gen = stream(); model.train(); t0 = time.time()
for it in range(ITERS):
    ids, cast, P, value, material, king, pawns, mob, bishop = make_batch(gen, BSZ)
    out = model(ids, cast)
    loss_p = -(P * F.log_softmax(out["policy"], dim=-1)).sum(dim=-1).mean()       # soft CE to action-values
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
                   os.path.join(HERE, "ckpt_chess_net.pt"))
torch.save({"model": model.state_dict(), "cfg": {"d": D, "layers": LAYERS}}, os.path.join(HERE, "ckpt_chess_net.pt"))
print("DONE -> chess/ckpt_chess_net.pt", flush=True)
