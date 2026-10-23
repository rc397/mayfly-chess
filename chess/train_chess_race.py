"""
train_chess_race compares the tropical encoder with a transformer encoder on identical chess data.

Same rich/*.jsonl stream, multi-task loss, optimiser, learning rate, batch order (the data RNG is seeded
the same for both) and held-out split. Only the encoder differs, and parameter counts match to within a
percent. Chess suits the tropical algebra because negamax is a maximum over moves of a sum along a line,
which is the tropical semiring itself.

  python chess/train_chess_race.py [ITERS] [BSZ] [D] [LAYERS]
  env: ARCH=trop|ctrl   (one architecture per run, run both to race)
       VAL_EVERY=500  VAL_BATCHES=20  LR=3e-4  TAG=...

Every 20th line of the stream is held out for validation and never trained on.
"""
import os, sys, glob, json, time, random
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import chess
from chess_net import ChessNet, board_to_ids, move_to_idx, N_MOVES
from mayfly_chess import MayflyChessNet, matched_dff_h

E = os.environ.get
# Corpus folders, comma separated. "rich" is Stockfish self-play at 100 to 200ms and "rich5" is
# tournament positions labelled at 5ms. List both to train on the union.
RICH = [os.path.join(HERE, d) for d in os.environ.get("DATA", "rich").split(",")]
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
BSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 128
D = int(sys.argv[3]) if len(sys.argv) > 3 else 256
LAYERS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
ARCH = E("ARCH", "trop")
H = int(E("H", 8))
LR = float(E("LR", 3e-4))
VAL_EVERY = int(E("VAL_EVERY", 500))
VAL_BATCHES = int(E("VAL_BATCHES", 20))
TAG = E("TAG", ARCH)
device = "cuda" if torch.cuda.is_available() else "cpu"
TEMP = 100.0                      # cp softmax temperature for the policy soft target
HOLDOUT = 20                      # every Nth record is validation, never trained on


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
    """Deterministic stream. want_val picks the held-out slice. The seed is fixed per split, so both
    architectures see the same records in the same order."""
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
def validate(model, vgen, n):
    model.eval()
    lp_sum = top1_sum = 0.0
    for _ in range(n):
        b = make_batch(vgen, BSZ)
        out = model(b[0], b[1])
        _, lp, _, _ = losses(out, *b[2:])
        lp_sum += lp.item()
        top1_sum += (out["policy"].argmax(-1) == b[2].argmax(-1)).float().mean().item()
    model.train()
    return lp_sum / n, top1_sum / n


def main():
    """The training run, kept in a function so other scripts can import the helpers above."""
    if ARCH == "trop":
        dff_h = matched_dff_h(D, H)
        model = MayflyChessNet(d=D, layers=LAYERS, dff_h=dff_h, H=H).to(device)
        label = f"MAYFLY tropical (H={H}, dff_h={dff_h})"
    else:
        model = ChessNet(d=D, nhead=max(4, D // 64), layers=LAYERS, dim_ff=4 * D).to(device)
        label = "CONTROL transformer"

    print(f"{label} | {model.num_params()/1e6:.3f}M params | d{D} L{LAYERS} bsz{BSZ} | {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05, betas=(0.9, 0.95))
    tgen, vgen = stream(False, 1234), stream(True, 99)
    model.train()
    t0 = time.time()
    best = (1e9, 0.0, 0)
    hist = []

    for it in range(ITERS + 1):
        if it % VAL_EVERY == 0:
            vl, vtop1 = validate(model, stream(True, 99), VAL_BATCHES)
            hist.append((it, vl, vtop1))
            if vl < best[0]:
                best = (vl, vtop1, it)
                torch.save({"model": model.state_dict(), "arch": ARCH,
                            "cfg": {"d": D, "layers": LAYERS, "H": H}, "val": vl, "top1": vtop1, "iter": it},
                           os.path.join(HERE, f"ckpt_race_{TAG}.pt"))
            print(f"  it {it:6d}  val_policy {vl:.4f}  val_top1 {vtop1:.4f}  {time.time()-t0:7.0f}s",
                  flush=True)
        if it == ITERS:
            break
        b = make_batch(tgen, BSZ)
        out = model(b[0], b[1])
        loss, lp, lv, la = losses(out, *b[2:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"\n{label}")
    print(f"  params        {model.num_params()/1e6:.3f}M")
    print(f"  best val loss {best[0]:.4f}  top1 {best[1]:.4f}  at iter {best[2]}")
    print(f"  final         {hist[-1][1]:.4f}  top1 {hist[-1][2]:.4f}")
    print(f"  wall clock    {time.time()-t0:.0f}s")
    print(f"  checkpoint    ckpt_race_{TAG}.pt", flush=True)


if __name__ == "__main__":
    main()
