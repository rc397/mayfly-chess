"""Score the model separately on held-out positions it has seen in training and ones it has not.

The split takes every twentieth line and positions repeat, so some held-out positions also sit in
the training slice (7.7 percent in one measurement). If seen positions score much better, the
held-out number is inflated. Works on the live checkpoint while training continues.

    python chess/audit_memorise.py [N_TRAIN_LINES] [TAG]
"""
import os, sys, glob, json, random
import torch, torch.nn.functional as F
import chess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet, N_MOVES
from mayfly_chess import MayflyChessNet, matched_dff_h
import train_chess_race as R

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 600000
TAGS = sys.argv[2:] or ["real_trop", "real_ctrl"]
RICH = os.path.join(HERE, "rich")
HOLDOUT = 20
WANT = 3000                      # validation positions per group
device = "cuda" if torch.cuda.is_available() else "cpu"
TEMP = 100.0


def board_key(fen):
    return " ".join(fen.split()[:4])


def load_live(tag):
    """Rebuild from the config the checkpoint carries, so this works for any run."""
    p = os.path.join(HERE, f"ckpt_long_{tag}.pt")
    ck = torch.load(p, map_location=device)
    c = ck.get("cfg", {})
    d, L = c.get("d", 256), c.get("layers", 4)
    sd = ck["model"]
    if c.get("arch", "trop") == "trop":
        H = int(c.get("H", 8))
        dff_h = sd["blocks.0.w2a"].shape[0]
        m = MayflyChessNet(d=d, layers=L, dff_h=dff_h, H=H, T=float(c.get("T", 0.66)),
                           sc=float(c.get("sc", 1.5)),
                           learn_ts="blocks.0.log_ts" in sd, mix_norm="blocks.0.mix_g" in sd)
    else:
        m = ChessNet(d=d, nhead=int(c.get("nhead", 8)), layers=L,
                     dim_ff=int(sd["enc.layers.0.linear1.weight"].shape[0]))
    m.load_state_dict(sd, strict=True); m.to(device).eval()
    return m, ck.get("iter", -1), c.get("arch", "trop")


@torch.no_grad()
def score(model, recs):
    """Policy cross-entropy and top-1 agreement over a list of raw records."""
    tot_loss = tot_top1 = 0.0
    B = 64
    for i in range(0, len(recs), B):
        chunk = [R.encode(r) for r in recs[i:i + B]]
        ids = torch.tensor([c["ids"] for c in chunk], dtype=torch.long, device=device)
        cast = torch.tensor([c["cast"] for c in chunk], dtype=torch.float, device=device)
        P = torch.zeros(len(chunk), N_MOVES, device=device)
        for j, c in enumerate(chunk):
            for idx, w in zip(c["p_idx"], c["p_w"]):
                P[j, idx] += w
        out = model(ids, cast)
        tot_loss += float(-(P * F.log_softmax(out["policy"], -1)).sum(-1).sum())
        tot_top1 += float((out["policy"].argmax(-1) == P.argmax(-1)).float().sum())
    n = max(len(recs), 1)
    return tot_loss / n, tot_top1 / n


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(RICH, "*.jsonl")))
    rng = random.Random(0); rng.shuffle(files)
    train_keys, seen, unseen, read = set(), [], [], 0

    print(f"indexing up to {N_TRAIN:,} training lines", flush=True)
    for fp in files:
        if read >= N_TRAIN:
            break
        for i, line in enumerate(open(fp, encoding="utf-8")):
            if read >= N_TRAIN:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            read += 1
            if i % HOLDOUT != 0:
                train_keys.add(board_key(rec["fen"]))

    print(f"{len(train_keys):,} distinct training positions indexed", flush=True)
    print("collecting held-out positions, split by whether training saw them", flush=True)
    for fp in files:
        if len(seen) >= WANT and len(unseen) >= WANT:
            break
        for i, line in enumerate(open(fp, encoding="utf-8")):
            if len(seen) >= WANT and len(unseen) >= WANT:
                break
            if i % HOLDOUT != 0:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if board_key(rec["fen"]) in train_keys:
                if len(seen) < WANT:
                    seen.append(rec)
            elif len(unseen) < WANT:
                unseen.append(rec)

    print(f"  {len(seen):,} seen during training, {len(unseen):,} never seen\n", flush=True)
    print(f"{'model':12s} {'iter':>8s} {'group':>8s} {'policy loss':>13s} {'top-1':>9s}")
    print("-" * 56)
    out = {}
    for tag in TAGS:
        try:
            model, it, arch = load_live(tag)
        except Exception as e:
            print(f"{tag:12s} could not load: {e}")
            continue
        row = {}
        for label, recs in (("seen", seen), ("unseen", unseen)):
            l, t1 = score(model, recs)
            row[label] = {"policy": l, "top1": t1, "n": len(recs)}
            print(f"{tag:12s} {it:8d} {label:>8s} {l:13.4f} {t1:9.4f}")
        d = row["unseen"]["policy"] - row["seen"]["policy"]
        row["gap"] = d
        print(f"{'':12s} {'':>8s} {'gap':>8s} {d:+13.4f} "
              f"{row['unseen']['top1'] - row['seen']['top1']:+9.4f}")
        # repeats are mostly openings, which are easier anyway, so compare within move-number bands too
        mvno = lambda r: int(r["fen"].split()[-1])
        row["matched"] = {}
        for lo, hi in ((1, 12), (12, 25), (25, 200)):
            a = [r for r in seen if lo <= mvno(r) < hi]
            b = [r for r in unseen if lo <= mvno(r) < hi]
            if len(a) < 80 or len(b) < 80:
                continue
            la, ta = score(model, a)
            lb, tb = score(model, b)
            row["matched"][f"{lo}-{hi}"] = {"seen": la, "unseen": lb, "gap": lb - la,
                                            "n_seen": len(a), "n_unseen": len(b)}
            print(f"{'':12s} {'':>8s} {f'mv {lo}-{hi}':>8s} seen {la:.4f}  unseen {lb:.4f}  "
                  f"gap {lb-la:+.4f}  (n {len(a)}/{len(b)})")
        out[tag] = row
    json.dump(out, open(os.path.join(HERE, "audit_memorise.json"), "w"), indent=1)
    print("\nA large positive gap means the held-out score is inflated by repeats.")
