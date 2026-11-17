"""Probe where counting information is lost in the tropical (max-plus) trunk.

In a max-plus trunk every path into the CLS token runs through a maximum, which reads one input, so
counts may not reach CLS. This fits a ridge probe for material, king, pawn and bishop targets on the
CLS token and on the mean over all 65 tokens, for both the tropical net and the transformer. A big
gain from pooling means the information is in the trunk but not routed to CLS. The probe is fitted
on one half and scored on the other.

    python chess/chess_diag.py [N_POSITIONS]
"""
import os, sys, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet
from mayfly_chess import MayflyChessNet, matched_dff_h
import train_chess_race as R

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
BSZ = 128
device = "cuda" if torch.cuda.is_available() else "cpu"


def tuned_cfg(arch):
    p = os.path.join(HERE, f"sweep_{arch}.json")
    return json.load(open(p, encoding="utf-8"))["final"] if os.path.exists(p) else {}


def load(arch):
    ck = torch.load(os.path.join(HERE, f"ckpt_long_{arch}.pt"), map_location=device)
    sd, cfg, tuned = ck["model"], ck.get("cfg", {}), tuned_cfg(arch)
    d, L = cfg.get("d", 256), cfg.get("layers", 4)
    if arch == "trop":
        H = int(tuned.get("H", cfg.get("H", 8)))
        m = MayflyChessNet(d=d, layers=L, dff_h=matched_dff_h(d, H), H=H,
                           T=float(tuned.get("T", 0.66)), sc=float(tuned.get("SC", 1.5)))
    else:
        m = ChessNet(d=d, nhead=int(tuned.get("NHEAD", 0)) or max(4, d // 64), layers=L,
                     dim_ff=int(sd["enc.layers.0.linear1.weight"].shape[0]))
    m.load_state_dict(sd, strict=True); m.to(device).eval()
    return m


@torch.no_grad()
def representations(model, ids, cast, is_trop):
    """Final-layer token stack, before the readout picks CLS out of it."""
    if is_trop:
        B = ids.shape[0]
        x = model.piece(ids) + model.pos(model.posidx)[None]
        cls = model.cls.expand(B, -1, -1) + model.cast(cast)[:, None, :]
        x = torch.cat([cls, x], dim=1)
        for blk in model.blocks:
            x = blk(x)
        return model.ln(x)                       # (B, 65, d), same norm the heads see
    B = ids.shape[0]
    x = model.piece(ids) + model.pos(model.posidx)[None]
    cls = model.cls.expand(B, -1, -1) + model.cast(cast)[:, None, :]
    x = torch.cat([cls, x], dim=1)
    return model.ln(model.enc(x))


def ridge_r2(X, y, lam=1.0):
    """Fit on the first half, score on the second. R2 of 1 is perfect, 0 is predicting the mean."""
    n = len(X) // 2
    Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float64)], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1), np.float64)], 1)
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w
    ss_res = ((yte - pred) ** 2).sum(0)
    ss_tot = ((yte - yte.mean(0, keepdims=True)) ** 2).sum(0) + 1e-12
    return float(np.mean(1.0 - ss_res / ss_tot))


if __name__ == "__main__":
    gen = R.stream(True, 99)
    feats = {a: {"cls": [], "pool": []} for a in ("trop", "ctrl")}
    targets = {"material": [], "bishop": [], "pawns": [], "king": []}
    models = {a: load(a) for a in ("trop", "ctrl")}
    print(f"probing {N} held-out positions on {device}\n", flush=True)

    done = 0
    while done < N:
        b = R.make_batch(gen, BSZ)
        ids, cast = b[0], b[1]
        for a, m in models.items():
            rep = representations(m, ids, cast, a == "trop")
            feats[a]["cls"].append(rep[:, 0].float().cpu().numpy())
            feats[a]["pool"].append(rep.mean(1).float().cpu().numpy())
        for k, t in (("material", b[4]), ("king", b[5]), ("pawns", b[6]), ("bishop", b[8])):
            targets[k].append(t.float().cpu().numpy())
        done += BSZ

    for k in targets:
        targets[k] = np.concatenate(targets[k]).astype(np.float64)
    for a in feats:
        for w in feats[a]:
            feats[a][w] = np.concatenate(feats[a][w]).astype(np.float64)

    print(f"{'target':10s} {'tropical CLS':>13s} {'tropical POOL':>14s} "
          f"{'transf CLS':>12s} {'transf POOL':>12s}")
    print("-" * 66)
    out = {}
    for k, y in targets.items():
        row = {}
        for a in ("trop", "ctrl"):
            for w in ("cls", "pool"):
                row[f"{a}_{w}"] = ridge_r2(feats[a][w], y)
        out[k] = row
        print(f"{k:10s} {row['trop_cls']:13.3f} {row['trop_pool']:14.3f} "
              f"{row['ctrl_cls']:12.3f} {row['ctrl_pool']:12.3f}")

    print()
    for a, nm in (("trop", "tropical"), ("ctrl", "transformer")):
        gains = [out[k][f"{a}_pool"] - out[k][f"{a}_cls"] for k in out]
        print(f"{nm:12s} mean R2 gain from pooling over tokens: {np.mean(gains):+.3f}")
    json.dump(out, open(os.path.join(HERE, "diag_probe.json"), "w"), indent=1)
    print("\nwrote diag_probe.json")
