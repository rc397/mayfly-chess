"""
Find whether the policy gap between the tropical and transformer models is in the trunk or the head.

Each trunk is frozen, its trained policy head is discarded, and a fresh linear head is fitted on the
frozen representation with identical optimisation for both. The cross-entropy each reaches is what
its representation can support.

  trunks equal, heads differ   the gap is optimisation or readout, and cheap to fix
  trunks differ                the representation is worse, and the fix is structural

  python chess/probe_policy.py [STEPS] [BSZ]
"""
import os, sys, json, time
import torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet, N_MOVES
from mayfly_chess import MayflyChessNet, matched_dff_h
import train_chess_race as R

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
BSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 128
VAL_BATCHES = 20
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
    for p_ in m.parameters():
        p_.requires_grad_(False)
    return m, d


@torch.no_grad()
def trunk(model, ids, cast, is_trop):
    """The exact vector the trained policy head was reading, and the pooled alternative."""
    B = ids.shape[0]
    x = model.piece(ids) + model.pos(model.posidx)[None]
    cls = model.cls.expand(B, -1, -1) + model.cast(cast)[:, None, :]
    x = torch.cat([cls, x], dim=1)
    if is_trop:
        for blk in model.blocks:
            x = blk(x)
        x = model.ln(x)
    else:
        x = model.ln(model.enc(x))
    return x[:, 0], x.mean(1)


def run(arch, readout):
    model, d = load(arch)
    is_trop = arch == "trop"
    head = nn.Linear(d if readout == "cls" else 2 * d, N_MOVES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=0.0)
    tgen, t0 = R.stream(False, 1234), time.time()

    def feats(ids, cast):
        c, p = trunk(model, ids, cast, is_trop)
        return c if readout == "cls" else torch.cat([c, p], -1)

    for it in range(STEPS):
        b = R.make_batch(tgen, BSZ)
        loss = -(b[2] * F.log_softmax(head(feats(b[0], b[1])), -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    vgen = R.stream(True, 99)
    tot = top1 = 0.0
    with torch.no_grad():
        for _ in range(VAL_BATCHES):
            b = R.make_batch(vgen, BSZ)
            lg = head(feats(b[0], b[1]))
            tot += float(-(b[2] * F.log_softmax(lg, -1)).sum(-1).mean())
            top1 += float((lg.argmax(-1) == b[2].argmax(-1)).float().mean())
    return tot / VAL_BATCHES, top1 / VAL_BATCHES, time.time() - t0


if __name__ == "__main__":
    print(f"fresh linear policy head on frozen trunks, {STEPS} steps, {device}\n", flush=True)
    print(f"{'trunk':12s} {'readout':8s} {'probe policy':>13s} {'probe top1':>11s} {'trained':>9s}")
    print("-" * 58)
    trained = {"trop": 3.4155, "ctrl": 3.0559}
    res = {}
    for arch, nm in (("trop", "tropical"), ("ctrl", "transformer")):
        for readout in ("cls", "cls+pool"):
            l, t1, secs = run(arch, readout)
            res[f"{arch}_{readout}"] = {"policy": l, "top1": t1}
            print(f"{nm:12s} {readout:8s} {l:13.4f} {t1:11.4f} {trained[arch]:9.4f}   ({secs:.0f}s)",
                  flush=True)
    gap = res["trop_cls"]["policy"] - res["ctrl_cls"]["policy"]
    print(f"\ntrunk gap on an identical fresh head: {gap:+.4f} nats")
    print(f"gap between the two trained models:   {trained['trop'] - trained['ctrl']:+.4f} nats")
    json.dump(res, open(os.path.join(HERE, "probe_policy.json"), "w"), indent=1)
