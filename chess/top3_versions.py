r"""top3_versions scores top-1 and top-3 move agreement for every frozen version and the live checkpoint.

Top-1 shows peak strength and top-3 shows general play. Every model is scored on the validation positions
the trainer uses (the cached val_policy_<TAG>.npy, legal moves only, same loader and seed), so versions
trained before the trainer logged top-3 still get a point on the chart.

  python top3_versions.py              # writes top3_versions.json in CKPT_DIR
"""
import glob
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from chess_net import ChessNet                       # noqa: E402
from data_packed import PackedLoader                 # noqa: E402
import paths                                         # noqa: E402

E = os.environ.get
SHARDS = E("SHARDS", paths.under("shards"))
CKPT_DIR = E("CKPT_DIR", paths.under("ckpt"))
VERSIONS = E("VERSION_DIR", paths.under("versions"))
TAG = E("TAG", "mayfly")
BATCH = int(E("BATCH", 256))
BATCHES = int(E("BATCHES", 80))                      # 20,480 positions, about +-0.35 points on top-1
OUT = os.path.join(CKPT_DIR, "top3_versions.json")


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    c = ck.get("cfg", {})
    m = ChessNet(d=c.get("d", 1024), nhead=c.get("nhead", 16), layers=c.get("layers", 16),
                 dim_ff=int(c.get("ff_mult", 2.0) * c.get("d", 1024)),
                 qhead=bool(c.get("qhead", "qvalue.weight" in ck["model"])),
                 att=bool(c.get("att", "att.q.weight" in ck["model"])),
                 dist=int(c.get("dist", 64 if "vdist.weight" in ck["model"] else 0)),
                 mlh=bool(c.get("mlh", "mleft.0.weight" in ck["model"]))).to(dev).eval()
    m.load_state_dict(ck["model"], strict=False)
    return m, ck.get("hours"), ck.get("version"), ck.get("iter")


@torch.no_grad()
def score(model, dev):
    pol = PackedLoader(SHARDS, batch=BATCH, split="val", device=dev, seed=1, sources=["lz", "lc"],
                       refresh=10 ** 9, legal=True,
                       val_cache=os.path.join(CKPT_DIR, f"val_policy_{TAG}.npy"))
    tot, n = {}, 0
    for _ in range(BATCHES):
        p = pol.next()
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev != "cpu")):
            op = model(p[0], p[1], p[2], p[3])
        want = p[4].argmax(-1, keepdim=True)
        for head, key in (("policy", ""), ("policy_att", "att_")):
            if head not in op:
                continue
            ml = op[head].float().masked_fill(~p[12], -1e9)
            top = ml.topk(3, -1).indices
            tot[key + "top1_legal"] = tot.get(key + "top1_legal", 0.0) + float((top[:, :1] == want).any(-1).float().sum())
            tot[key + "top3_legal"] = tot.get(key + "top3_legal", 0.0) + float((top == want).any(-1).float().sum())
        n += int(want.shape[0])
    return {k: v / n for k, v in tot.items()}, n


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(glob.glob(os.path.join(VERSIONS, f"{TAG}_v*.pt")),
                   key=lambda p: int(os.path.basename(p).split("_v")[1].split(".")[0]))
    paths.append(os.path.join(CKPT_DIR, f"ckpt_{TAG}_last.pt"))
    out = {}
    for path in paths:
        model, hours, version, it = load(path, dev)
        s, n = score(model, dev)
        name = f"v{version}" if version else "live"
        out[name] = {"hours": hours, "iter": it, "positions": n, **{k: round(v, 4) for k, v in s.items()}}
        print(f"{name:5} {hours or 0:6.1f}h  " + "  ".join(f"{k} {100 * v:5.1f}%" for k, v in s.items()), flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
