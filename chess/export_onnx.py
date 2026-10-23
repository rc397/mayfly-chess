"""Export a MayFly version to ONNX, quantise it to int8, and check the quantised model plays the same.

The int8 model is small enough for a browser to download and run with no server. A model whose loss
is unchanged but whose favourite move differs is a different player, so both files are run on the
same real positions and compared with PyTorch on top-move agreement and value errors.

    mayfly_v1.onnx        float32, about 573 MB, the reference
    mayfly_v1.int8.onnx   quantised, about a quarter of that

The graph takes the network's four inputs (ids, castling, halfmove clock, en passant square) and
returns policy, value, action values and the auxiliary heads, with a dynamic batch axis.

    python chess/export_onnx.py [mayfly_v1.pt]
    env OUT (output folder), CHECK=2000 (positions to compare on),
        EXTRA_SITE (an extra site-packages folder to search first, for example one holding onnx)
"""
import os, sys, time, glob
# onnx's nested test data can exceed Windows path limits in some site-packages folders, so it may live elsewhere
if os.environ.get("EXTRA_SITE"):
    sys.path.insert(0, os.environ["EXTRA_SITE"])
import numpy as np
import torch
import torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet
import paths

E = os.environ.get
OUT = E("OUT", paths.under("onnx"))
CHECK = int(E("CHECK", 2000))
VERSION_DIR = E("VERSION_DIR", paths.under("versions"))
SHARD_DIR = paths.under("shards")


class Wrapped(nn.Module):
    """Return the outputs as a tuple in a fixed order, since ONNX wants tensors, not a dictionary."""

    KEYS = ("policy", "value", "qvalue", "material", "king", "pawns", "mob", "bishop")

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, ids, cast, hm, ep):
        o = self.net(ids, cast, hm, ep)
        return tuple(o[k] for k in self.KEYS if o.get(k) is not None)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    c = ck.get("cfg", {})
    m = ChessNet(d=c.get("d", 1024), nhead=c.get("nhead", 16), layers=c.get("layers", 16),
                 dim_ff=int(c.get("ff_mult", 2.0) * c.get("d", 1024)),
                 qhead=bool(c.get("qhead", "qvalue.weight" in ck["model"])))
    m.load_state_dict(ck["model"], strict=False)
    return m.to(device).eval(), ck


def sample_inputs(n, device="cpu"):
    """Real positions from the corpus, because quantisation error depends on the inputs."""
    from pack_av import open_shard, unpack_ids
    paths = sorted(glob.glob(os.path.join(E("SHARDS", SHARD_DIR), "av_*.bin")))

    rng = np.random.default_rng(0)
    rows = []
    for p in rng.choice(paths, min(8, len(paths)), replace=False):
        mm = open_shard(p)
        if len(mm):
            rows.append(mm[np.sort(rng.choice(len(mm), min(n // 8 + 1, len(mm)), replace=False))])
    r = np.concatenate(rows)[:n]
    ids = unpack_ids(r["ids4"]).astype(np.int64)
    c = r["cast"]
    cast = np.stack([(c >> i) & 1 for i in range(4)], 1).astype(np.float32)
    hm = np.where(r["flags"] & 2, np.minimum(r["hm"], 100), 101).astype(np.int64)
    ep = np.where(r["flags"] & 4, r["ep"], 0).astype(np.int64)
    return ids, cast, hm, ep


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "mayfly_v1.pt"
    path = src if os.path.isabs(src) else os.path.join(VERSION_DIR, src)
    os.makedirs(OUT, exist_ok=True)
    name = os.path.basename(path).replace(".pt", "")
    f32 = os.path.join(OUT, f"{name}.onnx")
    i8 = os.path.join(OUT, f"{name}.int8.onnx")

    model, ck = load(path)
    print(f"{name}: {model.num_params()/1e6:.1f}M params, iteration {ck.get('iter', 0):,}")

    ids, cast, hm, ep = sample_inputs(CHECK)
    ex = (torch.from_numpy(ids[:4]), torch.from_numpy(cast[:4]),
          torch.from_numpy(hm[:4]), torch.from_numpy(ep[:4]))
    names = [k for k in Wrapped.KEYS if k != "qvalue" or model.qvalue is not None]

    print("exporting to onnx", flush=True)
    t0 = time.time()
    torch.onnx.export(
        Wrapped(model), ex, f32,
        input_names=["ids", "cast", "hm", "ep"], output_names=names,
        dynamic_axes={k: {0: "batch"} for k in ["ids", "cast", "hm", "ep"] + names},
        opset_version=17, dynamo=False)
    print(f"  {os.path.getsize(f32)/1e6:,.0f} MB in {time.time()-t0:.0f}s")

    print("quantising to int8", flush=True)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    t0 = time.time()
    quantize_dynamic(f32, i8, weight_type=QuantType.QInt8)
    print(f"  {os.path.getsize(i8)/1e6:,.0f} MB in {time.time()-t0:.0f}s, "
          f"{os.path.getsize(f32)/os.path.getsize(i8):.1f}x smaller")

    print(f"\ncomparing the two on {CHECK:,} real positions")
    import onnxruntime as ort
    with torch.inference_mode():
        ref = model(torch.from_numpy(ids), torch.from_numpy(cast),
                    torch.from_numpy(hm), torch.from_numpy(ep))
    ref_pol = ref["policy"].numpy()
    ref_q = torch.sigmoid(ref["qvalue"]).numpy() if model.qvalue is not None else None
    ref_val = torch.sigmoid(ref["value"]).numpy()

    for label, f in (("float32", f32), ("int8", i8)):
        s = ort.InferenceSession(f, providers=["CPUExecutionProvider"])
        t0 = time.time()
        out = s.run(None, {"ids": ids, "cast": cast, "hm": hm, "ep": ep})
        el = time.time() - t0
        got = dict(zip(names, out))
        pol = got["policy"]
        agree = float((pol.argmax(1) == ref_pol.argmax(1)).mean())
        line = (f"  {label:<8} {CHECK/el:>7,.0f} positions/s on cpu   "
                f"policy argmax matches pytorch {100*agree:5.1f}%")
        if ref_q is not None:
            q = 1.0 / (1.0 + np.exp(-got["qvalue"]))
            line += f"   action value mean error {np.abs(q - ref_q).mean():.4f}"
        v = 1.0 / (1.0 + np.exp(-got["value"]))
        line += f"   value error {np.abs(v - ref_val).mean():.4f}"
        print(line, flush=True)
    print(f"\nwritten to {OUT}")
