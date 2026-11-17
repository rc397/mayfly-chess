"""export_net -- a released version's network as one ONNX file for the volunteer app. ONNX Runtime with DirectML runs
it on any Windows graphics card (NVIDIA, AMD or Intel), so a volunteer needs no PyTorch.

The graph is selfplay_remote's torch server in one piece: the trunk and the two heads a search reads, then the policy
at the legal moves, every legal move's q and the policy-weighted value. So the app's server answers a worker exactly
as the torch server does. Inputs are _prepare's arrays padded to W_FIX moves, with the batch size free.

  python chess/volunteer/export_net.py data/versions/mayfly_v6.1.pt OUT.onnx [fp16|fp32]
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from selfplay_remote import W_FIX, _torch_net, load_weights  # noqa: E402

NAMES = ["ids", "cast", "hm", "ep", "I", "Ip", "M"]


class Served(torch.nn.Module):
    """The trunk plus what selfplay_remote's torch server computes after it."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, ids, cast, hm, ep, I, Ip, M):
        pol, qv = self.net(ids, cast, hm, ep)
        pg = torch.gather(pol.float(), 1, Ip)
        qg = torch.gather(torch.sigmoid(qv.float()), 1, I)
        w = torch.softmax(pg.masked_fill(~M, float("-inf")), -1)
        val = (w * qg).sum(1) * 2.0 - 1.0
        return pg, qg * 2.0 - 1.0, val


def example(b, half, dev):
    rng = np.random.default_rng(0)
    ids = torch.from_numpy(rng.integers(0, 13, (b, 64)))
    cast = torch.zeros(b, 4, dtype=torch.float16 if half else torch.float32)
    z = torch.zeros(b, dtype=torch.long)
    I = torch.from_numpy(rng.integers(0, 4000, (b, W_FIX)))
    Ip = torch.from_numpy(rng.integers(0, 1800, (b, W_FIX)))
    M = torch.ones(b, W_FIX, dtype=torch.bool)
    return tuple(a.to(dev) for a in (ids, cast, z, z.clone(), I, Ip, M))


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    half = (sys.argv[3] if len(sys.argv) > 3 else "fp16") == "fp16"
    cfg, params = load_weights(ckpt=ckpt)
    dev = torch.device("cuda" if half else "cpu")      # a half-precision trace runs on the card
    model = Served(_torch_net(cfg, params)).eval().to(dev)
    if half:
        model.net.half()
    torch.backends.mha.set_fastpath_enabled(False)    # the fused encoder layer has no ONNX form; same maths unfused
    with torch.no_grad():
        torch.onnx.export(model, example(8, half, dev), out, input_names=NAMES, output_names=["pg", "q", "val"],
                          dynamic_axes={n: {0: "batch"} for n in NAMES + ["pg", "q", "val"]},
                          opset_version=17, dynamo=False)
    print(f"{out}: {os.path.getsize(out) / 1e6:.0f} MB, {'float16' if half else 'float32'} weights, "
          f"{int(cfg['layers'])} layers of {int(cfg['d'])}")


if __name__ == "__main__":
    main()
