"""The search's forward pass as ONNX for TensorRT, the yardstick for cudanet.

The graph takes what cudanet takes and gives what it gives (cudanet_spec.h). Inputs are ids int64 [B, 64], cast f32
[B, 4], hm int64 [B], ep int64 [B], I int64 [B, W], M bool [B, W]. Outputs are pol f32 [B, W] (raw policy logits at the
legal moves, 0 on padding), val f32 [B] and qv f32 [B, W]. Inside are the embedding, the 16 pre-norm layers, the final
layer norm on the CLS row, the policy and qvalue heads and _collect's arithmetic (export_ts.post). B and W are dynamic.

There are two flavours.
  typed  for a strongly typed engine, laid out as cudanet computes, with residual stream and layer norms in fp32, the four
         products with fp16 weights and fp16 outputs (bias in the product), GELU (exact, erf) in fp32, attention in
         fp16 (or fp32 scores with --att32), the CLS norm rounded to fp16 and the heads in fp32 on the fp16 rounded
         weights (so the logits are not rounded to fp16), the post in fp32. TensorRT runs fp16 products with fp16
         outputs on fp16 accumulating kernels (as cudanet's ACC16=2); --acc32 writes each product as fp32 on the
         fp16 values (Cast, MatMul in fp32, no constant folding, so the weights stay fp16 in the file), which asks
         for fp32 accumulation as ACC16=0 has it. Whether TensorRT folds the casts into an f16 in, f32 accumulate
         kernel or falls back to TF32 shows in build_engine.py's layer dump.
  fp32   everything fp32 with the checkpoint's own weights, for TensorRT's FP16 flag (it picks the precisions) or an
         fp32 engine.

Usage: python export_onnx.py CKPT OUT.onnx [--mode typed|fp32] [--att32] [--acc32] [--check]
  --check   compare this module (in fp32) with ChessNet and export_ts.post on the CPU first
Runs on the CPU, no card needed. torch.onnx.export wants the onnx package, so run it in the TensorRT venv
(setup_venv.bat puts torch 2.6 for the CPU and onnx there).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CHESS = os.path.dirname(HERE)
for _d in (os.path.join(CHESS, "cudanet"), os.path.join(CHESS, "cpp"), CHESS):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import export_cudanet  # noqa: E402
import export_ts  # noqa: E402

N_MOVES = 4162
EPS = 1e-5


class Forward(nn.Module):
    """ChessNet's search pass from a state dict, written out op by op so the ONNX graph is plain."""

    def __init__(self, sd, arch, mode="typed", att32=False, acc32=False):
        super().__init__()
        self.mode, self.att32, self.acc32 = mode, att32, acc32 and mode == "typed"
        self.d, self.H, self.L = arch["d"], arch["nhead"], arch["layers"]
        self.dh = self.d // self.H
        wt = torch.float16 if mode == "typed" else torch.float32

        def buf(name, t, dt=torch.float32):
            self.register_buffer(name, t.detach().to(dt).contiguous().clone())

        buf("piece", sd["piece.weight"])
        buf("pos", sd["pos.weight"])
        buf("cls", sd["cls"].reshape(self.d))
        buf("cast_wt", sd["cast.weight"].t())          # [4, d]
        buf("cast_b", sd["cast.bias"])
        buf("hm_emb", sd["hm.weight"])
        buf("ep_mark", sd["ep_mark"])
        buf("sq1", torch.arange(1, 65), torch.long)    # square index plus one, as ep counts
        for i in range(self.L):
            p = f"enc.layers.{i}."
            buf(f"n1_g{i}", sd[p + "norm1.weight"])
            buf(f"n1_b{i}", sd[p + "norm1.bias"])
            buf(f"qkv_w{i}", sd[p + "self_attn.in_proj_weight"].t(), wt)   # [d, 3d]
            buf(f"qkv_b{i}", sd[p + "self_attn.in_proj_bias"], wt)
            buf(f"out_w{i}", sd[p + "self_attn.out_proj.weight"].t(), wt)
            buf(f"out_b{i}", sd[p + "self_attn.out_proj.bias"], wt)
            buf(f"n2_g{i}", sd[p + "norm2.weight"])
            buf(f"n2_b{i}", sd[p + "norm2.bias"])
            buf(f"ff1_w{i}", sd[p + "linear1.weight"].t(), wt)
            buf(f"ff1_b{i}", sd[p + "linear1.bias"], wt)
            buf(f"ff2_w{i}", sd[p + "linear2.weight"].t(), wt)
            buf(f"ff2_b{i}", sd[p + "linear2.bias"], wt)
        buf("ln_g", sd["ln.weight"])
        buf("ln_b", sd["ln.bias"])
        heads_w = torch.cat([sd["policy.weight"], sd["qvalue.weight"]], 0).t()   # [d, 2*moves]
        if mode == "typed":
            heads_w = heads_w.half()       # cudanet's fp16 weights, kept as fp32 numbers
        buf("heads_w", heads_w)
        buf("heads_b", torch.cat([sd["policy.bias"], sd["qvalue.bias"]], 0))

    def _attention(self, qkv):
        T, H, dh = 65, self.H, self.dh
        qkv = qkv.reshape(-1, T, 3, H, dh).permute(2, 0, 3, 1, 4)     # [3, B, H, T, dh]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * (1.0 / dh ** 0.5)                                      # 0.125, exact in fp16
        if self.att32:
            s = torch.matmul(q.float(), k.float().transpose(-1, -2))
            p = torch.softmax(s, -1).to(v.dtype)
        else:
            p = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), -1)
        a = torch.matmul(p, v)                                         # [B, H, T, dh]
        return a.transpose(1, 2).reshape(-1, T, self.d)

    def _lin(self, x, w, b):
        if self.acc32:
            return torch.matmul(x.float(), w.float()) + b.float()
        return torch.matmul(x, w) + b

    def forward(self, ids, cast, hm, ep, I, M):
        d = self.d
        wt = torch.float16 if self.mode == "typed" else torch.float32
        mark = (self.sq1[None, :] == ep[:, None]).float()              # [B, 64], none when ep is 0
        sq = self.piece[ids] + self.pos[None] + mark[:, :, None] * self.ep_mark
        cls = self.cls + torch.matmul(cast, self.cast_wt) + self.cast_b + self.hm_emb[hm]
        x = torch.cat([cls[:, None, :], sq], 1)                        # [B, 65, d] fp32
        for i in range(self.L):
            g = lambda n: getattr(self, f"{n}{i}")  # noqa: E731
            h = F.layer_norm(x, (d,), g("n1_g"), g("n1_b"), EPS).to(wt)
            a = self._attention(self._lin(h, g("qkv_w"), g("qkv_b")).to(wt))
            x = x + self._lin(a, g("out_w"), g("out_b")).float()
            h = F.layer_norm(x, (d,), g("n2_g"), g("n2_b"), EPS).to(wt)
            f = F.gelu(self._lin(h, g("ff1_w"), g("ff1_b")).float()).to(wt)
            x = x + self._lin(f, g("ff2_w"), g("ff2_b")).float()
        c = F.layer_norm(x[:, 0], (d,), self.ln_g, self.ln_b, EPS)
        if self.mode == "typed":
            c = c.half().float()           # cudanet's cls buffer is fp16
        logits = torch.matmul(c, self.heads_w) + self.heads_b         # [B, 2*moves] fp32
        # _collect's sigmoid on q, softmax of the policy at the legal moves, val and qv
        pl, q = logits[:, :N_MOVES], torch.sigmoid(logits[:, N_MOVES:])
        lg = torch.gather(pl, 1, I)
        w = torch.softmax(torch.where(M, lg, torch.full_like(lg, -1e30)), -1)
        qg = torch.gather(q, 1, I)
        val = (w * qg).sum(1) * 2.0 - 1.0
        pol = torch.where(M, lg, torch.zeros_like(lg))
        return pol, val, qg * 2.0 - 1.0


def example(fens):
    """One batch as torch tensors (ids, cast, hm, ep, I, M) and the numpy batch."""
    b = export_ts.make_batches(fens, [len(fens)])[0]
    t = (torch.from_numpy(b["ids"]), torch.from_numpy(b["cast"]), torch.from_numpy(b["hm"]),
         torch.from_numpy(b["ep"]), torch.from_numpy(b["I"]), torch.from_numpy(b["M"]))
    return t, b


def check(sd, arch, ckpt):
    """This module in fp32 against ChessNet and export_ts.post on a few positions (CPU)."""
    import test_cudanet
    fens = test_cudanet.positions(48)[:48]
    model = export_ts.load_v5(ckpt, "cpu")
    t, b = example(fens)
    ref = test_cudanet.reference(model, [b])[0]
    fw = Forward(sd, arch, "fp32").eval()
    with torch.no_grad():
        pol, val, qv = (a.numpy() for a in fw(*t))
    m = test_cudanet.metrics([ref], [(pol, val, qv)], [b])
    test_cudanet.print_metrics("export_onnx Forward (fp32, CPU) vs ChessNet", m)
    if m["pol"]["max"] > 1e-3 or m["val"]["max"] > 1e-4 or m["qv"]["max"] > 1e-4:
        raise SystemExit("check failed: the rewritten forward is not ChessNet's")


def main():
    ap = argparse.ArgumentParser(description="the search's forward pass as ONNX for TensorRT")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=("typed", "fp32"), default="typed")
    ap.add_argument("--att32", action="store_true", help="typed: attention scores and softmax in fp32")
    ap.add_argument("--acc32", action="store_true", help="typed: the four products in fp32 on fp16 values")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sd, arch = export_cudanet.load_state(a.ckpt)
    export_cudanet.check_arch(sd, arch, export_cudanet.spec_numbers())
    if a.check:
        check(sd, arch, a.ckpt)
    fw = Forward(sd, arch, a.mode, a.att32, a.acc32).eval()
    import test_cudanet
    t, _ = example(test_cudanet.FIXED[:2])
    dyn = {"ids": {0: "B"}, "cast": {0: "B"}, "hm": {0: "B"}, "ep": {0: "B"}, "I": {0: "B", 1: "W"},
           "M": {0: "B", 1: "W"}, "pol": {0: "B", 1: "W"}, "val": {0: "B"}, "qv": {0: "B", 1: "W"}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(fw, t, a.out, opset_version=a.opset, input_names=["ids", "cast", "hm", "ep", "I", "M"],
                          output_names=["pol", "val", "qv"], dynamic_axes=dyn, do_constant_folding=not fw.acc32,
                          dynamo=False)
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB, mode {a.mode}"
          f"{' att32' if a.att32 else ''}{' acc32' if fw.acc32 else ''}, opset {a.opset}", flush=True)


if __name__ == "__main__":
    main()
