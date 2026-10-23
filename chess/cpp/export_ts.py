"""MayFly as TorchScript for the C++ NetBackend, plus the Python reference the C++ is checked against.

The traced module is ChessNet.forward returning (policy, qvalue) only, which is all the v5 search reads
(value_from_q, policy_head cls, q_head cls, mlh off). There are two flavours.

  cpu   fp32, traced with grad on so nn.TransformerEncoderLayer takes its ordinary (slow) path. Used to
        check the export and the C++ code path without a card.
  cuda  what _replay_autocast_fast runs (chess/mcts.py). Every nn.Linear weight and bias and every
        MultiheadAttention in_proj weight and bias is replaced by its .to(bfloat16) copy (_bf16_copies)
        and traced under torch.autocast("cuda", bfloat16, cache_enabled=False) with grad off, so the
        trace records exactly the casts and kernels the Python graph runs. Run it without autocast.

Usage: python export_ts.py cpu|cuda OUT.ts [CKPT]     (CHESS_DIR = folder with chess_net.py)
"""
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("CHESS_DIR") or os.path.dirname(HERE))   # chess/, where chess_net.py is
import paths  # noqa: E402
CKPT = os.environ.get("CKPT", os.path.join(paths.VERSIONS, "mayfly_v5.pt"))
N_MOVES = 4162


def load_v5(path=CKPT, device="cpu"):
    from chess_net import ChessNet
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = ChessNet(d=c["d"], nhead=c["nhead"], layers=c["layers"], dim_ff=int(c.get("ff_mult", 2.0) * c["d"]),
                 qhead=bool(c["qhead"]), att=bool(c["att"]), dist=int(c["dist"]), mlh=bool(c["mlh"]))
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}, strict=True)
    return m.to(device).eval()


def bf16_copies(model):
    """mcts._bf16_copies in effect, the weights autocast casts for a matmul, each .to(bfloat16)."""
    out = {}
    for mname, mod in model.named_modules():
        names = (("weight", "bias") if isinstance(mod, nn.Linear) else
                 ("in_proj_weight", "in_proj_bias") if isinstance(mod, nn.MultiheadAttention) else ())
        for pn in names:
            p = getattr(mod, pn, None)
            if p is not None:
                out[f"{mname}.{pn}" if mname else pn] = p.detach().to(torch.bfloat16)
    return out


def put_bf16(model):
    """The bf16 copies put in the module (as functional_call puts them for the duration of a call)."""
    w = bf16_copies(model)
    for full, t in w.items():
        mname, pn = full.rsplit(".", 1) if "." in full else ("", full)
        mod = model.get_submodule(mname)
        setattr(mod, pn, nn.Parameter(t, requires_grad=False))
    return model


class Export(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, ids, cast, hm, ep):
        out = self.net(ids, cast, hm, ep)
        return out["policy"], out["qvalue"]


def encode(fens):
    """What _prepare builds for these positions, ids, cast, hm, ep and the legal moves' policy rows I."""
    import chess
    from encode import board_to_ids, clock_and_ep
    from moves import move_to_idx
    ids, cast, hm, ep, I = [], [], [], [], []
    for f in fens:
        b = chess.Board(f)
        i, c, flip = board_to_ids(b)
        h, e = clock_and_ep(b, flip)
        ids.append(i); cast.append(c); hm.append(h); ep.append(e)
        I.append([move_to_idx(m, flip) for m in b.legal_moves])
    return (np.array(ids, np.int64), np.array(cast, np.float32), np.array(hm, np.int64),
            np.array(ep, np.int64), I)


def test_fens(path=None):
    import json
    path = path or os.environ.get("POSITIONS") or os.path.join(HERE, "positions.json")
    import chess
    out = []
    for p in json.load(open(path)):
        b = chess.Board(p["fen"])
        if any(True for _ in b.legal_moves):
            out.append(p["fen"])
    return out


def make_batches(fens, sizes):
    """Consecutive slices of the positions, one batch per size, each with I/M padded to its widest row."""
    batches, k = [], 0
    for B in sizes:
        sel = [fens[(k + j) % len(fens)] for j in range(B)]
        k += B
        ids, cast, hm, ep, I = encode(sel)
        widths = [len(r) for r in I]
        wide = max(widths)
        Ia = np.zeros((B, wide), np.int64)
        Ma = np.zeros((B, wide), bool)
        for r, row in enumerate(I):
            Ia[r, :len(row)] = row
            Ma[r, :len(row)] = True
        batches.append(dict(ids=ids, cast=cast, hm=hm, ep=ep, I=Ia, M=Ma, widths=np.array(widths, np.int32)))
    return batches


def write_batches(path, batches):
    """The C++ test driver's input. int32 n, then per batch int32 B and wide, ids i64, cast f32, hm i64,
    ep i64, I i64, M u8 and widths i32."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(batches)))
        for b in batches:
            B, wide = b["I"].shape
            f.write(struct.pack("<ii", B, wide))
            for k, dt in (("ids", np.int64), ("cast", np.float32), ("hm", np.int64), ("ep", np.int64),
                          ("I", np.int64), ("M", np.uint8), ("widths", np.int32)):
                f.write(np.ascontiguousarray(b[k], dtype=dt).tobytes())


def read_results(path, batches):
    """The C++ driver's output, per batch policy f32 (B, 4162), val f32 (B) and qv f32 (B, wide)."""
    raw = open(path, "rb").read()
    out, o = [], 0
    for b in batches:
        B, wide = b["I"].shape
        r = {}
        for k, n, shape in (("policy", B * N_MOVES, (B, N_MOVES)), ("val", B, (B,)), ("qv", B * wide, (B, wide))):
            r[k] = np.frombuffer(raw, np.float32, n, o).reshape(shape)
            o += 4 * n
        out.append(r)
    assert o == len(raw), (o, len(raw))
    return out


def post(pol, qval, b, device):
    """_dispatch and _collect after the forward, with value_from_q and castle_prior king (Ip is I)."""
    It = torch.from_numpy(b["I"]).to(device)
    Mt = torch.from_numpy(b["M"]).to(device)
    q = torch.sigmoid(qval.float())
    pl = pol.float()
    w = torch.softmax(torch.gather(pl, 1, It).masked_fill(~Mt, float("-inf")), -1)
    qg = torch.gather(q, 1, It)
    val = (w * qg).sum(1) * 2.0 - 1.0
    return dict(policy=pl.cpu().numpy(), val=val.cpu().numpy(), qv=(qg * 2.0 - 1.0).cpu().numpy())


def inputs(b, device):
    L = lambda a: torch.tensor(a, dtype=torch.long, device=device)  # noqa: E731
    return (torch.tensor(b["ids"], dtype=torch.long, device=device), torch.tensor(b["cast"], device=device),
            L(b["hm"]), L(b["ep"]))


def compare(ref, got, widths=None):
    """Largest absolute differences and how many values are bit-identical, per output."""
    s = {}
    for k in ("policy", "val", "qv"):
        a = np.concatenate([r[k].ravel() for r in ref])
        g = np.concatenate([r[k].ravel() for r in got])
        if k == "qv":   # padding columns are garbage in both, so compare legal columns only
            a = np.concatenate([r[k][i, :w] for r, ws in zip(ref, widths) for i, w in enumerate(ws)])
            g = np.concatenate([r[k][i, :w] for r, ws in zip(got, widths) for i, w in enumerate(ws)])
        d = np.abs(a.astype(np.float64) - g.astype(np.float64))
        s[k] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()), exact=float((a == g).mean()), n=int(a.size))
    return s


def trace(flavour, out_path, ckpt=CKPT, B=48):
    device = "cuda" if flavour == "cuda" else "cpu"
    net = load_v5(ckpt, device)
    if flavour == "cuda":
        put_bf16(net)
    exp = Export(net).eval()
    fens = test_fens()
    ex = inputs(make_batches(fens, [B])[0], device)
    if flavour == "cuda":
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
            ts = torch.jit.trace(exp, ex, check_trace=False)
    else:
        for p in exp.parameters():
            p.requires_grad_(True)
        ts = torch.jit.trace(exp, ex, check_trace=False)
    ts.save(out_path)
    kinds = {}
    for n in ts.inlined_graph.nodes():
        kinds[n.kind()] = kinds.get(n.kind(), 0) + 1
    return kinds


if __name__ == "__main__":
    flav, outp = sys.argv[1], sys.argv[2]
    k = trace(flav, outp, sys.argv[3] if len(sys.argv) > 3 else CKPT)
    print({a: b for a, b in sorted(k.items()) if not a.startswith("prim::")})
