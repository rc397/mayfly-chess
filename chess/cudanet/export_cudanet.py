"""Writes a MayFly checkpoint as the weight file of the hand-tuned CUDA backend.

The format is the one in cudanet_spec.h (the weight file section). Magic "MFCNW001", a uint32 header length, a JSON
header, zero padding to 256, then every tensor at a 256-aligned offset from the data start. The five big matrices of
each layer and heads_w are fp16, everything else fp32. weights.cpp reads it.

The checkpoint is a versions file (ck["cfg"] names the architecture) or train_av's ckpt_<tag>_last.pt. With no "cfg"
the architecture comes from the tensor shapes and the head count from train_av's NHEAD default, 16. The fingerprint is
mcts_cpp.fingerprint of the loaded ChessNet, so it matches what mcts_cpp checks.

Usage: python export_cudanet.py CKPT OUT.cnw [--check]
  --check   reread the written file and compare every tensor to the checkpoint (exact for f32, fp16 rounding for f16)
"""
import json
import os
import re
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("CHESS_DIR") or os.path.dirname(HERE))   # chess/, where chess_net.py is

MAGIC = b"MFCNW001"
ALIGN = 256
VERSION = 1


def spec_numbers():
    """The architecture constants of cudanet_spec.h, read from the header so they live in one place."""
    with open(os.path.join(HERE, "cudanet_spec.h"), encoding="utf-8") as f:
        text = f.read()
    out = {}
    for name in ("kD", "kLayers", "kHeads", "kFF", "kTokens", "kMoves"):
        m = re.search(r"constexpr\s+int\s+" + name + r"\s*=\s*(\d+)\s*;", text)
        if not m:
            raise SystemExit(f"cudanet_spec.h has no {name}")
        out[name] = int(m.group(1))
    return out


def align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def load_state(path):
    """The checkpoint's state dict (prefix "_orig_mod." stripped) and its architecture."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    c = ck.get("cfg") if isinstance(ck, dict) else None
    d = sd["piece.weight"].shape[1]
    layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("enc.layers."))
    arch = {
        "d": d,
        "layers": layers,
        "nhead": int(c["nhead"]) if c else int(os.environ.get("NHEAD", 16)),
        "dim_ff": sd["enc.layers.0.linear1.weight"].shape[0],
        "qhead": "qvalue.weight" in sd,
        "att": any(k.startswith("att.") for k in sd),
        "dist": sd["vdist.weight"].shape[0] if "vdist.weight" in sd else 0,
        "mlh": any(k.startswith("mleft.") for k in sd),
    }
    if c:
        # the cfg wins where it speaks, the shapes fill the rest
        arch["d"], arch["layers"] = int(c["d"]), int(c["layers"])
        arch["dim_ff"] = int(c.get("ff_mult", 2.0) * c["d"])
    return sd, arch


def build_model(sd, arch):
    """The ChessNet mcts_cpp would hold, for the fingerprint."""
    from chess_net import ChessNet
    m = ChessNet(d=arch["d"], nhead=arch["nhead"], layers=arch["layers"], dim_ff=arch["dim_ff"],
                 qhead=arch["qhead"], att=arch["att"], dist=arch["dist"], mlh=arch["mlh"])
    m.load_state_dict(sd, strict=True)
    return m.eval()


def check_arch(sd, arch, spec):
    want = {"d": spec["kD"], "layers": spec["kLayers"], "nhead": spec["kHeads"], "dim_ff": spec["kFF"]}
    bad = [f"{k} {arch[k]} (spec {v})" for k, v in want.items() if arch[k] != v]
    if not arch["qhead"]:
        bad.append("no qvalue head")
    elif sd["qvalue.weight"].shape[0] != spec["kMoves"]:
        bad.append(f"qvalue rows {sd['qvalue.weight'].shape[0]} (spec {spec['kMoves']})")
    if sd["policy.weight"].shape[0] != spec["kMoves"]:
        bad.append(f"policy rows {sd['policy.weight'].shape[0]} (spec {spec['kMoves']})")
    if bad:
        raise SystemExit("architecture differs from cudanet_spec.h: " + ", ".join(bad))


def tensors(sd, spec):
    """[(name, numpy array)] in the spec's order, with the spec's dtypes."""
    def f32(t):
        return np.ascontiguousarray(t.detach().float().cpu().numpy(), dtype="<f4")

    def f16(t):
        return np.ascontiguousarray(t.detach().float().cpu().half().numpy(), dtype="<f2")

    d = spec["kD"]
    out = [("piece", f32(sd["piece.weight"])),
           ("pos", f32(sd["pos.weight"])),
           ("cls", f32(sd["cls"].reshape(d))),
           ("cast_w", f32(sd["cast.weight"])),
           ("cast_b", f32(sd["cast.bias"])),
           ("hm", f32(sd["hm.weight"])),
           ("ep_mark", f32(sd["ep_mark"]))]
    for i in range(spec["kLayers"]):
        p = f"enc.layers.{i}."
        out += [(f"L{i}.n1_g", f32(sd[p + "norm1.weight"])),
                (f"L{i}.n1_b", f32(sd[p + "norm1.bias"])),
                (f"L{i}.qkv_w", f16(sd[p + "self_attn.in_proj_weight"])),
                (f"L{i}.qkv_b", f32(sd[p + "self_attn.in_proj_bias"])),
                (f"L{i}.out_w", f16(sd[p + "self_attn.out_proj.weight"])),
                (f"L{i}.out_b", f32(sd[p + "self_attn.out_proj.bias"])),
                (f"L{i}.n2_g", f32(sd[p + "norm2.weight"])),
                (f"L{i}.n2_b", f32(sd[p + "norm2.bias"])),
                (f"L{i}.ff1_w", f16(sd[p + "linear1.weight"])),
                (f"L{i}.ff1_b", f32(sd[p + "linear1.bias"])),
                (f"L{i}.ff2_w", f16(sd[p + "linear2.weight"])),
                (f"L{i}.ff2_b", f32(sd[p + "linear2.bias"]))]
    out += [("ln_g", f32(sd["ln.weight"])),
            ("ln_b", f32(sd["ln.bias"])),
            ("heads_w", f16(torch.cat([sd["policy.weight"], sd["qvalue.weight"]], 0))),
            ("heads_b", f32(torch.cat([sd["policy.bias"], sd["qvalue.bias"]], 0)))]
    return out


def expected_shapes(spec):
    """{name: (dtype, shape)} as cudanet_spec.h lists them."""
    d, ff, m = spec["kD"], spec["kFF"], spec["kMoves"]
    e = {"piece": ("f32", [13, d]), "pos": ("f32", [64, d]), "cls": ("f32", [d]),
         "cast_w": ("f32", [d, 4]), "cast_b": ("f32", [d]), "hm": ("f32", [102, d]), "ep_mark": ("f32", [d])}
    for i in range(spec["kLayers"]):
        e.update({f"L{i}.n1_g": ("f32", [d]), f"L{i}.n1_b": ("f32", [d]),
                  f"L{i}.qkv_w": ("f16", [3 * d, d]), f"L{i}.qkv_b": ("f32", [3 * d]),
                  f"L{i}.out_w": ("f16", [d, d]), f"L{i}.out_b": ("f32", [d]),
                  f"L{i}.n2_g": ("f32", [d]), f"L{i}.n2_b": ("f32", [d]),
                  f"L{i}.ff1_w": ("f16", [ff, d]), f"L{i}.ff1_b": ("f32", [ff]),
                  f"L{i}.ff2_w": ("f16", [d, ff]), f"L{i}.ff2_b": ("f32", [d])})
    e.update({"ln_g": ("f32", [d]), "ln_b": ("f32", [d]),
              "heads_w": ("f16", [2 * m, d]), "heads_b": ("f32", [2 * m])})
    return e


def write_cnw(path, header, arrays):
    """header without "tensors", arrays as [(name, array)]. Written to path.part, then moved into place."""
    entries, off = [], 0
    for name, a in arrays:
        off = align(off)
        entries.append({"name": name, "dtype": "f16" if a.dtype == np.float16 else "f32",
                        "shape": list(a.shape), "offset": off, "nbytes": a.nbytes})
        off += a.nbytes
    hdr = dict(header, tensors=entries)
    js = json.dumps(hdr, separators=(",", ":")).encode("ascii")
    start = align(8 + 4 + len(js))
    part = path + ".part"
    with open(part, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(js)))
        f.write(js)
        f.write(b"\0" * (start - f.tell()))
        for e, (_, a) in zip(entries, arrays):
            f.write(b"\0" * (start + e["offset"] - f.tell()))
            f.write(a.tobytes())
        f.write(b"\0" * (align(f.tell()) - f.tell()))
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)
    return hdr


def read_cnw(path):
    """(header dict, {name: numpy array}) of a weight file, checked as weights.cpp checks it."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:8] != MAGIC:
        raise ValueError(f"{path}: bad magic {raw[:8]!r}")
    (n,) = struct.unpack("<I", raw[8:12])
    if 12 + n > len(raw):
        raise ValueError(f"{path}: header runs past the end")
    hdr = json.loads(raw[12:12 + n].decode("ascii"))
    start = align(12 + n)
    out = {}
    for e in hdr["tensors"]:
        dt = {"f16": "<f2", "f32": "<f4"}[e["dtype"]]
        count = int(np.prod(e["shape"])) if e["shape"] else 1
        if e["offset"] % ALIGN or e["nbytes"] != count * np.dtype(dt).itemsize:
            raise ValueError(f"{path}: tensor {e['name']} misaligned or wrong size")
        if start + e["offset"] + e["nbytes"] > len(raw):
            raise ValueError(f"{path}: tensor {e['name']} runs past the end")
        out[e["name"]] = np.frombuffer(raw, dtype=dt, count=count, offset=start + e["offset"]).reshape(e["shape"])
    return hdr, out


def check(path, arrays, header, spec):
    """Reread the file. Every tensor must match the checkpoint and every spec tensor must be there."""
    hdr, got = read_cnw(path)
    for k, v in header.items():
        if hdr.get(k) != v:
            raise SystemExit(f"check: header {k} {hdr.get(k)!r}, wrote {v!r}")
    want = expected_shapes(spec)
    for name, (dt, shape) in want.items():
        if name not in got:
            raise SystemExit(f"check: {name} missing")
        g = got[name]
        if ("f16" if g.dtype == np.float16 else "f32") != dt or list(g.shape) != shape:
            raise SystemExit(f"check: {name} is {g.dtype} {list(g.shape)}, spec {dt} {shape}")
    for name, a in arrays:
        # bit for bit what was written, f32 the checkpoint itself and f16 its rounding (checked next)
        if got[name].dtype != a.dtype or not np.array_equal(got[name].view(np.uint8), a.view(np.uint8)):
            raise SystemExit(f"check: {name} differs from the checkpoint")
    print(f"check ok: {len(got)} tensors read back bit for bit", flush=True)


def check_rounding(sd, arrays):
    """The f16 tensors against the fp32 weights. The largest relative error must be fp16's (2^-11)."""
    src = {"heads_w": torch.cat([sd["policy.weight"], sd["qvalue.weight"]], 0)}
    names = {"qkv_w": "self_attn.in_proj_weight", "out_w": "self_attn.out_proj.weight",
             "ff1_w": "linear1.weight", "ff2_w": "linear2.weight"}
    worst = 0.0
    for name, a in arrays:
        if a.dtype != np.float16:
            continue
        if name in src:
            ref = src[name]
        else:
            i, short = name[1:].split(".")
            ref = sd[f"enc.layers.{i}.{names[short]}"]
        ref = ref.detach().float().cpu().numpy()
        err = np.abs(a.astype(np.float32) - ref)
        tiny = np.abs(ref) < 6.2e-5          # fp16 subnormals get an absolute error test instead
        rel = np.where(tiny, 0.0, err / np.maximum(np.abs(ref), 1e-30))
        if np.any(err[tiny] > 6e-8) or rel.max() > 2.0 ** -11 * 1.0001:
            raise SystemExit(f"check: {name} is not the fp16 rounding of the checkpoint")
        worst = max(worst, float(rel.max()))
    print(f"check ok: f16 within fp16 rounding (worst relative {worst:.3g})", flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        raise SystemExit(__doc__)
    ckpt, out = args
    spec = spec_numbers()
    sd, arch = load_state(ckpt)
    check_arch(sd, arch, spec)
    from mcts_cpp import fingerprint
    fp = fingerprint(build_model(sd, arch))
    arrays = tensors(sd, spec)
    want = expected_shapes(spec)
    for name, a in arrays:
        dt, shape = want[name]
        if ("f16" if a.dtype == np.float16 else "f32") != dt or list(a.shape) != shape:
            raise SystemExit(f"{name} is {a.dtype} {list(a.shape)}, spec {dt} {shape}")
    header = {"version": VERSION, "d": spec["kD"], "layers": spec["kLayers"], "heads": spec["kHeads"],
              "ff": spec["kFF"], "tokens": spec["kTokens"], "moves": spec["kMoves"],
              "fingerprint": fp, "source": os.path.abspath(ckpt)}
    write_cnw(out, header, arrays)
    print(f"wrote {out}: {os.path.getsize(out) / 1e6:.1f} MB, {len(arrays)} tensors, fingerprint {fp}", flush=True)
    if "--check" in sys.argv[1:]:
        check(out, arrays, header, spec)
        check_rounding(sd, arrays)


if __name__ == "__main__":
    main()
