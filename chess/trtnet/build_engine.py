"""Builds a TensorRT engine from export_onnx.py's graph. Runs in the TensorRT venv (see setup_venv.bat), on the card.

  python build_engine.py IN.onnx OUT.engine [--mode typed|fp16|fp32] [--workspace-mb 512] [--opt-level 3]
                         [--batch 48] [--wide 256] [--timing-cache X.cache]

--mode typed   STRONGLY_TYPED network, the precisions are the ONNX graph's own (export_onnx.py --mode typed).
       fp16    weakly typed with the FP16 flag on the fp32 graph, TensorRT picks each layer's precision.
       fp32    weakly typed, TF32 off, a full precision engine to check the graph itself.
TensorRT 10 has no switch for fp16 accumulation in its products (no builder flag, no layer attribute). The tactic
it picks decides, and the layer dump written beside the engine (OUT.layers.json, ProfilingVerbosity.DETAILED) names
each kernel, so the accumulation type can be read off the tactic names.
One optimisation profile, B from 1 to --batch (opt at --batch) and W (the I and M width) from 1 to --wide (opt at
--wide, trtnet_py pads every batch to --wide so one CUDA graph a batch size serves every width).
"""
import argparse
import json
import os
import re
import time

import tensorrt as trt


def build(onnx, out, mode="typed", workspace_mb=512, opt_level=3, batch=48, wide=256, timing_cache=None, quant=()):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if mode == "typed" else 0
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(onnx):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise SystemExit("ONNX parse failed:\n" + "\n".join(errs))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    cfg.builder_optimization_level = opt_level
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if mode == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
        for q in quant:
            cfg.set_flag({"int8": trt.BuilderFlag.INT8, "fp8": trt.BuilderFlag.FP8}[q])
    elif mode == "fp32":
        cfg.clear_flag(trt.BuilderFlag.TF32)
    prof = builder.create_optimization_profile()
    shapes = {"ids": lambda b, w: (b, 64), "cast": lambda b, w: (b, 4), "hm": lambda b, w: (b,),
              "ep": lambda b, w: (b,), "I": lambda b, w: (b, w), "M": lambda b, w: (b, w)}
    for i in range(net.num_inputs):
        name = net.get_input(i).name
        f = shapes[name]
        prof.set_shape(name, f(1, 1), f(batch, wide), f(batch, wide))
    cfg.add_optimization_profile(prof)
    cache = None
    if timing_cache:
        blob = open(timing_cache, "rb").read() if os.path.exists(timing_cache) else b""
        cache = cfg.create_timing_cache(blob)
        cfg.set_timing_cache(cache, ignore_mismatch=False)
    t0 = time.perf_counter()
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        raise SystemExit("engine build failed (see TensorRT's messages above; a workspace cap too small for any "
                         "tactic shows here)")
    secs = time.perf_counter() - t0
    with open(out, "wb") as f:
        f.write(plan)
    if cache is not None:
        with open(timing_cache, "wb") as f:
            f.write(cache.serialize())
    print(f"wrote {out}: {os.path.getsize(out) / 1e6:.1f} MB, mode {mode}, built in {secs:.0f} s", flush=True)
    return dump_layers(out, logger)


def dump_layers(engine_path, logger=None):
    """The engine's layers and kernels as JSON beside it, and a short summary of what TensorRT chose."""
    runtime = trt.Runtime(logger or trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
    insp = engine.create_engine_inspector()
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    path = os.path.splitext(engine_path)[0] + ".layers.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    layers = info.get("Layers", [])
    tactics = [str(l.get("TacticName", "")) if isinstance(l, dict) else "" for l in layers]
    names = [str(l.get("Name", l)) if isinstance(l, dict) else str(l) for l in layers]
    kinds = {}
    for l in layers:
        k = l.get("LayerType", "?") if isinstance(l, dict) else "?"
        kinds[k] = kinds.get(k, 0) + 1
    # gemm kernel names carry the types as <A><B>_<C><Acc>, e.g. f16f16_f16f32 (fp32 accumulation)
    acc = {}
    for t in tactics:
        for m in re.findall(r"(f16f16_f16f(?:16|32)|f16f16_f32f32|h16816|s16816|h1688|s1688)", t):
            acc[m] = acc.get(m, 0) + 1
    mha = sum(1 for n, t in zip(names, tactics) if re.search(r"mha|fmha|flash|attention", n + t, re.I))
    summary = {"layers": len(layers), "layer_types": kinds, "gemm_types_in_tactic_names": acc,
               "attention_like_layers": mha, "device_memory_mb": engine.device_memory_size_v2 / 2 ** 20}
    print(f"layers {summary['layers']}, types {kinds}, gemm tactic types {acc}, attention-like layers {mha}, "
          f"activation memory {summary['device_memory_mb']:.0f} MB; full dump in {path}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description="a TensorRT engine from export_onnx.py's graph")
    ap.add_argument("onnx")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=("typed", "fp16", "fp32"), default="typed")
    ap.add_argument("--workspace-mb", type=int, default=512, help="builder workspace cap in MB")
    ap.add_argument("--opt-level", type=int, default=3, help="builder optimisation level 0..5")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--wide", type=int, default=256)
    ap.add_argument("--timing-cache")
    ap.add_argument("--quant", default="", help="fp16 mode on a Q/DQ graph (ModelOpt): int8 and/or fp8, comma "
                                                "separated, sets those builder flags too (typed needs none)")
    ap.add_argument("--dump-only", action="store_true", help="only write the layer dump of an existing engine")
    a = ap.parse_args()
    if a.dump_only:
        dump_layers(a.out)
        return
    print(f"TensorRT {trt.__version__}", flush=True)
    build(a.onnx, a.out, a.mode, a.workspace_mb, a.opt_level, a.batch, a.wide, a.timing_cache,
          [q for q in a.quant.split(",") if q])


if __name__ == "__main__":
    main()
