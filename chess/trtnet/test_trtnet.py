"""TensorRT engines of the search's forward pass measured exactly as test_cudanet.py measures cudanet, with the same
positions, batches, float32 reference, metrics and pass lines (imported from it), and the same timing loop. Runs in
the TensorRT venv.

  python test_trtnet.py numerics --engines A.engine[,B.engine] --ckpt CKPT [--n 256] [--json out.json]
  python test_trtnet.py speed --engines A.engine[,B.engine] [--sizes 1,8,16,32,48] [--cnw X.cnw --acc16 0,1,2]

numerics  test_cudanet.positions(N), cut into test_cudanet's mixed batch sizes. The reference is ChessNet in float32
          on the CPU through export_ts.post (test_cudanet.reference). Each engine runs every batch twice (the second
          time on the captured graphs, which must give the same bits), then test_cudanet.metrics and verdict judge it.
speed     evaluations a second at each size with test_cudanet.timed (the same 4 batches a size, warm-up, reps).
          --cnw adds cudanet itself, once per --acc16 setting, each in its own process (test_cudanet.py speed
          --paths cudanet with MAYFLY_CUDANET_ACC16 set), so the table compares like with like.
--no-graph  enqueueV3 at each call instead of the CUDA graphs (and with --no-pad at each call's own width).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHESS = os.path.dirname(HERE)
CUDANET = os.path.join(CHESS, "cudanet")
for _d in (HERE, CUDANET, os.path.join(CHESS, "cpp"), CHESS):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import export_ts  # noqa: E402
import test_cudanet  # noqa: E402
from trtnet_py import TrtNet, mem_info  # noqa: E402


def run(net, b):
    return net.evaluate(b["ids"], b["cast"], b["hm"], b["ep"], b["I"], b["M"])


def numerics(args, fens):
    sizes = test_cudanet.batch_sizes(len(fens))
    batches = export_ts.make_batches(fens, sizes)
    print(f"numerics: {len(fens)} positions in {len(batches)} batches (sizes {sorted(set(sizes))})", flush=True)
    model = export_ts.load_v5(args.ckpt, "cpu")
    t0 = time.perf_counter()
    ref = test_cudanet.reference(model, batches)
    print(f"  reference: float32 on the CPU, {time.perf_counter() - t0:.1f} s", flush=True)
    del model
    res = {"positions": len(fens), "batches": len(batches), "sizes": sizes, "engines": {}}
    for path in args.engines.split(","):
        net = TrtNet(path, args.device, graph=not args.no_graph, pad=not args.no_pad)
        got = [run(net, b) for b in batches]
        again = [run(net, b) for b in batches]
        net.close()
        m = test_cudanet.metrics(ref, got, batches)
        m["repeatable"] = all(all(np.array_equal(x, y) for x, y in zip(g1, g2)) for g1, g2 in zip(got, again))
        bad = test_cudanet.verdict(m)
        if not m["repeatable"]:
            bad.append("two runs of the same batches differ")
        m["fails"], m["pass"] = bad, not bad
        res["engines"][path] = m
        test_cudanet.print_metrics(f"TensorRT {os.path.basename(path)} vs float32", m)
        print("  PASS" if not bad else "  FAIL: " + "; ".join(bad), flush=True)
    return res


def cudanet_speed(args, acc16):
    """test_cudanet.py speed --paths cudanet in a child process with MAYFLY_CUDANET_ACC16 set."""
    fd, js = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    env = dict(os.environ, MAYFLY_CUDANET_ACC16=str(acc16))
    cmd = [sys.executable, os.path.join(CUDANET, "test_cudanet.py"), "speed", "--cnw", args.cnw, "--paths", "cudanet",
           "--sizes", args.sizes, "--warmup", str(args.warmup), "--reps", str(args.reps), "--n", str(args.n),
           "--json", js]
    if args.dll:
        cmd += ["--dll", args.dll]
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
    with open(js, encoding="utf-8") as f:
        row = json.load(f)["speed"]["paths"]["cudanet"]
    os.remove(js)
    return {int(k): v for k, v in row.items()}


def speed(args, fens):
    sizes = [int(x) for x in args.sizes.split(",")]
    sets = {B: export_ts.make_batches(fens, [B] * 4) for B in sizes}
    res = {"sizes": sizes, "warmup": args.warmup, "reps": args.reps, "graph": not args.no_graph,
           "pad": not args.no_pad, "paths": {}}
    for path in args.engines.split(","):
        net = TrtNet(path, args.device, graph=not args.no_graph, pad=not args.no_pad)
        row = {}
        for B in sizes:
            s = test_cudanet.timed(lambda b: run(net, b), sets[B], args.warmup, args.reps)
            row[B] = {"evals_s": B / s, "ms_batch": 1000.0 * s}
        free, total = mem_info()
        net.close()
        res["paths"]["trt " + os.path.splitext(os.path.basename(path))[0]] = row
        print(f"  {os.path.basename(path)}: done (free memory with it loaded {free / 2 ** 20:.0f} MB)", flush=True)
    if args.cnw:
        for a16 in args.acc16.split(","):
            res["paths"][f"cudanet acc16={a16}"] = cudanet_speed(args, a16)
    print(f"speed: evaluations a second (ms a batch), {args.warmup} warm-up then {args.reps} calls each")
    print(f"  {'path':34s}" + "".join(f"{'B=' + str(B):>18s}" for B in sizes))
    for name, row in res["paths"].items():
        print(f"  {name:34s}" + "".join(f"{row[B]['evals_s']:>10.0f} ({row[B]['ms_batch']:5.2f})" for B in sizes))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="TensorRT engines against float32 and against cudanet")
    ap.add_argument("mode", choices=("numerics", "speed"))
    ap.add_argument("--engines", required=True, help="comma separated engine files (build_engine.py)")
    ap.add_argument("--ckpt", help="the checkpoint (numerics)")
    ap.add_argument("--n", type=int, default=None, help="positions (numerics 256, speed 512 as test_cudanet)")
    ap.add_argument("--json")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--sizes", default=",".join(str(s) for s in test_cudanet.SIZES))
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--cnw", help="speed: also time cudanet on this weight file")
    ap.add_argument("--acc16", default="0,1,2", help="speed: the MAYFLY_CUDANET_ACC16 settings for cudanet")
    ap.add_argument("--dll", help="speed: cudanet.dll (default as test_cudanet)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-pad", action="store_true")
    args = ap.parse_args(argv)
    if args.mode == "numerics" and not args.ckpt:
        ap.error("numerics needs --ckpt")
    if args.n is None:
        args.n = 256 if args.mode == "numerics" else 512
    fens = test_cudanet.positions(max(args.n, 48 * 4), args.seed)
    import tensorrt as trt
    out = {"test": "trtnet", "when": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
           "tensorrt": trt.__version__, "engines": args.engines.split(",")}
    ok = True
    if args.mode == "numerics":
        out["numerics"] = numerics(args, fens[:args.n])
        ok = all(m["pass"] for m in out["numerics"]["engines"].values())
    else:
        out["speed"] = speed(args, fens)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=str)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
