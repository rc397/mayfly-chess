r"""Measure whether the action-value shard loader can keep up with training.

Runs the AV loader alone and reports batches a second end to end, against TARGET, the rate the
trainer needs (AV_BSZ=96 and POL_BSZ=32 a step at 475 positions a second is 3.71 steps a second).
The parts are also timed separately.

  draw        pool management and the read from the memory-mapped shards
  build       unpacking the packed records into the arrays a step wants
  to-device   the six host-to-device transfers

Keep runs short while training, since it reads the same disk.

  python chess/loader_bench.py [BATCHES] [BATCH_SIZE]
  env: SHARDS=<data>/shards
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

import paths

E = os.environ.get
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 96
SHARDS = E("SHARDS", paths.SHARDS)
TARGET = 3.71          # batches a second each loader must sustain at the current training rate


def timed(fn, n):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


if __name__ == "__main__":
    import numpy as np
    import torch
    from data_av import AVLoader
    from pack_av import batch_tensors_av

    print(f"target: each loader must sustain {TARGET:.2f} batches a second "
          f"to keep the trainer at 475 pos/s", flush=True)
    print("")

    av = AVLoader(SHARDS, batch=BSZ, split="train", device="cpu")
    print(f"  {av.describe()}", flush=True)

    # end to end, through the queue, which is what the trainer actually experiences
    for _ in range(20):
        av.next()
    t0 = time.perf_counter()
    for _ in range(N):
        av.next()
    el = time.perf_counter() - t0
    rate = N / el
    print("")
    print(f"  AV loader end to end: {rate:>7.1f} batches a second "
          f"({rate/TARGET:.1f}x what the trainer needs)")

    # the parts, on the producing side, without the queue
    t_draw = timed(av._draw, 200)
    rows = av._draw()
    t_build = timed(lambda: batch_tensors_av(rows), 200)
    built = batch_tensors_av(rows)
    if torch.cuda.is_available():
        def to_gpu():
            for a, d in zip(built, (torch.long, torch.float32, torch.long,
                                    torch.float32, torch.long, torch.long)):
                torch.as_tensor(a).to("cuda", dtype=d, non_blocking=True)
            torch.cuda.synchronize()
        t_to = timed(to_gpu, 100)
    else:
        t_to = float("nan")

    print("")
    print(f"  {'part':<26} {'ms a batch':>11} {'batches/s':>11}")
    for name, ms in (("draw from the pool", t_draw * 1000),
                     ("build the arrays", t_build * 1000),
                     ("six host-to-device copies", t_to * 1000)):
        print(f"  {name:<26} {ms:>11.3f} {1000/ms if ms == ms and ms > 0 else 0:>11.1f}")
    tot = (t_draw + t_build + (t_to if t_to == t_to else 0)) * 1000
    print(f"  {'one batch, all parts':<26} {tot:>11.3f} {1000/max(tot,1e-9):>11.1f}")

    # A refill is the expensive event, so report how often it happens and what it costs.
    t0 = time.perf_counter()
    av._refill()
    t_refill = time.perf_counter() - t0
    per_pool = len(av.pool) / max(BSZ, 1)
    print("")
    print(f"  a pool refill reads {len(av.pool):,} rows in {t_refill*1000:.0f} ms and then serves")
    print(f"  {per_pool:.0f} batches from memory, so it costs "
          f"{t_refill*1000/max(per_pool,1):.2f} ms a batch amortised")

    print("")
    if rate > TARGET * 4:
        print(f"  THE LOADER IS NOT THE BOTTLENECK. It produces {rate:.1f} batches a second alone")
        print(f"  against the {TARGET:.2f} the trainer needs, so the slowdown is somewhere else.")
        print(f"  Candidates then: the other loader, GIL contention between the two filler threads")
        print(f"  and the training loop, or the step itself.")
    elif rate > TARGET * 1.5:
        print(f"  MARGINAL. {rate:.1f} against {TARGET:.2f} needed is headroom on an idle machine")
        print(f"  and may not be headroom while the trainer is competing for the same cores.")
    else:
        print(f"  THE LOADER IS THE BOTTLENECK: {rate:.1f} batches a second against {TARGET:.2f}")
        print(f"  needed. The part table above says which half of it to fix.")
