r"""tpu_bench measures what a Kaggle TPU session can do with MayFly's network.

kaggle_selfplay.py runs it in its own process at the start of each TPU session, so the TPU is free again
when it exits.

  jax     the 146.1M network sharded across every chip. A training step (forward, backward and an Adam
          update, with a stand-in policy and q loss) at two global batches, and the forward pass alone at
          self-play's largest bucket. Reports samples a second and the implied TFLOPs (6 x params x 65
          tokens a sample to train, 2 x to evaluate).
  torch   whether PyTorch/XLA is installed (train_xla.py runs train_av.py's code through it), and if so a
          training step of ChessNet on one chip.

  python tpu_bench.py jax|torch NPZ OUT.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PARAMS = 146.1e6
TOKENS = 65


def batch(rng, B, W=48):
    ids = rng.integers(0, 13, (B, 64)).astype(np.int32)
    cast = rng.integers(0, 2, (B, 4)).astype(np.float32)
    hm = rng.integers(0, 60, B).astype(np.int32)
    ep = np.zeros(B, np.int32)
    I = rng.integers(0, 4162, (B, W)).astype(np.int32)
    M = np.zeros((B, W), bool)
    M[:, :30] = True
    tq = rng.random((B, W)).astype(np.float32) * 2 - 1
    return ids, cast, hm, ep, I, M, tq


def bench_jax(npz):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from selfplay_remote import load_weights
    from chessnet_jax import make_forward
    t0 = time.time()
    cfg, params = load_weights(npz=npz)
    devs = jax.devices()
    n = len(devs)
    res = {"jax": jax.__version__, "devices": n, "kind": devs[0].device_kind, "platform": devs[0].platform}
    mesh = Mesh(np.array(devs), ("d",))
    rep, dat = NamedSharding(mesh, P()), NamedSharding(mesh, P("d"))
    fwd = make_forward(int(cfg["nhead"]), "bfloat16")

    def loss_fn(p, ids, cast, hm, ep, I, M, tq):
        pg, qg, _ = fwd(p, ids, cast, hm, ep, I, I, M)
        lp = -jnp.mean(jax.nn.log_softmax(jnp.where(M, pg, -1e9), -1)[:, 0])
        lq = jnp.sum(jnp.where(M, (qg - tq) ** 2, 0.0)) / jnp.sum(M)
        return lp + lq

    def step(p, m, v, *b):
        loss, g = jax.value_and_grad(loss_fn)(p, *b)
        tm = jax.tree_util.tree_map
        m = tm(lambda m, g: 0.9 * m + 0.1 * g, m, g)
        v = tm(lambda v, g: 0.999 * v + 0.001 * g * g, v, g)
        p = tm(lambda p, m, v: p - 1e-6 * m / (jnp.sqrt(v) + 1e-8), p, m, v)
        return p, m, v, loss

    p = jax.device_put(params, rep)
    zeros = lambda: jax.device_put(jax.tree_util.tree_map(np.zeros_like, params), rep)  # noqa: E731
    m, v = zeros(), zeros()
    jstep = jax.jit(step, in_shardings=(rep, rep, rep) + (dat,) * 7, out_shardings=(rep, rep, rep, rep),
                    donate_argnums=(0, 1, 2))
    rng = np.random.default_rng(0)
    res["train"] = []
    for B in (1024, 4096):
        try:
            b = batch(rng, B)
            tc = time.time()
            p, m, v, loss = jstep(p, m, v, *b)
            jax.block_until_ready(loss)
            compile_s = time.time() - tc
            for _ in range(2):
                p, m, v, loss = jstep(p, m, v, *b)
            jax.block_until_ready(loss)
            k = 8
            ts = time.time()
            for _ in range(k):
                p, m, v, loss = jstep(p, m, v, *b)
            jax.block_until_ready(loss)
            dt = (time.time() - ts) / k
            sps = B / dt
            res["train"].append({"batch": B, "step_s": round(dt, 4), "samples_s": round(sps),
                                 "tflops": round(sps * 6 * PARAMS * TOKENS / 1e12, 1),
                                 "compile_s": round(compile_s, 1), "loss": float(loss)})
        except Exception as e:
            res["train"].append({"batch": B, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        print(json.dumps(res["train"][-1]), flush=True)
    del m, v
    # the forward pass alone, at self-play's largest bucket
    try:
        pf = jax.device_put(params, rep)
        f = jax.jit(fwd, in_shardings=(rep,) + (dat,) * 7, out_shardings=(dat, dat, dat))
        B = 8192
        ids, cast, hm, ep, I, M, _ = batch(rng, B)
        out = f(pf, ids, cast, hm, ep, I, I, M)
        jax.block_until_ready(out)
        ts = time.time()
        for _ in range(8):
            out = f(pf, ids, cast, hm, ep, I, I, M)
        jax.block_until_ready(out)
        dt = (time.time() - ts) / 8
        res["forward"] = {"batch": B, "step_s": round(dt, 4), "evals_s": round(B / dt),
                          "tflops": round(B / dt * 2 * PARAMS * TOKENS / 1e12, 1)}
    except Exception as e:
        res["forward"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    res["seconds"] = round(time.time() - t0)
    return res


def bench_torch():
    import torch
    res = {"torch": torch.__version__}
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        res["torch_xla"] = getattr(torch_xla, "__version__", "?")
    except Exception as e:
        res["torch_xla"] = f"missing ({type(e).__name__}: {str(e)[:200]})"
        return res
    try:
        from chess_net import ChessNet
        dev = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
        model = ChessNet(d=1024, nhead=16, layers=16, dim_ff=2048, qhead=True, att=True, dist=64, mlh=True).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999))
        rng = np.random.default_rng(0)
        out_rows = []
        for B in (128, 512):
            ids, cast, hm, ep, _, _, _ = batch(rng, B)
            T = lambda a, dt: torch.from_numpy(a).to(dev, dt)  # noqa: E731
            x = (T(ids, torch.long), T(cast, torch.float32), T(hm, torch.long), T(ep, torch.long))
            tgt = T(rng.integers(0, 4162, B), torch.long)

            def one():
                with torch.autocast("xla", dtype=torch.bfloat16):
                    o = model(*x)
                loss = (torch.nn.functional.cross_entropy(o["policy"].float(), tgt)
                        + torch.nn.functional.mse_loss(torch.sigmoid(o["qvalue"].float()).mean(-1),
                                                       torch.full((B,), 0.5, device=dev)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                xm.mark_step()
                return loss

            tc = time.time()
            for _ in range(3):
                loss = one()
            loss.item()
            compile_s = time.time() - tc
            ts = time.time()
            for _ in range(8):
                loss = one()
            loss.item()
            dt = (time.time() - ts) / 8
            out_rows.append({"batch": B, "step_s": round(dt, 4), "samples_s": round(B / dt),
                             "tflops": round(B / dt * 6 * PARAMS * TOKENS / 1e12, 1), "compile_s": round(compile_s, 1)})
            print(json.dumps(out_rows[-1]), flush=True)
        res["train_one_chip"] = out_rows
    except Exception as e:
        import traceback
        res["error"] = f"{type(e).__name__}: {str(e)[:300]} | {traceback.format_exc()[-800:]}"
    return res


if __name__ == "__main__":
    mode, npz, out = sys.argv[1], sys.argv[2], sys.argv[3]
    r = bench_jax(npz) if mode == "jax" else bench_torch()
    with open(out, "w") as fh:
        json.dump(r, fh, indent=1)
    print("BENCH", json.dumps(r), flush=True)
