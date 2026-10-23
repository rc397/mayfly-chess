r"""train_jax continues the main training run on a TPU in JAX, one process driving every chip.

torch_xla (train_xla.py) silently runs untranslatable ops on the CPU or stops the step to read values back, and
with a process per chip every chip must make the same collective calls in the same order. JAX compiles the whole
step once, rejects at compile time what the TPU cannot run, and drives all chips from one process with the batch
split across them.

The network is ChessNet's arithmetic for every head the main run trains. The losses are step_batch's with its
weights and detach schedules. The update is torch's AdamW (decoupled weight decay per group, bias correction from
each parameter's own step count) after train_av's clip to norm 1 and skip rule (no update when the norm is not
finite or over 50), then the policy is recentred. The rate is train_av's lr_at times XLA_LR_SCALE (default the
square root of the batch multiple). The checkpoint is read and written in train_av's exact format with the GPU
trainer's optimizer flags kept, so it resumes from it as from its own. kaggle_train.py watches the same files
train_xla.py writes (xla_started, xla_heartbeat, xla_summary.json).

  data        LOADERS worker processes, each running train_av's own loaders on its own seed, hand whole batches
              over a queue.
  one trunk   a chip's action-value rows and position rows go through the trunk together, as in step_batch. The
              host lays each chip's share of both side by side, so only the gradient sum moves between chips.
  precision   bfloat16 matrix products accumulating in float32. Weights, moments, every softmax, layer norm and
              loss in float32. XLA_DTYPE=float32 for the equivalence tests.
  iterations  a step is chips x XLA_BATCH_X GPU batches and adds that many to `it`, as in train_xla.py. Trained
              hours advance by samples over RIG_POS_PER_HOUR.
  saves       at step XLA_PROOF_STEP, every XLA_SAVE_MIN minutes and at the end. XLA_COOL_MIN cools a copy at the
              end (train_av's cool_factor), written as ckpt_cooled_last.pt.

  python chess/train_jax.py
  env: train_av's (CKPT_DIR, SHARDS, TAG, AV_BSZ, POL_BSZ, the shares, heads and weights), XLA_SECONDS, XLA_SAVE_MIN=30,
       XLA_PROOF_STEP=40, XLA_COOL_MIN=0, XLA_LR_SCALE=, XLA_BATCH_X=1, XLA_LOG_EVERY=200, XLA_MIN_SPS=500,
       XLA_MAX_STEPS=0, XLA_DTYPE=bfloat16, RIG_POS_PER_HOUR=972000, LOADERS (worker processes), DATA_SEED
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
E = os.environ.get
A_KEYS = ("ids", "cast", "hm", "ep", "mv", "q")
P_KEYS = ("ids", "cast", "hm", "ep", "pol", "val", "mat", "king", "pawns", "mob", "bish", "aux")
INTS = {"ids", "hm", "ep", "mv"}


# ------------------------------------------------------------------------------------------------ the data

def slices(total_av, total_pol):
    """train_av's split of a step's rows into action values (the ChessBench rest, route 1, Lichess deep, deep
    second) and positions (the rest, the moves-left rows)."""
    import train_av as T
    S = T.SHARDS
    has = lambda p: any(f.startswith(p + "_") for f in os.listdir(S))  # noqa: E731
    dp = int(round(total_av * T.DP_SHARE)) if T.DP_SHARE and has("dp") else 0
    lv = int(round(total_av * T.LV_SHARE)) if T.LV_SHARE and has("lv") else 0
    dh = int(round(total_av * T.DH_SHARE)) if T.DH_SHARE and has("dh") else 0
    mlh = int(round(total_pol * T.MLH_SHARE)) if T.MLH else 0
    return total_av - dp - lv - dh, dp, lv, dh, total_pol - mlh, mlh


def make_loaders(seed, sizes):
    """train_av's loaders at this session's slice sizes, as train_xla.py builds them for one replica."""
    import train_av as T
    from data_av import AVLoader
    from data_packed import PackedLoader
    S = T.SHARDS
    av_n, dp_n, lv_n, dh_n, pol_n, mlh_n = sizes
    mk = lambda **kw: AVLoader(S, split="train", device="cpu", refresh=0, **kw)  # noqa: E731
    avs = [mk(batch=av_n, seed=seed)]
    if dp_n:
        avs.append(mk(batch=dp_n, seed=seed + 2, prefix="dp", thin=T.DP_THIN, down_keys=T.DP_DOWN_KEYS or None,
                      down_weight=T.DP_DOWN_WEIGHT))
    if lv_n:
        avs.append(mk(batch=lv_n, seed=seed + 4, prefix="lv", thin=T.LV_THIN))
    if dh_n:
        avs.append(mk(batch=dh_n, seed=seed + 6, prefix="dh", thin=T.DH_THIN))
    pols = [PackedLoader(S, batch=pol_n, split="train", device="cpu", seed=seed + 3, sources=T.SOURCES, refresh=0,
                         legal=T.MASK_LEGAL, mleft=T.MLH)]
    if mlh_n:
        pols.append(PackedLoader(S, batch=mlh_n, split="train", device="cpu", seed=seed + 5, sources=("lz",),
                                 refresh=0, legal=T.MASK_LEGAL, mleft=True, need_flag=8))
    p_keys = P_KEYS + (("legal",) if T.MASK_LEGAL else ()) + (("mleft", "mlv") if T.MLH else ())
    return avs, pols, p_keys


def next_batch(avs, pols, p_keys):
    """One whole batch as numpy arrays, in step_batch's order (action values as the rest, route 1, Lichess, deep
    second, then positions as the rest and the moves-left rows)."""
    a = [np.concatenate(z) for z in zip(*[[t.numpy() for t in L.next()] for L in avs])]
    p = [np.concatenate(z) for z in zip(*[[t.numpy() for t in L.next()] for L in pols])]
    batch = {"a_" + k: (v.astype(np.int32) if k in INTS else v) for k, v in zip(A_KEYS, a)}
    batch.update({"p_" + k: (v.astype(np.int32) if k in INTS else v) for k, v in zip(p_keys, p)})
    return batch


def loader_worker(seed, sizes, out_q, stop):
    """One worker process, putting whole batches onto out_q until told to stop."""
    avs, pols, p_keys = make_loaders(seed, sizes)
    while not stop.is_set():
        batch = next_batch(avs, pols, p_keys)
        while not stop.is_set():
            try:
                out_q.put(batch, timeout=5)
                break
            except Exception:
                continue


def to_chips(batch, nd):
    """The inputs both kinds of row share (the board), each chip's action-value rows then its position rows, so a
    chip's share of the trunk is its own share of both. The targets stay in their own order, which splits over the
    chips the same way."""
    out = {}
    for k in ("ids", "cast", "hm", "ep"):
        a, p = batch["a_" + k], batch["p_" + k]
        ra, rp = len(a) // nd, len(p) // nd
        out["x_" + k] = np.concatenate([a.reshape(nd, ra, *a.shape[1:]), p.reshape(nd, rp, *p.shape[1:])],
                                       axis=1).reshape(nd * (ra + rp), *a.shape[1:])
    for k, v in batch.items():
        if k[2:] not in ("ids", "cast", "hm", "ep"):
            out[k] = v
    return out


# ------------------------------------------------------------------------------------------------ the network

def make_loss(cfg, dtype="bfloat16"):
    """loss(P, X, flags) -> (total, parts), step_batch's objective on a batch laid out by to_chips, for nd chips.
    P is the model's state dict as arrays under torch's own names and layouts (a Linear's weight is (out, in))."""
    import jax
    import jax.numpy as jnp
    from moves import IDX_PROMO, N_FROMTO, UNDERPROMOTIONS
    f32 = jnp.float32
    cdt = jnp.bfloat16 if dtype == "bfloat16" else f32
    d, nhead, L, nd = cfg["d"], cfg["nhead"], cfg["layers"], cfg["nd"]
    att, dist, mlh, legal_on = cfg["att"], cfg["dist"], cfg["mlh"], cfg["legal"]
    W = cfg["weights"]
    ftp = [IDX_PROMO[N_FROMTO + i] for i in range(len(IDX_PROMO))]
    PF = np.array([f for f, _, _ in ftp])
    PT = np.array([t for _, t, _ in ftp])
    PP = np.array([UNDERPROMOTIONS.index(p) for _, _, p in ftp])
    sg = jax.lax.stop_gradient

    def mm(x, w):
        return jnp.matmul(x.astype(cdt), w.astype(cdt), preferred_element_type=f32)

    def lin(P, name, x):
        y = mm(x, P[name + ".weight"].T)
        return y + P[name + ".bias"] if name + ".bias" in P else y

    def ln(P, name, x, eps=1e-5):
        x = x.astype(f32)
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) * jax.lax.rsqrt(var + eps) * P[name + ".weight"] + P[name + ".bias"]

    gelu = lambda x: jax.nn.gelu(x, approximate=False)  # noqa: E731

    def layer(P, i, x):
        k = f"enc.layers.{i}."
        B, S, _ = x.shape
        dh = d // nhead
        h = ln(P, k + "norm1", x)
        qkv = mm(h, P[k + "self_attn.in_proj_weight"].T) + P[k + "self_attn.in_proj_bias"]
        q, kk, v = (t.reshape(B, S, nhead, dh).transpose(0, 2, 1, 3) for t in jnp.split(qkv, 3, axis=-1))
        s = mm(q, kk.transpose(0, 1, 3, 2)) * (dh ** -0.5)
        o = mm(jax.nn.softmax(s, axis=-1), v).transpose(0, 2, 1, 3).reshape(B, S, d)
        x = x + lin(P, k + "self_attn.out_proj", o)
        h = gelu(lin(P, k + "linear1", ln(P, k + "norm2", x)))
        return x + lin(P, k + "linear2", h)

    def trunk(P, ids, cast, hm, ep):
        x = P["piece.weight"][ids] + P["pos.weight"][None]
        mark = jax.nn.one_hot(jnp.clip(ep - 1, 0, 63), 64, dtype=f32) * (ep > 0).astype(f32)[:, None]
        x = x + mark[:, :, None] * P["ep_mark"]
        cls = P["cls"].reshape(1, d) + lin(P, "cast", cast) + P["hm.weight"][hm]
        x = jnp.concatenate([cls[:, None, :], x], axis=1)
        for i in range(L):
            x = layer(P, i, x)
        return x

    def attn_head(P, name, s):
        q, k = lin(P, name + ".q", s), lin(P, name + ".k", s)
        ft = mm(q, k.transpose(0, 2, 1)) * (k.shape[-1] ** -0.5)
        under = ft[:, PF, PT] + lin(P, name + ".promo", k)[:, PT, PP]
        return jnp.concatenate([ft.reshape(ft.shape[0], -1), under], 1)

    def bce(x, y):
        return jnp.maximum(x, 0) - x * y + jnp.log1p(jnp.exp(-jnp.abs(x)))

    def q_loss(qv, mv, q):
        return bce(jnp.take_along_axis(qv, mv[:, None], 1)[:, 0], q).mean()

    def policy_ce(logits, pol, legal):
        if legal is not None:
            logits = jnp.where(legal, logits, -1e9)
        return -(pol * jax.nn.log_softmax(logits, -1)).sum(-1).mean()

    def hl_gauss(t, bins):
        sigma = 0.75 / bins
        edges = jnp.linspace(0.0, 1.0, bins + 1, dtype=f32)
        cdf = jax.scipy.special.erf((edges[None, :] - t.astype(f32)[:, None]) / (sigma * (2.0 ** 0.5)))
        p = cdf[:, 1:] - cdf[:, :-1]
        return p / jnp.maximum(p.sum(-1, keepdims=True), 1e-9)

    def dist_loss(logits, t):
        return -(hl_gauss(t, logits.shape[-1]) * jax.nn.log_softmax(logits, -1)).sum(-1).mean()

    def keep(flag, x):
        """x with its gradient cut while flag is 1, for train_av's detach schedules without a recompile."""
        return flag * sg(x) + (1.0 - flag) * x

    def loss(P, X, flags):
        z = trunk(P, X["x_ids"], X["x_cast"], X["x_hm"], X["x_ep"])
        ra = X["a_mv"].shape[0] // nd
        z = z.reshape(nd, -1, *z.shape[1:])
        za, zp = z[:, :ra].reshape(-1, *z.shape[2:]), z[:, ra:].reshape(-1, *z.shape[2:])
        ha, hp = ln(P, "ln", za[:, 0]), ln(P, "ln", zp[:, 0])
        legal = X.get("p_legal") if legal_on else None
        parts = {"q": q_loss(lin(P, "qvalue", ha), X["a_mv"], X["a_q"])}
        parts["policy"] = policy_ce(lin(P, "policy", hp), X["p_pol"], legal)
        w = X["p_aux"][:, None]
        n = jnp.maximum(X["p_aux"].sum(), 1.0)
        msew = lambda a, b: ((((a - b) ** 2).mean(-1, keepdims=True)) * w).sum() / n  # noqa: E731
        la = (msew(lin(P, "h_material", hp), X["p_mat"]) + msew(lin(P, "h_king", hp), X["p_king"])
              + msew(lin(P, "h_pawns", hp), X["p_pawns"]) + msew(lin(P, "h_mob", hp), X["p_mob"][:, None])
              + (bce(lin(P, "h_bishop", hp), X["p_bish"]).mean(-1, keepdims=True) * w).sum() / n)
        total = W["q"] * parts["q"] + parts["policy"] + 0.2 * la
        parts["aux"] = la
        if W["v"]:
            parts["value"] = bce(lin(P, "value", hp)[:, 0], X["p_val"]).mean()
            total = total + W["v"] * parts["value"]
        if att:
            sa = ln(P, "ln_sq", keep(flags["att"], za[:, 1:]))
            sp = ln(P, "ln_sq", keep(flags["att"], zp[:, 1:]))
            parts["att_q"] = q_loss(attn_head(P, "att_q", sa), X["a_mv"], X["a_q"])
            parts["att_policy"] = policy_ce(attn_head(P, "att", sp), X["p_pol"], legal)
            total = total + W["att"] * (W["q"] * parts["att_q"] + parts["att_policy"])
        if dist:
            hq = keep(flags["dist"], ha)
            e = P["qdist.mv.weight"][X["a_mv"]]
            qd = lin(P, "qdist.net.2", gelu(lin(P, "qdist.net.0", jnp.concatenate([hq, e], -1))))
            parts["qdist"] = dist_loss(qd, X["a_q"])
            parts["vdist"] = dist_loss(lin(P, "vdist", sg(hp)), X["p_val"])
            total = total + W["qdist"] * parts["qdist"] + W["vdist"] * parts["vdist"]
        if mlh:
            pred = lin(P, "mleft.2", gelu(lin(P, "mleft.0", keep(flags["mlh"], hp))))[:, 0]
            valid = X["p_mlv"]
            err = pred - X["p_mleft"] / 100.0
            a = jnp.abs(err)
            hub = jnp.where(a < 0.1, 0.5 * err * err, 0.1 * (a - 0.05))
            parts["mleft"] = (hub * valid).sum() / jnp.maximum(valid.sum(), 1.0)
            total = total + W["mlh"] * parts["mleft"]
        return total, parts

    return loss


def frozen_names(names, weights):
    """Parameters step_batch's loss never reaches, which torch's AdamW leaves alone (no weight decay, no moments,
    no step). That is the scalar value head while W_V is 0, its only conditional term. Every other head is added
    unconditionally, so it gets a gradient, zero or not, and steps."""
    return {n for n in names if n.startswith("value.")} if not weights["v"] else set()


def make_step(loss, wd, betas, eps, frozen=()):
    """step(state, X, lr, flags) -> (state, metrics). The gradient of `loss`, train_av's clip to norm 1 and skip
    rule, torch's AdamW per parameter (decoupled weight decay wd[name], bias correction from each parameter's own
    step count) except the frozen ones, then the policy recentred. A skipped step changes only the count of skips."""
    import jax
    import jax.numpy as jnp
    b1, b2 = betas

    def step(state, X, lr, flags):
        (total, parts), g = jax.value_and_grad(loss, has_aux=True)(state["p"], X, flags)
        gn = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(g)))
        coef = jnp.minimum(1.0, 1.0 / (gn + 1e-6))
        bad = ~jnp.isfinite(gn) | (gn > 50.0)
        P, M, V, S = {}, {}, {}, {}
        for k, p in state["p"].items():
            m, v, s = state["m"][k], state["v"][k], state["s"][k]
            if k in frozen:
                P[k], M[k], V[k], S[k] = p, m, v, s
                continue
            gk = g[k] * coef
            s2 = s + 1.0
            p2 = p * (1.0 - lr * wd[k])
            m2 = m + (gk - m) * (1.0 - b1)
            v2 = v * b2 + (1.0 - b2) * gk * gk
            denom = jnp.sqrt(v2) / jnp.sqrt(1.0 - b2 ** s2) + eps
            p2 = p2 - (lr / (1.0 - b1 ** s2)) * m2 / denom
            P[k], M[k], V[k], S[k] = (jnp.where(bad, a, b) for a, b in ((p, p2), (m, m2), (v, v2), (s, s2)))
        for k in ("policy.weight", "policy.bias"):
            w = P[k]
            P[k] = jnp.where(bad, w, w - w.mean(0, keepdims=True))
        gnorm = jnp.where(bad, state["gnorm"], 0.98 * state["gnorm"] + 0.02 * gn)
        new = {"p": P, "m": M, "v": V, "s": S, "gnorm": gnorm, "skipped": state["skipped"] + bad.astype(jnp.float32)}
        return new, {"loss": total, "gn": gn, "bad": bad, **parts}

    return step


# ------------------------------------------------------------------------------------------------ the checkpoint

def layout(ck, cfg_env):
    """train_av's model and optimizer as named arrays, the parameters in its optimizer's order (the trunk's matrices,
    the trunk's vectors, then the attention, distributional and moves-left heads, each a group) with each one's
    group, moments and step count. Built from ChessNet itself, so the order is the module's."""
    import torch
    from chess_net import ChessNet
    c = cfg_env
    model = ChessNet(d=c["d"], nhead=c["nhead"], layers=c["layers"], dim_ff=c["dim_ff"], qhead=True,
                     dropout=0.0, att=c["att"], dist=c["dist"], mlh=c["mlh"])
    model.load_state_dict(ck["model"], strict=True)
    is_att = lambda n: n.split(".")[0] in ("ln_sq", "att", "att_q")  # noqa: E731
    is_dist = lambda n: n.split(".")[0] in ("qdist", "vdist")  # noqa: E731
    is_mlh = lambda n: n.split(".")[0] == "mleft"  # noqa: E731
    named = [(n, p) for n, p in model.named_parameters() if not (is_att(n) or is_dist(n) or is_mlh(n))]
    groups = [[n for n, p in named if p.ndim > 1], [n for n, p in named if p.ndim <= 1]]
    for on, pred in ((c["att"], is_att), (c["dist"], is_dist), (c["mlh"], is_mlh)):
        if on:
            groups.append([n for n, _ in model.named_parameters() if pred(n)])
    og = ck["optim"]["param_groups"]
    if [len(g) for g in groups] != [len(g["params"]) for g in og]:
        raise SystemExit(f"the checkpoint's optimizer groups {[len(g['params']) for g in og]} are not ChessNet's "
                         f"{[len(g) for g in groups]}")
    sd = {n: p.detach() for n, p in model.named_parameters()}
    st = ck["optim"]["state"]
    P, M, V, S, WD, had = {}, {}, {}, {}, {}, set()
    for g, names in zip(og, groups):
        for i, n in zip(g["params"], names):
            s = st.get(i, {})
            if s:
                had.add(n)
            P[n] = sd[n].float().numpy().copy()
            M[n] = (s["exp_avg"].float().numpy().copy() if "exp_avg" in s else np.zeros_like(P[n]))
            V[n] = (s["exp_avg_sq"].float().numpy().copy() if "exp_avg_sq" in s else np.zeros_like(P[n]))
            if M[n].shape != P[n].shape:
                raise SystemExit(f"{n}: moment {M[n].shape} against parameter {P[n].shape}")
            S[n] = np.float32(float(s["step"]) if "step" in s else 0.0)
            WD[n] = float(g["weight_decay"])
    return P, M, V, S, WD, groups, had


def to_checkpoint(ck, groups, P, M, V, S, lr, keep_state):
    """The arrays back in train_av's format. The model's state dict in its own order (buffers as loaded), and the
    optimizer's state under its indices with the checkpoint's own groups and flags and the current rate. Only
    parameters in keep_state get optimizer state, as torch's own state dict holds only the ones it stepped."""
    import torch
    model = {}
    for k, v in ck["model"].items():
        model[k] = torch.from_numpy(np.asarray(P[k], np.float32).copy()) if k in P else v
    state, og = {}, ck["optim"]["param_groups"]
    for g, names in zip(og, groups):
        for i, n in zip(g["params"], names):
            if n not in keep_state:
                continue
            state[i] = {"step": torch.tensor(float(S[n]), dtype=torch.float32),
                        "exp_avg": torch.from_numpy(np.asarray(M[n], np.float32).copy()),
                        "exp_avg_sq": torch.from_numpy(np.asarray(V[n], np.float32).copy())}
    groups_out = [dict(g, lr=lr) for g in og]
    return model, {"state": state, "param_groups": groups_out}


# ------------------------------------------------------------------------------------------------ the run

def cpus():
    """CPUs this process may really use, from its affinity and the container's quota where there is one (Kaggle's
    TPU host shows 224 cores to os.cpu_count and gives the container 48)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 8)
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def get_batch(q, workers):
    """The next batch, or an error if every worker has died, never a silent wait."""
    import queue as _q
    while True:
        try:
            return q.get(timeout=30)
        except _q.Empty:
            dead = [w for w in workers if not w.is_alive()]
            if len(dead) == len(workers):
                raise RuntimeError(f"every loader process has died (exit codes {[w.exitcode for w in dead]})")


def run():
    import multiprocessing as mp
    import torch
    import train_av as T
    t_start = time.time()
    budget = float(E("XLA_SECONDS", 3600))
    dtype = E("XLA_DTYPE", "bfloat16")
    if T.DROPOUT:
        raise SystemExit("train_jax has no dropout (train_av's DROPOUT is set)")

    def say(msg):
        print(time.strftime("%H:%M:%S ") + msg, flush=True)

    # the loaders first, spawned (fresh interpreters with no TPU runtime), while the checkpoint loads
    import jax
    devs = jax.devices()
    nd = len(devs)
    batch_x = int(E("XLA_BATCH_X", 1))
    av_tot, pol_tot = T.AV_BSZ * batch_x * nd, T.POL_BSZ * batch_x * nd
    sizes = slices(av_tot, pol_tot)
    if sizes[0] % 1 or av_tot % nd or pol_tot % nd:
        raise SystemExit(f"a step's {av_tot} + {pol_tot} rows do not split over {nd} chips")
    data_seed = int(E("DATA_SEED") or int.from_bytes(os.urandom(4), "little"))
    # one worker makes about 2,800 samples a second and they add up, so eight cover the chips' 12,000 with room,
    # within the CPUs the container really has and leaving the trainer its own
    nw = int(E("LOADERS") or max(2, min(8, cpus() // 4)))
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=max(4, 2 * nw))
    stop = ctx.Event()
    workers = [ctx.Process(target=loader_worker, args=(data_seed + 1000 * w, sizes, q, stop), daemon=True)
               for w in range(nw)]
    for w in workers:
        w.start()

    if E("XLA_CACHE_DIR"):                      # a restart in the session loads its compiled step
        jax.config.update("jax_compilation_cache_dir", os.path.join(E("XLA_CACHE_DIR"), "jax"))
    # check the chips answer with a sum over all of them, then write xla_started for kaggle_train.py
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS
    mesh = Mesh(np.array(devs), ("d",))
    rep, dat = NamedSharding(mesh, PS()), NamedSharding(mesh, PS("d"))
    t_init = time.time()
    got = float(jax.jit(lambda x: x.sum(), out_shardings=rep)(jax.device_put(np.ones(nd, np.float32), dat)))
    if round(got) != nd:
        raise SystemExit(f"the sum over {nd} chips came to {got}")
    with open(os.path.join(T.CKPT_DIR, "xla_started"), "w") as fh:
        json.dump({"mode": "jax", "world": nd, "devices": nd, "seconds": round(time.time() - t_init, 1),
                   "jax": jax.__version__, "kind": devs[0].device_kind}, fh)
    say(f"jax {jax.__version__}: {nd} {devs[0].device_kind} chip(s) answer; a step is {nd} x {batch_x} GPU batches: "
        f"{av_tot} action values (rest {sizes[0]}, route 1 {sizes[1]}, Lichess {sizes[2]}, deep {sizes[3]}) + "
        f"{pol_tot} positions (moves left {sizes[5]}); {nw} loader processes; {dtype} products")

    ck = torch.load(T.LAST, map_location="cpu", weights_only=False)
    cfg = {"d": T.D, "nhead": T.NHEAD, "layers": T.LAYERS, "dim_ff": int(T.FF_MULT * T.D), "att": bool(T.ATT),
           "dist": int(T.DIST), "mlh": bool(T.MLH), "legal": bool(T.MASK_LEGAL), "nd": nd,
           "weights": {"q": T.W_Q, "v": T.W_V, "att": T.W_ATT, "qdist": T.W_QDIST, "vdist": T.W_VDIST,
                       "mlh": T.W_MLH}}
    P, M, V, S, WD, groups, had = layout(ck, cfg)
    frozen = frozen_names(P, cfg["weights"])
    keep_state = had | (set(P) - frozen)
    for k in ("policy.weight", "policy.bias"):          # train_av recentres at resume too
        P[k] = P[k] - P[k].mean(0, keepdims=True)
    og = ck["optim"]["param_groups"]
    betas, eps = (float(og[0]["betas"][0]), float(T.BETA2)), float(og[0].get("eps", 1e-8))
    it = int(ck["iter"]) + 1
    hist, best, trained = ck.get("hist", []), ck.get("best", 1e9), float(ck.get("trained_hours", 0.0))
    att_since, dist_since, mlh_since = ck.get("att_since"), ck.get("dist_since"), ck.get("mlh_since")
    yard = ck.get("yardstick", T.YARDSTICK)
    lr_scale = float(E("XLA_LR_SCALE") or math.sqrt(nd * batch_x))
    rig_sph = float(E("RIG_POS_PER_HOUR", 972000))
    step_it = nd * batch_x
    per = T.AV_BSZ + T.POL_BSZ
    state = jax.device_put({"p": P, "m": M, "v": V, "s": S, "gnorm": np.float32(1.0), "skipped": np.float32(0.0)},
                           rep)
    P = M = V = S = None
    loss = make_loss(cfg, dtype)
    jstep = jax.jit(make_step(loss, WD, betas, eps, frozen), donate_argnums=(0,),
                    in_shardings=(rep, dat, rep, rep), out_shardings=(rep, rep))
    say(f"RESUMED the main run at iter {it:,} ({trained:.1f}h); rate x{lr_scale:.3g} "
        f"({T.lr_at(it, trained) * lr_scale:.3g}); {budget / 3600:.2f}h budget; {sum(map(len, groups))} parameters "
        f"in {len(groups)} groups; left alone as in train_av: {sorted(frozen) or 'none'}")

    samples = 0
    cooled_path = os.path.join(T.CKPT_DIR, "ckpt_cooled_last.pt")
    cool_s = float(E("XLA_COOL_MIN", 0)) * 60

    def save(final=False, path=None, extra=None):
        path = path or T.LAST
        h = jax.device_get(state)
        model_sd, osd = to_checkpoint(ck, groups, h["p"], h["m"], h["v"], h["s"], T.lr_at(it, trained) * lr_scale,
                                      keep_state)
        obj = {"model": model_sd, "optim": osd, "iter": it - 1, "hist": hist, "best": best,
               "trained_hours": trained + samples / rig_sph, "att_since": att_since, "dist_since": dist_since,
               "mlh_since": mlh_since, "yardstick": yard, "data_seed": data_seed, "scaler": None,
               "xla": {"sessions": 1, "samples": samples, "world": nd, "lr_scale": lr_scale, "mode": "jax",
                       "devices": nd, "final": final, **(extra or {})}}
        torch.save(obj, path + ".part")
        os.replace(path + ".part", path)

    def put(b):
        return jax.device_put(to_chips(b, nd), dat)

    phase, k, el, last_save_el, saved_k = "main", 0, 0.0, 0.0, -1
    k_r0 = el_r0 = None
    log_every = int(E("XLA_LOG_EVERY", 200))
    save_every = float(E("XLA_SAVE_MIN", 30)) * 60
    proof_at = int(E("XLA_PROOF_STEP", 40))
    min_sps, paced, slow = float(E("XLA_MIN_SPS", 500)), False, None
    max_steps = int(E("XLA_MAX_STEPS", 0))
    t0 = last_log = time.time()
    last_beat, wait_s, host_s = 0.0, 0.0, 0.0
    tot = {"wait": 0.0, "host": 0.0, "steps": 0}
    nxt = put(get_batch(q, workers))
    m = None
    while True:
        if max_steps and k >= max_steps:
            break
        if phase == "cool" and it >= T.COOL_FROM + T.COOL_STEPS:
            break
        ta = time.time()
        lr = np.float32(T.lr_at(it, trained) * lr_scale)
        flags = {"att": np.float32((it - att_since) < T.ATT_DETACH) if T.ATT else np.float32(0),
                 "dist": np.float32((it - dist_since) < T.DIST_DETACH) if T.DIST else np.float32(0),
                 "mlh": np.float32((it - mlh_since) < T.MLH_DETACH) if T.MLH else np.float32(0)}
        X = nxt
        state, m = jstep(state, X, lr, flags)       # dispatched, the chips run it while the next batch comes in
        tb = time.time()
        nxt = put(get_batch(q, workers))
        tc = time.time()
        host_s, wait_s = host_s + (tb - ta), wait_s + (tc - tb)
        it += step_it
        samples += step_it * per
        k += 1
        if k > 20:
            tot["host"] += tb - ta
            tot["wait"] += tc - tb
            tot["steps"] += 1
        if k == 1:
            jax.block_until_ready(m["loss"])
            say(f"  first step done {time.time() - t0:.0f}s after the start (the compile)")
        if k <= 3 or time.time() - last_beat > 30:
            with open(os.path.join(T.CKPT_DIR, "xla_heartbeat"), "w") as fh:
                fh.write(f"{it} {k} {time.time():.0f}\n")
            last_beat = time.time()
        if k % log_every == 0:
            mh = jax.device_get(m)
            gnorm, skipped = float(jax.device_get(state["gnorm"])), int(jax.device_get(state["skipped"]))
            now = time.time()
            say(f"  it {it:,}  loss {float(mh['loss']):.4f}  q {float(mh['q']):.4f}  policy {float(mh['policy']):.4f}  "
                f"grad {gnorm:.2f}  lr {float(lr):.3g}  {log_every * step_it * per / (now - last_log):,.0f} samples/s  "
                f"{samples / rig_sph:.1f} GPU-trainer hours this session  (a step: {1000 * host_s / log_every:.0f} ms to "
                f"dispatch, {1000 * wait_s / log_every:.0f} ms waiting for data"
                + (f"; {skipped} skipped" if skipped else "") + ")")
            last_log, host_s, wait_s = now, 0.0, 0.0
        if k % 25 == 0:
            el = time.time() - t_start
            if el_r0 is None and k >= 20:
                k_r0, el_r0 = k, el
            if el >= budget:
                break
            if phase == "main" and cool_s and el >= budget - cool_s:
                save(final=True)
                saved_k = k
                say(f"  main run saved at it {it - 1:,} ({samples / rig_sph:.1f} GPU-trainer hours this session)")
                pace = (k - k_r0) / max(el - el_r0, 1.0) if el_r0 is not None and k > k_r0 else k / max(el, 1.0)
                steps = int(pace * max(60.0, budget - el - 300))
                T.COOL_FROM, T.COOL_STEPS = it, max(step_it, steps * step_it)
                phase = "cool"
                say(f"  cooling a copy: the rate falls to {T.COOL_FLOOR:g} of itself over iterations {T.COOL_FROM:,} to "
                    f"{T.COOL_FROM + T.COOL_STEPS:,} ({steps:,} steps at {pace:.2f} a second)")
            elif phase == "main" and el - last_save_el >= save_every:
                save()
                saved_k, last_save_el = k, el
                say(f"  saved at it {it - 1:,}")
            if not paced and el_r0 is not None and phase == "main" and min_sps and el - el_r0 > 1800:
                paced = True
                sps = (k - k_r0) * step_it * per / (el - el_r0)
                say(f"  pace past the compile: {sps:,.0f} samples/s")
                if sps < min_sps:
                    slow = round(sps)
                    say(f"  SLOWER THAN THE GPU TRAINER ({slow:,} < {min_sps:,.0f} samples/s): saving and stopping")
                    break
        if k == proof_at and phase == "main" and saved_k != k:
            save()
            saved_k, last_save_el = k, el
            say(f"  first save done at step {k} (it {it - 1:,})")
    if m is not None:
        jax.block_until_ready(m["loss"])
    skipped = int(jax.device_get(state["skipped"]))
    if phase == "cool":
        save(final=True, path=cooled_path, extra={"cool_from": T.COOL_FROM, "cool_steps": T.COOL_STEPS,
                                                  "cool_floor": T.COOL_FLOOR})
        say(f"  cooled copy saved at it {it - 1:,} ({os.path.basename(cooled_path)})")
    else:
        save(final=True)
    took = time.time() - t0
    say(f"DONE: {samples:,} samples ({k:,} steps, {skipped} skipped) in {took / 3600:.2f}h, "
        f"{samples / max(took, 1):,.0f} samples/s, {samples / rig_sph:.1f} GPU-trainer hours; ends at it {it - 1:,}")
    with open(os.path.join(T.CKPT_DIR, "xla_summary.json"), "w") as fh:
        json.dump({"samples": samples, "steps": k, "skipped": skipped, "seconds": round(took),
                   "samples_per_s": round(samples / max(took, 1)), "rig_hours": round(samples / rig_sph, 2),
                   "iter_end": it - 1, "mode": "jax", "world": nd, "devices": nd, "lr_scale": lr_scale,
                   "batch_x": batch_x, "loaders": nw, "stopped_slow": slow,
                   "ms_per_step": {n: round(1000 * tot[n] / max(tot["steps"], 1), 1) for n in ("host", "wait")},
                   "cooled": phase == "cool", "cool_from": T.COOL_FROM if phase == "cool" else None,
                   "cool_steps": T.COOL_STEPS if phase == "cool" else None}, fh, indent=1)
    stop.set()
    for w in workers:
        w.terminate()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    run()
