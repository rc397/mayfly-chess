r"""train_xla continues the main training run on a TPU through PyTorch/XLA, using train_av.py's own step.

One process per chip (XLA_MODE=procs), one process for every chip (spmd) or one chip (single), whichever
kaggle_train.py finds starts. The batch, every loss and head, the detach schedules, the rate (lr_at), the
policy recentring and the skip rule are train_av's own functions. What differs is listed here.

  device        each chip runs a replica on its own slice of the stream (loader seeds offset by replica) with
                train_av's batch, and gradients are averaged before the clip, so a step is world x the GPU batch.
  precision     bfloat16 autocast on XLA. Weights, moments and the checkpoint stay fp32.
  iterations    counted in GPU-trainer steps. A TPU step adds world x XLA_BATCH_X to `it`, so VAL_EVERY, the
                detach schedules and COOL_FROM keep their meaning in samples.
  rate          train_av's rate times XLA_LR_SCALE (default the square root of the batch multiple).
  hours         trained_hours advances by samples over RIG_POS_PER_HOUR, the GPU trainer's pace, so the
                checkpoint's hours stay comparable.
  never         a version freeze, a validation, a best-of checkpoint or a history row. Those stay with the GPU
                trainer, which validates when it resumes.
  saves         the resume checkpoint in train_av's exact format, once at XLA_PROOF_STEP (so a broken save shows
                in minutes), then every XLA_SAVE_MIN minutes and at the end.
  one clock     every decision about time (save, cooled copy, pace check, end) is taken every XLA_SYNC_EVERY
                steps from the replicas' agreed elapsed time. If each replica read its own clock, one could enter
                the save's collective while the others entered the next gradient sum, and the run would hang.
  optimiser     torch_xla's sync-free AdamW, told to skip by a flag computed on the chip (train_av's rule, grad
                norm not finite or over 50), so a step never reads the norm back to the host.
  cooldown      with XLA_COOL_MIN, the last that many minutes cool a copy (train_av's cool_factor) after the main
                run is saved, written as ckpt_cooled_last.pt to export as the next candidate.
  modes         procs is a process a chip (xmp.spawn). spmd is one process with the batch sharded over every chip.
                single (also XLA_PROCS=1) is one process on one chip. spmd and single make no collective call.
  started       each mode first sums over its replicas, then writes CKPT_DIR/xla_started, so kaggle_train.py can
                tell a failed start from a slow compile.
  stops         below XLA_MIN_SPS samples a second half an hour in, it saves and stops, since TPU time slower than
                the GPU trainer is wasted.

  python chess/train_xla.py
  env: train_av's (CKPT_DIR with ckpt_<TAG>_last.pt, SHARDS, TAG, AV_BSZ, POL_BSZ, the shares and heads)
       XLA_MODE=procs  XLA_SECONDS=3600  XLA_SAVE_MIN=30  XLA_LR_SCALE=  XLA_BATCH_X=1  RIG_POS_PER_HOUR=972000
       XLA_LOG_EVERY=200  XLA_COOL_MIN=0 (minutes of cooled copy at the end)  COOL_FLOOR (train_av's, 0.01)
       XLA_MIN_SPS=500  XLA_CACHE_DIR= (a persistent compile cache, so a restart in the session does not recompile)
       XLA_SYNC_EVERY=25  XLA_PROOF_STEP=40
"""
import os
import sys
import time
import math
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
E = os.environ.get
MODE = (E("XLA_MODE") or ("single" if E("XLA_PROCS") == "1" else "procs")).lower()


def procs_env(env=os.environ, chips=None):
    """Kaggle's TPU VM is set up for one process. Its image presets TPU_PROCESS_ADDRESSES=local, and torch_xla's
    per-process topology only uses setdefault, so a process per chip keeps "local" and fails with "Expected 8 worker
    addresses, got 1". So every per-process setting is removed before the spawn and each process gets torch_xla's
    own. A setting that describes the whole host as one process is first moved to the host-level name torch_xla
    reads when it skips the metadata server (TPU_HOST_BOUNDS, TPU_CHIPS_PER_HOST_BOUNDS, TPU_WORKER_ID), never over
    one already there. With the metadata server skipped and no topology at all, the host's own is filled in.
    Returns (taken, moved) for the log."""
    def size(s):
        try:
            return math.prod(int(x) for x in str(s).split(","))
        except ValueError:
            return 0
    taken, moved = {}, {}

    def move(k, host):
        taken[k] = env.pop(k)
        if host not in env:
            env[host] = moved[host] = taken[k]
    if "TPU_PROCESS_BOUNDS" in env and size(env["TPU_PROCESS_BOUNDS"]) <= 1:
        move("TPU_PROCESS_BOUNDS", "TPU_HOST_BOUNDS")           # one process for the host, so the host's bounds
    if "TPU_CHIPS_PER_PROCESS_BOUNDS" in env and size(env["TPU_CHIPS_PER_PROCESS_BOUNDS"]) != 1:
        move("TPU_CHIPS_PER_PROCESS_BOUNDS", "TPU_CHIPS_PER_HOST_BOUNDS")   # all its chips, so the host's chips
    if "CLOUD_TPU_TASK_ID" in env:
        move("CLOUD_TPU_TASK_ID", "TPU_WORKER_ID")
    for k in ("TPU_PROCESS_ADDRESSES", "TPU_VISIBLE_CHIPS", "TPU_VISIBLE_DEVICES", "TPU_PROCESS_PORT",
              "PJRT_LOCAL_PROCESS_RANK", "PJRT_LOCAL_PROCESS_COUNT"):
        if k in env:
            taken[k] = env.pop(k)
    if env.get("TPU_SKIP_MDS_QUERY"):
        if chips is None:
            try:
                from torch_xla._internal import tpu
                chips = tpu.num_available_chips()
            except Exception:
                chips = 0
        fill = {"TPU_HOST_BOUNDS": "1,1,1", "TPU_WORKER_ID": "0",
                "TPU_CHIPS_PER_HOST_BOUNDS": {8: "2,4,1", 4: "2,2,1", 1: "1,1,1"}.get(chips)}
        for k, alt in (("TPU_HOST_BOUNDS", "TPU_PROCESS_BOUNDS"), ("TPU_WORKER_ID", "CLOUD_TPU_TASK_ID"),
                       ("TPU_CHIPS_PER_HOST_BOUNDS", "TPU_CHIPS_PER_PROCESS_BOUNDS")):
            if k not in env and alt not in env and fill[k]:
                env[k] = moved[k] = fill[k]
    return taken, moved


def tpu_settings(env=os.environ):
    return {k: v for k, v in sorted(env.items()) if k.startswith(("TPU_", "PJRT_", "XLA_", "LIBTPU", "CLOUD_TPU"))}


class OnDevice:
    """A loader whose batches arrive on the device, all of a batch's tensors in one transfer. Its filler thread stays
    on the CPU, since XLA wants transfers made on the thread that runs the step. `spent` adds up the seconds every
    loader took, for the log."""
    spent = 0.0

    def __init__(self, inner, dev):
        import torch_xla.core.xla_model as xm
        self.inner, self.dev, self.send = inner, dev, xm.send_cpu_data_to_device

    def next(self):
        t = time.time()
        out = tuple(self.send(list(self.inner.next()), self.dev))
        OnDevice.spent += time.time() - t
        return out

    def __len__(self):
        return len(self.inner)

    def close(self):
        self.inner.close()


def run(index=0):
    import numpy as np
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    spmd = MODE == "spmd"
    if spmd:
        import torch_xla.distributed.spmd as xs
        xr.use_spmd()                             # before anything touches the device
    single = MODE in ("single", "spmd")           # one process, so no other replica to average with or wait for
    rank, world = (0, 1) if single else (xr.global_ordinal(), xr.world_size())
    dev = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
    master = rank == 0
    ndev = xr.global_runtime_device_count() if spmd else world
    import train_av as T
    from chess_net import ChessNet
    from data_av import AVLoader
    from data_packed import PackedLoader

    def say(msg):
        if master:
            print(time.strftime("%H:%M:%S ") + msg, flush=True)

    # A compile cache on the session's disk (kaggle_train.py sets it), so a trainer restarted in the same session
    # loads its graphs instead of compiling them again. Set before the first computation, as torch_xla requires.
    if E("XLA_CACHE_DIR"):
        try:
            xr.initialize_cache(os.path.join(E("XLA_CACHE_DIR"), f"{MODE}_r{rank}"), readonly=False)
        except Exception as e:
            say(f"no compile cache ({type(e).__name__}: {e})")
    # A sum over the replicas before anything else compiles, so a start that cannot work fails in seconds and
    # kaggle_train.py moves on, and a slow compile is not mistaken for it
    t_init = time.time()
    one = torch.ones(1, device=dev)
    if not single:
        one = xm.all_reduce(xm.REDUCE_SUM, one)
    got = float(one.sum())
    if not single and round(got) != world:
        raise SystemExit(f"replica {rank}: the sum over {world} replicas came to {got}")
    if master:
        with open(os.path.join(T.CKPT_DIR, "xla_started"), "w") as fh:
            json.dump({"mode": MODE, "world": world, "devices": ndev, "seconds": round(time.time() - t_init, 1),
                       "tpu_settings": tpu_settings()}, fh)
    say(f"{MODE}: {ndev} device(s) answer, {world} replica(s); settings {tpu_settings()}")

    # train_av's step reads autocast_ctx from its own module, so swap in bfloat16 on XLA instead of CUDA
    T.autocast_ctx = lambda mode=None: torch.autocast("xla", dtype=torch.bfloat16)
    # train_av.centres caches one tensor and remakes it when the device changes. Where XLA runs replicas as threads
    # of one process (its CPU backend) they would take each other's, so keep one cached tensor per device.
    _cent = {}

    def centres(device):
        k = str(device)
        if k not in _cent:
            from chess_net import bin_centres
            _cent[k] = bin_centres(T.DIST, device)
        return _cent[k]

    T.centres = centres

    # The moves-left loss without F.huber_loss. torch_xla has no huber_loss of its own, so it ran on the CPU and
    # split every step in two. This is train_av's formula (delta 0.1) from ops XLA has, same values and gradients.
    def mleft_loss(pred, plies, valid):
        w = valid.float()
        n = w.sum().clamp(min=1.0)
        err = pred.float() - plies / 100.0
        a = err.abs()
        hub = torch.where(a < 0.1, 0.5 * err * err, 0.1 * (a - 0.05))
        return (hub * w).sum() / n, (a * w).sum() / n * 100.0

    T.mleft_loss = mleft_loss
    budget = float(E("XLA_SECONDS", 3600))
    t_start = time.time()
    batch_x = int(E("XLA_BATCH_X", 1)) * (ndev if spmd else 1)   # spmd carries every chip's batch in one process
    lr_scale = float(E("XLA_LR_SCALE") or math.sqrt(world * batch_x))
    rig_sph = float(E("RIG_POS_PER_HOUR", 972000))
    per = T.AV_BSZ + T.POL_BSZ
    av_bsz, pol_bsz = T.AV_BSZ * batch_x, T.POL_BSZ * batch_x

    ck = torch.load(T.LAST, map_location="cpu", weights_only=False)
    data_seed = int(E("DATA_SEED") or int.from_bytes(os.urandom(4), "little")) + 1000 * rank
    S = T.SHARDS
    rows = lambda share, total: int(round(total * share))  # noqa: E731
    dp_rows = rows(T.DP_SHARE, av_bsz) if T.DP_SHARE and os.listdir(S) and any(
        f.startswith("dp_") for f in os.listdir(S)) else 0
    lv_rows = rows(T.LV_SHARE, av_bsz) if any(f.startswith("lv_") for f in os.listdir(S)) else 0
    dh_rows = rows(T.DH_SHARE, av_bsz) if any(f.startswith("dh_") for f in os.listdir(S)) else 0
    mk = lambda **kw: OnDevice(AVLoader(S, split="train", device="cpu", refresh=0, **kw), dev)  # noqa: E731
    av = mk(batch=av_bsz - dp_rows - lv_rows - dh_rows, seed=data_seed)
    dp = (mk(batch=dp_rows, seed=data_seed + 2, prefix="dp", thin=T.DP_THIN,
             down_keys=T.DP_DOWN_KEYS or None, down_weight=T.DP_DOWN_WEIGHT) if dp_rows else None)
    lv = mk(batch=lv_rows, seed=data_seed + 4, prefix="lv", thin=T.LV_THIN) if lv_rows else None
    dh = mk(batch=dh_rows, seed=data_seed + 6, prefix="dh", thin=T.DH_THIN) if dh_rows else None
    mlh_rows = int(round(pol_bsz * T.MLH_SHARE)) if T.MLH else 0
    pol = OnDevice(PackedLoader(S, batch=pol_bsz - mlh_rows, split="train", device="cpu", seed=data_seed + 3,
                                sources=T.SOURCES, refresh=0, legal=T.MASK_LEGAL, mleft=T.MLH), dev)
    mlh = (OnDevice(PackedLoader(S, batch=mlh_rows, split="train", device="cpu", seed=data_seed + 5,
                                 sources=("lz",), refresh=0, legal=T.MASK_LEGAL, mleft=True, need_flag=8), dev)
           if mlh_rows else None)
    say(f"{MODE}, replica {rank} of {world}: a step is {world} x {av_bsz} action values (route 1 {dp_rows}, Lichess "
        f"{lv_rows}, deep {dh_rows}) + {world} x {pol_bsz} positions (moves left {mlh_rows}); "
        f"{av.inner.describe()}; {pol.inner.describe()}")

    model = ChessNet(d=T.D, nhead=T.NHEAD, layers=T.LAYERS, dim_ff=int(T.FF_MULT * T.D), qhead=True,
                     dropout=T.DROPOUT, att=T.ATT, dist=T.DIST, mlh=T.MLH)
    model.load_state_dict(ck["model"], strict=True)
    model = model.to(dev)
    if spmd:
        # The batch sharded over the chips. The model's four inputs split along the batch and the weights are
        # replicated, and XLA partitions the step and sums the gradients itself.
        mesh = xs.Mesh(np.arange(ndev), (ndev,), ("data",))

        def shard_batch(module, args):
            for a in args:
                if torch.is_tensor(a) and a.dim() and a.shape[0] % ndev == 0:
                    xs.mark_sharding(a, mesh, ("data",) + (None,) * (a.dim() - 1))
        model.register_forward_pre_hook(shard_batch)
    is_att = lambda n: n.split(".")[0] in ("ln_sq", "att", "att_q")  # noqa: E731
    is_dist = lambda n: n.split(".")[0] in ("qdist", "vdist")  # noqa: E731
    is_mlh = lambda n: n.split(".")[0] == "mleft"  # noqa: E731
    named = [(n, p) for n, p in model.named_parameters() if not (is_att(n) or is_dist(n) or is_mlh(n))]
    from torch_xla.amp import syncfree
    opt = syncfree.AdamW([{"params": [p for _, p in named if p.ndim > 1], "weight_decay": 0.05},
                          {"params": [p for _, p in named if p.ndim <= 1], "weight_decay": 0.0}],
                         lr=T.LR, betas=(0.9, T.BETA2))
    for on, pred in ((T.ATT, is_att), (T.DIST, is_dist), (T.MLH, is_mlh)):
        if on:   # train_av's fixed order, which its checkpoint's groups follow
            opt.add_param_group({"params": [p for n, p in model.named_parameters() if pred(n)],
                                 "weight_decay": 0.05})
    opt.load_state_dict(ck["optim"])
    # Capturable on XLA, with the GPU trainer's own flags written back on save. Its groups say fused=True (CUDA only),
    # which puts each step count on the device, and unfused AdamW would then read each count back to the host for
    # its bias correction, a sync per parameter per step. Capturable keeps that on the device. The sync-free AdamW
    # also fuses each parameter's update. It wants an empty max_exp_avg_sq beside each moment, which save() drops.
    rig_flags = [{k: g.get(k) for k in ("fused", "capturable", "foreach")} for g in ck["optim"]["param_groups"]]
    for g in opt.param_groups:
        g["betas"] = (g["betas"][0], T.BETA2)
        g["fused"], g["foreach"], g["capturable"] = None, False, True
    for p, s in opt.state.items():
        if torch.is_tensor(s.get("step")):
            s["step"] = s["step"].to(device=p.device, dtype=torch.float32)
        s["max_exp_avg_sq"] = torch.empty(0, dtype=torch.float32, device=p.device)
    it = int(ck["iter"]) + 1
    hist, best, trained = ck.get("hist", []), ck.get("best", 1e9), float(ck.get("trained_hours", 0.0))
    att_since, dist_since, mlh_since = ck.get("att_since"), ck.get("dist_since"), ck.get("mlh_since")
    yard = ck.get("yardstick", T.YARDSTICK)
    ck = None
    T.recentre_policy(model)
    say(f"RESUMED the main run at iter {it:,} ({trained:.1f}h); rate x{lr_scale:.3g} "
        f"({T.lr_at(it, trained) * lr_scale:.3g}); {budget / 3600:.2f}h budget")

    def cpu(x):
        """A state with every tensor copied to the CPU (what xm.save does before torch.save)."""
        if isinstance(x, dict):
            return {k: cpu(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(cpu(v) for v in x)
        return x.detach().cpu() if hasattr(x, "detach") else x

    def save(final=False, path=None, extra=None):
        path = path or T.LAST
        osd = opt.state_dict()
        for g, f in zip(osd["param_groups"], rig_flags):
            g.update(f)
        # a copy of each parameter's state without the sync-free optimiser's extra (the live dicts are the optimiser's)
        osd["state"] = {i: {n: v for n, v in s.items() if n != "max_exp_avg_sq"} for i, s in osd["state"].items()}
        state = {"model": model.state_dict(), "optim": osd, "iter": it - 1, "hist": hist,
                 "best": best, "trained_hours": trained + samples / rig_sph, "att_since": att_since,
                 "dist_since": dist_since, "mlh_since": mlh_since, "yardstick": yard, "data_seed": data_seed,
                 "scaler": None, "xla": {"sessions": 1, "samples": samples, "world": world, "lr_scale": lr_scale,
                                         "mode": MODE, "devices": ndev, "final": final, **(extra or {})}}
        tmp = path + ".part"
        if single:
            torch.save(cpu(state), tmp)
            os.replace(tmp, path)
            return
        xm.save(state, tmp)                       # every replica calls it, the master writes
        if master:
            os.replace(tmp, path)
        xm.rendezvous("saved")

    # A cooled copy at the end. With XLA_COOL_MIN the main run trains until that many minutes are left and its resume
    # checkpoint is saved uncooled. The remaining time trains a copy whose rate falls by train_av's half cosine
    # (cool_factor, to COOL_FLOOR) over the steps the measured pace fits in, saved as ckpt_cooled_last.pt, the same
    # cooled copy cooldown.py makes. It is exported as the next candidate.
    cool_s = float(E("XLA_COOL_MIN", 0)) * 60
    cooled_path = os.path.join(T.CKPT_DIR, "ckpt_cooled_last.pt")
    phase = "main"

    step_it = world * batch_x
    samples, skipped, gnorm = 0, 0, 1.0
    t0, last_log, last_beat = time.time(), time.time(), 0.0
    log_every = int(E("XLA_LOG_EVERY", 200))
    save_every = float(E("XLA_SAVE_MIN", 30)) * 60
    sync_every = int(E("XLA_SYNC_EVERY", 25))
    proof_at = int(E("XLA_PROOF_STEP", 40))
    min_sps, paced, slow = float(E("XLA_MIN_SPS", 500)), False, None
    max_steps = int(E("XLA_MAX_STEPS", 0))

    def agreed(x):
        """The largest of every replica's x, the one number they all decide from. A collective, so it is only ever
        called at a step count, never because of a clock."""
        if single:
            return x
        return float(xm.all_reduce(xm.REDUCE_MAX, torch.tensor([x], device=dev)).sum())

    def counters():
        """Graphs compiled and graphs run so far. A compile count that keeps growing means a shape changes every
        step, and runs well above one a step are syncs the step should not have."""
        try:
            import torch_xla.debug.metrics as met
            c = met.metric_data("CompileTime")
            e = met.metric_data("ExecuteTime") or met.metric_data("ExecuteReplicatedTime")
            return (c[0] if c else 0), (e[0] if e else 0)
        except Exception:
            return 0, 0

    def fallbacks():
        """Ops torch_xla ran on the CPU instead (counters named aten::...). Each one stops the step to read its inputs
        back. The deliberate reads (_local_scalar_dense, for the log line and the agreed clock) are left out."""
        try:
            import torch_xla.debug.metrics as met
            return {n: met.counter_value(n) for n in met.counter_names()
                    if n.startswith("aten::") and n != "aten::_local_scalar_dense"}
        except Exception:
            return {}
    gnorm_t = torch.tensor(1.0, device=dev)       # the norm's running mean and the skips stay on the chip
    skipped_t = torch.zeros((), device=dev)
    model.train()
    k, el, last_save_el, saved_k = 0, 0.0, 0.0, -1
    k_r0 = el_r0 = None
    tr_s = wt_s = 0.0                             # host seconds tracing, and waiting in mark_step, since the last log
    tot = {"trace": 0.0, "wait": 0.0, "data": 0.0, "steps": 0}   # the same past the compile, for the summary
    OnDevice.spent = 0.0
    while True:
        if max_steps and k >= max_steps:          # a test's fixed length, so two modes can be compared step for step
            break
        if phase == "cool" and it >= T.COOL_FROM + T.COOL_STEPS:
            break
        ta, d0 = time.time(), OnDevice.spent
        for g in opt.param_groups:
            g["lr"] = T.lr_at(it, trained) * lr_scale
        if T.ATT:
            model.att_detach = (it - att_since) < T.ATT_DETACH
        if T.DIST:
            model.dist_detach = (it - dist_since) < T.DIST_DETACH
        if T.MLH:
            model.mlh_detach = (it - mlh_since) < T.MLH_DETACH
        loss, _ = T.step_batch(model, "bf16", av, pol, dp=dp, mlh=mlh, lv=lv, dh=dh)
        opt.zero_grad()
        loss.backward()
        if not single:
            xm.reduce_gradients(opt)              # the mean over replicas, one step of world x the batch
        # train_av's skip rule on the chip, no update when the norm is not finite or over 50 (every replica sees the
        # same norm), and the step never waits for the host to read it back
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        bad = ~torch.isfinite(gn) | (gn > 50.0)
        opt.step(found_inf=bad.float())
        T.recentre_policy(model)
        gnorm_t = torch.where(bad, gnorm_t, 0.98 * gnorm_t + 0.02 * gn)
        skipped_t = skipped_t + bad.float()
        tb = time.time()
        xm.mark_step()
        tc = time.time()
        tr_s, wt_s = tr_s + (tb - ta), wt_s + (tc - tb)
        it += step_it
        samples += step_it * per
        k += 1
        if k > 20:
            tot["trace"] += tb - ta
            tot["wait"] += tc - tb
            tot["data"] += OnDevice.spent - d0
            tot["steps"] += 1
        if k == 1:
            say(f"  first step done {time.time() - t0:.0f}s after the start (the compile)")
        if k == 3 and fallbacks():
            say(f"  ops run on the CPU, each a stop in every step: {fallbacks()}")
        if master and (k <= 3 or time.time() - last_beat > 30):
            # The heartbeat kaggle_train.py watches. A first step that never comes, or steps that stop, end the session
            # instead of leaving the TPU idle. Written by the clock, so a slow stretch (a graph compiling) still beats.
            with open(os.path.join(T.CKPT_DIR, "xla_heartbeat"), "w") as fh:
                fh.write(f"{it} {k} {time.time():.0f}\n")
            last_beat = time.time()
        if k % log_every == 0:
            now = time.time()
            nc, ne = counters()
            gnorm, skipped = float(gnorm_t), int(float(skipped_t))   # read here only
            say(f"  it {it:,}  loss {float(loss.detach()):.4f}  grad {gnorm:.2f}  lr {opt.param_groups[0]['lr']:.3g}  "
                f"{log_every * step_it * per / (now - last_log):,.0f} samples/s  "
                f"{samples / rig_sph:.1f} GPU-trainer hours this session  (a step: {1000 * tr_s / log_every:.0f} ms on the "
                f"host, {1000 * OnDevice.spent / log_every:.0f} of it data, then {1000 * wt_s / log_every:.0f} ms "
                f"waiting for the chip; {nc} graphs compiled, {ne / k:.1f} runs a step"
                + (f"; {skipped} skipped" if skipped else "") + ")")
            last_log, tr_s, wt_s, OnDevice.spent = now, 0.0, 0.0, 0.0
        # Every decision about time is made here and only here (see "one clock" above), at the same step count on every
        # replica and from the elapsed time they agree on, so they all save, switch to the cooled copy and stop together
        if k % sync_every == 0:
            el = agreed(time.time() - t_start)
            if el_r0 is None and k >= 20:
                k_r0, el_r0 = k, el               # the pace measured past the compile
            if el >= budget:
                break
            if phase == "main" and cool_s and el >= budget - cool_s:
                save(final=True)                  # the main run's resume point, as it stands
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
            # Slower than the GPU trainer, stop, since that TPU quota is wasted. Checked once, half an hour past the
            # compile.
            if not paced and el_r0 is not None and phase == "main" and min_sps and el - el_r0 > 1800:
                paced = True
                sps = (k - k_r0) * step_it * per / (el - el_r0)
                nc, ne = counters()
                say(f"  pace past the compile: {sps:,.0f} samples/s ({nc} graphs compiled, {ne / k:.1f} runs a step)")
                if sps < min_sps:
                    slow = round(sps)
                    say(f"  SLOWER THAN THE GPU TRAINER ({slow:,} < {min_sps:,.0f} samples/s): saving and stopping, so the TPU "
                        f"quota is not spent on it")
                    break
        # the first save early, so a save that cannot work shows minutes in, not half an hour
        if k == proof_at and phase == "main" and saved_k != k:
            save()
            saved_k, last_save_el = k, el
            say(f"  first save done at step {k} (it {it - 1:,}): every replica saves")
    skipped = int(float(skipped_t))
    if phase == "cool":
        save(final=True, path=cooled_path, extra={"cool_from": T.COOL_FROM, "cool_steps": T.COOL_STEPS,
                                                  "cool_floor": T.COOL_FLOOR})
        say(f"  cooled copy saved at it {it - 1:,} ({os.path.basename(cooled_path)})")
    else:
        save(final=True)
    took = time.time() - t0
    say(f"DONE: {samples:,} samples ({k:,} steps, {skipped} skipped) in {took / 3600:.2f}h, "
        f"{samples / max(took, 1):,.0f} samples/s, {samples / rig_sph:.1f} GPU-trainer hours; ends at it {it - 1:,}")
    if master:
        try:
            import torch_xla.debug.metrics as met
            print(met.short_metrics_report(), flush=True)
        except Exception:
            pass
        nc, ne = counters()
        with open(os.path.join(T.CKPT_DIR, "xla_summary.json"), "w") as fh:
            json.dump({"samples": samples, "steps": k, "skipped": skipped, "seconds": round(took),
                       "samples_per_s": round(samples / max(took, 1)), "rig_hours": round(samples / rig_sph, 2),
                       "iter_end": it - 1, "mode": MODE, "world": world, "devices": ndev, "lr_scale": lr_scale,
                       "batch_x": batch_x, "compiles": nc, "runs_per_step": round(ne / max(k, 1), 2),
                       "stopped_slow": slow, "ms_per_step": {n: round(1000 * tot[n] / max(tot["steps"], 1), 1)
                                                             for n in ("trace", "data", "wait")},
                       "cpu_fallbacks": fallbacks(),
                       "cooled": phase == "cool", "cool_from": T.COOL_FROM if phase == "cool" else None,
                       "cool_steps": T.COOL_STEPS if phase == "cool" else None}, fh, indent=1)
    for x in (av, dp, lv, dh, pol, mlh):
        if x is not None:
            x.close()
    if single:
        # Everything is written, so exit without the runtime's teardown, which can abort at exit ("terminate called
        # without an active exception"). Not under xmp.spawn, whose pool waits for each replica to return.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    if MODE == "procs":
        was = tpu_settings()
        taken, moved = procs_env()
        print(f"procs: the TPU settings were {was}; taken out for a process a chip {taken}; moved to the host's names "
              f"{moved}", flush=True)
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(run, args=())
    else:
        run(0)
