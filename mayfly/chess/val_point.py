"""val_point adds one point to the training curve for a checkpoint trained somewhere else.

The TPU trainer (train_jax.py) runs no validation, so the curve (hist_<TAG>.json, curve.py) would stop where the GPU
trainer last validated. This runs train_av's own
validate() on the checkpoint, on the same held-out sets the trainer uses (the same seeds and the val_*.npy caches in
CKPT_DIR), in the same precision, and appends the result to the history with the checkpoint's trained hours, which
curve.py reads for a point like this. An iteration already in the history is not added again.

  python val_point.py [CKPT]        default the main run's ckpt_<TAG>_last.pt
  env: the main run's own (read from autostart.json), DRY=1 prints the point without writing it, VAL_ROUNDS=8
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main_env():
    """The main run's settings, as autostart.json starts it, so the loaders and the model match the trainer's."""
    p = os.path.join(HERE, "autostart.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8-sig"))
        for k, v in d.get("env", d).items():
            if isinstance(v, (str, int, float)):
                os.environ.setdefault(k, str(v))


def main():
    main_env()
    import glob
    import torch
    import train_av as T
    from chess_net import ChessNet
    from data_av import AVLoader
    from data_packed import PackedLoader
    from precision import apply_precision

    ckpt = sys.argv[1] if len(sys.argv) > 1 else T.LAST
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    it, hours = int(ck.get("iter", 0)), float(ck.get("trained_hours", 0.0))
    hist = json.load(open(T.HIST)) if os.path.exists(T.HIST) else []
    if any(r.get("it") == it for r in hist) and os.environ.get("DRY") != "1":     # DRY=1 rescores it anyway
        print(f"iteration {it:,} is already on the curve; nothing to add", flush=True)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S, B = T.SHARDS, T.AV_BSZ
    has = lambda p: bool(glob.glob(os.path.join(S, p + "_*.bin")))  # noqa: E731
    # the trainer's validation loaders (train_av.main), only the ones its shares switch on
    val = lambda **kw: AVLoader(S, batch=B, split="val", device=device, seed=T.SEED + 1, refresh=600, **kw)  # noqa: E731
    lvv = val(prefix="lv") if T.LV_SHARE and has("lv") else None
    fsv = val(prefix="fs") if T.FS_SHARE and has("fs") else None
    dhv = val(prefix="dh") if T.DH_SHARE and has("dh") else None
    stv = val(prefix="st") if T.ST_SHARE and has("st") else None
    avv = val()
    mlv = None
    if T.MLH:
        since = time.mktime(time.strptime(T.MLH_SINCE, "%Y-%m-%d %H:%M"))
        mlv = PackedLoader(S, batch=T.POL_BSZ, split="val", device=device, seed=T.SEED + 2, sources=("lz",),
                           refresh=0, legal=True, mleft=True, need_flag=8, min_mtime=since, val_size=8192,
                           val_cache=os.path.join(T.CKPT_DIR, f"val_mleft_{T.TAG}.npy"))
    polv = PackedLoader(S, batch=T.POL_BSZ, split="val", device=device, seed=T.SEED + 1, sources=T.SOURCES,
                        refresh=600, legal=True, val_cache=os.path.join(T.CKPT_DIR, f"val_policy_{T.TAG}.npy"))
    model = ChessNet(d=T.D, nhead=T.NHEAD, layers=T.LAYERS, dim_ff=int(T.FF_MULT * T.D), qhead=True, dropout=0.0,
                     att=T.ATT, dist=T.DIST, mlh=T.MLH).to(device)
    model.load_state_dict(ck["model"])
    model, used = apply_precision(model)
    used_val = "bf16" if used == "fp16acc" else used      # the trainer validates in bf16 whatever it trains in
    t0 = time.time()
    # Several samples, not one. validate() scores VAL_BATCHES random batches of the frozen held-out set, and a fresh
    # process always draws the same first sample, which reads about 0.03 easier on policy than the run of samples a
    # trainer moves through, so every such point sat low. VAL_ROUNDS samples, averaged, read like the trainer's
    # curve does over time.
    rounds = max(1, int(os.environ.get("VAL_ROUNDS", 8)))
    vs = [T.validate(model, used_val, avv, polv, T.VAL_BATCHES, mlv=mlv, lvv=lvv, fsv=fsv, dhv=dhv, stv=stv)
          for _ in range(rounds)]
    v = {k: sum(x[k] for x in vs) / len(vs) for k in vs[0] if all(k in x for x in vs)}
    row = {"it": it, **v, "t": 0.0, "hours": round(hours, 3), "src": os.path.basename(os.path.dirname(ckpt)) or "ckpt",
           "skipped": 0}
    print(f"iteration {it:,} at {hours:.1f} h: q {v['q']:.4f}  policy (legal) {v.get('policy_legal', 0):.4f}  "
          f"top-1 (legal) {100 * v.get('top1_legal', 0):.1f}%  top-3 (legal) {100 * v.get('top3_legal', 0):.1f}%  "
          f"({time.time() - t0:.0f} s)", flush=True)
    if os.environ.get("DRY") == "1":
        return
    hist.append(row)
    tmp = T.HIST + ".tmp"
    json.dump(hist, open(tmp, "w"), indent=1)
    os.replace(tmp, T.HIST)
    print(f"added to {T.HIST}", flush=True)


if __name__ == "__main__":
    main()
