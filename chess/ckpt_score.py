"""Score checkpoints on the same frozen held-out batches, as arms_search.py does, so they can be compared.

kaggle_train_ctl.py uses this to score a TPU-trained checkpoint against the local one before swapping
them in, since a run at a much larger batch can train and still make the network worse.

    python chess/ckpt_score.py OUT.json CKPT [CKPT ...]
    env SCORE_BATCHES=60, SCORE_DIR, SCORE_DEVICE=cuda (cpu to test beside a busy GPU)

Writes {path: {iter, q, policy_legal, top1_legal, top3_legal, dp_q, lv_q, dh_q, ...}}.
"""
import json
import os
import shutil
import sys

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
CK = paths.CKPT
SCORE_DIR = paths.under("kaggle", "score")


def main(out, paths):
    env = json.load(open(os.path.join(HERE, "autostart.json"), encoding="utf-8-sig"))["env"]
    os.environ.update({k: str(v) for k, v in env.items()})
    ev = E("SCORE_DIR", SCORE_DIR)
    os.makedirs(ev, exist_ok=True)
    os.environ.update(TAG="score", CKPT_DIR=ev, VERSION_DIR=ev, COMPILE="0")
    sys.path.insert(0, HERE)
    import torch
    import train_av as T
    from data_av import AVLoader
    from data_packed import PackedLoader
    n = int(E("SCORE_BATCHES", 60))
    dev = E("SCORE_DEVICE", "cuda")
    shutil.copy(os.path.join(CK, "val_policy_mayfly.npy"), os.path.join(ev, "val_policy_score.npy"))
    S = T.SHARDS
    have = {f.split("_")[0] for f in os.listdir(S) if f.endswith(".bin")}
    sets = {p: AVLoader(S, batch=T.AV_BSZ, split="val", device=dev, seed=T.SEED + 1, refresh=0, prefix=p)
            for p in ("av", "lv", "dh", "dp") if p in have}
    polv = PackedLoader(S, batch=T.POL_BSZ, split="val", device=dev, seed=T.SEED + 1, sources=T.SOURCES,
                        refresh=0, legal=True, val_cache=os.path.join(ev, "val_policy_score.npy"))
    frozen = {k: [v.next() for _ in range(n)] for k, v in sets.items()}
    P = [polv.next() for _ in range(n)]

    class Replay:
        def __init__(self, xs):
            self.xs, self.i = xs, 0

        def next(self):
            x = self.xs[self.i % len(self.xs)]
            self.i += 1
            return x

    res = {}
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)   # only the weights go to the GPU
        model = T.ChessNet(d=T.D, nhead=T.NHEAD, layers=T.LAYERS, dim_ff=int(T.FF_MULT * T.D), qhead=True,
                           dropout=T.DROPOUT, att=T.ATT, dist=T.DIST, mlh=T.MLH)
        model.load_state_dict(ck["model"], strict=False)
        model = model.to(dev)
        it = int(ck.get("iter", 0))
        del ck
        model, used = T.apply_precision(model, "bf16")
        R = lambda k: Replay(frozen[k]) if k in frozen else None  # noqa: E731
        with torch.no_grad():
            v = T.validate(model, used, Replay(frozen["av"]), Replay(P), n, lvv=R("lv"), fsv=R("dp"), dhv=R("dh"))
        if "fs_q" in v:
            v["dp_q"], v["dp_mae"] = v.pop("fs_q"), v.pop("fs_mae")   # route 1's held-out rows ran in the fs slot
        res[path] = {"iter": it, **{k: round(float(x), 5) for k, x in v.items()}}
        print(path, json.dumps(res[path]), flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    for x in list(sets.values()) + [polv]:
        try:
            x.close()
        except Exception:
            pass
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
