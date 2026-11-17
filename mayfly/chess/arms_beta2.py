"""Test Adam's beta2 with three short runs from the same paused checkpoint and data stream.

Run while the main training is paused. Each arm is train_av.py with a fixed DATA_SEED, STEPS steps each.
    a  beta2 0.95, the main run's setting and the control
    b  beta2 0.999, a second-moment half-life of about 690 steps
    c  beta2 0.9995 at twice the learning rate
Each arm has its own folder with the main run's validation caches copied in and its own copy of
versions.json, so it can never freeze a version. The main checkpoint is only read. At the end all three
final checkpoints are scored on the same frozen validation batches, written to beta2_result.txt.

A finished arm is skipped and one that died resumes from its last save.
    python chess/arms_beta2.py            env ARM_STEPS=100000, ARM_EVAL_BATCHES=120
"""
import json
import os
import shutil
import subprocess
import sys
import time

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
CK = paths.CKPT
ROOT = paths.under("arms")
STEPS = int(E("ARM_STEPS", 100000))
EVAL_BATCHES = int(E("ARM_EVAL_BATCHES", 120))
SEED = 20260925
ARMS = [("a_b095", {"BETA2": "0.95"}),
        ("b_b0999", {"BETA2": "0.999"}),
        # the effective rate is LR halved per version owed (lr_at), so 2.4e-4 doubles the control's 3e-5
        ("c_b09995_lr2", {"BETA2": "0.9995", "LR": "2.4e-4"})]
STATUS = os.path.join(ROOT, "status.json")
CAP = os.path.join(ROOT, "shard_cap.json")
LOG = os.path.join(ROOT, "arms.log")


def say(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + m
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def trainers():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
                          "Where-Object { $_.CommandLine -match 'train_av\\.py' } | "
                          "ForEach-Object { $_.ProcessId }"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def main_env():
    return json.load(open(os.path.join(HERE, "autostart.json"), encoding="utf-8-sig"))["env"]


def load_status():
    return json.load(open(STATUS, encoding="utf-8")) if os.path.exists(STATUS) else {}


def save_status(st):
    json.dump(st, open(STATUS, "w", encoding="utf-8"), indent=1)


def write_cap():
    """Record every shard and its row count when the first arm starts, passed to every arm as SHARD_CAP.

    The corpus keeps growing while arms run, and a new or longer shard would change every draw. An
    existing file is kept, so a resumed run reads the corpus its first arm read."""
    if os.path.exists(CAP):
        return
    import glob
    shards = main_env().get("SHARDS", paths.SHARDS)
    rows = {}
    for p in glob.glob(os.path.join(shards, "*.bin")):
        b = os.path.basename(p)
        rec = 40 if b.split("_")[0] in ("av", "dp") else 112     # pack_av.REC_AV, pack.REC
        rows[b] = os.path.getsize(p) // rec
    tmp = CAP + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=0)
    os.replace(tmp, CAP)
    say(f"corpus frozen for the arms: {len(rows):,} shards, {sum(rows.values()):,} rows, {CAP}")


def run_arm(name, extra, st):
    d = os.path.join(ROOT, name)
    tag = f"arm_{name}"
    vdir = os.path.join(d, "versions")
    os.makedirs(vdir, exist_ok=True)
    last = os.path.join(d, f"ckpt_{tag}_last.pt")
    if not os.path.exists(last):
        shutil.copy(os.path.join(CK, "ckpt_mayfly_last.pt"), last)
        for src, dst in (("val_policy_mayfly.npy", f"val_policy_{tag}.npy"),
                         ("val_mleft_mayfly.npy", f"val_mleft_{tag}.npy")):
            if os.path.exists(os.path.join(CK, src)):
                shutil.copy(os.path.join(CK, src), os.path.join(d, dst))
        shutil.copy(os.path.join(paths.VERSIONS, "versions.json"), os.path.join(vdir, "versions.json"))
    env = dict(os.environ, **main_env())
    env.update(TAG=tag, CKPT_DIR=d, VERSION_DIR=vdir, RESUME="1", ITERS=str(st["iters"]),
               DATA_SEED=str(SEED), PLAYING_FILE=os.path.join(CK, "playing.flag"),
               PYTHONUNBUFFERED="1", **extra)
    if os.path.exists(CAP):
        env["SHARD_CAP"] = CAP             # the corpus as the first arm found it, see write_cap
    say(f"arm {name}: {extra}, to iteration {st['iters']:,}")
    with open(os.path.join(d, f"train_{tag}.log"), "a", encoding="utf-8") as fh:
        rc = subprocess.call([sys.executable, os.path.join(HERE, "train_av.py")], cwd=HERE, env=env,
                             stdout=fh, stderr=subprocess.STDOUT)
    say(f"arm {name}: train_av exited {rc}")
    return rc == 0


def common_eval(st):
    """All three final checkpoints on the same frozen validation batches."""
    env = main_env()
    os.environ.update(env)
    os.environ.update(TAG="armeval", CKPT_DIR=os.path.join(ROOT, "_eval"), VERSION_DIR=os.path.join(ROOT, "_eval"))
    os.makedirs(os.path.join(ROOT, "_eval"), exist_ok=True)
    sys.path.insert(0, HERE)
    import torch
    import train_av as T
    from data_av import AVLoader
    from data_packed import PackedLoader
    shutil.copy(os.path.join(CK, "val_policy_mayfly.npy"), os.path.join(ROOT, "_eval", "val_policy_armeval.npy"))
    avv = AVLoader(T.SHARDS, batch=T.AV_BSZ, split="val", device="cuda", seed=T.SEED + 1, refresh=0)
    polv = PackedLoader(T.SHARDS, batch=T.POL_BSZ, split="val", device="cuda", seed=T.SEED + 1,
                        sources=T.SOURCES, refresh=0, legal=True,
                        val_cache=os.path.join(ROOT, "_eval", "val_policy_armeval.npy"))
    A = [avv.next() for _ in range(EVAL_BATCHES)]
    P = [polv.next() for _ in range(EVAL_BATCHES)]

    class Replay:
        def __init__(self, xs):
            self.xs, self.i = xs, 0

        def next(self):
            x = self.xs[self.i % len(self.xs)]
            self.i += 1
            return x

    rows = []
    for name, extra in ARMS:
        tag = f"arm_{name}"
        path = os.path.join(ROOT, name, f"ckpt_{tag}_last.pt")
        ck = torch.load(path, map_location="cuda")
        model = T.ChessNet(d=T.D, nhead=T.NHEAD, layers=T.LAYERS, dim_ff=int(T.FF_MULT * T.D), qhead=True,
                           dropout=T.DROPOUT, att=T.ATT, dist=T.DIST, mlh=T.MLH).to("cuda")
        model.load_state_dict(ck["model"], strict=False)
        it = int(ck.get("iter", 0))
        del ck
        model, used = T.apply_precision(model)
        with torch.no_grad():
            v = T.validate(model, used, Replay(A), Replay(P), EVAL_BATCHES)
        rows.append((name, extra, it, v))
        del model
        torch.cuda.empty_cache()
    keys = [k for k in ("policy_legal", "top1_legal", "top3_legal", "q", "mae", "att_policy_legal",
                        "att_top1_legal", "att_q") if k in rows[0][3]]
    out = [f"beta2 arms, {STEPS:,} steps each from iteration {st['start']:,} (the main run paused there), "
           f"DATA_SEED {SEED}, the same {EVAL_BATCHES} validation batches for all three",
           "  " + f"{'arm':<16}{'iter':>11}" + "".join(f"{k:>18}" for k in keys)]
    for name, extra, it, v in rows:
        out.append("  " + f"{name:<16}{it:>11,}" + "".join(f"{float(v[k]):>18.4f}" for k in keys))
    out.append("  lower is better for the losses (policy_legal, q, mae, att_policy_legal, att_q); "
               "higher for top1/top3.")
    open(os.path.join(ROOT, "beta2_result.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    for line in out:
        say(line)


def main():
    os.makedirs(ROOT, exist_ok=True)
    if not os.path.exists(os.path.join(HERE, "training.paused")):
        raise SystemExit("training is not paused (chess/training.paused is missing); the arms need the card")
    st = load_status()
    if "start" not in st:
        if trainers():
            raise SystemExit(f"a trainer is still running (pid {trainers()}); stop it after a save first")
        import torch
        ck = torch.load(os.path.join(CK, "ckpt_mayfly_last.pt"), map_location="cpu", weights_only=False)
        # end on a validation step, since train_av only saves at a multiple of VAL_EVERY or SAVE_EVERY
        # and the last steps of an arm would otherwise be trained and never saved
        ve = int(main_env().get("VAL_EVERY", 10000))
        st = {"start": int(ck["iter"]), "iters": -(-(int(ck["iter"]) + STEPS) // ve) * ve, "done": []}
        del ck
        save_status(st)
        say(f"arms from the main checkpoint at iteration {st['start']:,}, {STEPS:,} steps each")
        # frozen at a fresh start only, so a run begun without it resumes the way it began
        write_cap()
    for name, extra in ARMS:
        if name in st["done"]:
            continue
        if trainers():
            raise SystemExit(f"a trainer is running (pid {trainers()}); not starting arm {name} beside it")
        if run_arm(name, extra, st):
            st["done"].append(name)
            save_status(st)
        else:
            raise SystemExit(f"arm {name} failed; see {os.path.join(ROOT, name)}")
    if not st.get("evaluated"):
        common_eval(st)
        st["evaluated"] = True
        save_status(st)


if __name__ == "__main__":
    main()
