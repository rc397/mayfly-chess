"""Test whether training on MayFly's own search targets helps, with short runs as in arms_beta2.py.

Each arm starts from the same checkpoint on the same data stream (fixed DATA_SEED, corpus frozen by SHARD_CAP),
and every final checkpoint is scored on the same frozen validation batches.
    a_ctrl   the main recipe, the control
    b_st15   15% of the action-value rows from search targets (ST_SHARE, st_* shards from pack_search.py)
    c_st30   30%, to see the direction and size of the effect
The deciding score is dp_q, the action-value loss on held-out Stockfish labels of MayFly's self-play positions.
st_q is reported too, but the search arms win it by construction.

The main run is paused at its next save, as cooldown.py does, and resumed at the end. Each arm runs train_av.py
copied to _arm_trainer.py, so supervise.py and kaggle_train_ctl.py do not take it for the main trainer, in its
own folder with its own copy of versions.json.

A finished arm is skipped and one that died resumes from its last save.
    python chess/arms_search.py          env ARM_STEPS=20000, ARM_EVAL_BATCHES=120
"""
import glob
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
ROOT = paths.under("arms_search")
STEPS = int(E("ARM_STEPS", 20000))
EVAL_BATCHES = int(E("ARM_EVAL_BATCHES", 120))
SEED = 20261020
ARMS = [("a_ctrl", {}), ("b_st15", {"ST_SHARE": "0.15"}), ("c_st30", {"ST_SHARE": "0.30"})]
STATUS = os.path.join(ROOT, "status.json")
CAP = os.path.join(ROOT, "shard_cap.json")
LOG = os.path.join(ROOT, "arms.log")
PAUSED = os.path.join(HERE, "training.paused")
MARK = os.path.join(ROOT, "paused_by_arms.txt")
SCRIPT = os.path.join(HERE, "_arm_trainer.py")
REC40 = ("av", "dp", "lv", "dh", "fs", "st")       # pack_av.REC_AV records, the rest are pack.REC


def say(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + m
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def trainers(name="train_av.py"):
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
                          f"Where-Object {{ $_.CommandLine -match '{name.replace('.', chr(92) + '.')}' }} | "
                          "ForEach-Object { $_.ProcessId }"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def main_env():
    return json.load(open(os.path.join(HERE, "autostart.json"), encoding="utf-8-sig"))["env"]


def load_status():
    return json.load(open(STATUS, encoding="utf-8")) if os.path.exists(STATUS) else {}


def save_status(st):
    json.dump(st, open(STATUS, "w", encoding="utf-8"), indent=1)


def pause_main():
    """Pause the main run at its next save, as cooldown.py does. Does nothing if it is already paused."""
    if os.path.exists(PAUSED):
        if not os.path.exists(MARK):
            say("the main run is already paused by someone else; the arms run in that pause and leave it alone")
        return
    last = os.path.join(CK, "ckpt_mayfly_last.pt")
    t0 = time.time()
    with open(PAUSED, "w", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M") + " paused by arms_search.py; it lifts this itself at the end\n")
    with open(MARK, "w", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M") + "\n")
    say("main run: training.paused written; waiting for its next save")
    while time.time() - t0 < 4500 and trainers():
        if os.path.getmtime(last) > t0 + 1:
            time.sleep(20)                    # the rename has landed
            break
        time.sleep(15)
    for pid in trainers():
        subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL)
    say("main run: stopped after its save")


def resume_main():
    if os.path.exists(MARK):
        for p in (PAUSED, MARK):
            try:
                os.remove(p)
            except OSError:
                pass
        say("main run: pause lifted; supervise.py restarts it from its own saved state")


def write_cap():
    if os.path.exists(CAP):
        return
    shards = main_env().get("SHARDS", paths.SHARDS)
    rows = {}
    for p in glob.glob(os.path.join(shards, "*.bin")):
        b = os.path.basename(p)
        rows[b] = os.path.getsize(p) // (40 if b.split("_")[0] in REC40 else 112)
    json.dump(rows, open(CAP + ".tmp", "w", encoding="utf-8"), indent=0)
    os.replace(CAP + ".tmp", CAP)
    say(f"corpus frozen for the arms: {len(rows):,} shards, {sum(rows.values()):,} rows")


def run_arm(name, extra, st):
    d = os.path.join(ROOT, name)
    tag = f"arm_{name}"
    vdir = os.path.join(d, "versions")
    os.makedirs(vdir, exist_ok=True)
    last = os.path.join(d, f"ckpt_{tag}_last.pt")
    if not os.path.exists(last):
        shutil.copy(st["start_ckpt"], last)
        for src, dst in (("val_policy_mayfly.npy", f"val_policy_{tag}.npy"),
                         ("val_mleft_mayfly.npy", f"val_mleft_{tag}.npy")):
            if os.path.exists(os.path.join(CK, src)):
                shutil.copy(os.path.join(CK, src), os.path.join(d, dst))
        shutil.copy(os.path.join(paths.VERSIONS, "versions.json"), os.path.join(vdir, "versions.json"))
    env = dict(os.environ, **main_env())
    env.update(TAG=tag, CKPT_DIR=d, VERSION_DIR=vdir, RESUME="1", ITERS=str(st["iters"]), DATA_SEED=str(SEED),
               PLAYING_FILE=os.path.join(CK, "playing.flag"), ASIDE_FILE="", PYTHONUNBUFFERED="1",
               SHARD_CAP=CAP, **extra)
    shutil.copy(os.path.join(HERE, "train_av.py"), SCRIPT)
    say(f"arm {name}: {extra or 'the main recipe'}, from {st['start']:,} to {st['iters']:,}")
    with open(os.path.join(d, f"train_{tag}.log"), "a", encoding="utf-8") as fh:
        rc = subprocess.call([sys.executable, SCRIPT], cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
    say(f"arm {name}: exited {rc}")
    return rc == 0


def common_eval(st):
    """Every final checkpoint on the same frozen batches of every held-out set."""
    os.environ.update(main_env())
    ev = os.path.join(ROOT, "_eval")
    os.makedirs(ev, exist_ok=True)
    os.environ.update(TAG="armeval", CKPT_DIR=ev, VERSION_DIR=ev, SHARD_CAP=CAP)
    sys.path.insert(0, HERE)
    import torch
    import train_av as T
    from data_av import AVLoader
    from data_packed import PackedLoader
    shutil.copy(os.path.join(CK, "val_policy_mayfly.npy"), os.path.join(ev, "val_policy_armeval.npy"))
    S = T.SHARDS
    mk = lambda prefix: AVLoader(S, batch=T.AV_BSZ, split="val", device="cuda", seed=T.SEED + 1, refresh=0,  # noqa
                                 prefix=prefix)
    sets = {"av": mk("av"), "lv": mk("lv"), "dh": mk("dh"), "dp": mk("dp"), "st": mk("st")}
    polv = PackedLoader(S, batch=T.POL_BSZ, split="val", device="cuda", seed=T.SEED + 1, sources=T.SOURCES,
                        refresh=0, legal=True, val_cache=os.path.join(ev, "val_policy_armeval.npy"))
    frozen = {k: [v.next() for _ in range(EVAL_BATCHES)] for k, v in sets.items()}
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
        path = os.path.join(ROOT, name, f"ckpt_arm_{name}_last.pt")
        ck = torch.load(path, map_location="cuda", weights_only=False)
        model = T.ChessNet(d=T.D, nhead=T.NHEAD, layers=T.LAYERS, dim_ff=int(T.FF_MULT * T.D), qhead=True,
                           dropout=T.DROPOUT, att=T.ATT, dist=T.DIST, mlh=T.MLH).to("cuda")
        model.load_state_dict(ck["model"], strict=False)
        it = int(ck.get("iter", 0))
        del ck
        model, used = T.apply_precision(model, "bf16")
        with torch.no_grad():
            v = T.validate(model, used, Replay(frozen["av"]), Replay(P), EVAL_BATCHES, lvv=Replay(frozen["lv"]),
                           fsv=Replay(frozen["dp"]), dhv=Replay(frozen["dh"]), stv=Replay(frozen["st"]))
        v["dp_q"], v["dp_mae"] = v.pop("fs_q"), v.pop("fs_mae")   # route 1's held-out rows ran in the fs slot
        rows.append((name, extra, it, v))
        del model
        torch.cuda.empty_cache()
    keys = [k for k in ("dp_q", "dh_q", "lv_q", "q", "policy_legal", "top1_legal", "top3_legal", "st_q")
            if k in rows[0][3]]
    out = [f"search-target arms, {STEPS:,} steps each from iteration {st['start']:,}, DATA_SEED {SEED}, the same "
           f"{EVAL_BATCHES} frozen batches of every held-out set for all three",
           "  " + f"{'arm':<10}{'iter':>12}" + "".join(f"{k:>14}" for k in keys)]
    for name, extra, it, v in rows:
        out.append("  " + f"{name:<10}{it:>12,}" + "".join(f"{float(v[k]):>14.4f}" for k in keys))
    out.append("  dp_q: Stockfish's labels of MayFly's own self-play positions (the question); dh_q Kaggle's deep "
               "labels of its mistakes; lv_q Lichess deep evaluations; q ChessBench; st_q held-out search targets "
               "(the search arms train towards them). Lower is better except top1/top3.")
    open(os.path.join(ROOT, "search_result.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    for line in out:
        say(line)


def main():
    os.makedirs(ROOT, exist_ok=True)
    st = load_status()
    try:
        if "start" not in st:
            if not glob.glob(os.path.join(main_env().get("SHARDS", paths.SHARDS), "st_*.bin")):
                raise SystemExit("no st_* shard: run pack_search.py and copy its shards first")
            pause_main()
            import torch
            start_ckpt = os.path.join(ROOT, "start_ckpt.pt")
            shutil.copy(os.path.join(CK, "ckpt_mayfly_last.pt"), start_ckpt)
            ck = torch.load(start_ckpt, map_location="cpu", weights_only=False)
            ve = int(main_env().get("VAL_EVERY", 10000))
            st = {"start": int(ck["iter"]), "iters": -(-(int(ck["iter"]) + STEPS) // ve) * ve, "done": [],
                  "start_ckpt": start_ckpt}
            del ck
            save_status(st)
            write_cap()
            say(f"arms from the main checkpoint at iteration {st['start']:,}, {STEPS:,} steps each")
        else:
            pause_main()
        for name, extra in ARMS:
            if name in st["done"]:
                continue
            if not run_arm(name, extra, st):
                raise SystemExit(f"arm {name} failed; rerun to resume it")
            st["done"].append(name)
            save_status(st)
        common_eval(st)
    finally:
        resume_main()
        try:
            os.remove(SCRIPT)
        except OSError:
            pass


if __name__ == "__main__":
    main()
