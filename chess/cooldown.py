"""Make a release candidate by cooling a copy of the main run's learning rate to near zero.

The main run holds a constant learning rate. At a milestone it is paused at a save, a copy is trained
with the rate falling on a half cosine to near zero over roughly 20 to 40 hours, and the main run then
resumes from its own state. The cooled copy is the candidate. The copy is train_av.py (COOL_FROM,
COOL_STEPS and COOL_FLOOR set the schedule) in its own folder with its own copy of versions.json, so it
can never freeze a version. It uses the main run's settings from autostart.json, a fresh DATA_SEED, and
ends on a validation step.

    PAUSE=1 python chess/cooldown.py      pause the main run at its next save, cool a copy, resume it
    python chess/cooldown.py              the same, with the main run already paused (chess/training.paused)
    COOL_TEST=1 python chess/cooldown.py  a short test beside the live trainer at a small batch
    env COOL_HOURS=30 (or COOL_STEPS), COOL_FLOOR=0.01, COOL_SEED=20261006, OUT

Writes OUT/<stamp>/ with the copy's checkpoints and log, v5_raw.pt (the starting weights) and
v5_cooled.pt (the final weights) in the release checkpoint format. A real run also writes
OUT/next_candidate.json, which a release step can use once in place of the main run's best checkpoint.
Rerun the same command to resume after a crash.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
TEST = E("COOL_TEST", "0") == "1"
OUT = E("OUT", paths.under("cool"))
HOURS = float(E("COOL_HOURS", 30))
FLOOR = E("COOL_FLOOR", "0.01")
SEED = E("COOL_SEED", "20261006")
PAUSE = E("PAUSE", "0") == "1"
PAUSED = os.path.join(HERE, "training.paused")
MARK = os.path.join(OUT, "paused_by_cooldown.txt")
CURRENT = os.path.join(OUT, "current_test.json" if TEST else "current.json")
NEXT = os.path.join(OUT, "next_candidate.json")
TAG = "cooltest" if TEST else "cool"
# the small batch the test runs at, beside the live trainer on the same GPU
TEST_ENV = {"POL_BSZ": "8", "AV_BSZ": "8", "VAL_EVERY": "500", "VAL_BATCHES": "4", "SAVE_EVERY": "0",
            "HEARTBEAT": "100"}
TEST_STEPS = int(E("COOL_TEST_STEPS", 1000))


def say(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + m
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cooldown.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main_env():
    return json.load(open(os.path.join(HERE, "autostart.json"), encoding="utf-8-sig"))["env"]


def trainers():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
                          "Where-Object { $_.CommandLine -match 'train_av\\.py' } | "
                          "ForEach-Object { $_.ProcessId }"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def steps_an_hour(ck_dir):
    """The main run's recent iterations per hour, the median over its last ten validation gaps.

    t resets at every restart, so only gaps within one run count."""
    try:
        hist = json.load(open(os.path.join(ck_dir, "hist_mayfly.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return 14000.0
    gaps = [(b["it"] - a["it"]) / (b["t"] - a["t"]) * 3600.0 for a, b in zip(hist, hist[1:])
            if b.get("t", 0) > a.get("t", 0) and b["it"] > a["it"]][-10:]
    return sorted(gaps)[len(gaps) // 2] if gaps else 14000.0


def pause_main(ck_dir):
    """Pause the main run at a save, so nothing trained is lost.

    training.paused stops supervise.py restarting it, and the trainer is stopped right after its next
    resume checkpoint."""
    last = os.path.join(ck_dir, "ckpt_mayfly_last.pt")
    t_pause = time.time()
    with open(PAUSED, "w", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M") + " paused by cooldown.py for a cooled copy; it removes this "
                 "file itself when the copy is done\n")
    with open(MARK, "w", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M") + "\n")
    say("main run: training.paused written; waiting for its next save")
    while time.time() - t_pause < 3600:
        if os.path.getmtime(last) > t_pause + 1:
            break
        time.sleep(15)
    else:
        raise SystemExit("no save within an hour: training.paused is left in place, nothing was stopped")
    time.sleep(20)                              # the rename has landed and the loop is back to training
    for pid in trainers():
        say(f"main run: stopping trainer pid {pid} right after its save")
        subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL)
    for _ in range(60):
        if not trainers():
            break
        time.sleep(2)
    if trainers():
        raise SystemExit(f"the trainer did not stop: {trainers()}")


def resume_main():
    if os.path.exists(MARK):
        for p in (PAUSED, MARK):
            try:
                os.remove(p)
            except OSError:
                pass
        say("main run: training.paused removed; supervise.py restarts it from its own saved state")
    else:
        say("main run: it was paused by hand, so it is left paused (delete chess/training.paused to resume)")


def export(src, dst, cfg, note):
    """Save the weights, iteration, last validation and config, the release checkpoint format."""
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    v = (ck.get("hist") or [{}])[-1]
    torch.save({"model": ck["model"], "iter": int(ck["iter"]), "val": v, "yardstick": ck.get("yardstick"),
                "cfg": cfg, "trained_hours": ck.get("trained_hours"), "made_by": note}, dst + ".part")
    os.replace(dst + ".part", dst)
    return int(ck["iter"]), v


def main():
    env0 = main_env()
    ck_dir = env0.get("CKPT_DIR", paths.under("ckpt"))
    main_last = os.path.join(ck_dir, "ckpt_mayfly_last.pt")
    os.makedirs(OUT, exist_ok=True)
    st = json.load(open(CURRENT, encoding="utf-8")) if os.path.exists(CURRENT) else None
    if st is None:
        if not TEST:
            if PAUSE and not os.path.exists(PAUSED):
                pause_main(ck_dir)
            if not os.path.exists(PAUSED):
                raise SystemExit("the main run is not paused: PAUSE=1, or write chess/training.paused and "
                                 "stop the trainer after a save")
            if trainers():
                raise SystemExit(f"a trainer is still running (pid {trainers()}); stop it after a save first")
        import torch
        d = os.path.join(OUT, ("test_" if TEST else "") + time.strftime("%Y%m%d_%H%M"))
        os.makedirs(os.path.join(d, "versions"), exist_ok=True)
        last = os.path.join(d, f"ckpt_{TAG}_last.pt")
        shutil.copy(main_last, last)
        for src, dst in (("val_policy_mayfly.npy", f"val_policy_{TAG}.npy"),
                         ("val_mleft_mayfly.npy", f"val_mleft_{TAG}.npy")):
            if os.path.exists(os.path.join(ck_dir, src)):
                shutil.copy(os.path.join(ck_dir, src), os.path.join(d, dst))
        shutil.copy(os.path.join(paths.VERSIONS, "versions.json"), os.path.join(d, "versions", "versions.json"))
        ck = torch.load(last, map_location="cpu", weights_only=False)
        start = int(ck["iter"]) + 1
        hours_behind = float(ck.get("trained_hours", 0.0))
        del ck
        best = torch.load(os.path.join(ck_dir, "ckpt_mayfly.pt"), map_location="cpu", weights_only=False)
        cfg = best.get("cfg", {})
        del best
        ve = int((TEST_ENV if TEST else env0).get("VAL_EVERY", 10000))
        steps = TEST_STEPS if TEST else int(E("COOL_STEPS") or round(HOURS * steps_an_hour(ck_dir)))
        iters = -(-(start + steps) // ve) * ve           # end on a validation step
        st = {"dir": d, "start": start, "iters": iters, "cool_from": start, "cool_steps": iters - start,
              "hours_behind": hours_behind, "cfg": cfg, "main_last_md5": None if not TEST else md5(main_last),
              "versions_md5": md5(os.path.join(paths.VERSIONS, "versions.json")), "done": False}
        json.dump(st, open(CURRENT, "w", encoding="utf-8"), indent=1)
        say(f"{'TEST ' if TEST else ''}cooled copy in {d}: from iteration {start:,} ({hours_behind:.1f}h), "
            f"{iters - start:,} steps to {iters:,}, floor {FLOOR}, DATA_SEED {SEED}")
        it0, v0 = export(last, os.path.join(d, "v5_raw.pt"), cfg, "cooldown.py: the raw weights it started from")
        say(f"raw weights exported (iteration {it0:,}, validation {json.dumps({k: v0.get(k) for k in ('q', 'policy_legal', 'top1_legal', 'att_top1_legal', 'lv_q')})})")
    d = st["dir"]
    env = dict(os.environ, **env0)
    env.update(TAG=TAG, CKPT_DIR=d, VERSION_DIR=os.path.join(d, "versions"), RESUME="1",
               ITERS=str(st["iters"]), COOL_FROM=str(st["cool_from"]), COOL_STEPS=str(st["cool_steps"]),
               COOL_FLOOR=FLOOR, DATA_SEED=SEED, PLAYING_FILE=os.path.join(ck_dir, "playing.flag"),
               PYTHONUNBUFFERED="1", SAVE_EVERY="0")
    if TEST:
        env.update(TEST_ENV)
    if not st["done"]:
        say(f"training the cooled copy to iteration {st['iters']:,}")
        script = os.path.join(HERE, "train_av.py")
        if TEST:
            # supervise.py kills the younger of two processes named train_av.py, and the test runs
            # beside the live one, so it runs the same code under another name
            script = os.path.join(HERE, "_cooltest_trainer.py")
            shutil.copy(os.path.join(HERE, "train_av.py"), script)
        with open(os.path.join(d, f"train_{TAG}.log"), "a", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable, script], cwd=HERE, env=env,
                                 stdout=fh, stderr=subprocess.STDOUT)
        if TEST:
            os.remove(script)
        if rc != 0:
            raise SystemExit(f"train_av exited {rc}; see {d}\\train_{TAG}.log. Rerun this command to carry on.")
        st["done"] = True
        json.dump(st, open(CURRENT, "w", encoding="utf-8"), indent=1)
    it1, v1 = export(os.path.join(d, f"ckpt_{TAG}_last.pt"), os.path.join(d, "v5_cooled.pt"), st["cfg"],
                     f"cooldown.py: cooled from {st['cool_from']:,} over {st['cool_steps']:,} steps to {FLOOR}")
    say(f"cooled weights exported (iteration {it1:,}, validation "
        f"{json.dumps({k: v1.get(k) for k in ('q', 'policy_legal', 'top1_legal', 'att_top1_legal', 'lv_q')})})")
    if TEST:
        ok = md5(os.path.join(paths.VERSIONS, "versions.json")) == st["versions_md5"]

        stray = [f for f in os.listdir(ck_dir) if TAG in f]
        say(f"TEST: versions.json untouched: {ok}; files named {TAG} in the main run's folder: {stray or 'none'}")
        if it1 != st["iters"]:
            say(f"TEST: FAILED, the copy ended at {it1:,}, not {st['iters']:,}")
        os.remove(CURRENT)
        say("TEST DONE")
        return
    json.dump({"ckpt": os.path.join(d, "v5_cooled.pt"), "raw": os.path.join(d, "v5_raw.pt"), "iter": it1,
               "made": time.strftime("%Y-%m-%d %H:%M"), "used": None}, open(NEXT, "w", encoding="utf-8"), indent=1)
    os.remove(CURRENT)
    say(f"next candidate: {d}\\v5_cooled.pt ({NEXT})")
    resume_main()


if __name__ == "__main__":
    main()
