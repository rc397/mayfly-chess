"""Hand the main training run to a Kaggle TPU session and take it back, one pass per run.

Run it regularly, for example every ten minutes from a scheduler. One cycle has four phases.
    idle      switched on (train_tpu.on), the weekly TPU quota has MIN_QUOTA_H left, the code and data
              datasets are ready on Kaggle, and no other TPU run is out
    pushing   the latest local save goes up as a new version of the checkpoint dataset and kaggle_train.py is
              pushed on the TPU for the quota left, at most SESSION_H. Local training is not paused yet, since
              the queue for a TPU can take many hours.
    out       when the session starts, the main run is paused as cooldown.py does (training.paused, which stops
              supervise.py restarting it, plus this controller's marker) and the trainer stopped, so the local
              GPU is free for testing while the TPU trains
    home      the new checkpoint is downloaded and checked (it loads, its iteration is past both the one handed
              over and the local one, its weights are finite, and ckpt_score.py does not find it worse). The
              local checkpoint is kept beside it (.pre_tpu_<run>), the new one put in its place, and the pause
              lifted, so supervise.py resumes from where the TPU left off. A failed session or a checkpoint that
              fails the check lifts the pause with the local checkpoint untouched.

A pause this controller did not make (training.paused without its marker) is never touched, and nothing starts
while one is in place, unless train_tpu.while_paused exists too. Then the session runs, its checkpoint is swapped
in at home, and the pause stays.

train_cool.json ({"minutes": 60}, used by the next push) asks the session to save the main run, then cool a copy
for those minutes (XLA_COOL_MIN). At home the main run resumes from the uncooled checkpoint and the cooled copy is
exported as cooldown.py would (COOL_OUT/tpu_<run>/v5_raw.pt and v5_cooled.pt, next_candidate.json).

With train_label.on the spare host cores label positions while the TPU trains. When a cycle starts kaggle_label.py
is asked for a batch (trainlab/request.json, about TRAIN_LABEL_PER_HOUR positions per session hour), the batch
rides in the code dataset with label.json, kaggle_train.py runs kaggle_kernel.py on the spare cores, and when the
session is home its results are handed back (trainlab/done.json) for kaggle_label.py to read in.

    python chess/kaggle_train_ctl.py         one pass
    env KAGGLE_DIR, CKPT_DIR, TRAIN_SESSION_H=8.5, TRAIN_MIN_QUOTA_H=1.5, KAGGLE_USER
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kaggle_label as kl
import paths

E = os.environ.get
BASE = E("KAGGLE_DIR", paths.under("kaggle"))
CKPT_DIR = E("CKPT_DIR", paths.under("ckpt"))
LAST = os.path.join(CKPT_DIR, "ckpt_mayfly_last.pt")
ON = os.path.join(BASE, "train_tpu.on")
STATE = os.path.join(BASE, "train_state.json")
LOG = os.path.join(BASE, "kaggle_train.log")
CK_DS = os.path.join(BASE, "train_ckpt")
OUT = os.path.join(BASE, "train_out")
PAUSED = os.path.join(HERE, "training.paused")
MARK = os.path.join(BASE, "paused_by_tpu.txt")
SLUG = f"{kl.USER}/mayfly-train-tpu"
DATASETS = [f"{kl.USER}/mayfly-train-code", f"{kl.USER}/mayfly-train-ckpt", f"{kl.USER}/mayfly-train-data"]
WHILE_PAUSED = os.path.join(BASE, "train_tpu.while_paused")   # someone else's pause does not stop a session
LABEL_ON = os.path.join(BASE, "train_label.on")    # the spare cores label a batch from the Kaggle queue
EXT = os.path.join(BASE, "trainlab")                # kaggle_label.external()'s hand-over folder
CODE_DIR = os.path.join(BASE, "train_code")        # make_train_code.py's folder, the code dataset
LABEL_PER_HOUR = float(E("TRAIN_LABEL_PER_HOUR", 1500))   # 176 engines at depth 35 (the host has 224 cores)
SESSION_H = float(E("TRAIN_SESSION_H", 8.5))     # Kaggle stops a TPU session at 9 h
MIN_QUOTA_H = float(E("TRAIN_MIN_QUOTA_H", 1.5))


def say(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state():
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except (OSError, ValueError):
        return {"phase": "idle", "runs": 0}


def save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, STATE)


def trainers():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train_av\\.py' -and "
                          "$_.Name -match 'python' } | ForEach-Object { $_.ProcessId }"],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def resume_rig(why):
    if os.path.exists(MARK):
        for p in (PAUSED, MARK):
            try:
                os.remove(p)
            except OSError:
                pass
        say(f"main run: pause lifted ({why}), supervise.py restarts it from {os.path.basename(LAST)}")


def status():
    rc, out = kl.kaggle("kernels", "status", SLUG, timeout=120)
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?([A-Za-z_]+)', out)
    if m:
        s = m.group(1).lower()
        return {"cancel_requested": "running", "cancel_acknowledged": "cancelled"}.get(s, s)
    return "missing" if ("404" in out or "not found" in out.lower()) else "unknown"


def other_tpu_out():
    try:
        sp = json.load(open(os.path.join(BASE, "selfplay_state.json"), encoding="utf-8"))["slots"].get("tpu")
        return bool(sp) and not sp.get("done")
    except (OSError, ValueError, KeyError):
        return False


def ready():
    for ds in DATASETS[0::2]:              # the code and the data, the checkpoint goes up at the handoff
        rc, out = kl.kaggle("datasets", "status", ds, timeout=120)
        if "ready" not in out.lower():
            return f"{ds} not ready ({out.strip()[-80:]})"
    return None


def check(path, handed):
    """Check the checkpoint from the TPU. It must load, carry everything train_av resumes from, have moved past the
    iteration it was handed, and have finite weights. Returns (ok, why)."""
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"does not load ({type(e).__name__}: {e})"
    need = ("model", "optim", "iter", "trained_hours", "att_since", "dist_since", "mlh_since", "yardstick")
    missing = [k for k in need if k not in ck]
    if missing:
        return False, f"lacks {missing}"
    if int(ck["iter"]) <= int(handed):
        return False, f"iteration {ck['iter']:,} is not past {handed:,}"
    bad = [k for k, v in ck["model"].items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        return False, f"non-finite weights in {bad[:4]}"
    return True, f"iteration {handed:,} -> {int(ck['iter']):,}, {ck.get('xla', {})}"


SCORE_TOL = float(E("TPU_SCORE_TOL", 0.005))


def never_worse(got):
    """Refuse a TPU checkpoint that scores worse than the local one.

    The TPU trains at eight times the batch with a scaled rate and different numerics, so a session can train and
    still leave the network worse. Both checkpoints are scored on the same frozen held-out batches (ckpt_score.py,
    on the local GPU, which the paused run has left free). The TPU's is refused when it is worse by more than
    SCORE_TOL on at least two of ChessBench's q, the legal policy and route 1's dp_q. A scoring error is reported and
    does not block. Returns (ok, note)."""
    out, r = os.path.join(BASE, "score_last.json"), None
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "ckpt_score.py"), out, got, LAST],
                           capture_output=True, text=True, timeout=3600)
        sc = json.load(open(out, encoding="utf-8"))
        a, b = sc[got], sc[LAST]
    except Exception as e:
        return True, f"not scored ({type(e).__name__}: {e}; {((r.stderr or '') if r else '')[-300:]})"
    keys = [k for k in ("q", "policy_legal", "dp_q") if k in a and k in b]
    worse = [k for k in keys if a[k] > b[k] * (1 + SCORE_TOL)]
    note = (f"scored beside the local one ({b['iter']:,}): "
            + ", ".join(f"{k} {b[k]:.4f} -> {a[k]:.4f}" for k in keys))
    return len(worse) < 2, note + (f"; worse on {worse}" if worse else "")


def label_back(batch, ok, d=None):
    """Tell kaggle_label.external() how a shipped batch ended, read its results from d, or give it back."""
    os.makedirs(EXT, exist_ok=True)
    with open(os.path.join(EXT, "done.json"), "w", encoding="utf-8") as fh:
        json.dump({"batch": batch, "ok": bool(ok), "dir": d}, fh)
    say(f"labelling batch {batch}: " + (f"results handed to the labeller ({d})" if ok else "given back to the queue"))


def ship_label():
    """The ready batch (if any) into the code dataset's folder with label.json, or the folder cleared of one. Returns
    the batch id shipped, or None."""
    import glob
    for f in glob.glob(os.path.join(CODE_DIR, "b*.txt")) + [os.path.join(CODE_DIR, "label.json")]:
        try:
            os.remove(f)
        except OSError:
            pass
    rdy = os.path.join(EXT, "ready.json")
    if not (os.path.exists(LABEL_ON) and os.path.exists(rdy)):
        return None
    r = json.load(open(rdy, encoding="utf-8"))
    shutil.copy2(r["file"], CODE_DIR)
    json.dump({"batch": r["batch"], "file": os.path.basename(r["file"]), "positions": r.get("positions"),
               "spare": int(E("TRAIN_LABEL_SPARE", 48))}, open(os.path.join(CODE_DIR, "label.json"), "w"))
    return r["batch"]


COOL_ASK = os.path.join(BASE, "train_cool.json")   # {"minutes": 60}, the next session ends with a cooled copy, once
COOL_OUT = E("COOL_OUT", paths.under("cool"))


def ship_cool():
    """The cooled copy asked for (train_cool.json, used once) into session.json beside the code, or session.json
    cleared. Returns the minutes, or 0."""
    p = os.path.join(CODE_DIR, "session.json")
    try:
        os.remove(p)
    except OSError:
        pass
    if not os.path.exists(COOL_ASK):
        return 0
    m = int(json.load(open(COOL_ASK, encoding="utf-8")).get("minutes", 60))
    json.dump({"cool_min": m}, open(p, "w"))
    return m


def export_candidate(main_ck, cooled_ck, run):
    """Export the TPU's cooled copy as cooldown.py writes one.

    v5_raw.pt (the main run's weights it was cooled from) and v5_cooled.pt go in COOL_OUT/tpu_<run>, in the release
    checkpoint format, with next_candidate.json pointing at them. The cfg is the main run's best checkpoint's."""
    import torch
    cfg = torch.load(os.path.join(CKPT_DIR, "ckpt_mayfly.pt"), map_location="cpu", weights_only=False).get("cfg", {})
    d = os.path.join(COOL_OUT, f"tpu_{run}")
    os.makedirs(d, exist_ok=True)
    out = {}
    for src, name, note in ((main_ck, "v5_raw.pt", "the main run's weights it was cooled from"),
                            (cooled_ck, "v5_cooled.pt", "cooled on the TPU")):
        ck = torch.load(src, map_location="cpu", weights_only=False)
        x = ck.get("xla", {})
        torch.save({"model": ck["model"], "iter": int(ck["iter"]), "val": (ck.get("hist") or [{}])[-1],
                    "yardstick": ck.get("yardstick"), "cfg": cfg, "trained_hours": ck.get("trained_hours"),
                    "made_by": f"kaggle_train_ctl.py run {run}: {note}"
                    + (f" (from {x.get('cool_from'):,} over {x.get('cool_steps'):,} steps to {x.get('cool_floor')})"
                       if x.get("cool_from") else "")}, os.path.join(d, name + ".part"))
        os.replace(os.path.join(d, name + ".part"), os.path.join(d, name))
        out[name] = int(ck["iter"])
    json.dump({"ckpt": os.path.join(d, "v5_cooled.pt"), "raw": os.path.join(d, "v5_raw.pt"),
               "iter": out["v5_cooled.pt"], "made": time.strftime("%Y-%m-%d %H:%M"), "used": None},
              open(os.path.join(COOL_OUT, "next_candidate.json"), "w", encoding="utf-8"), indent=1)
    return d, out


LOCK = os.path.join(BASE, ".train_ctl.lock")


def main():
    """One pass, never two at once, or two passes could each push a session. A lock older than two hours was left
    by a pass that died."""
    os.makedirs(BASE, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        if time.time() - os.path.getmtime(LOCK) < 7200:
            print("another pass is running", flush=True)
            return
        os.utime(LOCK, None)
    try:
        one_pass()
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


def one_pass():
    os.makedirs(OUT, exist_ok=True)
    st = load_state()
    ph = st.get("phase", "idle")
    if ph == "idle":
        rdy = os.path.join(EXT, "ready.json")
        if not os.path.exists(ON) and os.path.exists(rdy) and not os.path.exists(os.path.join(EXT, "done.json")):
            label_back(json.load(open(rdy, encoding="utf-8"))["batch"], False)   # switched off, the batch goes back
            return
        if not os.path.exists(ON) or time.time() < st.get("wait_until", 0):
            return
        if os.path.exists(PAUSED) and not os.path.exists(WHILE_PAUSED):
            return                             # someone else's pause (a cooldown, a gate), not ours to work around
        if other_tpu_out():
            return
        why = ready()
        if why:
            say(f"not starting: {why}")
            st["wait_until"] = time.time() + 1800
            save_state(st)
            return
        left = kl.quota("TPU")
        if left is None or left < MIN_QUOTA_H:
            st["wait_until"] = time.time() + 3600
            save_state(st)
            return
        req = os.path.join(EXT, "request.json")
        if os.path.exists(LABEL_ON) and not os.path.exists(rdy) and not os.path.exists(req):
            os.makedirs(EXT, exist_ok=True)       # the labeller's next pass makes it, long before the push
            json.dump({"budget": int(LABEL_PER_HOUR * min(SESSION_H, left - 0.25))}, open(req, "w"))
        st.update(phase="pushing", quota_h=round(left, 2))
        save_state(st)
        say(f"handing the run to the TPU ({left:.1f}h of TPU quota left), local training goes on until the session starts")
        return
    if ph == "pushing":
        # Local training is not paused here, since the queue for a TPU can take many hours. Its latest save goes up
        # and it trains on until the session starts, and what it trains meanwhile gives way to the TPU's checkpoint.
        import torch
        handed = int(torch.load(LAST, map_location="cpu", weights_only=False)["iter"])
        os.makedirs(CK_DS, exist_ok=True)
        shutil.copy2(LAST, os.path.join(CK_DS, "ckpt_mayfly_last.pt"))
        json.dump({"title": "mayfly-train-ckpt", "id": DATASETS[1], "licenses": [{"name": "other"}]},
                  open(os.path.join(CK_DS, "dataset-metadata.json"), "w"))
        t0 = time.time()
        rc, out = kl.kaggle("datasets", "version", "-p", CK_DS, "-m", f"iter {handed}", timeout=3600)
        say(f"checkpoint at iteration {handed:,} uploaded in {time.time() - t0:.0f}s (rc {rc})")
        for _ in range(60):
            rc, out = kl.kaggle("datasets", "status", DATASETS[1], timeout=120)
            if "ready" in out.lower():
                break
            time.sleep(20)
        # the code folder is rebuilt from this checkout at every push, so the next session runs the current code
        import make_train_code
        make_train_code.build()
        lab = ship_label()                     # a labelling batch rides in the code dataset, or none does
        cool = ship_cool()                     # and a cooled copy at the end, when one was asked for
        rc, out = kl.kaggle("datasets", "version", "-p", CODE_DIR, "-m", f"code, label {lab}, cool {cool}",
                            timeout=1800)
        for _ in range(60):
            rc2, out2 = kl.kaggle("datasets", "status", DATASETS[0], timeout=120)
            if "ready" in out2.lower():
                break
            time.sleep(20)
        st["runs"] = st.get("runs", 0) + 1
        run = time.strftime("%Y%m%d_%H%M") + f"_train_{st['runs']:04d}"
        hours = min(SESSION_H, st.get("quota_h", SESSION_H) - 0.25)
        d = os.path.join(BASE, "train_push")
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        src = open(os.path.join(HERE, "kaggle_train.py"), encoding="utf-8").read()
        src = (src.replace("TIME_BUDGET = 8 * 3600 + 15 * 60", f"TIME_BUDGET = {int(hours * 3600)}")
               .replace('"__MODE__"', '"tpu"').replace('"__RUN__"', f'"{run}"'))
        with open(os.path.join(d, "mayfly_train_kernel.py"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(src)
        json.dump({"id": SLUG, "title": "mayfly train tpu", "code_file": "mayfly_train_kernel.py",
                   "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
                   "enable_tpu": True, "enable_internet": True, "machine_shape": "TpuV5E8",
                   "dataset_sources": DATASETS, "competition_sources": [], "kernel_sources": []},
                  open(os.path.join(d, "kernel-metadata.json"), "w"))
        rc, out = kl.kaggle("kernels", "push", "-p", d, "--accelerator", "TpuV5E8")
        if rc == 0 and "successfully pushed" in out:
            st.update(phase="out", run=run, handed=handed, pushed=time.strftime("%Y-%m-%d %H:%M"), hours=hours,
                      label_batch=lab, cool=cool)
            if cool:
                os.replace(COOL_ASK, COOL_ASK + f".used_{run}")
            say(f"run {run} pushed on the TPU ({hours:.2f}h) from iteration {handed:,}"
                + (f", labelling batch {lab} on its spare cores" if lab else "")
                + (f", the last {cool} minutes cooling a copy" if cool else "")
                + "; local training goes on until the session starts")
        else:
            resume_rig("the push was refused")
            st.update(phase="idle", wait_until=time.time() + 1800)
            say(f"push refused: {out.strip()[-200:]}")
        save_state(st)
        return
    if ph == "out":
        s = status()
        if s == "running" and not st.get("rig_paused") and not (os.path.exists(PAUSED) and not os.path.exists(MARK)):
            # The session has started, so local training stops and the GPU is free for testing. Whatever it trained
            # since the handover gives way to the TPU's checkpoint when that is home.
            with open(PAUSED, "w", encoding="utf-8") as fh:
                fh.write(time.strftime("%Y-%m-%d %H:%M") + " paused by kaggle_train_ctl.py: the TPU trains the run; "
                         "it lifts this itself when the TPU's checkpoint is home\n")
            with open(MARK, "w", encoding="utf-8") as fh:
                fh.write(time.strftime("%Y-%m-%d %H:%M") + "\n")
            for pid in trainers():
                subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL)
            st["rig_paused"] = time.strftime("%Y-%m-%d %H:%M")
            save_state(st)
            say(f"run {st['run']} started on the TPU: the main run is paused locally, its GPU free for testing")
        if s in ("running", "queued"):
            if s != st.get("seen"):
                st["seen"] = s
                save_state(st)
                say(f"run {st['run']} {s}")
            return
        d = os.path.join(OUT, st["run"])
        os.makedirs(d, exist_ok=True)
        rc, out = kl.kaggle("kernels", "output", SLUG, "-p", d, timeout=3600)
        got = os.path.join(d, "ckpt", "ckpt_mayfly_last.pt")
        if not os.path.exists(got):
            got = os.path.join(d, "ckpt_mayfly_last.pt")
        ok, why = (check(got, st["handed"]) if s in ("complete", "error") and os.path.exists(got)
                   else (False, f"ended {s}, no checkpoint"))
        if ok:
            # never backwards, since local training went on while the session queued, so the TPU's checkpoint must
            # be past the local one as well as past the one it was handed
            import torch
            got_it = int(torch.load(got, map_location="cpu", weights_only=False)["iter"])
            rig_it = int(torch.load(LAST, map_location="cpu", weights_only=False)["iter"])
            if got_it <= rig_it:
                ok, why = False, f"the TPU's iteration {got_it:,} is not past the local {rig_it:,}"
        worse = False
        if ok:
            fine, note = never_worse(got)
            if fine:
                why = f"{why}; {note}"
            else:
                ok, worse, why = False, True, f"the TPU's checkpoint scores worse than the local one: {note}"
        summ = {}
        try:
            summ = json.load(open(os.path.join(d, "summary.json")))
        except (OSError, ValueError):
            pass
        if st.get("label_batch"):              # the labels stand on their own, whatever became of the training
            ld = os.path.join(d, "label")
            label_back(st["label_batch"], s == "complete" and os.path.exists(os.path.join(ld, "summary.json")), ld)
            st["label_batch"] = None
        if ok:
            if trainers():                     # never paused (the session ran between passes), so stop it before the
                if not os.path.exists(PAUSED):  # swap, or its next save would write over the TPU's checkpoint
                    with open(PAUSED, "w", encoding="utf-8") as fh:
                        fh.write(time.strftime("%Y-%m-%d %H:%M") + " paused by kaggle_train_ctl.py for the swap\n")
                    with open(MARK, "w", encoding="utf-8") as fh:
                        fh.write(time.strftime("%Y-%m-%d %H:%M") + "\n")
                for pid in trainers():
                    subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL)
                time.sleep(5)
            keep = LAST + f".pre_tpu_{st['run']}"
            shutil.copy2(LAST, keep)
            tmp = LAST + ".part"
            shutil.copy2(got, tmp)
            os.replace(tmp, LAST)
            say(f"run {st['run']} home: {why}; summary {summ}; the local one kept as {os.path.basename(keep)}")
            cooled = os.path.join(os.path.dirname(got), "ckpt_cooled_last.pt")
            if st.get("cool") and os.path.exists(cooled):
                try:
                    cd, its = export_candidate(LAST, cooled, st["run"])
                    say(f"cooled copy exported as the next candidate: {cd} (iterations {its})")
                except Exception as e:
                    cd = None
                    say(f"cooled copy not exported: {type(e).__name__}: {e}")
                # Measure both copies while the GPU is still free (the main run stays paused until resume_rig below).
                if cd:
                    try:
                        r = subprocess.run([sys.executable, os.path.join(HERE, "strength_probe.py"), "score",
                                            os.path.join(cd, "v5_cooled.pt"), os.path.join(cd, "v5_raw.pt")],
                                           capture_output=True, text=True, timeout=3600, cwd=HERE)
                        got_pz = [l.strip() for l in (r.stdout or "").splitlines() if "puzzles" in l]
                        say(f"strength probe on the TPU copies (exit {r.returncode}): "
                            f"{got_pz or (r.stdout or r.stderr or '')[-300:]}")
                    except Exception as e:
                        say(f"strength probe not run on the TPU copies: {type(e).__name__}: {e}")
            resume_rig("the TPU's checkpoint is in place")
            st.update(phase="idle", last_run=st["run"], wait_until=0)
        else:
            say(f"run {st['run']} not used: {why}; summary {summ}; output in {d}")
            resume_rig("the TPU session did not give a usable checkpoint")
            st.update(phase="idle", wait_until=time.time() + 3600)
            if (worse or (s in ("complete", "error") and not summ.get("steps"))) and os.path.exists(ON):
                # A session that ran and never trained a step, or trained and made it worse, would do the same again
                # with the same code, so the controller switches itself off until train_tpu.on is put back

                held = ON[:-3] + f".held_{st['run']}"
                os.replace(ON, held)
                say(f"no TPU session is pushed again until the cause is fixed: {os.path.basename(ON)} is now "
                    f"{os.path.basename(held)}")
        st["rig_paused"] = None
        save_state(st)


if __name__ == "__main__":
    main()
