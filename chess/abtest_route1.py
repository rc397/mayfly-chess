"""A/B test of route 1, MayFly's own positions scored by Stockfish, as a share of the action-value rows.

The main run is paused at its next save (iteration X). An arm then trains from X for STEPS steps with route 1
off (DP_SHARE=0), as train_av.py copied to abtest_train.py so supervise.py does not treat it as the main
trainer. It has its own checkpoint folder and its own copy of versions.json, so it can never freeze a version.
The main run resumes and its checkpoint at X + STEPS is kept. All three models are then scored on held-out
action values (own positions, ChessBench, Lichess deep evals) and on a set of real blunders with no search.
result.txt holds the table.

status.json keeps the phase, so a re-run carries on where it stopped.

    python abtest_route1.py       STEPS=300000, ROOT sets the work folder
    DRY=1                         quick code-path check, 200 steps, tiny batches, no pause
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
DRY = E("DRY", "0") == "1"
LIVE = E("LIVE", HERE)                    # the live code folder, where training.paused and autostart.json live
CK = paths.CKPT
SHARDS = paths.SHARDS
STEPS = int(E("STEPS", 200 if DRY else 300000))
ROOT = E("ROOT", paths.under("abtest_dry") if DRY else paths.under("abtest"))
TAG = "abtest_noself"
ARM = os.path.join(ROOT, "noself")
CODE = os.path.join(ROOT, "code")
STATUS = os.path.join(ROOT, "status.json")
LOG = os.path.join(ROOT, "abtest.log")
PAUSED = os.path.join(LIVE, "training.paused")
MARK = os.path.join(ROOT, "paused_by_abtest.txt")
MAIN_LOG = os.path.join(LIVE, "train_mayfly.log")
N_EVAL = 2000 if DRY else 15000
PHASES = ["start", "paused", "arm_ready", "arm_done", "main_resumed", "main_end", "evaluated"]


def say(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + m
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_status():
    try:
        return json.load(open(STATUS, encoding="utf-8"))
    except (OSError, ValueError):
        return {"phase": "start"}


def save_status(st):
    with open(STATUS + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)
    os.replace(STATUS + ".tmp", STATUS)


def past(st, phase):
    return PHASES.index(st["phase"]) >= PHASES.index(phase)


def trainers():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
                          "Where-Object { $_.CommandLine -match 'train_av\\.py' } | ForEach-Object { $_.ProcessId }"],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def main_env():
    return json.load(open(os.path.join(LIVE, "autostart.json"), encoding="utf-8-sig"))["env"]


def load_ckpt(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def keep_weights(src, dst):
    """Save only the model, its cfg and its iteration, written whole."""
    import torch
    ck = load_ckpt(src)
    torch.save({k: ck[k] for k in ("model", "cfg", "iter", "val") if k in ck}, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    return int(ck.get("iter", 0))


def pause_main():
    """Pause the main run as cooldown.py does. training.paused stops supervise.py restarting the trainer, and
    the trainer is stopped right after its next resume checkpoint, so nothing trained is lost."""
    if os.path.exists(PAUSED):
        say("main run: already paused by someone else; it will be left paused at the end")
        return
    last = os.path.join(CK, "ckpt_mayfly_last.pt")
    t_pause = time.time()
    with open(PAUSED, "w", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M") + " paused by abtest_route1.py for the route 1 A/B; it removes this "
                 "file itself when its arm is done\n")
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
            return
        time.sleep(2)
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
        say("main run: this did not pause it, so it is left as it is")


def set_up_arm(st):
    os.makedirs(os.path.join(ARM, "versions"), exist_ok=True)
    shutil.copy2(os.path.join(CK, "ckpt_mayfly_last.pt"), os.path.join(ARM, f"ckpt_{TAG}_last.pt"))
    for src, dst in (("val_policy_mayfly.npy", f"val_policy_{TAG}.npy"), ("val_mleft_mayfly.npy", f"val_mleft_{TAG}.npy")):
        if os.path.exists(os.path.join(CK, src)):
            shutil.copy2(os.path.join(CK, src), os.path.join(ARM, dst))
    shutil.copy2(os.path.join(paths.VERSIONS, "versions.json"), os.path.join(ARM, "versions", "versions.json"))
    os.makedirs(CODE, exist_ok=True)
    for f in glob.glob(os.path.join(LIVE, "*.py")):
        shutil.copy2(f, CODE)
    shutil.copy2(os.path.join(CODE, "train_av.py"), os.path.join(CODE, "abtest_train.py"))
    st["x"] = keep_weights(os.path.join(ARM, f"ckpt_{TAG}_last.pt"), os.path.join(ROOT, "start.pt"))
    say(f"arm: set up in {ARM} from the main run's iteration {st['x']:,}; it trains to {st['x'] + STEPS:,}")


def run_arm(st):
    env = dict(os.environ, **main_env())
    env.update(TAG=TAG, CKPT_DIR=ARM, VERSION_DIR=os.path.join(ARM, "versions"), RESUME="1",
               ITERS=str(st["x"] + STEPS), DP_SHARE="0",
               PLAYING_FILE=os.path.join(CK, "playing.flag"), ASIDE_FILE=os.path.join(CK, "gate_aside.flag"),
               PID_SUFFIX="_abtest", PYTHONUNBUFFERED="1")
    if DRY:
        env.update(AV_BSZ="8", POL_BSZ="4", SAVE_EVERY="100", VAL_EVERY="100000000")
    for attempt in range(1, 4):
        say(f"arm: route 1 off, training (attempt {attempt})")
        with open(os.path.join(ARM, "train.log"), "a", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable, os.path.join(CODE, "abtest_train.py")], cwd=CODE, env=env,
                                 stdout=fh, stderr=subprocess.STDOUT)
        it = keep_weights(os.path.join(ARM, f"ckpt_{TAG}_last.pt"), os.path.join(ROOT, "noself_end.pt"))
        say(f"arm: exited {rc} at iteration {it:,}")
        if it >= st["x"] + STEPS:
            st["arm_end"] = it
            return
    raise SystemExit("the arm did not reach its last step in three attempts; the main run is still paused")


def main_iteration():
    """The main run's latest iteration, from its log (cheap), or None."""
    try:
        with open(MAIN_LOG, "rb") as fh:
            fh.seek(max(0, os.path.getsize(MAIN_LOG) - 20000))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    its = [int(m.replace(",", "")) for m in re.findall(r"^\s+(?:it )?([\d,]{7,})\s", tail, re.M)]
    return max(its) if its else None


def wait_main_end(st):
    target = st["x"] + STEPS
    last = os.path.join(CK, "ckpt_mayfly_last.pt")
    while True:
        it = main_iteration()
        if it is not None and it >= target:
            seen = os.path.getmtime(last)
            got = keep_weights(last, os.path.join(ROOT, "main_end.pt"))
            if got >= target:
                st["main_end"] = got
                say(f"main run: kept its checkpoint at iteration {got:,}")
                return
            while os.path.getmtime(last) == seen:      # the log is past the target, so the next save will be too
                time.sleep(60)
            continue
        time.sleep(600)


def evaluate(st):
    import numpy as np
    import torch
    sys.path.insert(0, CODE)
    from chess_net import ChessNet
    from data_av import split_is_val_av
    from pack_av import DT_AV, batch_tensors_av

    def held_out(paths, n=N_EVAL):
        out = []
        for p in paths:
            mm = np.memmap(p, DT_AV, mode="r")
            r = np.array(mm[max(0, len(mm) - 2_000_000):])
            out.append(r[split_is_val_av(r)])
        r = np.concatenate(out) if out else np.zeros(0, DT_AV)
        return r[-n:]

    rig_dp = sorted((p for p in glob.glob(os.path.join(SHARDS, "dp_*.bin")) if re.match(r"dp_\d+\.bin$", os.path.basename(p))),
                    key=os.path.getmtime)[-1:]
    sets = {"own positions (route 1, held out)": held_out(rig_dp),
            "ChessBench, held out": held_out([os.path.join(SHARDS, "av_a0000.bin")]),
            "Lichess deep evals, held out": held_out([os.path.join(SHARDS, "lv_009.bin")])}
    models = [("start (X)", "start.pt"), ("main, route 1 on", "main_end.pt"), ("arm, route 1 off", "noself_end.pt")]

    def load(path):
        ck = load_ckpt(path)
        c, sd = ck.get("cfg", {}), ck["model"]
        m = ChessNet(d=c.get("d", 1024), nhead=c.get("nhead", 16), layers=c.get("layers", 16),
                     dim_ff=int(c.get("ff_mult", 2.0) * c.get("d", 1024)),
                     qhead=bool(c.get("qhead", "qvalue.weight" in sd)), att=bool(c.get("att", "att.q.weight" in sd)),
                     dist=int(c.get("dist", 64 if "vdist.weight" in sd else 0)),
                     mlh=bool(c.get("mlh", "mleft.0.weight" in sd)))
        m.load_state_dict(sd, strict=False)
        return m.cuda().eval(), ck.get("iter")

    @torch.no_grad()
    def score(m, rows, bs=512):
        lq = er = 0.0
        preds = []
        for i in range(0, len(rows), bs):
            ids, cast, mv, q, hm, ep = (torch.as_tensor(x).cuda() for x in batch_tensors_av(rows[i:i + bs]))
            with torch.autocast("cuda", torch.bfloat16):
                out = m(ids, cast, hm, ep)
            logit = out["qvalue"].float().gather(1, mv.view(-1, 1)).squeeze(1)
            lq += float(torch.nn.functional.binary_cross_entropy_with_logits(logit, q, reduction="sum"))
            er += float((torch.sigmoid(logit) - q).abs().sum())
            preds.append(torch.sigmoid(logit).cpu().numpy())
        return lq / len(rows), er / len(rows), np.concatenate(preds)

    def best_first(rows, pred):
        key = np.concatenate([rows["ids4"], rows["cast"][:, None], rows["hm"][:, None], rows["ep"][:, None]], 1)
        hit = tot = i = 0
        while i < len(rows):
            j = i + 1
            while j < len(rows) and (key[j] == key[i]).all():
                j += 1
            if j - i >= 2:
                lab = rows["q"][i:j].astype(np.int64)
                tot += 1
                hit += int(lab[int(np.argmax(pred[i:j]))] == lab.max())
            i = j
        return 100 * hit / max(1, tot)

    # Real blunders from MayFly's own games, seen with no search. For each, the q head's value of Stockfish's
    # refutation against Stockfish's own value, and where Stockfish's best move from the position before sits in
    # the policy. Some of these positions are in route 1, which favours the main run on exactly these.
    import chess
    from encode import board_to_ids, clock_and_ep
    from moves import move_to_idx
    from pack_av import encode_av
    an = paths.under("analysis")
    blunders = []
    try:
        keep = set(json.load(open(os.path.join(an, "real_keys.json")))["B"])
        for r in json.load(open(os.path.join(an, "diag2.json"))):
            if f"{r['id']}:{r['ply']}" in keep and r.get("ref") and r.get("best"):
                b = chess.Board(r["fen"])
                a = b.copy()
                a.push(b.parse_san(r["san"]))
                truth = 1 / (1 + np.exp(0.00368208 * r["sf_after"]))     # the replier's win chance, by Stockfish
                blunders.append((b, chess.Move.from_uci(r["best"]), a, chess.Move.from_uci(r["ref"]), truth))
    except (OSError, ValueError, KeyError) as e:
        say(f"blunder view skipped: {type(e).__name__} {e}")

    def row_for(board, move):
        ids, cast, flip = board_to_ids(board)
        hm, ep = clock_and_ep(board, flip)
        bits = int(cast[0]) | (int(cast[1]) << 1) | (int(cast[2]) << 2) | (int(cast[3]) << 3)
        row = np.zeros(1, DT_AV)[0]
        encode_av(ids, [bits & 1, (bits >> 1) & 1, (bits >> 2) & 1, (bits >> 3) & 1], move_to_idx(move, flip), 0.0,
                  (1 if move.promotion else 0) | 6, row, hm=hm, ep=ep)
        return row, flip

    @torch.no_grad()
    def blunder_view(m):
        if not blunders:
            return None
        after = np.array([row_for(a, ref)[0] for _, _, a, ref, _ in blunders])
        before = [row_for(b, best) for b, best, _, _, _ in blunders]
        ids, cast, mv, _, hm, ep = (torch.as_tensor(x).cuda() for x in batch_tensors_av(after))
        with torch.autocast("cuda", torch.bfloat16):
            qa = torch.sigmoid(m(ids, cast, hm, ep)["qvalue"].float().gather(1, mv.view(-1, 1)).squeeze(1)).cpu().numpy()
        ids, cast, mv, _, hm, ep = (torch.as_tensor(x).cuda()
                                    for x in batch_tensors_av(np.array([r for r, _ in before])))
        with torch.autocast("cuda", torch.bfloat16):
            pol = m(ids, cast, hm, ep)["policy"].float().cpu().numpy()
        ranks = []
        for k, (b, best, _, _, _) in enumerate(blunders):
            flip = before[k][1]
            legal = [move_to_idx(x, flip) for x in b.legal_moves]
            ranks.append(1 + sum(pol[k][i] > pol[k][move_to_idx(best, flip)] for i in legal))
        truth = np.array([t for *_, t in blunders])
        ranks = np.array(ranks)
        return (float(np.median(qa)), float(np.median(np.abs(qa - truth))), float(np.mean(qa > 0.5) * 100),
                float(np.mean(ranks == 1) * 100), float(np.mean(ranks <= 3) * 100))

    res, iters, views = {}, {}, {}
    for name, f in models:
        m, it = load(os.path.join(ROOT, f))
        iters[name] = it
        for s, rows in sets.items():
            lq, er, pred = score(m, rows)
            res[(name, s)] = (lq, er, best_first(rows, pred) if s.startswith("own") else None)
        views[name] = blunder_view(m)
        del m
        torch.cuda.empty_cache()
    lines = [f"route 1 A/B, {time.strftime('%Y-%m-%d %H:%M')}: from iteration {st['x']:,}, {STEPS:,} steps each "
             f"({', '.join(f'{n} {iters[n]:,}' for n, _ in models)})", ""]
    for s, rows in sets.items():
        lines.append(f"{s} ({len(rows):,} rows)")
        for n, _ in models:
            lq, er, bf = res[(n, s)]
            lines.append(f"  {n:18s} loss {lq:.4f}  error {er:.4f}" + (f"  best move first {bf:.1f}%" if bf is not None else ""))
        a, b, c = (res[(n, s)][1] for n, _ in models)
        lines.append(f"  error change from the start: route 1 on {100 * (b - a) / a:+.2f}%, off {100 * (c - a) / a:+.2f}%")
        lines.append("")
    if any(views.values()):
        lines.append(f"v4's {len(blunders)} real Lichess blunders, no search (the refutation for the side playing it; "
                     f"Stockfish's best move in the position before)")
        for n, _ in models:
            v = views[n]
            lines.append(f"  {n:18s} refutation's value median {v[0]:.3f}, error {v[1]:.3f}, rated good {v[2]:.0f}% | "
                         f"best move first {v[3]:.0f}%, in the first three {v[4]:.0f}%")
        lines.append("")
    open(os.path.join(ROOT, "result.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    for line in lines:
        say(line)


def main():
    os.makedirs(ROOT, exist_ok=True)
    st = load_status()
    say(f"route 1 A/B {'(DRY) ' if DRY else ''}at phase {st['phase']}, {STEPS:,} steps")
    if not past(st, "paused"):
        if not DRY:
            pause_main()
        st["phase"] = "paused"
        save_status(st)
    if not past(st, "arm_ready"):
        set_up_arm(st)
        st["phase"] = "arm_ready"
        save_status(st)
    if not past(st, "arm_done"):
        run_arm(st)
        st["phase"] = "arm_done"
        save_status(st)
    if not past(st, "main_resumed"):
        if not DRY:
            resume_main()
        st["phase"] = "main_resumed"
        save_status(st)
    if not past(st, "main_end"):
        if DRY:
            shutil.copy2(os.path.join(ROOT, "start.pt"), os.path.join(ROOT, "main_end.pt"))
            st["main_end"] = st["x"]
        else:
            wait_main_end(st)
        st["phase"] = "main_end"
        save_status(st)
    if not past(st, "evaluated"):
        evaluate(st)
        st["phase"] = "evaluated"
        save_status(st)
    say("route 1 A/B done: " + os.path.join(ROOT, "result.txt"))


if __name__ == "__main__":
    main()
