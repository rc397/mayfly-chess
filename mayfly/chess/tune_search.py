r"""tune_search tunes the search settings jointly with Optuna.

The settings were each tuned alone, so their interactions were untested. lc0 reports 30 to 70 Elo from
joint tuning. The released version is the test bed. Every trial plays v4 against v4 plus the trial's
settings, and a winner goes into the next version.

  1. tune     Optuna's TPE over eight settings, TRIALS trials of TRIAL_GAMES games at TRIAL_VISITS visits,
              paired openings with one seed for every trial. Trial 0 is v4's own values.
  2. confirm  the three best trials again on unseen openings (CONFIRM_GAMES each), then the best of those
              at CONFIRM_DEEP_VISITS, because the optimum can move with the budget.
Everything resumes. Optuna keeps its trials in TUNE_DIR/tune.db, a trial that died runs again with the same
settings, and each match resumes game by game from its own log. The answer is TUNE_DIR/result.txt.

  python chess/tune_search.py
  env: TRIALS=60 TRIAL_GAMES=32 TRIAL_VISITS=400 CONFIRM_GAMES=96 CONFIRM_DEEP_GAMES=200
       TUNE_DIR (default paths.under("tune"))
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
import paths  # noqa: E402

E = os.environ.get
OUT = E("TUNE_DIR", paths.under("tune"))
TRIALS = int(E("TRIALS", 60))
GAMES = int(E("TRIAL_GAMES", 32))
VISITS = int(E("TRIAL_VISITS", 400))
CONFIRM_GAMES = int(E("CONFIRM_GAMES", 96))
DEEP_GAMES = int(E("CONFIRM_DEEP_GAMES", 200))
DEEP_VISITS = int(E("CONFIRM_DEEP_VISITS", 1600))
VERSION = "mayfly_v4.pt"

# name: (scale, low, high, v4's sealed value). Every one is a sealed search key. explore_growth and
# explore_base are off for v1 to v4.
SPACE = {
    "c1": ("log", 0.5, 4.0, 1.4142135623730951),     # puct_v's variance-scaled exploration
    "c2": ("log", 0.75, 8.0, 3.0),                   # puct_v's log term
    "q_n0": ("log", 0.5, 8.0, 2.0),                  # visits' worth of the q head's prior
    "policy_temp": ("lin", 0.8, 1.6, 1.0),           # prior temperature
    "lcb_floor": ("lin", 0.02, 0.3, 0.1),            # the LCB's share of the leader's visits
    "lcb_alpha": ("log", 1e-7, 1e-2, 1e-5),          # the LCB's confidence
    "explore_growth": ("lin", 0.0, 1.5, 0.0),        # exploration growing with the parent's visits
    "explore_base": ("log", 100.0, 5000.0, 500.0),
}


def say(m):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + m
    print(line, flush=True)
    with open(os.path.join(OUT, "tune.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def match(opts, games, visits, seed, tag):
    """B = v4 + opts against A = v4, at fixed visits. Returns B's score. Resumes from its own files."""
    spec = ",".join(f"{k}={v:.6g}" for k, v in sorted(opts.items()))
    h = hashlib.md5(f"{spec}|{games}|{visits}|{seed}".encode()).hexdigest()[:10]
    res = os.path.join(OUT, f"{tag}_{h}.result.jsonl")
    if not os.path.exists(res):
        env = dict(os.environ, VERSION=VERSION, BASE=VERSION, RESULTS=res,
                   GAMES_LOG=os.path.join(OUT, f"{tag}_{h}.games.jsonl"), OPENING_SEED=str(seed),
                   SF_MS="100", PYTHONUNBUFFERED="1")
        with open(os.path.join(OUT, f"{tag}_{h}.log"), "a", encoding="utf-8") as fh:
            fh.write(f"{spec}  games {games}  visits {visits}  seed {seed}\n")
            fh.flush()
            rc = subprocess.call([sys.executable, os.path.join(HERE, "search_match.py"), spec,
                                  str(games), str(visits)], env=env, cwd=HERE, stdout=fh,
                                 stderr=subprocess.STDOUT)
        if rc != 0 or not os.path.exists(res):
            raise RuntimeError(f"match {tag}_{h} failed (exit {rc})")
    row = [json.loads(l) for l in open(res, encoding="utf-8") if l.strip()][-1]
    return float(row["score"]), row


def elo(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return -400.0 * math.log10(1.0 / p - 1.0)


def main():
    import optuna
    from optuna.distributions import FloatDistribution
    from optuna.trial import TrialState
    os.makedirs(OUT, exist_ok=True)
    dist = {k: FloatDistribution(lo, hi, log=(sc == "log")) for k, (sc, lo, hi, _) in SPACE.items()}
    study = optuna.create_study(study_name="v4_joint", storage=f"sqlite:///{OUT}/tune.db".replace("\\", "/"),
                                direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=7, n_startup_trials=12,
                                                                   multivariate=True))
    # A restart must not replay the random start. With a fixed seed it would draw the first trials'
    # settings again and count their cached results as new trials. Seeding by the number of trials on
    # file keeps the first start unchanged and gives a restart fresh points.
    if study.trials:
        study.sampler = optuna.samplers.TPESampler(seed=7 + len(study.trials), n_startup_trials=12,
                                                   multivariate=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # a trial a crash left running is run again with the same settings (its match resumes)
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(t.number, state=TrialState.FAIL)
        study.enqueue_trial(t.params)
    # Trials scored at other TRIAL_GAMES or TRIAL_VISITS are not comparable, so one study keeps one setting.
    got = (study.user_attrs.get("games", GAMES), study.user_attrs.get("visits", VISITS))
    if study.trials and got != (GAMES, VISITS):
        raise SystemExit(f"{OUT} holds a study at {got[0]} games and {got[1]} visits a trial; this run "
                         f"asks for {GAMES} and {VISITS}. Use another TUNE_DIR.")
    study.set_user_attr("games", GAMES)
    study.set_user_attr("visits", VISITS)
    if not study.trials:
        study.enqueue_trial({k: v for k, (_, _, _, v) in SPACE.items()})
    while len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))) < TRIALS:
        tr = study.ask(fixed_distributions=dist)
        n = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        score, _ = match(tr.params, GAMES, VISITS, 23, "trial")
        study.tell(tr, score)
        say(f"trial {n + 1}/{TRIALS}: {100 * score:.1f}% ({elo(score):+.0f})  "
            + ", ".join(f"{k}={v:.4g}" for k, v in sorted(tr.params.items()))
            + f"  | best so far {100 * study.best_value:.1f}%")
    done = sorted(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)),
                  key=lambda t: -t.value)
    top = done[:3]
    confirmed = []
    for t in top:
        s, row = match(t.params, CONFIRM_GAMES, VISITS, 77, "confirm")
        confirmed.append((s, t, row))
        say(f"confirm trial {t.number} on fresh openings: {100 * s:.1f}% ({elo(s):+.0f}, "
            f"{row.get('lo', '?')} to {row.get('hi', '?')}) at {VISITS} visits (tuning read {100 * t.value:.1f}%)")
    best_s, best, _ = max(confirmed, key=lambda c: c[0])
    deep_s, deep_row = match(best.params, DEEP_GAMES, DEEP_VISITS, 78, "deep")
    say(f"best trial {best.number} at {DEEP_VISITS:,} visits: {100 * deep_s:.1f}% ({elo(deep_s):+.0f}, "
        f"{deep_row.get('lo', '?')} to {deep_row.get('hi', '?')})")
    lines = [f"joint tuning of v4's search, {len(done)} trials of {GAMES} games at {VISITS} visits",
             "",
             f"best (confirmed on fresh openings, then at {DEEP_VISITS:,} visits):"]
    lines += [f"  {k:<16} {best.params[k]:.6g}   (v4: {SPACE[k][3]:.6g})" for k in sorted(SPACE)]
    lines += ["",
              f"  tuning score   {100 * best.value:.1f}%  ({elo(best.value):+.0f})",
              f"  fresh openings {100 * best_s:.1f}%  ({elo(best_s):+.0f}) at {VISITS} visits, {CONFIRM_GAMES} games",
              f"  {DEEP_VISITS:,} visits   {100 * deep_s:.1f}%  ({elo(deep_s):+.0f}, 95% {deep_row.get('lo', '?')} to "
              f"{deep_row.get('hi', '?')}), {DEEP_GAMES} games",
              "",
              "the other confirmed candidates:"]
    lines += [f"  trial {t.number}: tuning {100 * t.value:.1f}%, fresh {100 * s:.1f}%" for s, t, _ in confirmed]
    open(os.path.join(OUT, "result.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    for l in lines:
        say(l)


if __name__ == "__main__":
    main()
