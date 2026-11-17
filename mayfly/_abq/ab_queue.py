"""Run search_match experiments one at a time, from a frozen copy of the code.

queue.txt holds one experiment a line, name | ENV=value;ENV=value | opts | games | visits [| script].
A finished name goes into done.txt. New lines can be appended while it runs, since it polls for them,
and it gives up after IDLE_HOURS with nothing to do only when IDLE_HOURS is set. `snap/chess` is refreshed only by
hand, so an edit to the working tree cannot change an experiment halfway through a queue.

Night hours are optional, for a card that should stay quiet by day. Put "23 7" (the start and end hour in local
time) in NIGHT_HOURS_FILE (default night_hours.txt under the data root) and jobs start only at night unless their env
has DAY_OK=1. A night job still running when the day starts is stopped and resumes the next night (everything here
resumes game by game), neither done nor counted as a failure. Without the file, jobs run at any hour.

  python _abq/ab_queue.py
"""
import os, sys, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "chess"))
import paths  # noqa: E402
Q, DONE = os.path.join(HERE, "queue.txt"), os.path.join(HERE, "done.txt")
SNAP, LOGS = os.path.join(HERE, "snap", "chess"), os.path.join(HERE, "logs")
# It never gives up by default, so rows queued later by a scheduled job with nobody watching still run.
# Set IDLE_HOURS to have it leave on its own.
IDLE_HOURS = float(os.environ.get("IDLE_HOURS", "inf"))
NIGHT_FILE = os.environ.get("NIGHT_HOURS_FILE", paths.under("night_hours.txt"))


def night_now():
    """Inside the night hours, or always when no night hours are set."""
    try:
        x, y = open(NIGHT_FILE, encoding="utf-8").read().split()[:2]
        a, b = int(x), int(y)
    except (OSError, ValueError):
        return True
    if not (0 <= a < 24 and 0 <= b < 24 and a != b):
        return True
    h = time.localtime().tm_hour
    return (a <= h or h < b) if a > b else (a <= h < b)


def day_ok(line):
    parts = line.split("|")
    return len(parts) > 1 and "DAY_OK=1" in [kv.strip() for kv in parts[1].split(";")]


idle_since = time.time()
retried = set()
waiting_said = False
while True:
    lines = [l.strip() for l in open(Q, encoding="utf-8") if l.strip() and not l.startswith("#")]
    done = set(l.strip() for l in open(DONE, encoding="utf-8")) if os.path.exists(DONE) else set()
    todo = [l for l in lines if l.split("|")[0].strip() not in done]
    if not todo:
        if time.time() - idle_since > IDLE_HOURS * 3600:
            print("queue empty, stopping", flush=True)
            break
        time.sleep(60)
        continue
    night = night_now()
    todo = [l for l in todo if night or day_ok(l)]
    if not todo:                                # by day with only night jobs left, wait for the night
        if not waiting_said:
            print(f"{time.strftime('%H:%M')} day: night jobs wait for the night", flush=True)
            waiting_said = True
        idle_since = time.time()
        time.sleep(60)
        continue
    waiting_said = False
    # name | env | opts | games | visits [| script]. Anything but search_match.py is driven by its
    # environment alone (elo_ladder.py is), so the three middle fields are ignored for it.
    parts = [x.strip() for x in todo[0].split("|")]
    if len(parts) < 5:
        # one unreadable line is skipped rather than stopping the whole queue
        print(f"skipping a line I cannot read: {todo[0][:80]}", flush=True)
        with open(DONE, "a", encoding="utf-8") as fh:
            fh.write(todo[0].split("|")[0].strip() + chr(10))
        continue
    name, envs, opts, games, visits = parts[:5]
    script = parts[5] if len(parts) > 5 and parts[5] else "search_match.py"
    env = dict(os.environ)
    # the openings, chess/twic in the repository unless PGN says otherwise
    env.update({"PGN": os.environ.get("PGN", os.path.join(os.path.dirname(HERE), "chess", "twic")),
                "RESULTS": os.path.join(HERE, "results.jsonl"),
                "GAMES_LOG": os.path.join(LOGS, name + ".games.jsonl"), "PYTHONUNBUFFERED": "1"})
    for kv in filter(None, envs.split(";")):
        k, _, v = kv.partition("=")
        env[k.strip()] = v.strip()
    if script == "search_match.py":
        args = [opts, games, visits]
    elif script == "tactics_suite.py":
        args = ["score", visits, opts]          # score VISITS "opts"
    else:
        args = []                               # elo_ladder reads its whole job from the env
    print(f"{time.strftime('%H:%M')} start {name}: {script} {opts} {games} {visits}", flush=True)
    t0 = time.time()
    night_job = not day_ok(todo[0])
    stopped_for_day = False
    with open(os.path.join(LOGS, name + ".log"), "a", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, script] + args,
                                cwd=SNAP, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                rc = proc.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                if night_job and not night_now():
                    # the whole tree, since a clock match runs its two engines as children
                    subprocess.call(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    rc = proc.wait()
                    stopped_for_day = True
                    break
    if stopped_for_day:
        print(f"{time.strftime('%H:%M')} day: {name} stopped after {(time.time()-t0)/3600:.2f} h, "
              f"resumes the next night", flush=True)
        continue
    print(f"{time.strftime('%H:%M')} end {name}: rc {rc}, {(time.time()-t0)/3600:.2f} h", flush=True)
    # A failed run gets one more go (a driver reset can kill a long run), and everything the queue runs
    # resumes, so a retry costs a game, not a run. Twice failed is a real failure, except that a run
    # which lasted an hour before failing made progress, so its failure counts as a fresh one.
    if rc != 0 and (name not in retried or time.time() - t0 > 3600):
        retried.add(name)
        print(f"  {name} failed; one retry, resuming where it stopped", flush=True)
        continue
    with open(DONE, "a", encoding="utf-8") as fh:
        fh.write(name + "\n")
    idle_since = time.time()
