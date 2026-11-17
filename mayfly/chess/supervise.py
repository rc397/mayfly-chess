r"""supervise checks the training pipeline every few minutes and restarts only what died (Windows).

The jobs recover from most failures themselves (downloads resume, converters rewind, the trainer reloads
its optimiser state) but not from their own process disappearing. This reads the state files the jobs
write for themselves, starts only what should be running, and never starts a second copy.

  syzygy      restarted while any tablebase is missing, in parallel with everything else
  dlqueue     restarted while any download is unfinished
  autostart   restarted until training has been launched
  train_av    restarted if it was launched and is no longer there, or when its checkpoint stops
              advancing, since a live process is not the same as a working one
  play_web    restarted while play_web.on exists and it cannot think about a position
  selfplay    restarted while selfplay.on exists, with the settings recorded in it
  deep_label  restarted while deep_label.on exists, likewise

There is exactly one trainer. Two trainers on one GPU and one checkpoint file destroy each other, so the
younger one is killed on sight.

A logon task can run it in a loop (_logon.ps1), so a supervisor that dies starts again within a minute,
and with SUPERVISE_RELOAD=1 one whose code changed on disk exits after its pass so the loop starts the new
code. Starting beside jobs that are already running changes nothing. Each job is recognised by its pid
file or running.json (checked against the process's age) or by its command line.

  python chess/supervise.py            watch, every five minutes
  python chess/supervise.py once       one pass, say what it would do, change nothing
  env: EVERY=300  QUEUE_STATE (default paths.under(".dlqueue.json"))
"""
import os, sys, json, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths
E = os.environ.get
# Five minutes. The detection lag adds to whatever the crash cost, and a pass is cheap (a process
# list and one request to the board).
EVERY = int(E("EVERY", 300))
QUEUE_STATE = E("QUEUE_STATE", paths.under(".dlqueue.json"))
QUEUE_JOBS = ("chessbench", "lichess", "syzygy")
LAUNCHED = os.path.join(HERE, "autostart.json")
# Training paused on purpose. While this file exists the trainer is neither started nor restarted, and a
# missing trainer is not a crash. Removing autostart.json instead would read as "never launched" and start
# a fresh run. Delete the file to resume from ckpt_<tag>_last.pt on the next pass.
PAUSED = os.path.join(HERE, "training.paused")
def _launch_env():
    """The settings training was actually launched with. A scheduled task inherits none of a shell's
    environment, so the only reliable copy is the one autostart.json recorded."""
    try:
        return json.load(open(LAUNCHED, encoding="utf-8-sig")).get("env", {})
    except Exception:
        return {}


CKPT_DIR = _launch_env().get("CKPT_DIR") or E("CKPT_DIR", HERE)
TAG = _launch_env().get("TAG") or E("TAG", "mayfly")
# Its presence says the board is meant to be up, so stopping the board on purpose (removing the file)
# does not fight the supervisor.
BOARD = os.path.join(HERE, "play_web.on")
BOARD_PORT = int(E("BOARD_PORT", 8000))
# A board on a busy card takes a few seconds to answer, so the check waits rather than calling a
# slow board a dead one.
BOARD_TIMEOUT = int(E("BOARD_TIMEOUT", 90))
PASS = E("PASS", "")
# Same idea for generation, except this marker carries the settings it was started with, so a
# restart is the same job rather than a job with the defaults.
SELFPLAY = os.path.join(HERE, "selfplay.on")
# The labeller runs for days and its death is silent, since the parent can stay alive with no
# Stockfish under it.
LABELLER = os.path.join(HERE, "deep_label.on")
# What this supervisor started, by process id. A job started over SSH lives in session 0 and the
# supervisor's scheduled task in session 1, and across that boundary Win32_Process.CommandLine comes
# back NULL. A process id is visible from anywhere.
RUNNING = os.path.join(HERE, "running.json")
# How long the resume checkpoint (or train_heartbeat) may go unwritten before the trainer counts as
# stuck, since a live process can be doing nothing. This is tied to SAVE_EVERY and must move with it,
# roughly two missed saves, or the supervisor kills the trainer before it can ever save.
STALL_MINUTES = float(E("STALL_MINUTES", 60))
# The labeller writes a progress line at least every two minutes once its workers report every
# minute, so twenty is ten missed reports rather than ordinary quiet.
LABEL_STALL_MINUTES = float(E("LABEL_STALL_MINUTES", 20))
DRY = len(sys.argv) > 1 and sys.argv[1] == "once"
# When each process in the last procs() was created, as epoch seconds, by id. Process ids are reused
# as soon as their owner is gone, so an id in a pid file proves nothing on its own. Whoever wrote a pid
# file was alive when it was written, so the real owner is older than the file and a recycled id
# belongs to a process born after it.
BORN = {}
# Allowance for the clock being stepped by time sync between a process starting and its note being
# written.
NOTE_SLACK = 2.0
SAID = set()                                         # recycled ids already reported this pass
# This file's own age, and the two modules it leans on. Python never reloads, so every pass reports
# when the code on disk has changed, and under the logon wrapper (SUPERVISE_RELOAD=1) it exits after the
# pass so the wrapper starts the new code.
CODE = [os.path.join(HERE, f) for f in ("supervise.py", "pidfile.py", "rigctl.py")]
def _mtimes():
    out = {}
    for f in CODE:
        try:
            out[f] = os.path.getmtime(f)
        except OSError:
            out[f] = None
    return out
CODE_AT_START = _mtimes()
RELOAD = E("SUPERVISE_RELOAD", "0") == "1"
RELOAD_EXIT = 75                                     # the wrapper restarts at once on this code


def changed_code():
    """The files among CODE that have changed on disk since this process started."""
    now = _mtimes()
    return [os.path.basename(f) for f in CODE if now[f] != CODE_AT_START[f]]


def procs():
    """Every running python process as (pid, started, lowercased command line), oldest first, with
    each one's exact creation time left in BORN."""
    # Out-String -Width matters. PowerShell wraps string output at the host width, which is only eighty
    # columns under a scheduled task, and that would cut the script name off every command line.
    # The third field repeats the creation time as a Windows file time (100 ns ticks, UTC). The readable
    # second field is local time to the second, fine for a log line but not for the age test in BORN.
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } "
          "| Sort-Object CreationDate | ForEach-Object "
          "{ $_.ProcessId.ToString() + '|' + $_.CreationDate.ToString('yyyy-MM-dd HH:mm:ss') "
          "+ '|' + $_.CreationDate.ToFileTimeUtc().ToString() "
          "+ '|' + $_.CommandLine } | Out-String -Width 4096")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=90).stdout
    except Exception:
        return None                                  # cannot tell, so change nothing
    rows = []
    BORN.clear()
    for line in out.splitlines():
        parts = line.strip().split("|", 3)
        if len(parts) == 4 and parts[0].isdigit():
            rows.append((int(parts[0]), parts[1], parts[3].lower()))
            if parts[2].isdigit():
                BORN[int(parts[0])] = int(parts[2]) / 1e7 - 11644473600.0
    # This supervisor runs supervise.py itself, so a list with no .py file in it is a broken read, and
    # acting on it would start a second copy of everything. Returning None changes nothing.
    if not any(".py" in c for _, _, c in rows):
        return None
    return rows


CRASHES = os.path.join(HERE, "crashes")


def keep_crash(name):
    """Save what the dead process said before its log is overwritten by the new one.

    Start-Process truncates the file it redirects to, so without this every restart would destroy
    the evidence for the crash that caused it.
    """
    os.makedirs(CRASHES, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(CRASHES, f"{name}_{stamp}.txt")
    kept = 0
    with open(out, "w", encoding="utf-8", errors="replace") as f:
        for ext in (".log", ".err"):
            src = os.path.join(HERE, name + ext)
            if not os.path.exists(src):
                continue
            body = open(src, encoding="utf-8", errors="replace").read()
            f.write(f"===== {name}{ext}, {len(body):,} bytes =====\n")
            f.write(body[-20000:])
            f.write(chr(10))
            kept += 1
    if kept:
        print(f"    kept what it said in crashes/{os.path.basename(out)}", flush=True)


def running():
    try:
        return json.load(open(RUNNING, encoding="utf-8-sig"))
    except Exception:
        return {}


def remember(job, pid):
    d = running()
    d[job] = int(pid)
    # When it was written down, so a recycled id can be told from the job (see BORN). Kept under
    # its own key so the job to pid entries keep their format.
    at = d.get("_at") if isinstance(d.get("_at"), dict) else {}
    at[job] = time.time()
    d["_at"] = at
    json.dump(d, open(RUNNING, "w"), indent=1)


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def mine(plist, job):
    """The live process id this job is recorded under, or None. Two records, in order of trust.

    1. The job's own pid file, written by pidfile.claim when it starts, whoever launched it.
    2. The pid this supervisor recorded when it started the job itself (running.json).

    A record counts only if its process is live and no younger than the record (see BORN), so an id
    recycled onto a newer process is nobody's. When the age cannot be read the record is trusted,
    because calling a live job dead would start a second copy of it.
    """
    from pidfile import read as claimed, path_for
    live = {p for p, _, _ in plist}
    d = running()
    at = d.get("_at") if isinstance(d.get("_at"), dict) else {}
    for pid, noted in ((claimed(job.replace(".py", "")), _mtime(path_for(job.replace(".py", "")))),
                       (d.get(job), at.get(job) or _mtime(RUNNING))):
        if not pid or pid not in live:
            continue
        if noted is not None and BORN.get(pid) is not None and BORN[pid] > noted + NOTE_SLACK:
            if (job, pid) not in SAID:
                SAID.add((job, pid))
                print(f"    pid {pid} was recorded for {job} but that process is younger than "
                      f"the record, so the id was recycled: not {job}", flush=True)
            continue
        return pid
    return None


def up(plist, job):
    """Is this job running? Its recorded process id (mine), and failing that the command line,
    which is readable only within a session."""
    if mine(plist, job):
        return True
    return has([c for _, _, c in plist], job)


def kill_tree(roots):
    """Kill these processes and everything under them, children first, and nothing else.

    Never by image name, which would also kill other jobs' Stockfish (a match in progress would score
    its dead engine's games as draws). A tree reaches the labeller's own Stockfish through its workers
    and nobody else's.

    The tree comes from rigctl, which follows a parent link only when the child is no younger than
    the parent, so a recycled parent id cannot pull in a stranger. If the process table cannot be
    read, only the roots are killed.
    """
    roots = [int(r) for r in dict.fromkeys(roots) if r]
    if not roots:
        return []
    victims = []
    try:
        from rigctl import snapshot, descendants
        snap = snapshot(everything=True)
        for r in roots:
            victims += [v for v in descendants(snap, r) if v not in victims and v not in roots]
    except Exception as e:
        print(f"    could not read the process tree ({type(e).__name__}), killing the parents only",
              flush=True)
    victims += roots
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"Stop-Process -Id {','.join(str(v) for v in victims)} -Force "
                    f"-ErrorAction SilentlyContinue"], capture_output=True, timeout=120)
    return victims


def survivors(pids, tries=6):
    """Which of these processes are still the same live processes after a kill (same id and same
    creation time, so a recycled id is not mistaken for a survivor). Waits a few seconds for them to
    go. Returns [] only when every one is provably gone. A process list that cannot be read counts as
    everything surviving, because the caller starts a second copy on the strength of this answer."""
    was = {p: BORN.get(p) for p in pids}
    kept = dict(BORN)                  # the rest of this pass reads BORN against its own process list
    left = list(pids)
    try:
        for i in range(tries):
            if i:
                time.sleep(2)
            rows = procs()
            if rows is None:
                left = list(pids)
                continue
            live = {p for p, _, _ in rows}
            left = [p for p in pids if p in live and (was[p] is None or BORN.get(p) == was[p])]
            if not left:
                break
    finally:
        BORN.clear()
        BORN.update(kept)
    return left


def has(cmds, name):
    """A command line that runs this script, with the name as a whole path component, so that
    pack_selfplay.py is not read as selfplay.py, nor download_syzygy.py as syzygy.py."""
    import re
    pat = re.compile(r"(^|[\s\"'\\/])" + re.escape(name) + r"($|[\s\"'])")
    return any(pat.search(c) for c in cmds)


def start(script, env=None, log=None):
    log = log or script.replace(".py", ".log")
    setters = "; ".join(f"$env:{k}='{v}'" for k, v in (env or {}).items())
    cmd = (f"Start-Process -FilePath python -ArgumentList '{script}' "
           f"-WorkingDirectory '{HERE}' -RedirectStandardOutput '{os.path.join(HERE, log)}' "
           f"-RedirectStandardError '{os.path.join(HERE, log[:-4] + '.err')}' -WindowStyle Hidden")
    if DRY:
        print(f"    would start {script}", flush=True)
        return
    # The new process id comes back through a file, not a pipe. Start-Process hands its child handles
    # that a captured pipe would wait on forever.
    pidfile = os.path.join(HERE, ".pid_" + script.replace(".py", ".txt"))
    try:
        os.remove(pidfile)
    except OSError:
        pass
    cmd = f"$p = {cmd} -PassThru; Set-Content -Path '{pidfile}' -Value $p.Id"
    # None of the supervisor's handles may reach the job. Otherwise every job would hold the pipe
    # _logon.ps1 reads the supervisor through, and the wrapper would never see a reload or a crash.
    # A failed start still shows, as "reported no pid" below.
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"{setters}; {cmd}" if setters else cmd], cwd=HERE,
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid = None
    try:
        pid = int(open(pidfile).read().strip())
        remember(script, pid)
    except Exception:
        pass
    print(f"    started {script}" + (f", pid {pid}" if pid else " but it reported no pid"),
          flush=True)


SYZYGY_DIR = E("SYZYGY_DIR", paths.under("syzygy"))
SYZYGY_FILES = int(E("SYZYGY_FILES", 1020))


def syzygy_done():
    """All 1,020 tables recorded in the download manifest."""
    try:
        return len(json.load(open(os.path.join(SYZYGY_DIR, ".fetched.json"), encoding="utf-8-sig"))) >= SYZYGY_FILES
    except Exception:
        return False


def downloads_done():
    try:
        return set(QUEUE_JOBS) <= set(json.load(open(QUEUE_STATE, encoding="utf-8-sig")).get("done", []))
    except Exception:
        return False


def board_answers():
    """Whether the board can still play, not just whether its process is alive or its port answers.

    A board can keep its process with every connection refused, or answer on its port while its model
    is dead after a GPU fault. So this asks it to think about a position.
    """
    import urllib.request
    # /health, not /state. The board gives the GPU back after an hour with nobody playing, and a check
    # that counted as a player would keep it awake forever. /health does a real forward pass when the
    # board is awake and reports "sleeping" as healthy, since the next move wakes it.
    req = urllib.request.Request(f"http://127.0.0.1:{BOARD_PORT}/api/health", data=b"{}",
                                 headers={"Content-Type": "application/json"})
    # The passphrase the board was started with, the one in play_web.on. A scheduled task has no PASS
    # of its own, and without this every check would get a 401 and restart the board.
    pw = PASS
    try:
        pw = str(json.load(open(BOARD, encoding="utf-8-sig")).get("PASS") or PASS)
    except Exception:
        pass
    if pw:
        req.add_header("X-Pass", pw)
    try:
        body = urllib.request.urlopen(req, timeout=BOARD_TIMEOUT).read()
        alive = b'"awake"' in body or b'"sleeping"' in body
        try:
            h = json.loads(body.decode("utf-8", "replace"))
        except Exception:
            return alive, [], 0
        return alive, h.get("stale") or [], int(h.get("players") or 0)
    except Exception:
        # The same tuple shape as every other answer. check() indexes it while building its status
        # line, so a bare False would abort the whole pass.
        return False, [], 0


def check():
    SAID.clear()
    stale_code = changed_code()
    if stale_code:
        print(f"  THIS SUPERVISOR IS RUNNING OLD CODE: {', '.join(stale_code)} changed on disk "
              f"after it started" + (", reloading after this pass" if RELOAD and not DRY
                                     else "; restart it to pick the change up"), flush=True)
    plist = procs()
    if plist is None:
        print("  cannot read the process list, leaving everything alone", flush=True)
        return
    cmds = [c for _, _, c in plist]
    dl_done = downloads_done()
    launched = os.path.exists(LAUNCHED)
    paused = os.path.exists(PAUSED)
    # asked once a pass and reused below, since a board that is down costs the full BOARD_TIMEOUT
    alive, stale, players = board_answers() if os.path.exists(BOARD) else (True, [], 0)
    state = (f"downloads {'done' if dl_done else 'running'}, "
             f"queue {'up' if up(plist, 'dlqueue.py') else 'DOWN'}, "
             f"watcher {'up' if up(plist, 'autostart.py') else 'down'}, "
             f"trainer {'up' if up(plist, 'train_av.py') else 'down'}"
             f"{' (PAUSED on purpose)' if paused else ''}, "
             f"{sum(1 for c in cmds if 'chessbench.py' in c)} converter(s)"
             + (", generator " + ("up" if up(plist, "selfplay.py") else "DOWN")
                if os.path.exists(SELFPLAY) else "")
             + (", labeller " + ("up" if up(plist, "deep_label.py") else "DOWN")
                if os.path.exists(LABELLER) else "")
             + (", board " + ("up" if alive else "DOWN")
                if os.path.exists(BOARD) else ""))
    print(f"  {time.strftime('%H:%M:%S')}  {state}", flush=True)

    # Syzygy runs beside the Lichess conversion rather than after it. Converting is processor work and
    # downloading is network work, so they do not compete. The supervisor owns this one, not the queue.
    if not syzygy_done() and not up(plist, "syzygy.py"):
        print("    the tablebases are incomplete and nothing is fetching them", flush=True)
        start("syzygy.py", {"WORKERS": E("SYZ_WORKERS", "6"),
                            "MAX_MBPS": E("SYZ_MBPS", "20")}, "syzygy.log")

    if not dl_done and not up(plist, "dlqueue.py"):
        print("    the download queue is not running and the downloads are not finished",
              flush=True)
        start("dlqueue.py", {"SYZYGY_MBPS": E("SYZYGY_MBPS", "150")}, "dlqueue.log")

    if not launched and not up(plist, "autostart.py"):
        print("    training has not been launched and nothing is waiting to launch it",
              flush=True)
        start("autostart.py", {"NOT_BEFORE_HOURS": E("NOT_BEFORE_HOURS", "0")}, "autostart.log")

    # One trainer, never two. The younger one goes, since the checkpoint on disk belongs to the older.
    # Not named `running`, which is the function holding the recorded process ids.
    trainers = [(pid, when) for pid, when, c in plist if "train_av.py" in c]
    if len(trainers) > 1:
        print(f"    {len(trainers)} trainers are running and there must be one", flush=True)
        for pid, when in trainers[1:]:
            print(f"      killing the one started at {when}, pid {pid}", flush=True)
            if not DRY:
                kill_tree([pid])
        trainers = trainers[:1]

    def started_at(job):
        """When the running copy of this job started, as epoch seconds, or 0 if unknown.

        Looked up by process id, not by command line, because from a scheduled task the command line
        of a job started in another session is not readable. A recycled id is not the job (mine).
        """
        pid = mine(plist, job)
        for p, when, c in plist:
            if p == pid or (pid is None and job in c):
                if BORN.get(p) is not None:
                    return BORN[p]
                try:
                    return time.mktime(time.strptime(when, "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    return 0.0
        return 0.0

    def stop(job):
        """Kill every copy of this job with its tree, the recorded one (mine), which is the only one
        visible from another session, and any whose command line names it. True when all of them are
        provably gone, and only then does the caller start a replacement. A job whose kill did not
        take is left for the next pass rather than joined by a second copy."""
        roots = [p for p, _, c in plist if job in c and "spawn_main" not in c]
        pid = mine(plist, job)
        if pid and pid not in roots:
            roots.append(pid)
        if DRY or not roots:
            return True
        kill_tree(roots)
        left = survivors(roots)
        if left:
            print(f"    {job} did not die (pid {', '.join(map(str, left))}), NOT starting a "
                  f"second copy beside it", flush=True)
        return not left

    stalled = False
    stopped = True
    # The launched configuration as it is now. A stale CKPT_DIR or TAG would point the test at a file
    # the trainer no longer writes and kill a healthy trainer. No readable configuration, no stall test.
    lenv = _launch_env()
    if launched and not paused and lenv and up(plist, "train_av.py"):
        try:
            last = os.path.getmtime(os.path.join(lenv.get("CKPT_DIR") or E("CKPT_DIR", HERE),
                                                 f"ckpt_{lenv.get('TAG') or E('TAG', 'mayfly')}_last.pt"))
        except OSError:
            last = 0.0
        # A trainer standing aside for the release gate (aside.py) saves nothing for hours but touches
        # train_heartbeat every 30 s while it waits, which counts as alive
        try:
            last = max(last, os.path.getmtime(os.path.join(lenv.get("CKPT_DIR") or E("CKPT_DIR", HERE),
                                                           "train_heartbeat")))
        except OSError:
            pass
        # From whichever is later, the last checkpoint or the moment this trainer started. Timing it
        # from the checkpoint alone would kill a fresh trainer before its first save, forever.
        since = time.time() - max(last, started_at("train_av.py"))
        stalled = since > STALL_MINUTES * 60
        if stalled:
            print(f"    the trainer is alive but nothing has advanced in {since/60:.0f} minutes, "
                  f"treating it as stuck", flush=True)
            # its recorded id only if that id is really the trainer (mine), never a recycled one
            stopped = stop("train_av.py")

    if launched and paused and not up(plist, "train_av.py"):
        print(f"    training is paused ({os.path.basename(PAUSED)}), not starting it", flush=True)
    if launched and not paused and stopped and (stalled or not up(plist, "train_av.py")):
        tag = "mayfly"
        try:
            tag = json.load(open(LAUNCHED, encoding="utf-8-sig"))["env"].get("TAG", tag)
        except Exception:
            pass
        # train_av resumes from its own last checkpoint, optimiser state included, so restarting it
        # costs at most the steps since the last evaluation.
        print(f"    training was launched and is gone, bringing it back (it resumes from "
              f"ckpt_{tag}_last.pt)", flush=True)
        if not DRY:
            keep_crash(f"train_{tag}")
        try:
            env = json.load(open(LAUNCHED, encoding="utf-8-sig"))["env"]
        except Exception as e:
            # Never guess. Default settings without CKPT_DIR would find no checkpoint and start the
            # model again from zero. (A file written by PowerShell may carry a UTF-8 byte order mark,
            # hence utf-8-sig.)
            print(f"    {os.path.basename(LAUNCHED)} cannot be read ({type(e).__name__}: {e}); "
                  f"NOT starting training on guessed settings", flush=True)
            env = None
        # Only the trainer waits for a readable autostart.json. The rest of the pass carries on.
        if env is not None:
            env["RESUME"] = "1"
            start("train_av.py", env, f"train_{tag}.log")

    if os.path.exists(SELFPLAY) and not up(plist, "selfplay.py"):
        try:
            env = json.load(open(SELFPLAY, encoding="utf-8-sig"))
        except Exception:
            env = {}
        # selfplay counts the lines already in its output file and continues from there, so a
        # restart costs the game it was in the middle of and nothing else.
        print("    generation is meant to be running and is gone, bringing it back", flush=True)
        if not DRY:
            keep_crash("selfplay")
        start("selfplay.py", env, "selfplay.log")

    # Alive is not working, for the labeller too. Its parent can look healthy while a worker is blocked
    # inside a Stockfish that stopped answering. Its log gets a line at least every two minutes, so a
    # silent log means it is stuck.
    label_stalled = False
    label_stopped = True
    if os.path.exists(LABELLER) and up(plist, "deep_label.py"):
        try:
            idle = time.time() - os.path.getmtime(os.path.join(HERE, "deep_label.log"))
            label_stalled = idle > LABEL_STALL_MINUTES * 60
        except OSError:
            pass
        if label_stalled:
            print(f"    labelling is alive but has said nothing in {idle/60:.0f} minutes, "
                  f"treating it as stuck", flush=True)
            # Its own tree and nothing else, the labeller, its workers and their Stockfish (see
            # kill_tree), found by recorded id as well as by command line.
            label_stopped = stop("deep_label.py")

    if os.path.exists(LABELLER) and label_stopped and (label_stalled
                                                       or not up(plist, "deep_label.py")):
        try:
            env = json.load(open(LABELLER, encoding="utf-8-sig"))
        except Exception:
            env = {}
        # deep_label keeps its own place in .labelled.json, so a restart repeats at most the batch
        # it was in the middle of.
        print("    labelling is meant to be running and is gone, bringing it back", flush=True)
        if not DRY:
            keep_crash("deep_label")
        start("deep_label.py", env, "deep_label.log")

    # A stale board (running code that was replaced on disk) is restarted like a dead one, but not
    # while somebody is mid game. It is reported every pass and left alone until the last player goes.
    if os.path.exists(BOARD) and alive and stale:
        if players:
            print(f"    the board is STALE ({', '.join(stale)}) but {players} player(s) are on "
                  f"it, leaving it until they finish", flush=True)
        else:
            print(f"    the board is STALE, running code older than {', '.join(stale)}, "
                  f"and nobody is playing: restarting it", flush=True)
            alive = False
    if os.path.exists(BOARD) and not alive:
        print("    the board is meant to be up and cannot think about a position", flush=True)
        try:
            env = json.load(open(BOARD, encoding="utf-8-sig"))
        except Exception as e:
            # Never guess a version. The board stays as it is until the marker can be read.
            print(f"    {os.path.basename(BOARD)} cannot be read ({type(e).__name__}: {e}); "
                  f"NOT starting a board on guessed settings", flush=True)
            env = None
        if env is None:
            pass
        elif DRY:
            print("    would restart the board", flush=True)
        else:
            keep_crash("play")
            # start(), the same fire and forget path as everything else, not a captured pipe that the
            # board would inherit and hold open. The old board goes by its recorded id as well as by
            # name, and a new one starts only once the old is provably gone, since Windows lets two
            # processes share port 8000 and the new one would get no traffic.
            if stop("play_web.py"):
                start("play_web.py", env, "play.log")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "adopt":
        # Run this from an ordinary shell, where command lines are readable, to hand the supervisor
        # the ids of things that are already running. Without it the first pass after a code change
        # sees nothing it recognises and starts a second copy of everything.
        plist = procs() or []
        found = {}
        for job in ("train_av.py", "selfplay.py", "deep_label.py", "play_web.py"):
            # the script itself, not a name inside another's (pack_selfplay.py is not selfplay.py),
            # and the longest running copy, by start time rather than by pid
            hits = sorted((w, p) for p, w, c in plist if has([c], job))
            if hits:
                found[job] = hits[0][1]
                remember(job, hits[0][1])
                print(f"  adopted {job:<16} pid {hits[0][1]}, started {hits[0][0]}")
            else:
                print(f"  {job:<16} not running")
        print(f"  wrote {RUNNING}")
        raise SystemExit(0)

    if DRY:
        print("one pass, changing nothing:")
        check()
        raise SystemExit(0)
    print(f"watching every {EVERY // 60} minutes"
          + (", reloading when my code changes" if RELOAD else ""), flush=True)
    while True:
        try:
            check()
        except Exception as e:
            print(f"  check failed, {type(e).__name__}: {e}", flush=True)
        # After a full pass, never in the middle of one, so a reload cannot leave a kill without its
        # restart. Only under the wrapper (_logon.ps1), which starts the new code on this exit code.
        if RELOAD and changed_code():
            print("  exiting so the logon wrapper starts the new code", flush=True)
            raise SystemExit(RELOAD_EXIT)
        time.sleep(EVERY)
