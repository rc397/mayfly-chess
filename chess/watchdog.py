"""
watchdog brings the tropical sweep pipeline back up after a reboot.

The trainer resumes from its own checkpoint and full_sweep.py relaunches it if it dies, but a reboot takes
the parent processes with it. This script looks at what is running, starts only what is missing, and exits
at once when the pipeline has finished. It is safe to run every few minutes, at boot, or twice at once.

  python chess/watchdog.py          check once and start anything missing
  python chess/watchdog.py --dry    say what it would do and change nothing
"""
import os, sys, json, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DRY = "--dry" in sys.argv
DETACHED = 0x00000008 | 0x00000200      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
BSZ, D, L = "128", "256", "4"


def running():
    """Command lines of every live python process, lowercased."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } "
          "| ForEach-Object { $_.CommandLine }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    return [ln.strip().lower() for ln in out.splitlines() if ln.strip()]


def arch_done(arch):
    p = os.path.join(HERE, f"long_{arch}.log")
    if not os.path.exists(p):
        return False
    s = open(p, encoding="utf-8", errors="ignore").read()
    return ("DONE MAYFLY" in s) or ("DONE CONTROL" in s)


def spawn(script, env_extra, log_name):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    target = os.path.join(HERE, script)
    if DRY:
        print(f"would start {script} with {env_extra}")
        return
    f = open(os.path.join(HERE, log_name), "a", encoding="utf-8")
    f.write(f"\n===== watchdog restart {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    f.flush()
    subprocess.Popen([sys.executable, target, "400000", BSZ, D, L] if script.startswith("train_")
                     else [sys.executable, target],
                     cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT,
                     creationflags=DETACHED, close_fds=True)
    print(f"started {script}")


if os.path.exists(os.path.join(HERE, "FINAL_SUMMARY.json")):
    print("pipeline already complete, nothing to do")
    sys.exit(0)

live = running()
started = []

for arch in ("trop", "ctrl"):
    if arch_done(arch):
        continue
    busy = any("train_chess_long" in c and f"tag={arch}" not in c for c in live)
    owned = any("train_chess_long" in c or "full_sweep" in c for c in live)
    if owned:
        continue
    sweep = os.path.join(HERE, f"sweep_{arch}.json")
    if not os.path.exists(sweep):
        print(f"[{arch}] no tuned config on disk yet, leaving it alone")
        continue
    cfg = json.load(open(sweep))["final"]
    env = dict(cfg)
    env.update({"ARCH": arch, "TAG": arch, "VAL_EVERY": "1000", "VAL_BATCHES": "20",
                "CRAWL_WINDOW": "10", "CRAWL_THRESH": "0.05", "MAX_HOURS": "6", "RESUME": "1"})
    spawn("train_chess_long.py", env, f"long_{arch}.log")
    started.append(arch)

if not any("finisher" in c for c in live):
    spawn("finisher.py", {}, "finisher.log")
    started.append("finisher")

if not started:
    print("everything that should be running is running")
