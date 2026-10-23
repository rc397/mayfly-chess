r"""
Start and stop this project's jobs on Windows without leaving orphans behind.

Killing a multiprocessing parent on Windows does not kill its workers, and their command line is a
generic spawn_main stub that never names the script. Orphaned Stockfish workers then hold the CPU.
So stopping a job means stopping its tree, finding the parent, collecting every descendant, and
killing the children first so none is re-parented and survives, then the parent.

A parent id is only a clue. Windows records ParentProcessId at creation and never clears it, and
process ids are reused. A real child is never older than its parent, so a link to a parent created
after the child is a recycled id and is not followed. The tree is read from every process, not just
python, because the workers' Stockfish children hold the cores.

  python chess/rigctl.py status                  what is running
  python chess/rigctl.py stop gen_selfplay       one job and all its workers
  python chess/rigctl.py stop-all                every job this project started, supervisor too
  python chess/rigctl.py orphans                 workers whose parent is already gone

stop-all first stops the scheduled task named by SUPERVISOR_TASK, if it exists, since a wrapper that
restarts a dead supervisor would otherwise bring it back within a minute.
"""
import os, sys, subprocess, time, json

TASK = os.environ.get("SUPERVISOR_TASK", "MayFly-Supervisor")

OURS = ("gen_selfplay", "gen_packed", "gen_gm_rich", "train_chess_long", "train_real",
        "ablate_mixer", "sweep_width", "full_sweep", "train_ttt", "finisher", "sf_match",
        "sf_time_ladder", "chess_strength", "train_engine", "lc0_queue", "lc0_convert",
        "fetch_lichess_evals", "chessbench", "eqdepth", "mcts", "train_av", "play_web",
        "dlqueue", "autostart", "supervise", "syzygy", "analyse",
        "selfplay", "deep_label", "lz_parallel", "curve")

# Created is the creation time as a Windows file time (100 ns ticks, UTC), an exact integer with no
# time zone or daylight saving, which the parent check compares. A process with no creation time
# (the idle process) gets null and is never judged by it.
_SELECT = (" | Select-Object ProcessId,ParentProcessId,Name,CommandLine,"
           "@{n='Created';e={ if ($_.CreationDate) { $_.CreationDate.ToFileTimeUtc() } else { $null } }}"
           " | ConvertTo-Json -Compress")
PS = "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' }" + _SELECT
PS_ALL = "Get-CimInstance Win32_Process" + _SELECT


def _flagged(cl, name):
    """The value after a named flag on a command line, for telling stripes of one job apart."""
    bits = cl.split()
    try:
        return bits[bits.index(name) + 1]
    except (ValueError, IndexError):
        return ""


def snapshot(everything=False):
    """Python processes, or with everything=True every process on the machine.

    Collect a tree from the full snapshot before killing it."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command", PS_ALL if everything else PS],
                         capture_output=True, text=True, timeout=90).stdout.strip()
    if not out:
        return []
    d = json.loads(out)
    return d if isinstance(d, list) else [d]


def label(cl):
    cl = cl or ""
    for name in OURS:
        if name in cl:
            tag = _flagged(cl, "--tag")
            return f"{name}:{tag}" if tag else name
    return "worker" if "spawn_main" in cl else "other"


def descendants(procs, root):
    """Every process under root, deepest first, so children die before their parent.

    A child is followed only if it is no older than the process it names as its parent (see the
    top of this file). The same check keeps the walk finite, since with recycled ids two processes
    can name each other as parent. A root missing from the snapshot has already gone, and whatever
    still names its id is not provably its child, so it has no tree here.
    """
    born = {p["ProcessId"]: p.get("Created") for p in procs}
    if root not in born:
        return []
    kids = {}
    for p in procs:
        kids.setdefault(p["ParentProcessId"], []).append(p["ProcessId"])
    out, stack, seen = [], [root], {root}
    while stack:
        pid = stack.pop()
        for c in kids.get(pid, []):
            if c in seen:
                continue
            if born.get(c) is not None and born.get(pid) is not None and born[c] < born[pid]:
                continue                  # the parent id was recycled after this one's parent died
            seen.add(c)
            out.append(c)
            stack.append(c)
    return out[::-1]                      # reversed pre-order: every process after its descendants


def orphaned(p, by_pid):
    """A worker whose parent is gone, including when the parent's id now belongs to a newer
    process."""
    par = by_pid.get(p["ParentProcessId"])
    if par is None:
        return True
    return (p.get("Created") is not None and par.get("Created") is not None
            and par["Created"] > p["Created"])


def kill(pids):
    if not pids:
        return 0
    args = ",".join(str(p) for p in pids)
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"Stop-Process -Id {args} -Force -ErrorAction SilentlyContinue"],
                   capture_output=True, timeout=90)
    return len(pids)


def cmd_status():
    procs = snapshot()
    groups = {}
    for p in procs:
        groups.setdefault(label(p["CommandLine"]), []).append(p["ProcessId"])
    print(f"{len(procs)} python processes")
    for k in sorted(groups, key=lambda k: -len(groups[k])):
        print(f"  {k:22s} {len(groups[k]):4d}")
    by_pid = {p["ProcessId"]: p for p in snapshot(everything=True)} or {p["ProcessId"]: p for p in procs}
    orph = [p for p in procs if label(p["CommandLine"]) == "worker" and orphaned(p, by_pid)]
    if orph:
        print(f"\n{len(orph)} ORPHANED workers, parent already gone. "
              f"Run: python chess/rigctl.py orphans")


def cmd_stop(name):
    # Every process, not python only. The roots are python jobs found by name, but the tree under
    # them runs through Stockfish and anything else they started.
    procs = snapshot(everything=True)
    roots = [p["ProcessId"] for p in procs if (p.get("Name") or "").lower().startswith("python")
             and name in (p["CommandLine"] or "") and "spawn_main" not in (p["CommandLine"] or "")]
    if not roots:
        print(f"no running {name}")
        return
    victims = []
    for r in roots:
        victims += [v for v in descendants(procs, r) if v not in victims and v not in roots]
    kids = len(victims)
    victims += roots                                  # parents last
    print(f"stopping {name}: {len(roots)} parent(s), {kids} worker(s)")
    kill(victims)
    time.sleep(4)
    # Still alive means the same id and the same creation time, since a recycled id is another process.
    was = {p["ProcessId"]: p.get("Created") for p in procs}
    left = [p for p in snapshot(everything=True)
            if p["ProcessId"] in set(victims) and p.get("Created") == was.get(p["ProcessId"])]
    print(f"  {len(left)} still alive" if left else "  all stopped")


def cmd_orphans():
    procs = snapshot()
    by_pid = {p["ProcessId"]: p for p in snapshot(everything=True)} or {p["ProcessId"]: p for p in procs}
    orph = [p["ProcessId"] for p in procs
            if label(p["CommandLine"]) == "worker" and orphaned(p, by_pid)]
    print(f"killing {len(orph)} orphaned workers")
    kill(orph)


def stop_task():
    """Stop the logon task that keeps the supervisor alive, so stop-all means stop."""
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Stop-ScheduledTask -TaskName {TASK} -ErrorAction Stop; 'stopped'"],
                       capture_output=True, text=True, timeout=90)
    if "stopped" in r.stdout:
        print(f"stopped the {TASK} task, so the supervisor stays down")
    else:
        why = " ".join((r.stderr or "").split())[:160]
        print(f"could NOT stop the {TASK} task ({why or 'no reason given'}): if it is "
              f"running, its wrapper restarts the supervisor within a minute")


if __name__ == "__main__":
    a = sys.argv[1:] or ["status"]
    if a[0] == "status":
        cmd_status()
    elif a[0] == "orphans":
        cmd_orphans(); cmd_status()
    elif a[0] == "stop" and len(a) > 1:
        cmd_stop(a[1]); cmd_status()
    elif a[0] == "stop-all":
        stop_task()
        for n in OURS:
            cmd_stop(n)
        cmd_orphans(); cmd_status()
    else:
        print(__doc__)
