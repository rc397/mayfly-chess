"""Debug supervise.py by calling its process listing and its trainer start command directly.

The supervisor's log records what it intended, not what happened, so this prints what procs() really
returns, compares it with a raw process listing, and can run the exact start command with its output
captured.

    python chess/dbg_supervise.py           look only
    python chess/dbg_supervise.py start     also try to start the trainer and show what came back
"""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import supervise

print("=== procs() ===", flush=True)
rows = supervise.procs()
if rows is None:
    print("  returned None: it could not read the process list at all")
    raise SystemExit(1)
print(f"  {len(rows)} python processes parsed")
cmds = [c for _, _, c in rows]
for name in ("train_av.py", "selfplay.py", "deep_label.py", "play_web.py", "supervise.py"):
    hits = [(pid, when) for pid, when, c in rows if name in c]
    print(f"  has({name}) = {supervise.has(cmds, name)}   rows: {hits}")

print("")
print("=== the same question asked straight, without the parser in the way ===", flush=True)
ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } "
      "| ForEach-Object { $_.ProcessId.ToString() + '|' + $_.CommandLine }")
out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                     capture_output=True, text=True, timeout=90)
raw = [l for l in out.stdout.splitlines() if l.strip()]
print(f"  {len(raw)} lines back, {len(out.stderr)} bytes of stderr")
for name in ("selfplay.py", "deep_label.py", "play_web.py"):
    print(f"  {name}: {sum(1 for l in raw if name in l.lower())}")
if out.stderr.strip():
    print("  stderr:", out.stderr.strip()[:400])

if len(sys.argv) > 1 and sys.argv[1] == "start":
    print("")
    print("=== the exact command start() runs for the trainer, with its output captured ===")
    env = json.load(open(supervise.LAUNCHED))["env"]
    env["RESUME"] = "1"
    log = "train_mayfly.log"
    setters = "; ".join(f"$env:{k}='{v}'" for k, v in env.items())
    cmd = (f"Start-Process -FilePath python -ArgumentList 'train_av.py' "
           f"-WorkingDirectory '{HERE}' -RedirectStandardOutput '{os.path.join(HERE, log)}' "
           f"-RedirectStandardError '{os.path.join(HERE, log[:-4] + '.err')}' -WindowStyle Hidden")
    full = f"{setters}; {cmd}"
    print("  " + full[:200] + " ...")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", full],
                       cwd=HERE, capture_output=True, text=True, timeout=120)
    print(f"  exit {r.returncode}")
    print(f"  stdout: {r.stdout.strip()[:800] or '(nothing)'}")
    print(f"  stderr: {r.stderr.strip()[:1500] or '(nothing)'}")
