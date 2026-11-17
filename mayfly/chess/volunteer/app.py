"""MayFly Volunteer: lend your graphics card to train MayFly. Press Run and it plays practice games against itself
(self-play) on the card; the games become MayFly's training data and count toward your place on the leaderboard.

How it fits together
  the window     ui/index.html, shown by pywebview (Edge WebView2, part of Windows 10 and 11) or, without it, in
                 the default browser. It talks to this file over a small local server on 127.0.0.1.
  the engine     this same program started again with --engine: selfplay_remote with the "onnx" server (the
                 network in ONNX Runtime on DirectML, so any NVIDIA, AMD or Intel card) and its search workers
                 on torch_stub (no PyTorch). Same sealed search, same settings as Kaggle's self-play.
  your data      %LOCALAPPDATA%\\MayFly: settings.json (your name and a private key that proves uploads are yours),
                 the network, and out\\ with the games until they are sent.

  python chess/volunteer/app.py            the app
  python chess/volunteer/app.py --engine   the engine alone (what Run starts)
"""
import collections
import http.server
import json
import multiprocessing
import os
import re
import secrets
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser

FROZEN = getattr(sys, "frozen", False)
# A windowed build starts every process (the engine and its server and workers too) with no sys.stdout or
# sys.stderr. The engine's processes are given the app's pipe, so they get it back here and their errors reach the
# app; the app itself has no pipe and keeps None.
for _name, _fd in (("stdout", 1), ("stderr", 2)):
    if getattr(sys, _name) is None:
        try:
            setattr(sys, _name, open(_fd, "w", encoding="utf-8", buffering=1, closefd=False))
        except OSError:
            pass
# The engine's server and workers (multiprocessing children) start with no output at all on Windows, so whatever
# they print goes to a log file of their own. MAYFLY_DEBUG=N also dumps every thread's stack there after N seconds.
if "--multiprocessing-fork" in sys.argv:
    try:
        _logs = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "MayFly", "logs")
        os.makedirs(_logs, exist_ok=True)
        _log = open(os.path.join(_logs, f"child_{os.getpid()}.log"), "w", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = _log
        if os.environ.get("MAYFLY_DEBUG"):
            import faulthandler
            faulthandler.dump_traceback_later(int(os.environ["MAYFLY_DEBUG"]), file=_log)
    except OSError:
        pass
HERE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
CHESS = HERE if FROZEN else os.path.dirname(HERE)       # selfplay_remote and the search code
ASSETS = os.path.join(HERE, "ui")
KIT = os.path.join(HERE, "kit")             # versions.json (the sealed searches), corpus_starts.txt
DATA = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "MayFly")
VERSION = "mayfly_v6.1.pt"
MODEL = os.path.join(DATA, "mayfly_v6.1.onnx")
# the network, from the public repo's release; checked against its SHA-256 before it is used
MODEL_URL = "https://github.com/rc397/mayfly-chess/releases/download/v6.1/mayfly_v6.1.onnx"
MODEL_SHA256 = "1e6972836c8657a26a0128790c83921974de264983946782dc9939360f0ecbc7"
SITE = "https://rc397.github.io/mayfly-chess/"
APP_VERSION = "0.2"
# how much of the computer each level gives: the share of CPU cores running searches (the card follows them)
LEVELS = {"light": 0.25, "half": 0.5, "full": 0.9}


def games_at_once(level):
    """One search process per game, on this share of the cores."""
    return max(1, int((os.cpu_count() or 4) * LEVELS.get(level, 0.5)))

sys.path.insert(0, CHESS)
# a source checkout can keep onnxruntime and pywebview in a folder of their own (MAYFLY_LIBS); appended, so they
# never shadow the packages already installed
for _p in filter(None, os.environ.get("MAYFLY_LIBS", "").split(os.pathsep)):
    sys.path.append(_p)


# ---------------------------------------------------------------- settings

def load_settings():
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, "settings.json")
    try:
        s = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        s = {}
    s.setdefault("key", secrets.token_hex(16))
    s.setdefault("name", "")
    s.setdefault("gpu", 0)
    s.setdefault("level", "half")
    save_settings(s)
    return s


def save_settings(s):
    p = os.path.join(DATA, "settings.json")
    with open(p + ".part", "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=1)
    os.replace(p + ".part", p)


def clean_name(name):
    """Letters, digits, spaces, dots, dashes and underscores, at most 24: it is shown on a public page."""
    return re.sub(r"[^A-Za-z0-9 ._-]", "", name or "").strip()[:24]


# ---------------------------------------------------------------- the network, the first time

class Download:
    """MODEL_URL into MODEL: resumes a part file, checks the SHA-256, and only then puts the file in place."""

    def __init__(self):
        self.done = os.path.exists(MODEL)
        self.active = False
        self.frac = 0.0
        self.error = ""

    def start(self, then=None):
        if self.done or self.active:
            return
        self.active, self.error = True, ""
        threading.Thread(target=self._run, args=(then,), daemon=True).start()

    def _run(self, then):
        import hashlib
        import urllib.request
        part = MODEL + ".part"
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(MODEL_URL, headers={"User-Agent": f"MayFly-Volunteer/{APP_VERSION}",
                                                             **({"Range": f"bytes={have}-"} if have else {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    have = 0                  # the server ignored the range: start again
                total = have + int(r.headers.get("Content-Length") or 0)
                with open(part, "ab" if have else "wb") as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        self.frac = have / total if total else 0.0
            h = hashlib.sha256()
            with open(part, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            if h.hexdigest() != MODEL_SHA256:
                os.remove(part)
                raise ValueError("the download was damaged")
            os.replace(part, MODEL)
            self.done = True
            if then:
                then()
        except Exception as e:
            self.error = f"Couldn't download MayFly's network ({e}). Check your internet and press Run again."
        finally:
            self.active = False


# ---------------------------------------------------------------- the engine, from the app's side

class Engine:
    """The --engine process: started by Run, stopped by Stop, its progress read from what it prints."""

    def __init__(self):
        self.proc = None
        self.started = None
        self.games = self.positions = 0
        self.rate = self.evals = 0
        self.recent = collections.deque(maxlen=6)     # the last minute of speed reports
        self.status = "idle"
        self.lines = []
        self.error = ""
        self.stopping = False

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, settings):
        if self.running():
            return
        cmd = [sys.executable, "--engine"] if FROZEN else [sys.executable, os.path.abspath(__file__), "--engine"]
        env = dict(os.environ, MAYFLY_GPU=str(settings["gpu"]), MAYFLY_LEVEL=settings["level"],
                   PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", creationflags=flags)
        self.started, self.games, self.positions, self.rate, self.evals = time.time(), 0, 0, 0, 0
        self.recent.clear()
        self.status, self.error, self.stopping = "starting", "", False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip()
            self.lines = (self.lines + [line])[-200:]
            m = re.search(r"([\d,]+) games, ([\d,]+) positions \(([\d,]+) an hour\)", line)
            if m:
                self.games, self.positions, self.rate = (int(x.replace(",", "")) for x in m.groups())
                self.status = "playing"
                # the search speed, so a slow card shows it's working long before its first game ends;
                # averaged over a minute from the first nonzero report, since a slow card answers in bursts
                e = re.search(r"([\d,]+) evals a second", line)
                if e:
                    v = int(e.group(1).replace(",", ""))
                    if v or self.recent:
                        self.recent.append(v)
                        self.evals = round(sum(self.recent) / len(self.recent))
            elif "workers x" in line:
                self.status = "playing"
            elif line.startswith("server onnx"):
                self.status = "warming up"
        self.proc.wait()
        if self.proc.returncode and not self.stopping:      # it ended by itself: say so, with its last words
            last = next((l for l in reversed(self.lines) if l.strip()), "no message")
            self.error = f"MayFly stopped unexpectedly ({last[:160]}). Press Run to try again."
        self.status = "stopped"

    def stop(self):
        self.stopping = True
        if self.running():
            subprocess.call(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.proc.wait()
        self.status = "idle"


# ---------------------------------------------------------------- the engine itself

def engine_main():
    """selfplay_remote on this computer: the onnx server on the chosen card, searches on a share of the cores."""
    out = os.path.join(DATA, "out", time.strftime("%Y%m%d_%H%M%S"))
    os.environ.update({
        "SP_BACKEND": "onnx", "SP_ONNX": MODEL, "SP_DEVICE": os.environ.get("MAYFLY_GPU", "0"),
        "SP_WORKERS": str(games_at_once(os.environ.get("MAYFLY_LEVEL", "half"))), "SP_TORCH_STUB": "1", "SP_REPORT": "10",
        "SP_SECONDS": str(10 * 365 * 86400), "SP_OUT": out, "SP_STARTS": os.path.join(KIT, "corpus_starts.txt"),
        "VERSION": VERSION, "VERSION_DIR": KIT, "VISITS": "1600", "FULL_FRAC": "0.25", "FAST_VISITS": "200"})
    if os.environ.get("MAYFLY_TEST_VISITS"):        # tests only: small searches, short games
        os.environ.update({"VISITS": os.environ["MAYFLY_TEST_VISITS"], "FAST_VISITS": "16", "MAX_PLIES": "30",
                           "SP_WORKERS": os.environ.get("MAYFLY_TEST_WORKERS", "2")})
    import glob
    logs = os.path.join(DATA, "logs")
    for f in glob.glob(os.path.join(logs, "child_*.log")):      # the last run's
        try:
            os.remove(f)
        except OSError:
            pass
    import selfplay_remote
    selfplay_remote.run()
    # run() only comes back this early when every worker has stopped, so say why for the app to show
    why = "no message"
    for f in sorted(glob.glob(os.path.join(logs, "child_*.log")), key=os.path.getmtime, reverse=True):
        lines = [l.strip() for l in open(f, encoding="utf-8", errors="replace") if l.strip()]
        if lines:
            why = lines[-1]
            break
    print(f"the search workers stopped: {why}", flush=True)
    sys.exit(1)


# ---------------------------------------------------------------- the local server the window talks to

class App:
    def __init__(self):
        self.settings = load_settings()
        self.engine = Engine()
        self.download = Download()
        self.lock = threading.Lock()

    def state(self):
        e, d = self.engine, self.download
        return {"name": self.settings["name"], "gpu": self.settings["gpu"], "level": self.settings["level"],
                "gpus": self.gpus, "running": e.running(), "status": e.status if e.running() else "idle",
                "games": e.games, "positions": e.positions, "rate": e.rate,
                "evals_per_s": e.evals if e.running() else 0,
                "games_at_once": games_at_once(self.settings["level"]),
                "since": e.started if e.running() else None, "model_ready": d.done,
                "downloading": d.active, "download_frac": round(d.frac, 3),
                "download_error": d.error or (e.error if not e.running() else ""),
                "version": "v6.1", "app_version": APP_VERSION, "site": SITE, "uploads": "not yet"}

    def handle(self, path, body):
        with self.lock:
            err = self._handle(path, body)
        return dict(self.state(), error=err) if err else self.state()

    def _handle(self, path, body):
        """None, or what to tell the volunteer."""
        if path == "/api/settings":
            if "name" in body:
                self.settings["name"] = clean_name(body["name"])
            if body.get("level") in LEVELS:
                self.settings["level"] = body["level"]
            if isinstance(body.get("gpu"), int):
                self.settings["gpu"] = body["gpu"]
            save_settings(self.settings)
        elif path == "/api/run":
            if not self.settings["name"]:
                return "Pick a name for the leaderboard first."
            if not self.download.done:        # the first Run fetches the network, then starts by itself
                self.download.start(then=lambda: self.engine.start(self.settings))
                return None
            self.engine.start(self.settings)
        elif path == "/api/stop":
            self.engine.stop()
        elif path == "/api/site":
            webbrowser.open(SITE)
        return None


def serve(app):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, data, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/state":
                return self._send(200, json.dumps(app.state()).encode(), "application/json")
            name = "index.html" if self.path in ("/", "/index.html") else os.path.basename(self.path)
            p = os.path.join(ASSETS, name)
            if not os.path.isfile(p):
                return self._send(404, b"not found", "text/plain")
            kind = {"html": "text/html; charset=utf-8", "svg": "image/svg+xml", "png": "image/png",
                    "ico": "image/x-icon"}.get(name.rsplit(".", 1)[-1], "application/octet-stream")
            self._send(200, open(p, "rb").read(), kind)

        def do_POST(self):
            # only the app's own page may drive it: a page elsewhere can't set this header without a CORS check
            if self.headers.get("X-MayFly") != "1":
                return self._send(403, b"forbidden", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            self._send(200, json.dumps(app.handle(self.path, body)).encode(), "application/json")

    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


def dark_mode():
    """Windows' app colour setting, so the window opens in the page's colours instead of flashing white."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def main():
    app = App()
    from volunteer.gpus import list_gpus
    app.gpus = list_gpus()
    if app.settings["gpu"] not in [g["index"] for g in app.gpus] and app.gpus:
        app.settings["gpu"] = app.gpus[0]["index"]
    url = f"http://127.0.0.1:{serve(app)}/"
    try:
        if os.environ.get("MAYFLY_NO_BROWSER"):   # tests: the local server alone, no window
            raise ImportError
        import webview
        webview.create_window("MayFly Volunteer", url, width=480, height=760, min_size=(400, 620),
                              background_color="#141816" if dark_mode() else "#ffffff")
        webview.start()
    except ImportError:
        print(f"MayFly Volunteer is open at {url} (close this window to quit)", flush=True)
        if not os.environ.get("MAYFLY_NO_BROWSER"):
            webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    finally:
        app.engine.stop()


if __name__ == "__main__":
    multiprocessing.freeze_support()      # the engine's server and workers are this program started again
    sys.path.insert(0, HERE)
    if "--engine" in sys.argv:
        try:
            engine_main()
        except Exception:                 # printed for the app to show, never a dialog box from the background
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        main()
