"""Let timed games have the GPU to themselves while the trainer waits.

A search spends most of its time on the GPU, so a clock match beside the trainer would be unfair. The
holder keeps a flag file fresh while it plays, and the trainer waits between steps while the flag is fresh.

    holder (clock_gate.py)   with aside.hold(path): ...   touches the flag every 2 s, removes it at the end
    trainer (train_av.py)    Aside(path, beat).wait(it, say) before each step, returns the seconds waited

A holder that dies stops touching the flag, so the trainer resumes within FRESH seconds. One hold lasts at
most CAP seconds, after which the trainer ignores the flag until it has gone stale once. A waiting trainer
touches a heartbeat file so supervise.py does not take it for a stuck one.
"""
import os
import threading
import time

FRESH = 15.0                    # seconds a hold stays fresh without a touch
CAP = 24 * 3600.0               # the longest one hold may last
TOUCH = 2.0                     # how often the holder touches the flag
BEAT = 30.0                     # how often a waiting trainer touches its heartbeat


def _touch(path):
    with open(path, "a", encoding="utf-8"):
        pass
    os.utime(path, None)


def fresh(path, fresh_for=FRESH):
    try:
        return time.time() - os.stat(path).st_mtime < fresh_for
    except OSError:
        return False


class hold:
    """Hold the flag while the block runs. It is removed at the end, or goes stale if the process dies."""

    def __init__(self, path, touch=TOUCH):
        self.path, self.touch, self._stop = path, touch, threading.Event()

    def _run(self):
        while not self._stop.is_set():
            try:
                _touch(self.path)
            except OSError:
                pass
            self._stop.wait(self.touch)

    def __enter__(self):
        _touch(self.path)
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join(timeout=10)
        try:
            os.remove(self.path)
        except OSError:
            pass
        return False


class Aside:
    """The trainer's side, call wait() between steps."""

    def __init__(self, path, beat_path, fresh_for=FRESH, cap=CAP, beat=BEAT, poll=0.5):
        self.path, self.beat_path = path, beat_path
        self.fresh_for, self.cap, self.beat, self.poll = fresh_for, cap, beat, poll
        self.capped = False                 # the last hold ran out its cap and the flag has not gone stale since
        self.total = 0.0

    def wait(self, it=0, say=print):
        held = fresh(self.path, self.fresh_for)
        if not held:
            self.capped = False
            return 0.0
        if self.capped:
            return 0.0
        t = time.time()
        say(f"    {it:,}  standing aside: the release gate holds the card ({os.path.basename(self.path)})")
        last_beat = last_say = 0.0
        while fresh(self.path, self.fresh_for):
            now = time.time()
            if now - t >= self.cap:
                self.capped = True
                say(f"    {it:,}  back to training after {self.cap / 3600:.0f} h standing aside, the most one "
                    f"hold may take; the flag is ignored until it has been stale once")
                break
            if now - last_beat >= self.beat:
                try:
                    _touch(self.beat_path)
                except OSError:
                    pass
                last_beat = now
            if now - last_say >= 600 and now - t >= 600:
                say(f"    {it:,}  still standing aside, {(now - t) / 60:.0f} min")
                last_say = now
            time.sleep(self.poll)
        d = time.time() - t
        self.total += d
        try:
            _touch(self.beat_path)
        except OSError:
            pass
        say(f"    {it:,}  back after standing aside {d / 60:.1f} min ({self.total / 3600:.2f} h since this trainer started)")
        return d
