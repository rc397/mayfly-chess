r"""When the next version's candidate is cut, and what it has to beat.

Time is the main run's own training hours (train_av's trained_hours, read off its log, so a cooldown that
pauses the run never counts), from one release cut to the next.
  - A version's first candidate is cut WINDOW hours after the last release's cut, and the window doubles
    with each version.
  - After a candidate fails, another is cut every RETEST_HOURS (336) until one passes.
  - The gate (release_verdict.py) is more than half the points against Stockfish Skill BAR at 6,400 visits.
    v5's bar is Skill 15, and each later bar is 3 above the first rung its predecessor's released candidate
    failed. A bar rung between 40 and 50 percent plays 24 more games on a second book and is decided on all
    48. A release whose climb beat every rung it was given leaves the next bar unset (None), with a note.

release_plan.json, beside the candidates, is the state. It goes from "waiting" to "cooling" (a cooldown
launched at the cut) to "testing" (the candidate's climb queued), and release_verdict.py moves it back to
"waiting" for the next version or the retest.
  {"version": chased, "bar": its Skill, "window": its window in hours, "retest_hours": 336,
   "last_release": {"version", "cut_hours"}, "next_cut_hours": when the next candidate is cut,
   "state": "waiting" | "cooling" | "testing", "current": the attempt under way, or null,
   "history": every attempt, passed or not}
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

E = os.environ.get
CAND = E("CAND_DIR", paths.under("candidates"))
PLAN = E("RELEASE_PLAN", os.path.join(CAND, "release_plan.json"))
TRAIN_LOG = E("TRAIN_LOG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_mayfly.log"))
FIRST_BAR, BAR_STEP, FIRST_VERSION = 15, 3, 5
RETEST_HOURS = 336.0
NEAR_MISS = (0.40, 0.50)
HOURS_RE = re.compile(r"\s(\d+(?:\.\d+)?)h\s*$")


def bar_for(version):
    """The bar a run without release_plan.json can know, v5's.

    Every later bar comes from the plan, set by the verdict that released the version before (bar_after)."""
    return FIRST_BAR if version == FIRST_VERSION else None


def bar_after(failed_rung):
    """The next version's bar, BAR_STEP above the first rung the released candidate failed.

    None when its climb failed none of the rungs it was given."""
    return None if failed_rung is None else failed_rung + BAR_STEP


def load(path=PLAN):
    return json.load(open(path, encoding="utf-8"))


def save(plan, path=PLAN):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=1)
    os.replace(tmp, path)


def main_hours(log=TRAIN_LOG):
    """The main run's trained hours, from the last line of its log that ends in one ("... 450.94h"), or None
    when the log has none. Only the tail is read, so a month-long log costs nothing."""
    try:
        with open(log, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 65536))
            tail = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    for line in reversed(tail):
        m = HOURS_RE.search(line)
        if m:
            return float(m.group(1))
    return None


def on_verdict(plan, candidate, passed, cut_hours, detail, failed_rung=None):
    """The plan after a verdict, changed in place and returned.

    On a pass, the next version, its doubled window and its bar (3 above failed_rung, the first rung the
    released candidate failed). On a fail, a retest RETEST_HOURS after this cut."""
    plan.setdefault("history", []).append(
        {"version": plan["version"], "candidate": candidate, "cut_hours": cut_hours, "passed": bool(passed),
         "bar": plan["bar"], "failed_rung": failed_rung, "at": time.strftime("%Y-%m-%d %H:%M"),
         "detail": detail})
    if passed:
        plan["last_release"] = {"version": plan["version"], "cut_hours": cut_hours, "failed_rung": failed_rung}
        plan["version"] += 1
        plan["window"] = 2 * plan["window"]
        plan["bar"] = bar_after(failed_rung)
        if plan["bar"] is None:
            plan["bar_note"] = (f"v{plan['version'] - 1}'s climb failed none of its rungs, so v{plan['version']}'s "
                                f"bar (3 above the first rung v{plan['version'] - 1} fails) is not known yet")
        else:
            plan.pop("bar_note", None)
        plan["next_cut_hours"] = round(cut_hours + plan["window"], 2)
    else:
        plan["next_cut_hours"] = round(cut_hours + plan.get("retest_hours", RETEST_HOURS), 2)
    plan["state"], plan["current"] = "waiting", None
    return plan
