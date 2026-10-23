r"""Turn a candidate that passed the release gate into the next official version.

The gate is more than half the points against Stockfish Skill BAR at 6,400 visits, with BAR from
release_plan.json (Skill 15 for v5, then 3 above the first rung the previous released candidate failed,
which this verdict records and hands to the plan). A clock climb's score is recorded beside the verdict but
does not decide.

24 games give a 50 percent score an interval of about plus or minus 20 points, so a bar rung between 40 and
50 percent is played again with 24 more games on elo_vs_sf's second book (BOOK=b, twelve other openings under
its own protocol tag), and the verdict is on all 48.

Run after a candidate's climbs. A candidate that passed is copied to the versions folder as mayfly_v<N>.pt and
given v<N>'s row in versions.json, sealed with the search it was cut and measured with, and its ladder and
clock files are copied to v<N>'s names. One that did not pass stays a candidate, and the reason goes to
candidates.log. Either way release_plan.json moves on (release_plan.on_verdict). Once a v<N> exists nothing
here touches it again.

  python chess/release_verdict.py
  env: CANDIDATE=v5c_YYYYMMDD.pt  CAND_DIR=<data>/candidates  OUT=<data>/abq
       VERSION_DIR=<data>/versions  DRY=1 (say what it would do, no games, nothing written)
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import release_plan  # noqa: E402
import paths  # noqa: E402

E = os.environ.get
CAND = E("CAND_DIR", paths.under("candidates"))
OUT = E("OUT", paths.under("abq"))
VDIR = E("VERSION_DIR", paths.under("versions"))
NAME = E("CANDIDATE", "")
DRY = E("DRY", "0") == "1"
PLAN = E("RELEASE_PLAN", os.path.join(CAND, "release_plan.json"))
GATE_VISITS, GATE_GAMES, GATE_SF_MS, GATE_TC = 6400, 24, 100, E("GATE_TC", "60+1")
BOOK_B_TAG = "book2b-draw2"


def say(m):
    line = time.strftime("%Y-%m-%d %H:%M  ") + m
    print(line, flush=True)
    if not DRY:
        with open(os.path.join(CAND, "candidates.log"), "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def rows(path):
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass                        # blank, or the torn last line of a killed run
    return out


def top(rs):
    """The highest level held at half the points, or None."""
    held = [int(r["level"]) for r in rs if float(r.get("score", 0)) >= 0.5]
    return max(held) if held else None


def write_manifest(path, rows_, n):
    """versions.json as the trainer writes it (indent 1, its own newline, BOM and line ends), whole or not at all."""
    raw = open(path, "rb").read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    s = json.dumps(rows_, indent=1, ensure_ascii=False)
    if text.endswith("\n"):
        s += "\n"
    if "\r\n" in text:
        s = s.replace("\n", "\r\n")
    shutil.copy2(path, path + ".bak_" + time.strftime("%Y%m%d_%H%M") + f"_v{n}")
    with open(path + ".tmp", "wb") as fh:
        fh.write((b"\xef\xbb\xbf" if bom else b"") + s.encode("utf-8"))
    os.replace(path + ".tmp", path)


def bar_scores(stem, bar):
    """(the ladder's own score at the bar, the second book's there or None), both at the gate's protocol."""
    a = b = None
    for r in rows(os.path.join(OUT, f"elo_ladder_{stem}.jsonl")):
        if (int(r.get("visits", -1)) == GATE_VISITS and int(r.get("level", -1)) == bar
                and int(r.get("games", 0)) == GATE_GAMES and int(r.get("sf_ms", -1)) == GATE_SF_MS):
            if r.get("protocol") == BOOK_B_TAG:
                b = float(r["score"])
            else:
                a = float(r["score"])
    return a, b


def play_book_b(name, bar):
    """The near miss's 24 games at the bar on the second book, through elo_ladder.py like any cell.

    A crash resumes game by game. True when it exited cleanly."""
    env = dict(os.environ, BOOK="b", VERSIONS=name, VERSION_DIR=CAND, VISITS=str(GATE_VISITS),
               GAMES=str(GATE_GAMES), LEVEL=str(bar), SF_MS=str(GATE_SF_MS), OUT=OUT, PONDER="0",
               AUTOCAST_GRAPH=E("AUTOCAST_GRAPH", "1"))          # as the climb was played
    return subprocess.call([sys.executable, os.path.join(HERE, "elo_ladder.py")], env=env, cwd=HERE) == 0


def advance(plan, passed, cut, summary, failed=None):
    """Move release_plan.json on by this verdict, if this candidate is the attempt it is waiting on.

    On a pass the next bar is 3 above failed, the first rung this candidate's climb failed."""
    if plan is None:
        say("verdict: no release_plan.json, so no next cut was scheduled")
        return
    if cut is None:
        say(f"verdict: release_plan.json is not waiting on {NAME}, so it was left as it is")
        return
    release_plan.on_verdict(plan, NAME, passed, cut, summary, failed_rung=failed)
    bar = (f"bar Skill {plan['bar']}, 3 above the Skill {failed} it failed" if plan["bar"] is not None
           else f"bar not known yet: {plan.get('bar_note')}")
    nxt = (f"v{plan['version']}'s first candidate at {plan['next_cut_hours']:.1f} training hours "
           f"(a {plan['window']:.0f}-hour window, {bar})" if passed else
           f"the next v{plan['version']} candidate at {plan['next_cut_hours']:.1f} training hours")
    if DRY:
        say(f"verdict (DRY): release_plan.json would move on to {nxt}")
        return
    release_plan.save(plan, PLAN)
    say(f"verdict: release_plan.json moved on: {nxt}")


def main():
    if not NAME:
        raise SystemExit("CANDIDATE= is required, e.g. v5c_20261008.pt")
    stem = NAME.replace(".pt", "")
    plan = release_plan.load(PLAN) if os.path.exists(PLAN) else None
    n = int(plan["version"]) if plan else int(stem[1:].split("c_", 1)[0])
    if not stem.startswith(f"v{n}c_"):
        raise SystemExit(f"{NAME} is not a v{n} candidate, and v{n} is the version release_plan.json is chasing")
    bar = E("BAR_SKILL") or (plan["bar"] if plan else release_plan.bar_for(n))
    if bar is None:
        raise SystemExit(f"no bar is set for v{n} ({(plan or {}).get('bar_note', 'release_plan.json has none')}): "
                         f"set release_plan.json's bar before this verdict")
    bar = int(bar)
    cur = (plan or {}).get("current") or {}
    cut = cur.get("cut_hours") if cur.get("candidate") == NAME else None
    ladder = [r for r in rows(os.path.join(OUT, f"elo_ladder_{stem}.jsonl"))
              if int(r.get("visits", -1)) == GATE_VISITS and r.get("protocol") != BOOK_B_TAG]
    clock = [r for r in rows(os.path.join(OUT, f"clock_gate_{stem}.jsonl")) if r.get("tc") == GATE_TC]
    s_a, s_b = bar_scores(stem, bar)
    lo, hi = release_plan.NEAR_MISS
    fmt = lambda s: "not reached" if s is None else f"{100 * s:.1f}%"
    if s_a is not None and lo <= s_a <= hi and s_b is None:
        if DRY:
            say(f"verdict (DRY): {NAME} took {fmt(s_a)} against Skill {bar}, a near miss: WOULD PLAY 24 more "
                f"games there on the second book and decide on all 48")
            return
        say(f"verdict: {NAME} took {fmt(s_a)} against Skill {bar}, between {100 * lo:.0f}% and {100 * hi:.0f}%: "
            f"24 more games there on the second book, and the verdict on all 48")
        if not play_book_b(NAME, bar):
            say("verdict: the second book's games did not finish; the queue's retry resumes them")
            sys.exit(1)
        s_a, s_b = bar_scores(stem, bar)
        if s_b is None:
            say("verdict: the second book's games left no result; the queue's retry plays them")
            sys.exit(1)
    s6 = s_a if (s_a is None or s_b is None) else (s_a + s_b) / 2
    at_bar = lambda rs: next((float(r["score"]) for r in reversed(rs) if int(r.get("level", -1)) == bar), None)
    s1 = at_bar(clock)
    passed = s6 is not None and s6 > 0.5          # the clock climb's s1 is recorded, not required
    # Where the climb failed, the first rung under half on the ladder's own book, or None if it failed none.
    # A bar passed only on the near miss's 48 games is not a fail, and the rungs above it were never played,
    # so that also leaves it None and the next bar unset.
    failed = min((int(r["level"]) for r in ladder if float(r.get("score", 0)) < 0.5
                  and not (passed and int(r["level"]) == bar)), default=None)
    summary = (f"{NAME}: Skill {bar} at {GATE_VISITS} visits {fmt(s6)}"
               + (f" over 48 games ({fmt(s_a)} on the ladder's book, {fmt(s_b)} on the second)"
                  if s_b is not None else "")
               + f" (highest held {top(ladder)}, "
               + (f"first failed Skill {failed}" if failed is not None else "failed none of its rungs")
               + f"), on {GATE_TC} {fmt(s1)} (highest held {top(clock)})")
    vpath = os.path.join(VDIR, "versions.json")
    vrows = json.load(open(vpath, encoding="utf-8-sig"))
    if any(int(r.get("version", 0)) == n for r in vrows):
        say(f"verdict: {summary}; v{n} already exists, nothing changed")
        return
    if not passed:
        say(f"verdict: {summary}; NOT released, it stays a candidate (the gate needs more than half against "
            f"Skill {bar} at {GATE_VISITS} visits)")
        advance(plan, False, cut, summary, failed)
        return
    crow = next(r for r in json.load(open(os.path.join(CAND, "versions.json"), encoding="utf-8"))
                if r.get("file") == NAME)
    row = {"version": n, "file": f"mayfly_v{n}.pt", "iter": crow.get("iter"), "hours": crow.get("hours"),
           "at": time.strftime("%Y-%m-%d %H:%M:%S"), "from_candidate": NAME, "cut": crow.get("cut"),
           "cooled": crow.get("cooled"), "cut_hours": cut,
           "gate": {"skill": bar, f"{GATE_VISITS} visits": s6, "book_b": s_b, GATE_TC: s1, "failed_rung": failed},
           **{k: crow[k] for k in ("q", "policy_legal", "top1_legal", "att_top1_legal") if k in crow},
           "search": crow["search"]}
    if DRY:
        say(f"verdict (DRY): {summary}; WOULD RELEASE as mayfly_v{n}.pt with {len(row['search'])} sealed settings")
        advance(plan, True, cut, summary, failed)
        return
    dst = os.path.join(VDIR, f"mayfly_v{n}.pt")
    shutil.copy2(os.path.join(CAND, NAME), dst + ".part")
    os.replace(dst + ".part", dst)
    # Keep the candidate's C++ network trace with the release (mcts_cpp finds it via VERSION).
    for ext in (".cuda.ts", ".cuda.ts.json"):
        if os.path.exists(os.path.join(CAND, stem + ext)):
            shutil.copy2(os.path.join(CAND, stem + ext), os.path.join(VDIR, f"mayfly_v{n}{ext}"))
    write_manifest(vpath, vrows + [row], n)
    for kind in ("elo_ladder", "clock_gate"):
        src, to = os.path.join(OUT, f"{kind}_{stem}.jsonl"), os.path.join(OUT, f"{kind}_mayfly_v{n}.jsonl")
        if os.path.exists(src) and not os.path.exists(to):
            shutil.copy2(src, to)
    with open(os.path.join(CAND, f"RELEASED_v{n}.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M')} {NAME} released as the official v{n} (mayfly_v{n}.pt): "
                 f"{summary}.\nNot done here, each its own step: copying versions.json to other machines, "
                 f"and putting v{n} on the Lichess bot.\n")
    say(f"verdict: {summary}; RELEASED as the official v{n} (mayfly_v{n}.pt, v{n}'s row in {vpath})")
    advance(plan, True, cut, summary, failed)


if __name__ == "__main__":
    main()
