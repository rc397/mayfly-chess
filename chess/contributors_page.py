"""Build the contributor leaderboard (index.html, topcontributors.txt, contributors.json) from the volunteer runs
kaggle_volunteer_collect.py kept.

  python chess/contributors_page.py [OUT]
"""
import glob
import html
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

E = os.environ.get
VOL_OUT = E("VOL_OUT", paths.under("selfplay", "volunteer"))
OUT = sys.argv[1] if len(sys.argv) > 1 else E("PAGE_OUT", paths.under("selfplay", "contributors_page"))
NOTEBOOK = "https://www.kaggle.com/code/rc7530/mayfly-selfplay-volunteer"
REPO = "https://github.com/rc397/mayfly-chess"
VERSION = E("VOL_VERSION_NAME", "v6")


def gather():
    people = defaultdict(lambda: {"positions": 0, "games": 0, "labels": 0, "runs": 0, "first": None, "last": None})
    for path in glob.glob(os.path.join(VOL_OUT, "*", "*", "summary.json")):
        user = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if user.startswith("_"):
            continue                                   # skip _rejected and _incoming
        try:
            s = json.load(open(path))
        except (OSError, ValueError):
            continue
        day = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(path)))
        p = people[user]
        if s.get("kind") == "label":
            p["labels"] += int(s.get("labelled", 0))
        else:
            p["positions"] += int(s.get("positions_written", s.get("positions", 0)))
            p["games"] += int(s.get("games", 0))
        p["runs"] += 1
        p["first"] = min(p["first"] or day, day)
        p["last"] = max(p["last"] or day, day)
    # a deep label takes a CPU far longer than a self-play position takes a TPU, so weight it by 100
    return sorted(({"name": k, **v} for k, v in people.items()), key=lambda r: -(r["positions"] + 100 * r["labels"]))


def text(rows):
    lines = [f"MayFly {VERSION} top contributors (self-play and deep labels donated)", f"updated {time.strftime('%Y-%m-%d')}", ""]
    if not rows:
        lines.append(f"No contributions yet. Be the first at {NOTEBOOK}")
    for i, r in enumerate(rows, 1):
        lines.append(f"{i:>3}. {r['name']:<28} {r['positions']:>12,} positions  {r['games']:>8,} games  "
                     f"{r['labels']:>8,} deep labels")
    lines += ["", f"Join at {NOTEBOOK}", f"Code at {REPO}"]
    return "\n".join(lines) + "\n"


def page(rows):
    total_p = sum(r["positions"] for r in rows)
    total_g = sum(r["games"] for r in rows)
    total_l = sum(r["labels"] for r in rows)
    body = "".join(
        f"<tr><td>{i}</td><td>{html.escape(r['name'])}</td><td class=n>{r['positions']:,}</td>"
        f"<td class=n>{r['games']:,}</td><td class=n>{r['labels']:,}</td></tr>" for i, r in enumerate(rows, 1)) or (
        "<tr><td colspan=5>No contributions yet.</td></tr>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MayFly Contributors</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root {{ --bg:#ffffff; --ink:#1d2320; --soft:#66706a; --line:#dfe3dd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141816; --ink:#e7ece8; --soft:#9aa69f; --line:#2c332f; }} }}
body {{ margin:0; padding:32px 16px; background:var(--bg); color:var(--ink);
  font:15px/1.6 "JetBrains Mono", ui-monospace, "Cascadia Mono", Consolas, monospace; }}
main {{ max-width:640px; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 6px; }} .soft {{ color:var(--soft); margin:0 0 20px; }}
table {{ border-collapse:collapse; width:100%; }}
th, td {{ padding:8px 10px; text-align:left; border-bottom:1px solid var(--line); }}
th {{ color:var(--soft); font-weight:500; font-size:.85rem; }}
.n {{ text-align:right; font-variant-numeric:tabular-nums; }} a {{ color:inherit; }}
h2 {{ font-size:1.1rem; margin:36px 0 8px; }} h3 {{ font-size:1rem; margin:24px 0 6px; }}
ol {{ margin:0 0 12px; padding-left:1.4em; }} li {{ margin:6px 0; }}
</style></head><body><main>
<h1>MayFly Contributors</h1>
<p class=soft>{len(rows)} contributors, {total_g:,} games, {total_p:,} positions, {total_l:,} deep labels.
<a href="#join">Join</a></p>
<table><tr><th>#</th><th>Contributor</th><th class=n>Positions</th><th class=n>Games</th><th class=n>Deep labels</th></tr>{body}</table>
<p class=soft style="margin-top:16px">Updated {time.strftime('%d %b %Y')}.
<a href="topcontributors.txt">topcontributors.txt</a></p>

<h2 id="join">How to join the cause</h2>
<p class=soft>Two ways to help. Both are free.</p>

<h3>Option 1. Kaggle, in your browser</h3>
<p class=soft>Works on any computer, even a Mac or a laptop with no graphics card. Kaggle's cloud computers do the work.</p>
<ol>
<li>Make a free account at <a href="https://www.kaggle.com">kaggle.com</a>. Sign in with Google is the quickest.</li>
<li>Optional, but it makes your help about 100 times bigger. Click your picture, then Settings, then Phone
verification. That unlocks Kaggle's free TPU.</li>
<li>Open the <a href="{NOTEBOOK}">MayFly notebook</a> and click <b>Copy &amp; Edit</b>.</li>
<li>If you verified your phone, set Accelerator to <b>TPU VM v5e-8</b> under Session options. If not, leave it.</li>
<li>Click <b>Save Version</b>, keep Save &amp; Run All (Commit), and click <b>Save</b>.</li>
<li>Click <b>Share</b> and set it to <b>Public</b> so we can collect what your run made.</li>
</ol>
<p class=soft>That's it. It runs for 8 to 11 hours and you can close the tab. With a TPU it plays MayFly against itself,
and without one it analyses hard positions deeply with Stockfish. Kaggle gives fresh free hours every week, so open
your copy under Your Work and press Save Version again.</p>

<h3>Option 2. Your own graphics card</h3>
<p class=soft>Coming soon. A small app you download and open, pick your GPU and how much of it to give, and press Run.
It plays MayFly self-play on your computer and sends the games in under your name.</p>

<p class=soft>Your name shows up on the board once your games have been checked.</p>
</main></body></html>
"""


if __name__ == "__main__":
    rows = gather()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page(rows))
    with open(os.path.join(OUT, "topcontributors.txt"), "w", encoding="utf-8") as fh:
        fh.write(text(rows))
    with open(os.path.join(OUT, "contributors.json"), "w", encoding="utf-8") as fh:
        json.dump({"updated": time.strftime("%Y-%m-%d %H:%M"), "version": VERSION, "contributors": rows}, fh,
                  indent=1)
    print(f"{len(rows)} contributor(s) -> {OUT}")
