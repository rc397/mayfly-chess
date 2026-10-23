"""Replay positions where MayFly blundered in real games and check whether it still plays the blunder.

Search guards such as q_floor and forcing are judged here on the same positions, rather than on game
results where one blunder is lost among many moves.

    python blunder_suite.py build
        env REVIEWS  semicolon-separated game review folders
            OUT      the position file (blunder_suite.json)
        One position per blunder game, the position before MayFly's decisive move. Kept only if
        Stockfish at depth 22 confirms the move drops 0.3 or more win chance (the Lichess blunder
        line) from a position that was not already lost.

    python blunder_suite.py run
        env VERSION (mayfly_v4.pt), VISITS (1600), OPTS (search settings "k=v,k=v" on top of the
            version's sealed search), NAME (a label), POSITIONS (blunder_suite.json),
            RESULTS (blunder_results.jsonl)
        A position passes if the version avoids the blunder and Stockfish at depth 18 says its move
        drops less than 0.3. Resumable by NAME and VISITS.
"""
import glob
import json
import math
import os
import sys
import time

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
E = os.environ.get
BLUNDER = 0.3
V4_FROM = "2026-09-22 22:33:00"          # UTC, games before this were played by v3


def sf_path():
    return open(os.path.join(HERE, "stockfish_path.txt")).read().strip()


def wc(cp):
    """The Lichess win-chance curve on centipawns from the mover's side, mate as 10 pawns."""
    return 2 / (1 + math.exp(-0.00368 * cp)) - 1


def cp_of(score, side):
    s = score.pov(side)
    if s.is_mate():
        return 1000 if s.mate() > 0 else -1000
    return max(-1000, min(1000, s.score()))


def candidates():
    """(fen, move uci, game id, source) for every blunder game in the reviews."""
    out = []
    for d in filter(None, E("REVIEWS", paths.under("analysis_20260926") + ";" + paths.under("analysis_20260928")).split(";")):
        res = os.path.join(d, "result.json")
        cls = os.path.join(d, "classes.json")
        if os.path.exists(res):
            # result.json reviews, the first move of the game that dropped 0.3 from above -3
            for g in json.load(open(res)):
                if g.get("cls") != "blunder":
                    continue
                m = [x for x in g["moves"] if x["drop"] >= BLUNDER and x["before"] > -300][0]
                b = chess.Board(m["fen"])
                out.append((m["fen"], b.parse_san(m["san"]).uci(), g["id"], os.path.basename(d)))
        elif os.path.exists(cls):
            # classes.json reviews name the decisive ply, v4's games only
            games = {g["id"]: g for g in json.load(open(os.path.join(d, "games.json")))}
            for c in json.load(open(cls)):
                if not c["cause"].startswith("blunder"):
                    continue
                g = games[c["id"]]
                if g["utc"] < V4_FROM:
                    continue
                out.append((g["fens"][c["dec_ply"]], g["moves"][c["dec_ply"]], g["id"], os.path.basename(d)))
    return out


def build():
    sf = chess.engine.SimpleEngine.popen_uci(sf_path())
    sf.configure({"Threads": int(E("SF_THREADS", 4)), "Hash": 512})
    kept = []
    for fen, uci, gid, src in candidates():
        b = chess.Board(fen)
        mv = chess.Move.from_uci(uci)
        if mv not in b.legal_moves:
            print(f"  {gid}: {uci} is not legal in {fen}; skipped", flush=True)
            continue
        me = b.turn
        info = sf.analyse(b, chess.engine.Limit(depth=22))
        before = cp_of(info["score"], me)
        a = b.copy(); a.push(mv)
        after = cp_of(sf.analyse(a, chess.engine.Limit(depth=22))["score"], me)
        drop = wc(before) - wc(after)
        ok = drop >= BLUNDER and before > -300
        print(f"  {gid} {b.san(mv):8s} {before / 100:+.2f} -> {after / 100:+.2f} drop {drop:.2f} "
              f"{'kept' if ok else 'not a blunder at depth 22'}", flush=True)
        if ok:
            kept.append({"fen": fen, "blunder": uci, "san": b.san(mv), "game": gid, "source": src,
                         "sf_before": before, "sf_after": after, "sf_best": info["pv"][0].uci()})
    sf.quit()
    out = E("OUT", os.path.join(HERE, "blunder_suite.json"))
    json.dump(kept, open(out, "w"), indent=1)
    print(f"{len(kept)} positions written to {out}", flush=True)


def parse_opts(s):
    out = {}
    for kv in filter(None, (x.strip() for x in s.split(","))):
        k, _, v = kv.partition("=")
        out[k.strip()] = {"true": True, "false": False}.get(v.strip().lower(), v.strip())
    return out


def run():
    from mcts import MCTS, pinned_search
    from conversion_suite import load
    version = E("VERSION", "mayfly_v4.pt")
    visits = int(E("VISITS", 1600))
    opts = E("OPTS", "")
    name = E("NAME", opts or "sealed")
    positions = json.load(open(E("POSITIONS", os.path.join(HERE, "blunder_suite.json"))))
    results = E("RESULTS", os.path.join(HERE, "blunder_results.jsonl"))
    done = set()
    if os.path.exists(results):
        for line in open(results, encoding="utf-8"):
            r = json.loads(line)
            if r.get("name") == name and r.get("visits") == visits:
                done.add(r["i"])
    model = load(version)
    sf = chess.engine.SimpleEngine.popen_uci(sf_path())
    sf.configure({"Threads": int(E("SF_THREADS", 2)), "Hash": 256})
    print(f"{name}: {version} at {visits} visits, opts [{opts}], {len(positions)} positions, "
          f"{len(done)} already scored", flush=True)
    for i, p in enumerate(positions):
        if i in done:
            continue
        tree = MCTS(model, device="cuda", visits=visits)
        tree.apply(pinned_search(version))
        if opts:
            tree.apply(parse_opts(opts))
        tree.open_temp = 0.0
        b = chess.Board(p["fen"])
        me = b.turn
        t0 = time.time()
        mv, info = tree.search(b.copy(), visits=visits, reuse=False)
        secs = time.time() - t0
        same = mv.uci() == p["blunder"]
        if same:
            drop = wc(p["sf_before"]) - wc(p["sf_after"])
        else:
            a = b.copy(); a.push(mv)
            after = cp_of(sf.analyse(a, chess.engine.Limit(depth=18))["score"], me)
            drop = wc(p["sf_before"]) - wc(after)
        row = {"name": name, "visits": visits, "i": i, "game": p["game"], "played": b.san(mv),
               "blunder": p["san"], "same": same, "drop": round(drop, 3), "passed": (not same) and drop < BLUNDER,
               "forced": int(tree.stats.get("forced", 0)), "secs": round(secs, 1)}
        with open(results, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"  {i:2d} {p['game']} {p['san']:7s} -> {b.san(mv):7s} {'PASS' if row['passed'] else ('same' if same else 'other blunder')}"
              f"  drop {drop:.2f}  forced {row['forced']}  {secs:.0f}s", flush=True)
    sf.quit()
    rows = [json.loads(l) for l in open(results, encoding="utf-8")]
    rows = [r for r in rows if r["name"] == name and r["visits"] == visits]
    print(f"{name} at {visits}: passed {sum(r['passed'] for r in rows)} of {len(rows)}, "
          f"the same blunder {sum(r['same'] for r in rows)}, another {sum((not r['same']) and not r['passed'] for r in rows)}",
          flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else E("MODE", "")
    {"build": build, "run": run}.get(mode, lambda: print(__doc__))()
