"""Check whether MayFly converts won positions, judged on fixed positions against a fixed defender.

A known weakness is winning positions (+3 or more for Stockfish) that get shuffled into draws or losses.
Endgame fixes in the search are judged here rather than on game results.

    python conversion_suite.py build
        env PGNS  semicolon-separated globs of PGNs with MayFly in them
            OUT   the position file (conversion_suite.json)
        For every game MayFly did not win, keeps the first position where MayFly, to move, stands at
        MIN_CP or better (default 300) by Stockfish. One position per game.

    python conversion_suite.py build_mates
        From the same games, positions where Stockfish sees a forced mate in MATE_MIN to MATE_MAX
        (2 to 6) for the side to move, written to mate_suite.json. run on that file reports mates
        delivered, extra plies over the shortest mate, and seconds taken.

    python conversion_suite.py run
        env VERSION (mayfly_v4.pt), VISITS (1600), OPTS (search settings "k=v,k=v" on top of the
            version's sealed search), NAME (a label), SF_SKILL (10), SF_MS (100),
            POSITIONS (conversion_suite.json), RESULTS (conversion_results.jsonl)
        MayFly plays each position out against Stockfish to the end or MAX_PLIES, with Syzygy
        adjudicating at seven pieces. Positions already scored for NAME are skipped.
"""
import glob
import json
import os
import sys
import time

import chess
import chess.engine
import chess.pgn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths
E = os.environ.get
MIN_CP = int(E("MIN_CP", 300))
MAX_PLIES = int(E("MAX_PLIES", 300))
SYZYGY = E("SYZYGY", paths.under("syzygy"))


def sf_path():
    return open(os.path.join(HERE, "stockfish_path.txt")).read().strip()


def mayfly_side(game):
    for colour, key in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        if game.headers.get(key, "").startswith("MayFly"):
            return colour
    return None


def build():
    out = E("OUT", os.path.join(HERE, "conversion_suite.json"))
    depth = int(E("BUILD_DEPTH", 12))
    paths = sorted({p for g in E("PGNS", "").split(";") if g for p in glob.glob(g)})
    eng = chess.engine.SimpleEngine.popen_uci(sf_path())
    eng.configure({"Threads": int(E("THREADS", 4)), "Hash": 128})
    seen, found, games = set(), [], 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while True:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                me = mayfly_side(g)
                res = g.headers.get("Result", "*")
                if me is None or res == "*":
                    continue
                if (res == "1-0" and me == chess.WHITE) or (res == "0-1" and me == chess.BLACK):
                    continue                              # won, nothing to convert
                games += 1
                b = g.board()
                ply0 = len(b.move_stack)
                for i, mv in enumerate(g.mainline_moves()):
                    if b.turn == me and i >= 16 and (i // 2) % 2 == 0 and len(b.piece_map()) > 5:
                        s = eng.analyse(b, chess.engine.Limit(depth=depth))["score"].pov(me)
                        cp = s.score(mate_score=10000)
                        if cp is not None and cp >= MIN_CP and s.mate() is None:
                            key = b.board_fen() + (" w" if b.turn else " b")
                            if key not in seen:
                                seen.add(key)
                                found.append({"fen": b.fen(), "sf_cp": cp, "result": res,
                                              "mayfly": "white" if me else "black",
                                              "source": os.path.basename(path),
                                              "event": g.headers.get("Event", ""),
                                              "ply": ply0 + i, "pieces": len(b.piece_map())})
                            break
                    b.push(mv)
        print(f"{os.path.basename(path)}: {len(found)} positions from {games} non-won games so far", flush=True)
    eng.quit()
    json.dump(found, open(out, "w"), indent=1)
    print(f"wrote {len(found)} positions to {out}")


def build_mates():
    """Positions from the same games where Stockfish sees a forced mate in MATE_MIN to MATE_MAX moves
    for the side to move (mate in one is handled by root_mate). Only the first such position per game
    and side, so one mating sequence is not counted several times."""
    out = E("OUT", os.path.join(HERE, "mate_suite.json"))
    depth = int(E("BUILD_DEPTH", 14))
    lo, hi = int(E("MATE_MIN", 2)), int(E("MATE_MAX", 6))
    paths = sorted({p for g in E("PGNS", "").split(";") if g for p in glob.glob(g)})
    eng = chess.engine.SimpleEngine.popen_uci(sf_path())
    eng.configure({"Threads": int(E("THREADS", 4)), "Hash": 128})
    seen, found = set(), []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while True:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                if mayfly_side(g) is None:
                    continue
                b, done_sides = g.board(), set()
                for i, mv in enumerate(g.mainline_moves()):
                    if i >= 20 and b.turn not in done_sides:
                        s = eng.analyse(b, chess.engine.Limit(depth=depth))["score"].relative
                        m = s.mate()
                        if m is not None and lo <= m <= hi:
                            key = b.board_fen() + (" w" if b.turn else " b")
                            done_sides.add(b.turn)
                            if key not in seen:
                                seen.add(key)
                                found.append({"fen": b.fen(), "mate": m, "sf_cp": 10000, "pieces": len(b.piece_map()),
                                              "source": os.path.basename(path), "result": g.headers.get("Result", "*")})
                    b.push(mv)
                    if len(done_sides) == 2:
                        break
        print(f"{os.path.basename(path)}: {len(found)} mate positions so far", flush=True)
    eng.quit()
    json.dump(found, open(out, "w"), indent=1)
    print(f"wrote {len(found)} mate positions to {out}")


def load(name):
    import torch
    from chess_net import ChessNet
    path = os.path.join(E("VERSION_DIR", paths.under("versions")), name)

    ck = torch.load(path, map_location="cuda")
    c = ck.get("cfg", {})
    m = ChessNet(d=c.get("d", 1024), nhead=c.get("nhead", 16), layers=c.get("layers", 16),
                 dim_ff=int(c.get("ff_mult", 2.0) * c.get("d", 1024)),
                 qhead=bool(c.get("qhead", "qvalue.weight" in ck["model"])),
                 att=bool(c.get("att", "att.q.weight" in ck["model"])),
                 dist=int(c.get("dist", 64 if "vdist.weight" in ck["model"] else 0)),
                 mlh=bool(c.get("mlh", "mleft.0.weight" in ck["model"]))).cuda().eval()
    m.load_state_dict(ck["model"], strict=False)
    return m


def parse_opts(s):
    out = {}
    for kv in filter(None, (x.strip() for x in s.split(","))):
        k, _, v = kv.partition("=")
        out[k.strip()] = {"true": True, "false": False}.get(v.strip().lower(), v.strip())
    return out


def run():
    from mcts import MCTS, pinned_search
    from elo_vs_sf import game_over, game_result
    version = E("VERSION", "mayfly_v4.pt")
    visits = int(E("VISITS", 1600))
    opts = E("OPTS", "")
    name = E("NAME", opts or "sealed")
    skill, sf_ms = int(E("SF_SKILL", 10)), int(E("SF_MS", 100))
    positions = json.load(open(E("POSITIONS", os.path.join(HERE, "conversion_suite.json"))))
    results = E("RESULTS", os.path.join(HERE, "conversion_results.jsonl"))
    done = set()
    if os.path.exists(results):
        for line in open(results, encoding="utf-8"):
            r = json.loads(line)
            if r.get("name") == name:
                done.add(r["i"])
    model = load(version)
    tb = None
    try:
        import chess.syzygy
        tb = chess.syzygy.open_tablebase(SYZYGY)
    except Exception:
        pass
    sf = chess.engine.SimpleEngine.popen_uci(sf_path())
    sf.configure({"Skill Level": skill, "Threads": 1, "Hash": 64})
    print(f"{name}: {version} at {visits} visits, opts [{opts}], against Stockfish Skill {skill} at "
          f"{sf_ms} ms, {len(positions)} positions, {len(done)} already scored", flush=True)
    for i, p in enumerate(positions):
        if i in done:
            continue
        tree = MCTS(model, device="cuda", visits=visits)
        tree.apply(pinned_search(version))
        if opts:
            tree.apply(parse_opts(opts))
        tree.open_temp = 0.0                            # no opening variety in an ending
        board = chess.Board(p["fen"])
        me = board.turn
        t0, plies, how = time.time(), 0, "max plies"
        score = 0.5
        while plies < MAX_PLIES:
            if game_over(board):
                r = game_result(board)
                score = 0.5 if r == "1/2-1/2" else (1.0 if (r == "1-0") == (me == chess.WHITE) else 0.0)
                how = "played out"
                break
            if tb is not None and len(board.piece_map()) <= 7 and not board.castling_rights:
                try:
                    wdl = tb.probe_wdl(board)
                    stm_wins = wdl > 0
                    score = 0.5 if wdl == 0 else (1.0 if (stm_wins == (board.turn == me)) else 0.0)
                    how = "tablebase"
                    break
                except (KeyError, IndexError, chess.syzygy.MissingTableError):
                    pass
            if board.turn == me:
                mv, _ = tree.search(board, visits=visits, reuse=True)
            else:
                mv = sf.play(board, chess.engine.Limit(time=sf_ms / 1000.0)).move
            board.push(mv)
            plies += 1
        rec = {"name": name, "i": i, "score": score, "plies": plies, "how": how, "mate": p.get("mate"),
               "secs": round(time.time() - t0, 1), "sf_cp": p["sf_cp"], "pieces": p["pieces"],
               "version": version, "visits": visits, "opts": opts, "skill": skill}
        with open(results, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  {i:3d}  {'WIN ' if score == 1 else 'draw' if score == 0.5 else 'LOSS'}  "
              f"{plies:3d} plies  ({how}, {rec['secs']:.0f}s)  from {p['sf_cp']:+d} cp, {p['pieces']} pieces",
              flush=True)
    sf.quit()
    rows = [json.loads(l) for l in open(results, encoding="utf-8") if json.loads(l).get("name") == name]
    wins = [r for r in rows if r["score"] == 1.0]
    print(f"{name}: converted {len(wins)} of {len(rows)} ({100 * len(wins) / max(1, len(rows)):.1f}%), "
          f"drew {sum(r['score'] == 0.5 for r in rows)}, lost {sum(r['score'] == 0.0 for r in rows)}, "
          f"median plies to win {int(np.median([r['plies'] for r in wins])) if wins else '-'}", flush=True)
    mates = [r for r in rows if r.get("mate")]
    if mates:
        # a forced mate in m moves takes 2m - 1 plies from the mating side's move
        extra = [r["plies"] - (2 * r["mate"] - 1) for r in mates if r["score"] == 1.0]
        print(f"{name}: forced mates delivered {len(extra)} of {len(mates)}; extra plies over the shortest "
              f"mate: median {int(np.median(extra)) if extra else '-'}, mean {np.mean(extra) if extra else float('nan'):.1f}; "
              f"median seconds a position {np.median([r['secs'] for r in mates]):.1f}", flush=True)


if __name__ == "__main__":
    # the queue runner passes no arguments to a script it does not know, so MODE can say it instead
    mode = sys.argv[1] if len(sys.argv) > 1 else E("MODE", "")
    {"build": build, "build_mates": build_mates, "run": run}.get(mode, lambda: print(__doc__))()
