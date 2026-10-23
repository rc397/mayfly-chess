"""Grade a game with the network and explain its evaluations.

The action-value head scores every legal move in one forward pass, so grading a game costs one pass
a position. For each move it reports the cost in win probability against the best move the model saw,
and the auxiliary heads (material, king pressure, pawns, mobility, bishop pair) explain the evaluation.
Mistakes are grouped by the kind of position they happen in. With VISITS set, the search runs too, and
positions where it overrules the bare network are reported as blind spots.

    python chess/analyse.py game.pgn            grade a game
    python chess/analyse.py fen "<FEN>"         one position in full
    env TAG=engine, VISITS=0, OUT=analysis.json
"""
import os, sys, json, math
import numpy as np
import chess, chess.pgn
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import board_to_ids, clock_and_ep
from moves import move_to_idx

E = os.environ.get
PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


@torch.inference_mode()
def read(model, board, device):
    """One forward pass gives every legal move's value, the position's value, and the reasons."""
    ids, cast, flip = board_to_ids(board)
    hm, ep = clock_and_ep(board, flip)
    t = lambda a, d=torch.long: torch.tensor([a], dtype=d, device=device)
    out = model(t(ids), t(cast, torch.float32), t(hm), t(ep))
    legal = list(board.legal_moves)
    if not legal:
        return None
    idx = torch.tensor([move_to_idx(m, flip) for m in legal], device=device)
    q = torch.sigmoid(out["qvalue"][0][idx]).float().cpu().numpy() if out.get("qvalue") is not None \
        else torch.softmax(out["policy"][0][idx].float(), -1).cpu().numpy()
    has_q = out.get("qvalue") is not None
    f = lambda k, s: (out[k][0].float().cpu().numpy() * s)
    return {
        "moves": [m.uci() for m in legal],
        "legal": legal,
        "q": q,
        "has_q": has_q,
        "value": float(torch.sigmoid(out["value"][0].float())),
        "material": f("material", 39.0),
        "king": f("king", 8.0),
        "pawns": f("pawns", 8.0),
        "mobility": float(f("mob", 50.0)),
        "bishop_pair": torch.sigmoid(out["bishop"][0].float()).cpu().numpy(),
    }


def why(r):
    """The evaluation taken apart. Everything is from the side to move's point of view."""
    mat = r["material"]
    pw = r["pawns"]
    return {
        "win_probability": round(r["value"], 4),
        "material_mine": round(float(mat[0]), 2),
        "material_theirs": round(float(mat[1]), 2),
        "material_edge": round(float(mat[0] - mat[1]), 2),
        # not a safety score, the head counts attacked squares next to the king, so bigger is worse
        "king_pressure_on_me": round(float(r["king"][0]), 2),
        "king_pressure_on_them": round(float(r["king"][1]), 2),
        "pawns_mine": [round(float(x), 2) for x in pw[:3]],
        "pawns_theirs": [round(float(x), 2) for x in pw[3:]],
        "mobility": round(r["mobility"], 1),
        "bishop_pair_mine": round(float(r["bishop_pair"][0]), 3),
        "bishop_pair_theirs": round(float(r["bishop_pair"][1]), 3),
    }


def bucket(r, board):
    """The kind of position this is, so mistakes can be grouped by it."""
    mat = float(r["material"][0] - r["material"][1])
    npieces = sum(1 for _ in board.piece_map())
    ks = float(r["king"][0])
    # Thresholds measured over 480,000 Lichess positions. King pressure is 0 at the 33rd percentile,
    # 1 at the median and 4 at the 95th. Mobility is 28 at the 33rd and 37 at the 67th.
    return {
        "material": "level" if abs(mat) < 1 else ("ahead" if mat > 0 else "behind"),
        "phase": "opening" if npieces >= 26 else ("middlegame" if npieces >= 14 else "endgame"),
        "king": "quiet" if ks < 0.5 else ("pressured" if ks < 2.5 else "under fire"),
        "space": "cramped" if r["mobility"] < 28 else ("free" if r["mobility"] > 37 else "normal"),
    }


def analyse_game(model, game, device, visits=0, searcher=None):
    """Grade every move of a game and record why each position looked the way it did."""
    board = game.board() if hasattr(game, "board") else chess.Board()
    plies = []
    node = game
    while node.variations:
        node = node.variation(0)
        mv = node.move
        r = read(model, board, device)
        if r is None:
            break
        j = r["moves"].index(mv.uci()) if mv.uci() in r["moves"] else None
        best = int(np.argmax(r["q"]))
        rec = {
            "ply": board.ply(),
            "side": "white" if board.turn == chess.WHITE else "black",
            "san": board.san(mv),
            "played": mv.uci(),
            "played_q": None if j is None else round(float(r["q"][j]), 4),
            "best": r["moves"][best],
            "best_q": round(float(r["q"][best]), 4),
            "rank": None if j is None else int((r["q"] > r["q"][j]).sum()) + 1,
            "legal_moves": len(r["moves"]),
            "why": why(r),
            "kind": bucket(r, board),
        }
        rec["cost"] = None if j is None else round(rec["best_q"] - rec["played_q"], 4)
        if visits and searcher is not None:
            smove, info = searcher.search(board, visits=visits, reuse=False)
            rec["search_move"] = None if smove is None else smove.uci()
            rec["search_q"] = round(float(info.get("q", 0.0)), 4)
            rec["network_alone"] = r["moves"][best]
            rec["needed_search"] = (rec["search_move"] is not None
                                    and rec["search_move"] != rec["network_alone"])
        plies.append(rec)
        board.push(mv)
    return plies


def leaks(plies, side):
    """Where this side loses the most, grouped by the kind of position rather than by move number."""
    mine = [p for p in plies if p["side"] == side and p["cost"] is not None]
    if not mine:
        return {}
    out = {"moves": len(mine),
           "mean_cost": round(float(np.mean([p["cost"] for p in mine])), 4),
           "worst": sorted(mine, key=lambda p: -p["cost"])[:5],
           "by": {}}
    for axis in ("material", "phase", "king", "space"):
        rows = {}
        for p in mine:
            rows.setdefault(p["kind"][axis], []).append(p["cost"])
        out["by"][axis] = {k: {"moves": len(v), "mean_cost": round(float(np.mean(v)), 4)}
                           for k, v in sorted(rows.items(), key=lambda kv: -np.mean(kv[1]))}
    out["worst"] = [{k: p[k] for k in ("ply", "san", "cost", "best", "kind")} for p in out["worst"]]
    return out


def blind_spots(plies):
    """Positions where the network alone picked one move and its own search picked another."""
    got = [p for p in plies if p.get("needed_search")]
    if not got:
        return {}
    out = {"positions": len(got), "of": sum(1 for p in plies if "search_move" in p), "by": {}}
    for axis in ("material", "phase", "king", "space"):
        tot, bad = {}, {}
        for p in plies:
            if "search_move" not in p:
                continue
            k = p["kind"][axis]
            tot[k] = tot.get(k, 0) + 1
            bad[k] = bad.get(k, 0) + int(p["needed_search"])
        out["by"][axis] = {k: {"positions": tot[k],
                               "search_changed_the_move": round(100 * bad[k] / tot[k], 1)}
                           for k in sorted(tot, key=lambda k: -bad[k] / max(tot[k], 1))}
    return out


def show(plies, report):
    pct = lambda x: f"{100*x:.1f}%"
    print(f"\n{len(plies)} moves graded\n")
    for side in ("white", "black"):
        L = report[side]
        if not L:
            continue
        print(f"{side.upper()}  {L['moves']} moves, average cost {pct(L['mean_cost'])} "
              f"of win probability a move")
        for axis, rows in L["by"].items():
            bits = "  ".join(f"{k} {pct(v['mean_cost'])} ({v['moves']})" for k, v in rows.items())
            print(f"  by {axis:<9} {bits}")
        print("  worst moments:")
        for w in L["worst"]:
            print(f"    ply {w['ply']:>3}  {w['san']:<7} cost {pct(w['cost']):>6}  "
                  f"better was {w['best']}  [{w['kind']['phase']}, king {w['kind']['king']}, "
                  f"{w['kind']['space']}]")
        print()
    if report.get("blind_spots"):
        b = report["blind_spots"]
        print(f"THE MODEL'S OWN BLIND SPOTS: the search changed its mind in "
              f"{b['positions']} of {b['of']} positions")
        for axis, rows in b["by"].items():
            bits = "  ".join(f"{k} {v['search_changed_the_move']}%" for k, v in rows.items())
            print(f"  by {axis:<9} {bits}")


if __name__ == "__main__":
    from mcts import load, MCTS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load(E("TAG", "engine"), dev)
    print(f"{model.num_params()/1e6:.1f}M params, checkpoint at iter {ck.get('iter','?')}")
    visits = int(E("VISITS", 0))
    searcher = MCTS(model, dev, batch=32) if visits else None

    if len(sys.argv) > 2 and sys.argv[1] == "fen":
        b = chess.Board(sys.argv[2])
        r = read(model, b, dev)
        print(f"\n{b.fen()}\n")
        w = why(r)
        print(f"  it says {100*w['win_probability']:.1f}% for "
              f"{'white' if b.turn else 'black'}, and attributes it to:")
        for k, v in w.items():
            if k != "win_probability":
                print(f"    {k:<22} {v}")
        order = np.argsort(-r["q"])[:8]
        print(f"\n  every legal move scored in one pass "
              f"({'action-value head' if r['has_q'] else 'policy head, no q head in this ckpt'}):")
        for i in order:
            print(f"    {r['moves'][i]:<6} {100*r['q'][i]:5.1f}%")
        raise SystemExit(0)

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path or not os.path.exists(path):
        raise SystemExit("give a pgn file, or: analyse.py fen \"<FEN>\"")
    games = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            games.append(g)
    print(f"{len(games)} game(s) in {os.path.basename(path)}"
          + (f", searching {visits} visits a move" if visits else ""))

    allp = []
    for g in games:
        allp += analyse_game(model, g, dev, visits, searcher)
    report = {"white": leaks(allp, "white"), "black": leaks(allp, "black"),
              "blind_spots": blind_spots(allp), "plies": allp}
    show(allp, report)
    out = E("OUT", "analysis.json")
    json.dump(report, open(os.path.join(HERE, out), "w"), indent=1, default=float)
    print(f"\nwritten to {out}")
