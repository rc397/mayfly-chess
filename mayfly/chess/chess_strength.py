"""Compare the tropical and transformer networks by playing games, policy only with no search.

Plays the two architectures head to head with colours alternating, then each against Stockfish at
capped skill levels. Scores are converted to Elo differences with the logistic formula. A small
number of games gives a wide interval, so the game count is printed too.

    python chess/chess_strength.py [GAMES=40] [SF_SKILL=0,2] [SF_TIME=0.05]
"""
import os, sys, math, json, time, random
import torch, chess, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet, board_to_ids, move_to_idx
from mayfly_chess import MayflyChessNet, matched_dff_h

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SKILLS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "2"])]
SF_TIME = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
MAX_PLIES = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
SF = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()


def tuned_cfg(arch):
    """The configuration the convergence run actually used. The checkpoint only records d, layers
    and H, so anything the sweep tuned has to come back from the sweep's own record."""
    p = os.path.join(HERE, f"sweep_{arch}.json")
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p, encoding="utf-8"))["final"]
    except Exception:
        return {}


def load(path):
    ck = torch.load(path, map_location=device)
    sd = ck["model"]
    cfg = ck.get("cfg", {})
    arch = ck.get("arch", "ctrl")
    tuned = tuned_cfg(arch)
    d, L = cfg.get("d", 256), cfg.get("layers", 4)
    if arch == "trop":
        H = int(tuned.get("H", cfg.get("H", 8)))
        T, sc = float(tuned.get("T", 0.66)), float(tuned.get("SC", 1.5))
        m = MayflyChessNet(d=d, layers=L, dff_h=matched_dff_h(d, H), H=H, T=T, sc=sc)
        name = f"MAYFLY({d}x{L} H{H} T{T} sc{sc})"
    else:
        # read the feed-forward width from the tensors rather than trust a default
        dim_ff = int(sd["enc.layers.0.linear1.weight"].shape[0])
        nhead = int(tuned.get("NHEAD", 0)) or max(4, d // 64)
        m = ChessNet(d=d, nhead=nhead, layers=L, dim_ff=dim_ff)
        name = f"CONTROL({d}x{L} nhead{nhead} ff{dim_ff})"
    # strict, since temperature and head count do not change tensor shapes and a wrong config
    # would otherwise load silently
    m.load_state_dict(sd, strict=True); m.to(device).eval()
    print(f"loaded {name}  <- {os.path.basename(path)}", flush=True)
    return m, name, ck.get("val", {})


def elo(score, n):
    """Logistic Elo difference from a score fraction, clipped so a clean sweep is not infinite."""
    s = min(max(score, 0.5 / (n + 1)), 1 - 0.5 / (n + 1))
    return -400.0 * math.log10(1.0 / s - 1.0)


def result_for(board, side_white):
    r = board.result(claim_draw=True)
    if r == "1/2-1/2":
        return 0.5
    return 1.0 if (r == "1-0") == side_white else 0.0


def net_vs_net(a, b, games, seed=0):
    rng = random.Random(seed)
    score = 0.0
    for g in range(games):
        board = chess.Board()
        a_white = (g % 2 == 0)
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
            net = a if (board.turn == chess.WHITE) == a_white else b
            mv = net.best_move(board, device=device)
            if mv is None or mv not in board.legal_moves:
                mv = rng.choice(list(board.legal_moves))
            board.push(mv); plies += 1
        score += 0.5 if plies >= MAX_PLIES and not board.is_game_over(claim_draw=True) \
            else result_for(board, a_white)
    return score / games


def net_vs_sf(net, eng, skill, games, seed=0):
    rng = random.Random(seed)
    eng.configure({"Skill Level": skill})
    score = 0.0
    for g in range(games):
        board = chess.Board()
        net_white = (g % 2 == 0)
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
            if (board.turn == chess.WHITE) == net_white:
                mv = net.best_move(board, device=device)
                if mv is None or mv not in board.legal_moves:
                    mv = rng.choice(list(board.legal_moves))
            else:
                mv = eng.play(board, chess.engine.Limit(time=SF_TIME)).move
            board.push(mv); plies += 1
        score += 0.5 if plies >= MAX_PLIES and not board.is_game_over(claim_draw=True) \
            else result_for(board, net_white)
    return score / games


if __name__ == "__main__":
    paths = {"trop": os.path.join(HERE, "ckpt_long_trop.pt"),
             "ctrl": os.path.join(HERE, "ckpt_long_ctrl.pt")}
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    A, na, va = load(paths["trop"])
    B, nb, vb = load(paths["ctrl"])
    print(f"{na}  val_policy {va.get('policy', float('nan')):.4f}")
    print(f"{nb}  val_policy {vb.get('policy', float('nan')):.4f}")
    out = {"a": na, "b": nb, "games": GAMES, "sf_time": SF_TIME}

    print(f"\n=== head to head, {GAMES} games, colours alternated ===", flush=True)
    t0 = time.time()
    s = net_vs_net(A, B, GAMES)
    out["h2h_score_trop"] = s
    out["h2h_elo_trop"] = elo(s, GAMES)
    print(f"  {na} scores {s*100:.1f}% vs {nb}   Elo diff {elo(s, GAMES):+.0f}   "
          f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== versus Stockfish, {GAMES} games each, {SF_TIME}s per move ===", flush=True)
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    out["vs_sf"] = {}
    try:
        for skill in SKILLS:
            for tag, net, name in (("trop", A, na), ("ctrl", B, nb)):
                sc = net_vs_sf(net, eng, skill, GAMES)
                out["vs_sf"].setdefault(str(skill), {})[tag] = sc
                print(f"  skill {skill}: {name:18s} {sc*100:5.1f}%   Elo diff {elo(sc, GAMES):+.0f}",
                      flush=True)
    finally:
        eng.quit()

    json.dump(out, open(os.path.join(HERE, "strength.json"), "w"), indent=1)
    print("\nwrote strength.json", flush=True)
