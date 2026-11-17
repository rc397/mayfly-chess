r"""tune_ttt sweeps the opponent adapter's settings on real games against MayFly.

With lr 0.02 and 8 steps per observed move the adapter memorises its last sixty moves and predicts the
next one worse over time. The trunk features and base logits do not depend on the adapter settings, so
each position goes through the network once, is cached on the CPU, and every setting in the sweep is
small matrix maths on that cache. The score for each move is the probability the adapter gave it before
learning from it.

  python chess/tune_ttt.py [GAMES_GLOB]
  env: VERSION=mayfly_v1.pt  PLAYER=  (only games with this name)  GAMES (default paths.under("games"))
"""
import os, sys, glob, itertools, collections
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from paths import under

E = os.environ.get
GAMES = E("GAMES", under("games"))
PLAYER = E("PLAYER", "")


def human_moves(path, engine_hint="MayFly"):
    """Every position where the PERSON was to move, with the move they chose."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        g = chess.pgn.read_game(fh)
    if g is None:
        return None, []
    white, black = g.headers.get("White", ""), g.headers.get("Black", "")
    if engine_hint in white:
        human, colour = black, chess.BLACK
    elif engine_hint in black:
        human, colour = white, chess.WHITE
    else:
        return None, []
    if PLAYER and PLAYER.lower() not in human.lower():
        return None, []
    b, out = g.board(), []
    for m in g.mainline_moves():
        if b.turn == colour and b.legal_moves.count() >= 2:
            out.append((b.copy(stack=False), m))
        b.push(m)
    return human, out


if __name__ == "__main__":
    import torch
    from ttt import OpponentModel
    from play_web import load_version
    from chess_net import board_to_ids, clock_and_ep
    from moves import move_to_idx

    paths = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else os.path.join(GAMES, "*.pgn")),
                   key=os.path.getmtime)
    if not paths:
        raise SystemExit(f"no games in {GAMES}")

    model, label, _ = load_version(E("VERSION", "mayfly_v1.pt"))
    dev = str(next(model.parameters()).device)

    print(f"{len(paths)} game file(s)", flush=True)
    who = collections.Counter()
    cached = []
    # no_grad, not inference_mode. An inference tensor can never take part in autograd, and the
    # adapter runs backward against these cached tensors many times.
    with torch.no_grad():
        for p in paths:
            name, moves = human_moves(p)
            if not moves:
                continue
            who[name] += len(moves)
            for board, mv in moves:
                ids, cast, flip = board_to_ids(board)
                hm, ep = clock_and_ep(board, flip)
                t = lambda a, d: torch.tensor([a], dtype=d, device=dev)
                out = model(t(ids, torch.long), t(cast, torch.float32),
                            t(hm, torch.long), t(ep, torch.long))
                idx = torch.tensor([move_to_idx(m, flip) for m in board.legal_moves], device=dev)
                tgt = int((idx == move_to_idx(mv, flip)).nonzero()[0])
                # straight to the CPU, the sweep is tiny and leaves the GPU to other work
                cached.append((out["h"].detach().float().cpu().clone(),
                               out["policy"][0].detach().float().cpu().clone(),
                               idx.cpu(), tgt, len(list(board.legal_moves))))
    if not cached:
        raise SystemExit("no human moves found in those games")
    for n, c in who.most_common():
        print(f"  {n}: {c} moves")
    chance = 100.0 / (sum(c[4] for c in cached) / len(cached))
    print(f"{len(cached)} of their moves cached, "
          f"{sum(c[4] for c in cached)/len(cached):.1f} legal moves on average, "
          f"so guessing at random scores {chance:.1f}%", flush=True)
    print("  the card is free from here: the sweep never calls the model again", flush=True)
    print("")

    def run(lr, steps, rank, hist):
        om = OpponentModel(model, "cpu", rank=rank, lr=lr, steps=steps, max_history=hist)
        for h, base, idx, tgt, _ in cached:
            om.observe_cached(h, base, idx, tgt)
        b = om.before
        half = max(len(b) // 2, 1)
        return (100 * sum(b) / len(b), 100 * sum(b[:half]) / half,
                100 * sum(b[half:]) / max(len(b) - half, 1))

    base_mean, base_early, base_late = run(0.02, 8, 4, 60)
    print(f"  {'setting':<34} {'mean':>7} {'early':>7} {'late':>7} {'gain':>7}")
    print(f"  {'CURRENT lr.02 steps8 rank4 h60':<34} {base_mean:>6.2f}% {base_early:>6.2f}% "
          f"{base_late:>6.2f}% {base_late-base_early:>+6.2f}")
    print("")

    rows = []
    for lr, steps, rank, hist in itertools.product(
            (0.0005, 0.002, 0.005, 0.02), (1, 2, 8), (4, 16), (60, 400)):
        m, e, l = run(lr, steps, rank, hist)
        rows.append((l, m, e, lr, steps, rank, hist))
    rows.sort(reverse=True)
    print("  best twelve, ranked by how well they predict the LATE moves")
    for l, m, e, lr, steps, rank, hist in rows[:12]:
        tag = f"lr {lr} steps {steps} rank {rank} hist {hist}"
        print(f"  {tag:<34} {m:>6.2f}% {e:>6.2f}% {l:>6.2f}% {l-e:>+6.2f}")
    print("")
    l, m, e, lr, steps, rank, hist = rows[0]
    print(f"  best late prediction: lr={lr} steps={steps} rank={rank} max_history={hist}")
    print(f"    {l:.2f}% against {base_late:.2f}% now, and {chance:.1f}% for guessing")
