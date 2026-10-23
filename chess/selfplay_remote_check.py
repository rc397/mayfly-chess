r"""selfplay_remote_check tests whether a RemoteMCTS search plays what the local search plays.

It runs two checks.
  1. lean   selfplay_remote's lean network against ChessNet's own forward pass, float32 on the CPU.
  2. search N self-play positions searched twice at VISITS with the version's sealed search and no root
            noise, once by MCTS in this process and once by RemoteMCTS over the wire to a server thread
            running the same network on padded batches. Reports how often the move agrees and the mean
            total variation between the two root visit shares.

  python chess/selfplay_remote_check.py [positions file] [N] [VISITS]
  env: SP_CKPT (default mayfly_v5.pt in paths.VERSIONS)
"""
import os
import sys
import time
import threading
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def check_lean(ckpt, n=16):
    import torch
    from chessnet_jax import params_from_torch, check_against_torch  # noqa: F401
    from selfplay_remote import _torch_net
    from chess_net import ChessNet
    import random
    import chess
    from encode import board_to_ids, clock_and_ep
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    full = ChessNet(d=c["d"], nhead=c["nhead"], layers=c["layers"], dim_ff=int(c["ff_mult"] * c["d"]), qhead=True,
                    att=bool(c.get("att")), dist=int(c.get("dist", 0)), mlh=bool(c.get("mlh"))).eval()
    full.load_state_dict(ck["model"], strict=False)
    lean = _torch_net(c, params_from_torch(ck["model"], c)).eval()
    rng = random.Random(1)
    rows = []
    while len(rows) < n:
        b = chess.Board()
        for _ in range(rng.randrange(0, 60)):
            mv = list(b.legal_moves)
            if not mv:
                break
            b.push(rng.choice(mv))
        if b.ep_square is None and len(rows) % 4 == 0:
            continue                          # keep some positions with an en passant square
        if not b.is_game_over():
            ids, cast, flip = board_to_ids(b)
            hm, ep = clock_and_ep(b, flip)
            rows.append((ids, cast, hm, ep))
    ids = torch.tensor([r[0] for r in rows]); cast = torch.tensor([r[1] for r in rows], dtype=torch.float32)
    hm = torch.tensor([r[2] for r in rows]); ep = torch.tensor([r[3] for r in rows])
    with torch.no_grad():
        o = full(ids, cast, hm, ep)
        p, q = lean(ids, cast, hm, ep)
    return {"policy": float((o["policy"] - p).abs().max()), "q": float((o["qvalue"] - q).abs().max()),
            "with_ep": int((ep > 0).sum())}


def check_search(fens, visits):
    import torch
    import chess
    os.environ["VISITS"] = str(visits)
    import selfplay as sp
    from mcts import MCTS, pinned_search
    from selfplay_remote import remote_tree_class, Merger
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = sp.load_model(dev)
    sealed = pinned_search(os.path.basename(sp.model_path()))
    local = MCTS(model, dev, batch=sp.BATCH).apply(sealed)
    a, b = mp.Pipe()
    stop = threading.Event()
    merger = Merger([64, 128, 256])

    def server():
        # the same network object, the RemoteMCTS wire and the server's padding and tail
        with torch.inference_mode():
            while not stop.is_set():
                got = merger.take([b], 0.05, 192)
                if not got:
                    continue
                (ids, cast, hm, ep, I, Ip, M), spans, n = merger.stack(got)
                T = lambda x, dt=torch.long: torch.from_numpy(x).to(dev, dt)  # noqa: E731
                with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                    out = model(T(ids), T(cast, torch.float32), T(hm), T(ep))
                pg = torch.gather(out["policy"].float(), 1, T(Ip))
                qg = torch.gather(torch.sigmoid(out["qvalue"].float()), 1, T(I))
                w = torch.softmax(pg.masked_fill(~torch.from_numpy(M).to(dev), float("-inf")), -1)
                val = (w * qg).sum(1) * 2.0 - 1.0
                merger.reply(spans, pg.cpu().numpy(), (qg * 2.0 - 1.0).cpu().numpy(), val.cpu().numpy())

    th = threading.Thread(target=server, daemon=True)
    th.start()
    remote = remote_tree_class()(a, sp.BATCH).apply(sealed)
    same, tvs, t0 = 0, [], time.time()
    for fen in fens:
        board = chess.Board(fen)
        got = []
        for tree in (local, remote):
            tree.reset()
            mv, info = tree.search(board, visits=visits)
            r = tree.root
            n = r.N.astype(np.float64)
            got.append((mv, {r.moves[i].uci(): n[i] / max(n.sum(), 1) for i in range(len(r.moves))}))
        same += got[0][0] == got[1][0]
        keys = set(got[0][1]) | set(got[1][1])
        tvs.append(0.5 * sum(abs(got[0][1].get(k, 0) - got[1][1].get(k, 0)) for k in keys))
    stop.set()
    return {"positions": len(fens), "same_move": same, "mean_tv": round(float(np.mean(tvs)), 4),
            "max_tv": round(float(np.max(tvs)), 4), "seconds": round(time.time() - t0)}


if __name__ == "__main__":
    import paths
    ck = os.environ.get("SP_CKPT", paths.under("versions", "mayfly_v5.pt"))
    os.environ.setdefault("VERSION", "mayfly_v5.pt")
    print("lean:", check_lean(ck), flush=True)
    src = sys.argv[1] if len(sys.argv) > 1 else paths.under("selfplay", "positions.txt")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    visits = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    lines = [l.strip() for l in open(src, encoding="utf-8") if l.strip()]
    step = max(1, len(lines) // (n * 3))
    fens = lines[-step * n * 3::step][:n]
    print("search:", check_search(fens, visits), flush=True)
