"""MayFly's forward pass in JAX, for running self-play on a TPU (see selfplay_remote.py).

JAX drives every TPU chip from one process with the batch sharded across them. Only what a v5 search
reads is here, namely the trunk, the CLS policy and action-value heads, and the tail that
MCTS._dispatch_one_copy runs on the GPU (policy at the legal moves, every legal move's q, and the
policy-weighted leaf value). The arithmetic matches chess_net.ChessNet.forward, and
check_against_torch() compares the two. With dtype bfloat16, matrix products run in bfloat16 with
float32 accumulation and layer norms and softmaxes in float32, as under CUDA autocast. float32 is for
checking.

    python chessnet_jax.py [checkpoint.pt]
"""
import numpy as np


def params_from_torch(sd, cfg):
    """The weights a forward pass needs, as float32 numpy arrays, from a ChessNet state dict and its cfg."""
    g = lambda k: np.asarray(sd[k].detach().cpu().float().numpy() if hasattr(sd[k], "detach") else sd[k],
                             np.float32)
    d = int(cfg.get("d", 1024))
    layers = int(cfg.get("layers", 16))
    p = {"piece": g("piece.weight"), "pos": g("pos.weight"), "cls": g("cls").reshape(d),
         "cast_w": g("cast.weight").T.copy(), "cast_b": g("cast.bias"), "hm": g("hm.weight"),
         "ep_mark": g("ep_mark"), "ln_g": g("ln.weight"), "ln_b": g("ln.bias"),
         "pol_w": g("policy.weight").T.copy(), "pol_b": g("policy.bias"),
         "q_w": g("qvalue.weight").T.copy(), "q_b": g("qvalue.bias"), "layers": []}
    for i in range(layers):
        k = f"enc.layers.{i}."
        p["layers"].append({
            "in_w": g(k + "self_attn.in_proj_weight").T.copy(), "in_b": g(k + "self_attn.in_proj_bias"),
            "out_w": g(k + "self_attn.out_proj.weight").T.copy(), "out_b": g(k + "self_attn.out_proj.bias"),
            "l1_w": g(k + "linear1.weight").T.copy(), "l1_b": g(k + "linear1.bias"),
            "l2_w": g(k + "linear2.weight").T.copy(), "l2_b": g(k + "linear2.bias"),
            "n1_g": g(k + "norm1.weight"), "n1_b": g(k + "norm1.bias"),
            "n2_g": g(k + "norm2.weight"), "n2_b": g(k + "norm2.bias")})
    return p


def make_forward(nhead, dtype="bfloat16"):
    """forward(params, ids, cast, hm, ep, I, Ip, M) returns (pg, qg, val), the tail of v5's pass.

    ids is (B, 64) int, cast (B, 4) float, hm and ep (B,) int, I and Ip (B, W) int (the q head's and the
    policy's legal-move indices), M (B, W) bool. pg is the policy logits at Ip, qg every legal move's q
    in [-1, 1], val the policy-weighted leaf value."""
    import jax
    import jax.numpy as jnp

    cdt = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32

    def mm(x, w, b=None):
        y = jnp.matmul(x.astype(cdt), w.astype(cdt), preferred_element_type=jnp.float32)
        return y if b is None else y + b

    def ln(x, gamma, beta, eps=1e-5):
        x = x.astype(jnp.float32)
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) * jax.lax.rsqrt(var + eps) * gamma + beta

    def layer(x, q):
        B, S, d = x.shape
        dh = d // nhead
        h = ln(x, q["n1_g"], q["n1_b"])
        qkv = mm(h, q["in_w"], q["in_b"])                                   # (B, S, 3d)
        qq, kk, vv = jnp.split(qkv, 3, axis=-1)
        qq = qq.reshape(B, S, nhead, dh).transpose(0, 2, 1, 3)
        kk = kk.reshape(B, S, nhead, dh).transpose(0, 2, 1, 3)
        vv = vv.reshape(B, S, nhead, dh).transpose(0, 2, 1, 3)
        s = jnp.matmul(qq.astype(cdt), kk.astype(cdt).transpose(0, 1, 3, 2),
                       preferred_element_type=jnp.float32) * (dh ** -0.5)
        a = jax.nn.softmax(s, axis=-1)
        o = jnp.matmul(a.astype(cdt), vv.astype(cdt), preferred_element_type=jnp.float32)
        o = o.transpose(0, 2, 1, 3).reshape(B, S, d)
        x = x + mm(o, q["out_w"], q["out_b"])
        h = ln(x, q["n2_g"], q["n2_b"])
        h = jax.nn.gelu(mm(h, q["l1_w"], q["l1_b"]), approximate=False)
        return x + mm(h, q["l2_w"], q["l2_b"])

    def forward(params, ids, cast, hm, ep, I, Ip, M):
        x = params["piece"][ids] + params["pos"][None]                     # (B, 64, d)
        has = (ep > 0).astype(jnp.float32)
        onehot = jax.nn.one_hot(jnp.clip(ep - 1, 0, 63), 64, dtype=jnp.float32) * has[:, None]
        x = x + onehot[:, :, None] * params["ep_mark"]
        cls = params["cls"][None] + mm(cast, params["cast_w"], params["cast_b"]) + params["hm"][hm]
        x = jnp.concatenate([cls[:, None, :], x], axis=1)                 # (B, 65, d)
        for q in params["layers"]:
            x = layer(x, q)
        h = ln(x[:, 0], params["ln_g"], params["ln_b"])
        pol = mm(h, params["pol_w"], params["pol_b"])                      # (B, 4162)
        qv = jax.nn.sigmoid(mm(h, params["q_w"], params["q_b"]))
        pg = jnp.take_along_axis(pol, Ip, axis=1)
        qg = jnp.take_along_axis(qv, I, axis=1)
        w = jax.nn.softmax(jnp.where(M, pg, -jnp.inf), axis=-1)
        val = (w * qg).sum(1) * 2.0 - 1.0
        return pg, qg * 2.0 - 1.0, val

    return forward


def check_against_torch(ckpt, n=96, seed=0):
    """Compare the JAX pass at float32 with ChessNet's forward on n positions from random play.

    Returns the largest differences. Runs on a CPU."""
    import random
    import chess
    import jax
    import torch
    from chess_net import ChessNet
    from encode import board_to_ids, clock_and_ep
    from moves import move_to_idx

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = ChessNet(d=c["d"], nhead=c["nhead"], layers=c["layers"], dim_ff=int(c["ff_mult"] * c["d"]),
                 qhead=True, att=bool(c.get("att")), dist=int(c.get("dist", 0)), mlh=bool(c.get("mlh"))).eval()
    m.load_state_dict(ck["model"], strict=False)
    rng = random.Random(seed)
    # two positions with a legal en passant capture (a separate code path) and the start position
    boards = [chess.Board(f) for f in ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
                                       "rnbqkbnr/pppp1ppp/8/8/3PpP2/8/PPP1P1PP/RNBQKBNR b KQkq f3 0 3",
                                       chess.STARTING_FEN)]
    while len(boards) < n:
        b = chess.Board()
        for _ in range(rng.randrange(0, 80)):
            mv = list(b.legal_moves)
            if not mv:
                break
            b.push(rng.choice(mv))
        if not b.is_game_over():
            boards.append(b)
    ids, cast, hm, ep = [], [], [], []
    W = max(b.legal_moves.count() for b in boards)
    I = np.zeros((n, W), np.int64)
    M = np.zeros((n, W), bool)
    for i, b in enumerate(boards):
        r, cs, fl = board_to_ids(b)
        ids.append(r); cast.append(cs)
        h, e = clock_and_ep(b, fl)
        hm.append(h); ep.append(e)
        flip = b.turn == chess.BLACK
        mv = list(b.legal_moves)
        I[i, :len(mv)] = [move_to_idx(x, flip) for x in mv]
        M[i, :len(mv)] = True
    ids = np.asarray(ids, np.int64); cast = np.asarray(cast, np.float32)
    hm = np.asarray(hm, np.int64); ep = np.asarray(ep, np.int64)
    with torch.no_grad():
        out = m(torch.from_numpy(ids), torch.from_numpy(cast), torch.from_numpy(hm), torch.from_numpy(ep))
        It, Mt = torch.from_numpy(I), torch.from_numpy(M)
        pg = torch.gather(out["policy"].float(), 1, It)
        qg = torch.gather(torch.sigmoid(out["qvalue"].float()), 1, It)
        w = torch.softmax(pg.masked_fill(~Mt, float("-inf")), -1)
        val = ((w * qg).sum(1) * 2 - 1).numpy()
        pg, qg = pg.numpy(), (qg * 2 - 1).numpy()
    p = params_from_torch(ck["model"], c)
    f = jax.jit(make_forward(c["nhead"], "float32"))
    jp, jq, jv = (np.asarray(x) for x in f(p, ids.astype(np.int32), cast, hm.astype(np.int32),
                                            ep.astype(np.int32), I.astype(np.int32), I.astype(np.int32), M))
    return {"policy": float(np.abs((jp - pg)[M]).max()), "q": float(np.abs((jq - qg)[M]).max()),
            "value": float(np.abs(jv - val).max()), "positions": n}


if __name__ == "__main__":
    import os
    import sys
    import paths
    print(check_against_torch(sys.argv[1] if len(sys.argv) > 1 else os.path.join(paths.VERSIONS, "mayfly_v5.pt")))

