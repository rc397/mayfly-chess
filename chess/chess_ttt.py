"""Test-time training on the transformer encoder, a game memory that adapts to the opponent.

After the opponent moves, their real move is known, so the inner loss is a true prediction error on
a real event. It cannot be gamed by a self-chosen objective and needs no labels at inference.
GameMemory only reads and writes the CLS representation, so it works with any encoder. The copy in
mayfly_chess.py is the older tropical version, this is the canonical one.
"""
import os, sys, math
import torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet, N_MOVES


class GameMemory(nn.Module):
    """A fast weight carried across a game, updated on the opponent's actual moves.

    The state is a small linear map W. Reading it asks what this opponent tends to do in this
    position, and writing it takes a descent step on the move they really played. W0, the step
    size and the three projections are learned end to end.
    """

    def __init__(self, d, sc=None):
        super().__init__()
        s = sc if sc is not None else 1.0 / math.sqrt(d)
        self.d = d
        self.thK = nn.Parameter(torch.randn(d, d) * s)      # what to learn from
        self.thV = nn.Parameter(torch.randn(d, d) * s)      # what to reconstruct
        self.thQ = nn.Parameter(torch.randn(d, d) * s)      # what to ask at prediction time
        self.W0 = nn.Parameter(torch.randn(d, d) * s)       # where every game starts
        self.logeta = nn.Parameter(torch.tensor(math.log(0.5)))   # learned inner step size
        self.gate = nn.Parameter(torch.tensor(-2.0))        # how much to trust the fast weight

    def start(self, B, device):
        return self.W0.unsqueeze(0).expand(B, self.d, self.d).contiguous()

    def read(self, h, W):
        """Blend the base representation with what has been learned about this opponent."""
        q = (h @ self.thQ).unsqueeze(1)                     # (B, 1, d)
        adapted = torch.bmm(q, W).squeeze(1)                # (B, d)
        return h + torch.sigmoid(self.gate) * adapted

    def update(self, h, h_actual, W):
        """One descent step on ||k W - v||^2 / 2, where v comes from what actually happened.

        h is the position before the move, h_actual the position after the move the opponent played.
        """
        k = h @ self.thK
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)       # unit norm keeps one step stable
        k = k.unsqueeze(1)                                  # (B, 1, d)
        v = (h_actual @ self.thV).unsqueeze(1)              # (B, 1, d)
        r = torch.bmm(k, W) - v                             # residual, the prediction error
        return W - self.logeta.exp() * torch.bmm(k.transpose(1, 2), r), (r * r).mean(dim=(1, 2))


class ChessTTT(nn.Module):
    """Transformer encoder plus a game memory. Plays, watches, adapts, all within one game."""

    def __init__(self, d=256, nhead=16, layers=4, dim_ff=512):
        super().__init__()
        self.net = ChessNet(d=d, nhead=nhead, layers=layers, dim_ff=dim_ff)
        self.mem = GameMemory(d)
        self.d = d

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def load_base(self, path, map_location="cpu"):
        """Warm start the encoder from a supervised checkpoint. The memory stays at its init."""
        ck = torch.load(path, map_location=map_location)
        self.net.load_state_dict(ck["model"], strict=True)
        return ck.get("val", {})

    def encode(self, ids, cast, hm=None, ep=None):
        """Board to the CLS representation, mirroring ChessNet.forward up to the heads, including
        the halfmove clock embedding and the en passant mark. hm None means unknown, bucket 101."""
        n = self.net
        B = ids.shape[0]
        x = n.piece(ids) + n.pos(n.posidx)[None]                        # (B,64,d)
        if ep is not None and hasattr(n, "ep_mark"):
            has = (ep > 0).to(x.dtype)[:, None]
            rows = torch.arange(B, device=x.device)
            sq = (ep - 1).clamp(min=0)
            x[rows, sq] = x[rows, sq] + n.ep_mark.to(x.dtype) * has
        cls = n.cls.expand(B, -1, -1) + n.cast(cast)[:, None, :]        # (B,1,d)
        if hasattr(n, "hm"):
            if hm is None:
                hm = torch.full((B,), 101, dtype=torch.long, device=ids.device)
            cls = cls + n.hm(hm)[:, None, :]
        x = torch.cat([cls, x], dim=1)                                  # (B,65,d)
        return n.ln(n.enc(x)[:, 0])                                     # (B,d)

    def heads(self, h):
        n = self.net
        return {"policy": n.policy(h), "value": n.value(h).squeeze(-1),
                "material": n.h_material(h), "king": n.h_king(h), "pawns": n.h_pawns(h),
                "mob": n.h_mob(h).squeeze(-1), "bishop": n.h_bishop(h)}

    def forward(self, ids, cast, W=None):
        """Single position. Pass W to use the game memory, omit it for the plain net."""
        h = self.encode(ids, cast)
        if W is not None:
            h = self.mem.read(h, W)
        return self.heads(h)

    def play_game(self, ids_seq, cast_seq, bptt=0):
        """Walk a whole game, predicting each ply then learning from what was actually played.

        ids_seq and cast_seq are (B, P, ...) successive positions from real games. With bptt > 0
        the fast weight is detached every bptt plies, which bounds memory on long games.

        The inner update at ply t uses only position t+1, which at inference is the move the
        opponent made, so nothing needs a label. Gradients flow through every inner step that is
        not detached.
        """
        B, P = ids_seq.shape[0], ids_seq.shape[1]
        W = self.mem.start(B, ids_seq.device)
        hs = [self.encode(ids_seq[:, t], cast_seq[:, t]) for t in range(P)]
        outs, surprise = [], []
        for t in range(P):
            outs.append(self.heads(self.mem.read(hs[t], W)))
            if t + 1 < P:
                W, s = self.mem.update(hs[t], hs[t + 1], W)
                surprise.append(s)
                if bptt and (t + 1) % bptt == 0:
                    W = W.detach()
        return outs, surprise


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ChessTTT(d=64, nhead=4, layers=2, dim_ff=128).to(dev)
    print(f"ChessTTT {m.num_params()/1e6:.3f}M params  ({dev})")

    B, P = 3, 8
    ids = torch.randint(0, 13, (B, P, 64), device=dev)
    cast = torch.rand(B, P, 4, device=dev)

    outs, surprise = m.play_game(ids, cast)
    assert len(outs) == P and len(surprise) == P - 1
    assert outs[0]["policy"].shape == (B, N_MOVES)
    print(f"walked {P} plies, surprise {surprise[0].mean():.4f} -> {surprise[-1].mean():.4f}")

    # the memory must actually change the prediction, or the read is a no-op
    W = m.mem.start(B, dev)
    h = m.encode(ids[:, 0], cast[:, 0])
    W2, _ = m.mem.update(h, m.encode(ids[:, 1], cast[:, 1]), W)
    d0 = (m.mem.read(h, W2) - m.mem.read(h, W)).abs().max().item()
    assert d0 > 1e-6, "game memory does not affect the representation"
    print(f"memory shifts the representation by {d0:.4f} after one update")

    # gradients must reach every inner-loop parameter

    loss = sum(o["policy"].square().mean() for o in outs)
    loss.backward()
    for nm in ("thK", "thV", "thQ", "W0", "logeta", "gate"):
        g = getattr(m.mem, nm).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f"no gradient: {nm}"
        print(f"  grad {nm:7s} |g| {g.abs().sum():.6f}")
    print("gradients reach every inner-loop parameter, end to end")
