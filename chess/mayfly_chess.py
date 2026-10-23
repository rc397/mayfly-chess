"""
ChessNet's twin with the transformer encoder replaced by a tropical (max-plus) one.

Everything except the encoder matches chess_net.ChessNet (embeddings, the CLS token carrying
castling rights, the seven heads, board canonicalisation, move encoding), so a difference in
strength comes from the block alone. Negamax is a maximum over moves of a sum along a line, which
is the tropical semiring this network is built from. The prediction to test is that a tropical net
gains more over the transformer at search depth 1 than at depth 5, if each max-plus layer acts like
a minimax backup.

Two details matter. The mixer output is centred, because max_k(x_k + w_kj) sits far above zero and
added raw to the residual stream it swamps the signal after normalisation. The temperature is
constant, not annealed, since annealing hurt deeper models. The mixer is not causal, every square
sees every other square.

  python chess/mayfly_chess.py        # self-test, parameter match against ChessNet and a forward pass
"""
import math
import chess
import torch
import torch.nn as nn

from chess_net import board_to_ids, move_to_idx, idx_to_move, N_MOVES


# ---- the primitive ----------------------------------------------------------
# A soft tropical product is a matmul in exponential space, exactly:
#
#     T * logsumexp_k((x_k + w_kj) / T) == a_i + b_j + T * log(exp((x - a)/T) @ exp((w - b)/T))
#
# with a_i the row maximum of x and b_j the column maximum of w, pulled out so nothing overflows.
# The naive form builds an (n, d_out, d_in) tensor, while this one runs as a cuBLAS matmul. Only as
# T approaches zero would the exponentials underflow.
EPS = 1e-30


def tmatmul(x, w, T=0.66, chunk=None, ckpt=False):
    """out[..., i, j] = softmax-max_k ( x[..., i, k] + w[k, j] ), as a matmul."""
    a = x.max(dim=-1, keepdim=True).values                   # (..., n, 1)
    b = w.max(dim=0, keepdim=True).values                    # (1, d_out)
    ex = torch.exp((x - a) / T)
    ew = torch.exp((w - b) / T)
    return a + b + T * torch.log(ex @ ew + EPS)


def tpoly(x, w, H, T=0.66, chunk=None):
    """Sum of H max-plus terms per output.

    One maximum reads one input, so H terms let each output read H inputs while every selection
    stays tropical."""
    y = tmatmul(x, w, T)
    return y.reshape(*y.shape[:-1], y.shape[-1] // H, H).sum(-1)


def trational(x, wa, wb, H, T=0.66, chunk=None):
    """Difference of two tropical polynomials.

    max_k(x_k + w_k) is monotone in x, and a monotone stack cannot model chess. Differences of
    tropical polynomials are exactly the piecewise-linear functions."""
    return tpoly(x, wa, H, T) - tpoly(x, wb, H, T)


def tmix(q, k, v, T=0.66, Ts=None, ckpt=False, return_scores=False):
    """Best-path mixing over the board, not causal, both stages as matmuls.

        score[i, s] = max_c ( q[i, c] + k[s, c] )          tropical inner product
        out[i, j]   = max_s ( score[i, s] + v[s, j] )      strongest route into square i

    Attention averages over the squares it sees, this selects the strongest one.

    Ts is the temperature of the second stage, the contraction over squares, and it alone decides
    how many squares each output reads. With Ts equal to T a trained model read about 46 of 65
    squares, a blend rather than a selection. Ts defaults to T so older checkpoints reproduce.
    A small Ts is numerically safe, since the row maximum is subtracted first and the winning term
    always contributes exp(0) = 1.
    """
    Ts = T if Ts is None else Ts
    aq = q.max(dim=-1, keepdim=True).values                  # (B, n, 1)
    ak = k.max(dim=-1, keepdim=True).values                  # (B, n, 1)
    A = aq + ak.transpose(1, 2) + T * torch.log(
        torch.exp((q - aq) / T) @ torch.exp((k - ak) / T).transpose(1, 2) + EPS)   # (B, n, n)
    aA = A.max(dim=-1, keepdim=True).values                  # (B, n, 1)
    av = v.max(dim=1, keepdim=True).values                   # (B, 1, dv)
    out = aA + av + Ts * torch.log(
        torch.exp((A - aA) / Ts) @ torch.exp((v - av) / Ts) + EPS)                 # (B, n, dv)
    return (out, A, v) if return_scores else out


@torch.no_grad()
def read_width(A, v, Ts, max_items=4):
    """How many squares each output reads, and how much weight sits on the winner.

    Width is the exponential of the attention entropy, so 1 is a clean selection and 65 a flat
    average over the board. The weights are not softmax(A / Ts). The second stage is a logsumexp
    of A[i, s] + v[s, j], so the value vector takes part and the weights differ per output
    feature. Scoring A alone overstates the width.

    Full weights are (batch, out, src, feature), so only a few batch items are scored.
    """
    n = min(max_items, A.shape[0])
    W = (A[:n].unsqueeze(-1) + v[:n].unsqueeze(1)) / Ts      # (b, n_out, n_src, dv)
    P = torch.softmax(W, dim=2)
    ent = -(P * torch.log(P + 1e-12)).sum(2)
    return float(ent.exp().mean()), float(P.max(2).values.mean())


@torch.no_grad()
def calibrate_ts(A, v, target_width, lo=1e-4, hi=100.0, iters=30):
    """Solve for the square-stage temperature that makes the mixer read about target_width squares.

    Effective width increases with Ts, so bisection on the model's real scores and values works.
    """
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if read_width(A, v, mid)[0] > target_width:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


class TropicalBlock(nn.Module):
    """learn_ts and mix_norm default to off, so a block built without them loads older checkpoints
    strictly."""

    def __init__(self, d, dff_h, H, T, chunk, sc, learn_ts=False, mix_norm=False):
        super().__init__()
        self.H, self.T, self.chunk = H, T, chunk
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        p = lambda a, b: nn.Parameter(torch.randn(a, b) * sc)
        self.wq, self.wk, self.wv = p(d, d), p(d, d), p(d, d)
        self.w1a, self.w1b = p(d, dff_h * H), p(d, dff_h * H)
        self.w2a, self.w2b = p(dff_h, d * H), p(dff_h, d * H)
        # Learned square-stage temperature, calibrated from real scores before training.
        self.log_ts = nn.Parameter(torch.tensor(math.log(T))) if learn_ts else None
        # Rescales the centred mixer output, which otherwise writes only a small share of the
        # residual next to the feed-forward.
        self.mix_g = nn.Parameter(torch.ones(d)) if mix_norm else None

    def ts(self):
        return self.T if self.log_ts is None else self.log_ts.exp()

    def mix(self, x, return_scores=False):
        h = self.n1(x)
        out = tmix(tmatmul(h, self.wq, self.T, self.chunk),
                   tmatmul(h, self.wk, self.T, self.chunk),
                   tmatmul(h, self.wv, self.T, self.chunk),
                   self.T, Ts=self.ts(), return_scores=return_scores)
        m, A, v = out if return_scores else (out, None, None)
        m = m - m.mean(-1, keepdim=True)        # centre the mixer output, see the module docstring
        if self.mix_g is not None:
            m = m * torch.rsqrt(m.pow(2).mean(-1, keepdim=True) + 1e-6) * self.mix_g
        return (m, A, v) if return_scores else m

    def forward(self, x):
        x = x + self.mix(x)
        h = self.n2(x)
        h = trational(h, self.w1a, self.w1b, self.H, self.T, self.chunk)
        return x + trational(h, self.w2a, self.w2b, self.H, self.T, self.chunk)


class MayflyChessNet(nn.Module):
    """Drop-in twin of ChessNet. Same inputs, same outputs, same heads, tropical encoder."""

    def __init__(self, d=256, layers=4, dff_h=72, H=8, T=0.66, chunk=64, sc=1.5,
                 learn_ts=False, mix_norm=False):
        super().__init__()
        self.piece = nn.Embedding(13, d)
        self.pos = nn.Embedding(64, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.cast = nn.Linear(4, d)
        self.blocks = nn.ModuleList([TropicalBlock(d, dff_h, H, T, chunk, sc, learn_ts, mix_norm)
                                     for _ in range(layers)])
        self.ln = RMSNorm(d)
        self.policy = nn.Linear(d, N_MOVES)
        self.value = nn.Linear(d, 1)
        self.h_material = nn.Linear(d, 2)
        self.h_king = nn.Linear(d, 2)
        self.h_pawns = nn.Linear(d, 6)
        self.h_mob = nn.Linear(d, 1)
        self.h_bishop = nn.Linear(d, 2)
        self.register_buffer("posidx", torch.arange(64))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def embed(self, ids, cast):
        B = ids.shape[0]
        x = self.piece(ids) + self.pos(self.posidx)[None]
        cls = self.cls.expand(B, -1, -1) + self.cast(cast)[:, None, :]
        return torch.cat([cls, x], dim=1)                    # (B, 65, d)

    @torch.no_grad()
    def calibrate(self, ids, cast, target_width=4.0):
        """Set each layer's square-stage temperature so the mixer starts out selecting.

        Layer by layer on real data, because each layer's scores depend on the layers before it.
        """
        if all(b.log_ts is None for b in self.blocks):
            return []
        x, chosen = self.embed(ids, cast), []
        for blk in self.blocks:
            _, A, v = blk.mix(x, return_scores=True)
            ts = calibrate_ts(A, v, target_width)
            blk.log_ts.fill_(math.log(ts))
            chosen.append(ts)
            x = blk(x)
        return chosen

    @torch.no_grad()
    def mixer_stats(self, ids, cast):
        """Per layer, the squares read, the weight on the winner, and the mixer's share of the
        residual."""
        x, out = self.embed(ids, cast), []
        for blk in self.blocks:
            m, A, v = blk.mix(x, return_scores=True)
            width, top1 = read_width(A, v, blk.ts())
            x = x + m
            h = trational(blk.n2(x), blk.w1a, blk.w1b, blk.H, blk.T, blk.chunk)
            f = trational(h, blk.w2a, blk.w2b, blk.H, blk.T, blk.chunk)
            x = x + f
            out.append({"width": width, "top1": top1, "ts": float(blk.ts()),
                        "mix_std": float(m.std()), "ffwd_std": float(f.std()),
                        "mix_share": float(m.std()) / (float(x.std()) + 1e-9)})
        return out

    def forward(self, ids, cast):
        x = self.embed(ids, cast)
        for blk in self.blocks:
            x = blk(x)
        h = self.ln(x[:, 0])                                 # CLS
        return {"policy": self.policy(h), "value": self.value(h).squeeze(-1),
                "material": self.h_material(h), "king": self.h_king(h),
                "pawns": self.h_pawns(h), "mob": self.h_mob(h).squeeze(-1),
                "bishop": self.h_bishop(h)}

    @torch.no_grad()
    def best_move(self, board, device="cpu"):
        ids, cast, flip = board_to_ids(board)
        out = self.forward(torch.tensor([ids], device=device), torch.tensor([cast], device=device))
        legal = list(board.legal_moves)
        if not legal:
            return None
        idxs = torch.tensor([move_to_idx(m, flip) for m in legal], device=device)
        return legal[int(torch.argmax(out["policy"][0][idxs]))]


def matched_dff_h(d, H):
    """dff_h that makes a tropical block cost the same as a transformer block of width d.

    transformer layer = 4d^2 (q,k,v,o) + 2*d*(4d) (feed-forward) = 12d^2
    tropical layer    = 3d^2 (q,k,v)   + 4*d*dff_h*H
    so dff_h*H = 2.25d.
    """
    return max(1, round(2.25 * d / H))


if __name__ == "__main__":
    from chess_net import ChessNet
    d, layers, H = 256, 4, 8
    dff_h = matched_dff_h(d, H)
    trop = MayflyChessNet(d=d, layers=layers, dff_h=dff_h, H=H)
    ctrl = ChessNet(d=d, nhead=max(4, d // 64), layers=layers, dim_ff=4 * d)
    print(f"d={d} layers={layers} H={H} dff_h={dff_h}")
    print(f"  MAYFLY  {trop.num_params()/1e6:7.3f}M")
    print(f"  CONTROL {ctrl.num_params()/1e6:7.3f}M")
    print(f"  ratio   {trop.num_params()/ctrl.num_params():.3f}")

    b = chess.Board()
    ids, cast, flip = board_to_ids(b)
    ii = torch.tensor([ids]); cc = torch.tensor([cast])
    out = trop(ii, cc)
    print(f"\n  forward ok: policy {tuple(out['policy'].shape)}  value {tuple(out['value'].shape)}")
    print(f"  untrained best move from start: {trop.best_move(b)}")

    # A bare maximum is not zero-centred, so check the centring holds inside each block.
    with torch.no_grad():
        x = trop.piece(ii) + trop.pos(trop.posidx)[None]
        cls = trop.cls.expand(1, -1, -1) + trop.cast(cc)[:, None, :]
        x = torch.cat([cls, x], dim=1)
        for i, blk in enumerate(trop.blocks):
            x = blk(x)
            r = (x.std() / (x.mean().abs() + 1e-9)).item()
            print(f"  layer {i}: residual mean {x.mean():+.3f} std {x.std():.3f} signal ratio {r:.2f}")


# =============================================================================
# Test-time training, a fast weight that learns the opponent during the game
# =============================================================================
# After the opponent moves, the move they played is known, so the inner loss is a real prediction
# error that cannot be gamed and needs no labels at inference. A fast weight W is carried across
# the game and updated by gradient descent on predicting the opponent's move. Outer training
# differentiates through the inner updates, so the network learns what to remember, how fast to
# learn, and where to start.


class GameMemory(nn.Module):
    """A fast weight carried across a game, updated on the opponent's actual moves.

    The state is a small linear map W. Reading it asks what this opponent tends to do in this
    position. Writing it takes a descent step on the move they played. W0, the step size and the
    three projections are learned end to end.
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
        """Blend the base representation with what this opponent has taught us so far."""
        q = (h @ self.thQ).unsqueeze(1)                     # (B, 1, d)
        adapted = torch.bmm(q, W).squeeze(1)                # (B, d)
        return h + torch.sigmoid(self.gate) * adapted

    def update(self, h, h_actual, W):
        """One descent step on ||k W - v||^2 / 2, where v comes from what actually happened.

        h        the position before the move
        h_actual the position after the move the opponent played
        """
        k = h @ self.thK
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)       # unit norm keeps one step stable
        k = k.unsqueeze(1)                                  # (B, 1, d)
        v = (h_actual @ self.thV).unsqueeze(1)              # (B, 1, d)
        r = torch.bmm(k, W) - v                             # prediction error
        return W - self.logeta.exp() * torch.bmm(k.transpose(1, 2), r), (r * r).mean(dim=(1, 2))


class MayflyChessTTT(nn.Module):
    """Tropical encoder plus a game memory. Plays, watches, adapts, all in one game."""

    def __init__(self, d=256, layers=4, dff_h=None, H=8, T=0.66, chunk=64, sc=1.5):
        super().__init__()
        dff_h = dff_h if dff_h is not None else matched_dff_h(d, H)
        self.net = MayflyChessNet(d=d, layers=layers, dff_h=dff_h, H=H, T=T, chunk=chunk, sc=sc)
        self.mem = GameMemory(d)
        self.d = d

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def encode(self, ids, cast):
        """Board to the CLS representation, before any head reads it."""
        B = ids.shape[0]
        x = self.net.piece(ids) + self.net.pos(self.net.posidx)[None]
        cls = self.net.cls.expand(B, -1, -1) + self.net.cast(cast)[:, None, :]
        x = torch.cat([cls, x], dim=1)
        for blk in self.net.blocks:
            x = blk(x)
        return self.net.ln(x[:, 0])

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

    def play_game(self, ids_seq, cast_seq):
        """Walk a whole game, predicting each move then learning from what was actually played.

        ids_seq and cast_seq are (B, plies, ...) tensors of successive positions in real games.

        Returns per-ply outputs and the inner surprise at each ply. The inner update uses only the
        next position, which at inference is the move the opponent made, so nothing needs a label.
        Gradients flow through every inner step.
        """
        B, P = ids_seq.shape[0], ids_seq.shape[1]
        W = self.mem.start(B, ids_seq.device)
        hs = [self.encode(ids_seq[:, t], cast_seq[:, t]) for t in range(P)]
        outs, surprise = [], []
        for t in range(P):
            outs.append(self.heads(self.mem.read(hs[t], W)))
            if t + 1 < P:                                    # learn from what actually happened
                W, s = self.mem.update(hs[t], hs[t + 1], W)
                surprise.append(s)
        return outs, surprise
