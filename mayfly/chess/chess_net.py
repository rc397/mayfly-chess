"""The MayFly chess network, a transformer board encoder with several heads.

Heads are the policy, the value (win probability for the side to move), an optional action-value
head that scores every move, and auxiliary heads for material, king pressure, pawn structure,
mobility and bishop pair. Optional extras are attention policy heads on the square tokens,
distributional value heads and a moves-left head. The board is always seen from the side to move.

Moves are numbered from*64 + to, then 66 more indices for underpromotions (22 promotion from-to
pairs times three pieces), appended after 4095 so no older index moved. A rook move a7 to a8 shares
an index with a queen promotion there, but only one of the two can ever be legal.
"""
import chess
import torch, torch.nn as nn
# re-exported for older imports, moves.py has the numbering without torch
from moves import (N_MOVES, N_FROMTO, PROMO_IDX, IDX_PROMO, UNDERPROMOTIONS,
                   UCI_PROMO, move_to_idx, idx_to_move)

# the board encoding lives in encode.py, which needs no torch, and is re-exported here
from encode import PIECE_ORD, board_to_ids, clock_and_ep  # noqa: F401

class AttnHead(nn.Module):
    """Leela-style attention policy, a move scored from its two squares as (W_q s_from) . (W_k s_to) / sqrt(dk).

    The 4,096 from-to scores come out in move_to_idx order, because token j is square j in the side to
    move's frame. Each underpromotion is its from-to score plus a per-piece offset from the target square."""

    def __init__(self, d, dk=256):
        super().__init__()
        from moves import IDX_PROMO, UNDERPROMOTIONS
        self.q = nn.Linear(d, dk)
        self.k = nn.Linear(d, dk)
        self.promo = nn.Linear(dk, len(UNDERPROMOTIONS))
        self.scale = dk ** -0.5
        ftp = [IDX_PROMO[N_FROMTO + i] for i in range(len(IDX_PROMO))]
        self.register_buffer("pf", torch.tensor([f for f, _, _ in ftp]), persistent=False)
        self.register_buffer("pt", torch.tensor([t for _, t, _ in ftp]), persistent=False)
        self.register_buffer("pp", torch.tensor([UNDERPROMOTIONS.index(p) for _, _, p in ftp]),
                             persistent=False)

    def forward(self, s):
        """s is (B, 64, d), the square tokens after their own LayerNorm."""
        q, k = self.q(s), self.k(s)
        ft = torch.matmul(q, k.transpose(1, 2)) * self.scale              # (B, from, to)
        under = ft[:, self.pf, self.pt] + self.promo(k)[:, self.pt, self.pp]
        return torch.cat([ft.flatten(1), under], 1)                     # (B, 4162), works for a batch of 0


def hl_gauss(target, bins, sigma=None):
    """A win probability in [0, 1] as an HL-Gauss target over `bins` equal buckets.

    Classification over bins learns a value better than scalar regression (Imani and White, and
    "Stop Regressing"). The target is a Gaussian around the true value integrated over each bucket,
    so neighbouring buckets share the mass and the head stays calibrated.
    """
    sigma = sigma or 0.75 / bins           # three quarters of a bucket, as in the paper
    edges = torch.linspace(0.0, 1.0, bins + 1, device=target.device, dtype=torch.float32)
    z = (edges[None, :] - target.float()[:, None]) / (sigma * (2.0 ** 0.5))
    cdf = torch.erf(z)
    p = cdf[:, 1:] - cdf[:, :-1]
    return p / p.sum(-1, keepdim=True).clamp(min=1e-9)


def bin_centres(bins, device=None):
    return (torch.arange(bins, device=device, dtype=torch.float32) + 0.5) / bins


class MoveDist(nn.Module):
    """The win probability of one move as a distribution over `bins`.

    The trunk's summary vector and a move embedding go through a small network. Conditioning on
    the move keeps this near two million parameters, where a bins-per-move output layer would need
    about 270 million."""

    def __init__(self, d, bins=64, dm=256):
        super().__init__()
        self.mv = nn.Embedding(N_MOVES, dm)
        self.net = nn.Sequential(nn.Linear(d + dm, 512), nn.GELU(), nn.Linear(512, bins))
        self.bins = bins

    def forward(self, h, idx):
        """h (B, d) and idx (B,) or (B, M) move indices -> (B, bins) or (B, M, bins)."""
        e = self.mv(idx)
        if e.dim() == 3:
            h = h[:, None, :].expand(-1, e.shape[1], -1)
        return self.net(torch.cat([h, e], -1))


class ChessNet(nn.Module):
    def __init__(self, d=384, nhead=6, layers=6, dim_ff=1536, qhead=False, att=False, dist=0, mlh=False,
                 dropout=0.0):
        super().__init__()
        self.piece = nn.Embedding(13, d)
        self.pos = nn.Embedding(64, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.cast = nn.Linear(4, d)
        # Dropout is passed explicitly because nn.TransformerEncoderLayer defaults to 0.1. At
        # one pass over billions of examples there is nothing to regularise, so the default is 0.
        layer = nn.TransformerEncoderLayer(d, nhead, dim_ff, dropout=dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.ln = nn.LayerNorm(d)
        self.policy = nn.Linear(d, N_MOVES)
        self.value = nn.Linear(d, 1)
        # The action-value head scores every move. Each ChessBench record names one move, so the
        # loss is masked to that index. Off by default so older checkpoints still load.
        self.qvalue = nn.Linear(d, N_MOVES) if qhead else None
        self.h_material = nn.Linear(d, 2)
        self.h_king = nn.Linear(d, 2)
        self.h_pawns = nn.Linear(d, 6)
        self.h_mob = nn.Linear(d, 1)
        self.h_bishop = nn.Linear(d, 2)
        # The halfmove clock as a bucket rather than a number, since 98 and 99 mean very different
        # things. Index 101 means the corpus carried no clock, which is not the same as zero.
        self.hm = nn.Embedding(102, d)
        # en passant is marked on the capturable square, since the fact is geometric
        self.ep_mark = nn.Parameter(torch.zeros(d))
        nn.init.zeros_(self.hm.weight)
        self.register_buffer("posidx", torch.arange(64))
        # Attention heads on the square tokens, which the CLS heads ignore. The optional heads
        # below are registered last, so existing parameters keep their place in the optimiser
        # state, and are off by default so earlier checkpoints load as before.
        self.att_detach = False
        if att:
            self.ln_sq = nn.LayerNorm(d)
            self.att = AttnHead(d)
            self.att_q = AttnHead(d)
        else:
            self.ln_sq = self.att = self.att_q = None
        # Distributional heads. qdist is one move's action value over `dist` buckets, and vdist is
        # the position's value the same way, trained on the value Leela's search recorded.
        self.dist = int(dist)
        self.dist_detach = False
        if self.dist:
            self.qdist = MoveDist(d, self.dist)
            self.vdist = nn.Linear(d, self.dist)
        else:
            self.qdist = self.vdist = None
        # Moves-left head as in Lc0, plies to the end of the game in units of 100, trained on the
        # plies_left that Leela records carry. Helps endgame conversion.
        self.mlh_detach = False
        self.mleft = (nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))
                      if mlh else None)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids, cast, hm=None, ep=None):
        """hm is the halfmove clock bucket, 0 to 100 or 101 for unknown. ep is the en passant
        target as square plus one, zero for none, already in the side to move's frame."""
        B = ids.shape[0]
        x = self.piece(ids) + self.pos(self.posidx)[None]              # (B,64,d)
        if ep is not None:
            has = (ep > 0).to(x.dtype)[:, None]
            rows = torch.arange(B, device=x.device)
            sq = (ep - 1).clamp(min=0)
            x[rows, sq] = x[rows, sq] + self.ep_mark.to(x.dtype) * has
        if hm is None:
            hm = torch.full((B,), 101, dtype=torch.long, device=ids.device)
        cls = (self.cls.expand(B, -1, -1) + self.cast(cast)[:, None, :]
               + self.hm(hm)[:, None, :])                              # (B,1,d)
        x = torch.cat([cls, x], dim=1)                                 # (B,65,d)
        z = self.enc(x)
        h = self.ln(z[:, 0])                                           # CLS -> (B,d)
        # h is returned so adapters and probes can use the trunk without a second forward pass
        out = {"h": h, "policy": self.policy(h), "value": self.value(h).squeeze(-1),
               "material": self.h_material(h), "king": self.h_king(h),
               "pawns": self.h_pawns(h), "mob": self.h_mob(h).squeeze(-1),
               "bishop": self.h_bishop(h)}
        if self.qvalue is not None:
            out["qvalue"] = self.qvalue(h)
        if self.att is not None:
            # while a new head is still random its gradient would only shake the trunk, so for a
            # while it learns on the trunk as it stands and is then trained end to end
            sq = z[:, 1:].detach() if self.att_detach else z[:, 1:]
            s = self.ln_sq(sq)
            out["policy_att"] = self.att(s)
            out["qvalue_att"] = self.att_q(s)
        if self.vdist is not None:
            # read-out only, it learns from the trunk but sends no gradient back into it, since
            # the search does not read the value head
            out["vdist"] = self.vdist(h.detach())
        if self.mleft is not None:
            out["mleft"] = self.mleft(h.detach() if self.mlh_detach else h).squeeze(-1)
        return out

    def qdist_of(self, h, idx):
        """The action-value distribution for given moves, (B, M) at a leaf or (B,) in training."""

        if self.qdist is None:
            return None
        return self.qdist(h.detach() if self.dist_detach else h, idx)

    @torch.no_grad()
    def best_move(self, board, device="cpu"):
        ids, cast, flip = board_to_ids(board)
        hm, ep = clock_and_ep(board, flip)
        out = self.forward(torch.tensor([ids], device=device),
                           torch.tensor([cast], device=device),
                           torch.tensor([hm], device=device),
                           torch.tensor([ep], device=device))
        logits = out["policy"][0]
        legal = list(board.legal_moves)
        if not legal:
            return None
        idxs = torch.tensor([move_to_idx(m, flip) for m in legal], device=device)
        return legal[int(torch.argmax(logits[idxs]))]

if __name__ == "__main__":
    m = ChessNet()
    b = chess.Board()
    print(f"ChessNet {m.num_params()/1e6:.1f}M | best from start: {m.best_move(b)}")
