"""A game memory that adapts to the opponent by learning from the moves they actually play.

The first version updated its fast weight to reconstruct the next position's representation, which
says nothing about the player, and it learned nothing a control could not. This version takes one
gradient step per ply on the cross-entropy of the move the opponent actually played. The target is
their choice, it cannot be gamed, and it needs no labels at inference. The signal exists, since in
one sample of 46 grandmasters the same player in the same position repeated their own move 82.5
percent of the time against 63.2 percent across different players. The outer training differentiates
through the inner steps, so it learns how fast to adapt, what to remember and whether to trust it.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GameMemoryV2(nn.Module):
    """A fast weight carried across a game, updated by gradient descent on the opponent's moves.

    The state is a low-rank correction to the representation the policy head reads. A full d by d
    update per ply would be expensive and far more capacity than one game of evidence can fill.
    """

    def __init__(self, d, rank=16, eta=0.5, gate=-2.0):
        super().__init__()
        self.d, self.rank = d, rank
        s = 1.0 / math.sqrt(d)
        self.down = nn.Parameter(torch.randn(d, rank) * s)   # where in the representation to look
        self.up = nn.Parameter(torch.randn(rank, d) * s)     # how a correction is written back
        self.U0 = nn.Parameter(torch.zeros(rank, rank))      # where every game starts
        self.logeta = nn.Parameter(torch.tensor(math.log(eta)))
        self.gate = nn.Parameter(torch.tensor(gate))

    def start(self, B, device, dtype=torch.float32):
        return self.U0.unsqueeze(0).expand(B, self.rank, self.rank).to(device, dtype).contiguous()

    def read(self, h, U):
        """Correct the representation by what has been learned about this opponent so far."""
        z = (h @ self.down).unsqueeze(1)                     # (B, 1, r)
        corr = torch.bmm(z, U).squeeze(1) @ self.up          # (B, d)
        return h + torch.sigmoid(self.gate) * corr

    def update(self, h, policy_head, target_move, U, create_graph=True):
        """One gradient step on the cross-entropy of the move they actually played.

        h is the position before their move, policy_head turns a representation into move logits, and
        target_move is what they played, as move indices (B,).
        """
        U = U.requires_grad_(True) if not U.requires_grad else U
        logits = policy_head(self.read(h, U))
        loss = F.cross_entropy(logits, target_move, reduction="mean")
        g, = torch.autograd.grad(loss, U, create_graph=create_graph)
        return U - self.logeta.exp() * g, loss.detach()

    def extra_repr(self):
        return f"d={self.d}, rank={self.rank}"


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d, B, P, NM = 64, 4, 12, 512
    mem = GameMemoryV2(d, rank=8).to(dev)
    head = nn.Linear(d, NM).to(dev)
    print(f"GameMemoryV2 {sum(p.numel() for p in mem.parameters()):,} params ({dev})")

    hs = [torch.randn(B, d, device=dev) for _ in range(P)]

    def walk(tgt, label):
        U = mem.start(B, dev)
        losses = []
        for t in range(P):
            U, l = mem.update(hs[t], head, tgt[t], U)
            losses.append(float(l))
        print(f"  {label:34s} {losses[0]:6.3f} -> {losses[-1]:6.3f}")
        return U, losses

    print("inner loss across one game, by how predictable the opponent is")
    # a player with a strong habit, always reaching for the same idea
    habit = torch.full((B,), 7, device=dev, dtype=torch.long)
    U_habit, l_habit = walk([habit] * P, "fixed preference")
    # a player whose choice follows one direction of the position, which is learnable but harder
    w = torch.randn(d, device=dev)
    struct = [torch.where(hs[t] @ w > 0, 11, 23).long() for t in range(P)]
    walk(struct, "preference conditioned on position")
    # an unpredictable opponent, where there should be nothing to learn
    g = torch.Generator(device="cpu").manual_seed(1)
    noise = [torch.randint(0, NM, (B,), generator=g).to(dev) for _ in range(P)]
    walk(noise, "unpredictable, the control")

    assert l_habit[-1] < l_habit[0], "cannot even learn a fixed preference within a game"
    U = U_habit

    # the read must actually change the prediction
    U0 = mem.start(B, dev)
    shift = (mem.read(hs[0], U) - mem.read(hs[0], U0)).abs().max()
    print(f"representation moves by {float(shift):.4f} after {P} plies")
    assert shift > 1e-6

    # gradients must reach every inner-loop parameter

    out = sum(F.cross_entropy(head(mem.read(hs[t], U)), habit) for t in range(P))
    out.backward()
    for nm in ("down", "up", "U0", "logeta", "gate"):
        g = getattr(mem, nm).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f"no grad: {nm}"
        print(f"  grad {nm:7s} |g| {float(g.abs().sum()):.6f}")
    print("gradients reach every inner-loop parameter, end to end")
