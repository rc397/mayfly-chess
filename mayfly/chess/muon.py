r"""muon is orthogonalised momentum for the trunk's weight matrices, kept in AdamW's own state.

Muon (Jordan et al., 2024) steps each 2D weight by its Nesterov momentum with the singular values pushed
towards 1 by five Newton-Schulz iterations, so every direction in the matrix moves by about the same amount
instead of the few largest ones taking the whole step. Scaled by scale * sqrt(max(rows, cols)) the step has
an RMS of about scale, so AdamW's learning rate and weight decay carry over unchanged. Embeddings, heads,
norms and biases stay on AdamW.

The state is AdamW's, so a run can switch either way at any checkpoint. The momentum is AdamW's exp_avg for
that weight, so an AdamW checkpoint starts Muon with its momentum already warm. exp_avg_sq and step keep
being updated too, so a Muon checkpoint resumes under plain AdamW with every moment current. The AdamW
optimiser never sees these weights' gradients: step() clears them after using them, and AdamW skips a weight
with no gradient.

  OPTIM=muon python chess/train_av.py
"""
import math

import torch

# the quintic from the Muon paper: it pushes every singular value of a matrix with spectral norm <= 1
# into about 0.7 to 1.2
NS_COEF = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def orthogonalise(G, steps=5):
    """About U V^T of each matrix's SVD, for a batch (k, m, n), in bf16."""
    X = G.bfloat16()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = NS_COEF
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.mT if tall else X


def is_trunk_matrix(name):
    """The transformer layers' four weight matrices: attention in and out, feed-forward in and out."""
    return name.startswith("enc.layers.") and name.endswith(
        ("self_attn.in_proj_weight", "self_attn.out_proj.weight", "linear1.weight", "linear2.weight"))


class Muon:
    """Steps `named` (name, weight) pairs with orthogonalised momentum, reading the learning rate and weight
    decay from the AdamW group each weight sits in. Call step() after the clip and before opt.step()."""

    def __init__(self, opt, named, momentum=0.95, scale=0.2, steps=5):
        self.opt, self.momentum, self.scale, self.steps = opt, momentum, scale, steps
        self.params = [p for _, p in named]
        self.group = {id(p): g for g in opt.param_groups for p in g["params"]}
        missing = [n for n, p in named if id(p) not in self.group]
        if missing:
            raise ValueError(f"Muon weights must sit in an AdamW group: {missing[:3]}")
        # the attention's in_proj stacks three square matrices (q, k, v), each orthogonalised on its own
        self.blocks = {id(p): (3 if n.endswith("in_proj_weight") else 1) for n, p in named}

    def _state(self, p):
        st = self.opt.state[p]
        if "exp_avg" not in st:                        # a fresh run: AdamW's own layout, so its saves match
            g = self.group[id(p)]
            on_dev = g.get("fused") or g.get("capturable")
            st["step"] = torch.zeros((), dtype=torch.float32, device=p.device if on_dev else "cpu")
            st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return st

    @torch.no_grad()
    def step(self):
        live = [p for p in self.params if p.grad is not None]
        if not live:
            return
        mu = self.momentum
        grads = [p.grad.float() for p in live]
        sts = [self._state(p) for p in live]
        m = [s["exp_avg"] for s in sts]
        v = [s["exp_avg_sq"] for s in sts]
        torch._foreach_lerp_(m, grads, 1.0 - mu)
        # AdamW's second moment, kept current so a switch back to AdamW finds it ready
        b2 = self.group[id(live[0])]["betas"][1]
        torch._foreach_mul_(v, b2)
        torch._foreach_addcmul_(v, grads, grads, value=1.0 - b2)
        for s in sts:
            s["step"] += 1
        # Nesterov: the gradient pulled towards the momentum
        u = torch._foreach_lerp(grads, m, mu)
        # blocks of one shape go through Newton-Schulz together, one batched matmul chain a shape
        by_shape = {}
        for i, (p, x) in enumerate(zip(live, u)):
            k = self.blocks[id(p)]
            blk = x.view(k, x.shape[0] // k, x.shape[1])
            by_shape.setdefault(tuple(blk.shape[1:]), []).append((i, blk))
        out = [None] * len(live)
        for shape, items in by_shape.items():
            o = orthogonalise(torch.cat([b for _, b in items], 0), self.steps)
            o = o * (self.scale * math.sqrt(max(shape)))
            j = 0
            for i, b in items:
                out[i] = o[j:j + b.shape[0]].reshape(live[i].shape)
                j += b.shape[0]
        for p, o in zip(live, out):
            g = self.group[id(p)]
            if g.get("weight_decay"):
                p.mul_(1.0 - g["lr"] * g["weight_decay"])
            p.add_(o.to(p.dtype), alpha=-g["lr"])
            p.grad = None                              # AdamW skips a weight with no gradient
