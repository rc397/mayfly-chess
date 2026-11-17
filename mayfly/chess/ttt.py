r"""ttt is test-time training that learns the opponent during a game.

The network is frozen once a game starts, so it cannot notice an opponent repeating the same idea. After
each opponent move there is a free training example, the position and the move they chose. A low rank
adapter on the policy head (about twenty thousand parameters, starting at zero so it begins as the frozen
model) is trained on the cross entropy of that move. The target is their choice, it needs no labels, and
the model cannot pick an easy objective for itself.

The search uses the adapter only at nodes where the opponent is to move, so the tree predicts what this
opponent plays. The model's own moves are unaffected. Each ply records the probability given to their
actual move before adapting on it. If that number rises through a game, the adapter is learning them.

  from ttt import OpponentModel
  om = OpponentModel(model, device)
  om.observe(board_before_their_move, their_move)
  om.report()
"""
import os, sys
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import board_to_ids, clock_and_ep
from moves import move_to_idx, N_MOVES
# king-takes-rook castling in the side to move's frame, where v1 to v4's training put castling
OO_ROOK = move_to_idx(chess.Move.from_uci("e1h1"), False)
OOO_ROOK = move_to_idx(chess.Move.from_uci("e1a1"), False)

E = os.environ.get


class OpponentModel:
    """A small adapter over the policy head, trained during a game on the opponent's own moves."""

    # Tuned by tune_ttt.py on 410 real moves from saved games, scoring each setting by the probability
    # it gave the next move before learning from it. History length mattered most. All twelve best
    # settings used 400, because at 60 the adapter forgets the player and never accumulates anything.
    #
    #   lr .02  steps 8 rank 4  hist 60    18.40% -> 18.33%   -0.06
    #   lr .0005 steps 2 rank 16 hist 400  18.39% -> 29.15%  +10.76    (random guessing is 2.4%)
    def __init__(self, model, device, rank=16, lr=0.0005, steps=2, max_history=400,
                 blend=1.0, head="policy", castle="king"):
        self.model, self.device = model, str(device)
        # The head it corrects. The tree reads `policy` or `policy_att` depending on what the version
        # was sealed with, and a delta fitted to one head's logits means nothing added to the other's.
        self.head = head
        # Where this version's tree reads castling (its sealed castle_prior). v4 reads a castle's prior
        # from the king-takes-rook index, so the adapter must correct that index and not e1g1.
        self.castle_rook = (castle == "rook")
        self.rank, self.lr, self.steps = rank, lr, steps
        self.max_history, self.blend = max_history, blend
        d = model.policy.in_features
        # B projects the trunk down and A projects back out to the move space. A starts at zero, so
        # the adapter contributes nothing until it has seen something.
        self.B = torch.zeros(rank, d, device=self.device, requires_grad=True)
        self.A = torch.zeros(N_MOVES, rank, device=self.device, requires_grad=True)
        with torch.no_grad():
            self.B.normal_(0, 0.02)
        self.opt = torch.optim.Adam([self.A, self.B], lr=lr)
        self.history = []                 # (h, base logits, legal indices, their move's slot)
        self.skipped = []               # one entry per observe(), True if it learned nothing
        self.before = []                  # probability given to their move BEFORE adapting on it
        self.after = []
        self._stacked = None              # the same history as flat tensors, rebuilt each ply

    # ---------------------------------------------------------------- the adapter
    def delta(self, h):
        """The adapter's contribution to the policy logits, given the trunk representation."""
        return (h @ self.B.t()) @ self.A.t() * self.blend

    def _idx(self, board, m, flip):
        """A legal move's policy index as this version's tree reads it."""
        if self.castle_rook and board.is_castling(m):
            return OO_ROOK if board.is_kingside_castling(m) else OOO_ROOK
        return move_to_idx(m, flip)

    def _encode(self, board):
        ids, cast, flip = board_to_ids(board)
        hm, ep = clock_and_ep(board, flip)
        t = lambda a, d: torch.tensor([a], dtype=d, device=self.device)
        # the castling input in the network's own dtype, as mcts.policy_move does, because a FAST_EVAL
        # tree casts a shared network to bfloat16 and float32 input would then fail
        wd = getattr(next(self.model.parameters(), None), "dtype", torch.float32)
        return t(ids, torch.long), t(cast, wd), t(hm, torch.long), t(ep, torch.long), flip

    @torch.no_grad()
    def _key(self, out):
        """The head this adapter corrects, falling back when a checkpoint does not carry it."""
        return self.head if self.head in out else "policy"

    def predict(self, board):
        """The adapted distribution over the opponent's legal moves in this position."""
        ids, cast, hm, ep, flip = self._encode(board)
        out = self.model(ids, cast, hm, ep)
        legal = list(board.legal_moves)
        if not legal:
            return [], np.zeros(0)
        idx = torch.tensor([self._idx(board, m, flip) for m in legal], device=self.device)
        logits = (out[self._key(out)][0] + self.delta(out["h"])[0])[idx]
        return legal, torch.softmax(logits.float(), -1).cpu().numpy()

    # ---------------------------------------------------------------- learning from them
    def _stack(self):
        """The history as flat tensors, so a training step is a handful of kernels rather than one
        per remembered move (a per-move loop cost seconds a ply in launch overhead). Move indices are
        padded to the longest legal list and the padding is masked to minus infinity.
        """
        H = torch.cat([h.float() for h, _, _, _ in self.history], 0)
        B = torch.stack([b.float() for _, b, _, _ in self.history], 0)
        wide = max(len(i) for _, _, i, _ in self.history)
        I = torch.zeros(len(self.history), wide, dtype=torch.long, device=self.device)
        M = torch.zeros(len(self.history), wide, dtype=torch.bool, device=self.device)
        for k, (_, _, i, _) in enumerate(self.history):
            I[k, :len(i)] = i
            M[k, :len(i)] = True
        T = torch.tensor([t for _, _, _, t in self.history], device=self.device)
        self._stacked = (H, B, I, M, T)
        return self._stacked

    def _loss(self):
        H, B, I, M, T = self._stacked
        logits = torch.gather(B + self.delta(H), 1, I).masked_fill(~M, float("-inf"))
        return F.cross_entropy(logits.float(), T)

    def observe(self, board, move):
        """They played `move` from `board`. Record it, score the prediction, then adapt."""
        legal = list(board.legal_moves)
        if move not in legal or len(legal) < 2:
            self.skipped.append(True)        # nothing learned, but undo must know that
            return
        self.skipped.append(False)
        ids, cast, hm, ep, flip = self._encode(board)
        idx = torch.tensor([self._idx(board, m, flip) for m in legal], device=self.device)
        target = legal.index(move)
        with torch.no_grad():
            out = self.model(ids, cast, hm, ep)
            h = out["h"].detach()
            base = out[self._key(out)][0].detach()
        self.observe_cached(h, base, idx, target)

    def observe_cached(self, h, base, idx, target):
        """The learning half, given the network outputs already computed.

        Split out so a sweep can compute the forward pass once per position and try many settings on
        the cached result, since the trunk representation does not depend on the adapter settings.
        """
        # score BEFORE adapting on this move, or the number is circular
        with torch.no_grad():
            p = torch.softmax((base + self.delta(h)[0])[idx].float(), -1)
            self.before.append(float(p[target]))

        self.history.append((h, base, idx, target))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self._stack()

        for _ in range(self.steps):
            self.opt.zero_grad()
            self._loss().backward()
            self.opt.step()

        with torch.no_grad():
            p = torch.softmax((base + self.delta(h)[0])[idx].float(), -1)
            self.after.append(float(p[target]))

    def forget_last(self):
        """They took the move back, so it was not their choice and it is not evidence about them.

        The adapter cannot be un-stepped, but the move stops being fitted from here on and is not
        counted in the report.
        """
        # A skipped move (forced, or not legal) added nothing, so taking it back must not erase an
        # earlier real observation.
        if getattr(self, "skipped", None):
            if self.skipped.pop():
                return
        if not self.history:
            return
        self.history.pop()
        if self.before:
            self.before.pop()
        if self.after:
            self.after.pop()
        self._stacked = self._stack() if self.history else None

    # ---------------------------------------------------------------- is it working
    def report(self):
        """Prediction before and after adapting, early against late in the game."""
        if not self.before:
            return {"plies": 0, "adapting": False}
        b, a = np.array(self.before), np.array(self.after)
        half = max(len(b) // 2, 1)
        return {
            "plies": len(b),
            # how well it predicts them before seeing the move, early against late. This rises if
            # the adapter is learning the player rather than the position.
            "predicted_early": float(b[:half].mean()),
            "predicted_late": float(b[half:].mean()),
            "gain": float(b[half:].mean() - b[:half].mean()),
            "fit_to_the_last_move": float(a.mean() - b.mean()),
            "adapting": bool(float(self.A.detach().abs().mean()) > 0),
            "adapter_size": float(self.A.detach().abs().mean()),
        }

    def summary(self):
        r = self.report()
        if not r["plies"]:
            return "learning you: nothing seen yet"
        # Early against late needs enough moves for both halves. Below eight the gain is noise.
        if r["plies"] < 8:
            return f"learning you: {r['plies']} of your moves seen, too early to say"
        return (f"learning you: {r['plies']} of your moves seen, predicting your next one "
                f"{100*r['predicted_early']:.1f}% at the start against "
                f"{100*r['predicted_late']:.1f}% now ({100*r['gain']:+.1f})")

    def reset(self):
        with torch.no_grad():
            self.A.zero_()
            self.B.normal_(0, 0.02)
        self.opt = torch.optim.Adam([self.A, self.B], lr=self.lr)
        self.history, self.before, self.after = [], [], []
        self._stacked = None


if __name__ == "__main__":
    # The test is whether prediction improves more for a consistent opponent than for one who changes
    # style partway. Improving equally on both would mean it learns chess, not the player.
    from mcts import load
    import random, time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load(E("TAG", "mayfly"), dev)

    def personality(board, rng, bias):
        """Crude but consistent, strongly prefers one kind of move."""
        legal = list(board.legal_moves)

        def score(m):
            s = rng.random()
            if bias == "pawns" and board.piece_type_at(m.from_square) == chess.PAWN:
                s += 2.0
            if bias == "knights" and board.piece_type_at(m.from_square) == chess.KNIGHT:
                s += 2.0
            if bias == "captures" and board.is_capture(m):
                s += 2.0
            return s
        return max(legal, key=score)

    def run(label, biases, plies=120, seed=0):
        """biases is a list. The opponent uses biases[i] for the i-th third of the game."""
        rng = random.Random(seed)
        om = OpponentModel(model, dev)
        b = chess.Board()
        n = 0
        t0 = time.time()
        while n < plies:
            if b.is_game_over() or not list(b.legal_moves):
                b = chess.Board()          # a finished game is not evidence, start another
                continue
            if b.turn == chess.BLACK:
                bias = biases[min(int(3 * n / plies), len(biases) - 1)]
                mv = personality(b, rng, bias)
                om.observe(b, mv)
                n += 1
            else:
                mv = rng.choice(list(b.legal_moves))
            b.push(mv)
        r = om.report()
        print(f"  {label:<34} early {100*r['predicted_early']:5.2f}%  "
              f"late {100*r['predicted_late']:5.2f}%  gain {100*r['gain']:+6.2f}  "
              f"({r['plies']} plies, {time.time()-t0:.0f}s)")
        return r["gain"]

    print(f"a random legal move is worth about {100/35:.1f}%")
    print("")
    g1 = run("one personality all game", ["pawns"])
    g2 = run("one personality, different seed", ["pawns"], seed=7)
    g3 = run("CONTROL: personality changes", ["pawns", "knights", "captures"])
    print()
    print(f"consistent opponent gains {100*(g1+g2)/2:+.2f} points, "
          f"the changing control gains {100*g3:+.2f}")
    print("the first should beat the second, or it is learning positions rather than people")
