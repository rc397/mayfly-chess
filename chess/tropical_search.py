"""
tropical_search is a small negamax search over the network's own shortlist of moves.

The bare policy plays by reflex and can hang an attacked piece, which one ply of lookahead would see.
Rather than brute force, the network proposes a short list and the search checks only that list. The
search uses the same algebra as the network.

    value(position) = max over moves of ( -value(child) )

A maximum over moves of a sum along a line is the tropical semiring that every block of the network is
built from.

  from tropical_search import search
  move, score, stats = search(model, board, depth=3, topk=4)
"""
import chess
import torch

INF = 1e9
MATE = 1e6


class Searcher:
    """Negamax with alpha-beta over the network's own shortlist.

    Nothing here is hand-written chess knowledge. The candidate list comes from the policy head,
    the leaf score comes from the value head, and the search only decides which of the network's
    own suggestions survives contact with a reply.
    """

    # Eight, not four. In the test position below the tropical model's four best moves all lose the
    # knight and the saving moves rank fifth to eighth, so a top four shortlist blunders at any depth.
    def __init__(self, model, device=None, topk=8, mat_weight=0.35):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.topk = topk
        # The value head badly under-prices material swings (a lost knight moves it about six points
        # instead of thirty), and search amplifies that. The material head is accurate (R squared
        # 0.962), so the leaf reads both heads.
        self.mat_weight = mat_weight
        self.pol_cache = {}
        self.val_cache = {}
        self.nodes = 0

    def _forward(self, board):
        from chess_net import board_to_ids, move_to_idx
        key = board._transposition_key()
        if key in self.pol_cache:
            return self.pol_cache[key], self.val_cache[key]
        ids, cast, flip = board_to_ids(board)
        with torch.no_grad():
            out = self.model(torch.tensor([ids], device=self.device),
                             torch.tensor([cast], device=self.device))
        legal = list(board.legal_moves)
        if legal:
            idx = torch.tensor([move_to_idx(m, flip) for m in legal], device=self.device)
            p = torch.softmax(out["policy"][0][idx], dim=-1)
            order = torch.argsort(p, descending=True)
            moves = [(legal[int(i)], float(p[int(i)])) for i in order]
        else:
            moves = []
        # both heads are already from the side to move, same as the board canonicalisation
        val = float(torch.sigmoid(out["value"])[0]) * 2.0 - 1.0
        if self.mat_weight:
            m = out["material"][0]                       # [mine, theirs], each divided by 39
            edge = float(m[0] - m[1]) * 39.0 / 10.0      # back to pawns, then into roughly [-1,1]
            val = val + self.mat_weight * max(-1.0, min(1.0, edge))
        self.pol_cache[key], self.val_cache[key] = moves, val
        return moves, val

    def evaluate(self, board):
        self.nodes += 1
        return self._forward(board)[1]

    def candidates(self, board):
        """The shortlist. Captures are added even when the policy does not rank them, since missing a
        free piece is the failure this search is meant to fix."""
        moves = self._forward(board)[0]
        top = [m for m, _ in moves[:self.topk]]
        for m, _ in moves:
            if len(top) >= self.topk + 4:
                break
            if m not in top and board.is_capture(m):
                top.append(m)
        return top

    def negamax(self, board, depth, alpha, beta):
        if board.is_checkmate():
            return -MATE - depth          # prefer mating sooner, being mated later
        if board.is_stalemate() or board.is_insufficient_material() or \
                board.can_claim_fifty_moves() or board.is_repetition(3):
            return 0.0
        if depth == 0:
            return self.evaluate(board)

        best = -INF
        for mv in self.candidates(board):
            board.push(mv)
            score = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break                     # the opponent would never allow this line
        return best if best > -INF else self.evaluate(board)


def search(model, board, depth=2, topk=8, device=None):
    """Pick a move. Returns (move, score from the side to move, stats)."""
    s = Searcher(model, device, topk)
    best, best_score = None, -INF
    alpha = -INF
    scored = []
    for mv in s.candidates(board):
        board.push(mv)
        sc = -s.negamax(board, depth - 1, -INF, -alpha)
        board.pop()
        scored.append((mv, sc))
        if sc > best_score:
            best, best_score = mv, sc
        if sc > alpha:
            alpha = sc
    scored.sort(key=lambda x: -x[1])
    return best, best_score, {"nodes": s.nodes, "depth": depth, "topk": topk,
                              "lines": scored[:5]}


if __name__ == "__main__":
    import sys, os, time
    HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
    from play_race import load

    model, name, _ = load("trop")
    print(f"{name} with tropical search\n")

    # A known blunder position, built from the moves rather than a hand-typed FEN
    b = chess.Board()
    for s in "e4 c5 Nf3 Nc6 d4 d6 d5".split():
        b.push(b.parse_san(s))
    print("after 4.d5 the knight on c6 is attacked. In the game it played Nf6 and lost it.\n")

    def loses_knight(bd, mv):
        t = bd.copy(); t.push(mv)
        grab = chess.Move.from_uci("d5c6")
        if grab not in t.legal_moves:
            return False
        t.push(grab)
        for r in t.legal_moves:
            if r.to_square == chess.C6:
                t.push(r); break
        V = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        f = lambda c: sum(V[p] * len(t.pieces(p, c)) for p in V)
        return f(chess.BLACK) - f(chess.WHITE) < 0

    for topk in (4, 8):
        for depth in (1, 2, 3):
            t0 = time.time()
            mv, sc, st = search(model, b, depth=depth, topk=topk)
            verdict = "LOSES THE KNIGHT" if loses_knight(b, mv) else "saves it"
            print(f"  shortlist {topk:2d}  depth {depth}  {b.san(mv):6s} {verdict:18s} "
                  f"{st['nodes']:5d} nodes  {time.time() - t0:5.2f}s")
        print()
