"""
Combine a strength net (ckpt_chess_net.pt, Stockfish-distilled) and a style net (ckpt_human_net.pt,
a behavioural clone of defensive masters) at play time.

The engine proposes its strongest candidate moves and the style net picks among them, so the move
stays near-best while following the style.

  score(m) = engine_logp[m] + ALPHA * human_logp[m], argmax over the engine's TOPK legal moves
  ALPHA 0 is pure engine strength, a larger ALPHA gives more style.
  TOPK limits the choice to the engine's K strongest legal moves, the guard against blunders.

Falls back to the pure engine if the style checkpoint is missing.

  python chess/play_blend.py [ALPHA=1.0]      # prints a sample opening from the blend
"""
import os, sys, torch, torch.nn.functional as F, chess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet, board_to_ids, move_to_idx

def _load(path, device):
    ck = torch.load(path, map_location=device)
    cfg = ck["cfg"]
    m = ChessNet(d=cfg["d"], nhead=max(4, cfg["d"] // 64), layers=cfg["layers"], dim_ff=4 * cfg["d"])
    m.load_state_dict(ck["model"]); m.to(device).eval()
    return m

class BlendPlayer:
    def __init__(self, engine_ckpt=None, human_ckpt=None, device="cpu", alpha=1.0, topk=5):
        self.device = device; self.alpha = alpha; self.topk = topk
        self.engine = _load(engine_ckpt or os.path.join(HERE, "ckpt_chess_net.pt"), device)
        hp = human_ckpt or os.path.join(HERE, "ckpt_human_net.pt")
        self.human = _load(hp, device) if os.path.exists(hp) else None

    @torch.no_grad()
    def _logp(self, model, ids, cast):
        return F.log_softmax(model(ids, cast)["policy"][0], dim=-1)

    @torch.no_grad()
    def best_move(self, board, alpha=None, topk=None):
        alpha = self.alpha if alpha is None else alpha
        topk = self.topk if topk is None else topk
        legal = list(board.legal_moves)
        if not legal:
            return None
        ids_l, cast_l, flip = board_to_ids(board)
        ids = torch.tensor([ids_l], device=self.device); cast = torch.tensor([cast_l], device=self.device)
        idxs = [move_to_idx(m, flip) for m in legal]
        elp = self._logp(self.engine, ids, cast)
        e_scores = [float(elp[i]) for i in idxs]
        if self.human is None or alpha == 0:
            return legal[int(max(range(len(legal)), key=lambda j: e_scores[j]))]
        k = min(topk, len(legal))
        top = sorted(range(len(legal)), key=lambda j: e_scores[j], reverse=True)[:k]   # the engine's strongest
        hlp = self._logp(self.human, ids, cast)
        return legal[max(top, key=lambda j: e_scores[j] + alpha * float(hlp[idxs[j]]))]  # best blended score

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    bp = BlendPlayer(device=dev, alpha=alpha)
    print(f"engine + human={'loaded' if bp.human else 'MISSING (pure engine)'} | alpha {alpha} | device {dev}", flush=True)
    b = chess.Board(); moves = []
    for _ in range(16):
        m = bp.best_move(b)
        if m is None:
            break
        moves.append(b.san(m)); b.push(m)
    print("blend opening:", " ".join(moves), flush=True)
