"""
search.py is an early play-time search, policy-filtered negamax alpha-beta on the engine net.

  leaf eval    the engine net's value head (win probability for the side to move)
  move order   its policy head, top-K priors per node so the branching stays small
  style        at the root, among moves within a small value margin of the best, the human style net
               picks the move it likes most
  endgame      Syzygy tablebases give exact play with five men or fewer, and the candidate filter
               widens once few pieces remain

  python chess/search.py [DEPTH=3] [TOPK=5]      # self-play sanity check, prints moves, nodes and timing
"""
import os, sys, time, torch, torch.nn.functional as F, chess, chess.syzygy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet, board_to_ids, move_to_idx
import paths

def _load(path, device):
    ck = torch.load(path, map_location=device); cfg = ck["cfg"]
    m = ChessNet(d=cfg["d"], nhead=max(4, cfg["d"] // 64), layers=cfg["layers"], dim_ff=4 * cfg["d"])
    m.load_state_dict(ck["model"]); m.to(device).eval(); return m

class SearchPlayer:
    def __init__(self, engine_ckpt=None, human_ckpt=None, device="cuda", depth=3, topk=5,
                 style_margin=0.06, style_alpha=1.0, tb_path=None):
        self.device = device; self.depth = depth; self.topk = topk
        self.style_margin = style_margin; self.style_alpha = style_alpha
        self.engine = _load(engine_ckpt or os.path.join(HERE, "ckpt_chess_net.pt"), device)
        hp = human_ckpt or os.path.join(HERE, "ckpt_human_net.pt")
        self.human = _load(hp, device) if os.path.exists(hp) else None
        self.tb = None                                       # Syzygy endgame tablebases (exact <=5-man play)
        tb_path = tb_path or paths.under("syzygy")
        try:
            if os.path.isdir(tb_path) and any(f.endswith(".rtbw") for f in os.listdir(tb_path)):
                self.tb = chess.syzygy.open_tablebase(tb_path)
        except Exception:
            self.tb = None
        self.nodes = 0

    @torch.no_grad()
    def _eval(self, board):                                  # one net forward -> (win-prob, policy logits, flip)
        ids, cast, flip = board_to_ids(board)
        out = self.engine(torch.tensor([ids], device=self.device), torch.tensor([cast], device=self.device))
        self.nodes += 1
        return torch.sigmoid(out["value"][0]).item(), out["policy"][0], flip

    def _ordered(self, board, logits, flip, k):              # top-k legal moves by policy prior
        legal = list(board.legal_moves)
        scored = sorted(((m, float(logits[move_to_idx(m, flip)])) for m in legal), key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:k]]

    def _men(self, board):
        return bin(board.occupied).count("1")

    def _eff_topk(self, board):                              # widen the candidate filter in the endgame
        return max(self.topk, 8) if self._men(board) <= 12 else self.topk

    def _tb_score(self, board):                              # exact <=5-man value, side-to-move POV, else None
        if self.tb is None or self._men(board) > 5:
            return None
        try:
            wdl = self.tb.probe_wdl(board)
        except Exception:
            return None
        return 0.99 if wdl > 0 else (-0.99 if wdl < 0 else 0.0)

    def _tb_root_move(self, board):                          # optimal tablebase move (WDL, DTZ tiebreak), else None
        if self.tb is None or self._men(board) > 5:
            return None
        best, best_key = None, None
        for m in board.legal_moves:
            board.push(m)
            try:
                wdl = -self.tb.probe_wdl(board); dtz = self.tb.probe_dtz(board)
            except Exception:
                board.pop(); return None
            board.pop()
            key = (2, -abs(dtz)) if wdl > 0 else ((1, 0) if wdl == 0 else (0, abs(dtz)))
            if best_key is None or key > best_key:
                best_key, best = key, m
        return best

    @torch.no_grad()
    def _negamax(self, board, depth, alpha, beta):
        if board.is_checkmate():
            return -1.0                                      # side to move is mated (worst)
        if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3) or board.can_claim_fifty_moves():
            return 0.0
        tb = self._tb_score(board)
        if tb is not None:
            return tb                                        # exact endgame result, stop searching here
        val, logits, flip = self._eval(board)
        if depth == 0:
            return 2.0 * val - 1.0                            # win-prob -> [-1,1], side-to-move POV
        best = -2.0
        for m in self._ordered(board, logits, flip, self._eff_topk(board)):
            board.push(m)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break                                        # alpha-beta cutoff
        return best

    @torch.no_grad()
    def best_move(self, board, return_scores=False):
        legal = list(board.legal_moves)
        if not legal:
            return None
        self.nodes = 0
        tb_mv = self._tb_root_move(board)                    # perfect play once <=5 men on the board
        if tb_mv is not None:
            return (tb_mv, [(tb_mv, 0.99)]) if return_scores else tb_mv
        _, logits, flip = self._eval(board)
        roots = self._ordered(board, logits, flip, max(self._eff_topk(board), 6))   # candidates at the root
        scored = []
        for m in roots:
            board.push(m)
            s = -self._negamax(board, self.depth - 1, -2.0, 2.0)          # full window at root = accurate scores
            board.pop()
            scored.append((m, s))
        scored.sort(key=lambda ms: ms[1], reverse=True)
        if self.human is None or self.style_alpha == 0:
            return (scored[0][0], scored) if return_scores else scored[0][0]
        best_s = scored[0][1]
        cands = [m for m, s in scored if s >= best_s - self.style_margin]  # the strong-enough moves
        hids, hcast, hflip = board_to_ids(board)
        hlog = F.log_softmax(self.human(torch.tensor([hids], device=self.device),
                                        torch.tensor([hcast], device=self.device))["policy"][0], dim=-1)
        pick = max(cands, key=lambda m: float(hlog[move_to_idx(m, hflip)]))  # the style net's favourite
        return (pick, scored) if return_scores else pick

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    sp = SearchPlayer(device=dev, depth=depth, topk=topk)
    print(f"SearchPlayer depth {depth} topk {topk} | human={'on' if sp.human else 'off'} | "
          f"tb={'on' if sp.tb else 'off'} | {dev}", flush=True)
    b = chess.Board(); sans = []
    t0 = time.time()
    for ply in range(10):
        mv, scored = sp.best_move(b, return_scores=True)
        if mv is None:
            break
        if ply == 0:
            print("root move scores:", [(b.san(m), round(s, 3)) for m, s in scored[:5]], f"| {sp.nodes} nodes", flush=True)
        sans.append(b.san(mv)); b.push(mv)
    print("search self-play:", " ".join(sans), flush=True)
    print(f"{(time.time()-t0)/10:.2f}s/move avg", flush=True)
