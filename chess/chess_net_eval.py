"""Evaluate the multi-head ChessNet in ckpt_chess_net.pt with Stockfish as the judge.

The net picks moves with its policy head, masked to legal moves, and opens from the repertoire book
(repertoire.book_move). Reports the legal-move rate, top-1 agreement with Stockfish, average
centipawn loss against Stockfish's best move, and results against Stockfish at Skill 0.

    python chess/chess_net_eval.py [N_POS=300] [N_GAMES=10]
"""
import os, sys, random
import torch, chess, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet
from repertoire import book_move
random.seed(7)

N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 10
device = "cuda" if torch.cuda.is_available() else "cpu"
SF = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()

ck = torch.load(os.path.join(HERE, "ckpt_chess_net.pt"), map_location=device)
cfg = ck["cfg"]
model = ChessNet(d=cfg["d"], nhead=max(4, cfg["d"] // 64), layers=cfg["layers"], dim_ff=4 * cfg["d"]).to(device).eval()
model.load_state_dict(ck["model"])
print(f"ChessNet d{cfg['d']} L{cfg['layers']} ({model.num_params()/1e6:.1f}M) on {device}", flush=True)

eng = chess.engine.SimpleEngine.popen_uci(SF)

def sf_eval(board):
    info = eng.analyse(board, chess.engine.Limit(depth=10))
    return info["pv"][0], info["score"].pov(board.turn).score(mate_score=100000)

# legal rate, top-1 and centipawn loss on fresh positions
legal = match = cploss = n_cp = 0
b = chess.Board(); seen = 0
while seen < N_POS:
    if b.is_game_over():
        b = chess.Board()
    best, sc_best = sf_eval(b)
    mv = model.best_move(b, device=device)
    is_legal = bool(mv) and mv in b.legal_moves
    legal += int(is_legal); match += int(is_legal and mv == best)
    if is_legal:
        b2 = b.copy(); b2.push(mv)
        _, sc_opp = sf_eval(b2)            # eval from the opponent's pov after our move
        cploss += max(0, sc_best - (-sc_opp)); n_cp += 1
    seen += 1
    b.push(best if random.random() < 0.6 else random.choice(list(b.legal_moves)))
print(f"legal {legal/N_POS:.1%} | top-1 vs SF {match/N_POS:.1%} | avg centipawn loss "
      f"{cploss/max(1,n_cp):.0f}cp  (n={N_POS})", flush=True)

# games against Stockfish Skill 0, opening from the repertoire book
eng.configure({"Skill Level": 0})
w = d = l = 0; op = []
for g in range(N_GAMES):
    board = chess.Board(); model_white = (g % 2 == 0)
    while not board.is_game_over():
        if (board.turn == chess.WHITE) == model_white:
            mv = book_move(board) or model.best_move(board, device=device)
            if (mv is None) or (mv not in board.legal_moves):
                mv = random.choice(list(board.legal_moves))
            if g < 2: op.append(board.san(mv))
            board.push(mv)
        else:
            board.push(eng.play(board, chess.engine.Limit(time=0.05)).move)
    r = board.result()
    if r == "1/2-1/2": d += 1
    elif (r == "1-0") == model_white: w += 1
    else: l += 1
eng.quit()
print(f"vs Stockfish Skill-0 ({N_GAMES} games): {w}W {d}D {l}L", flush=True)
if op:
    print(f"opening (model White): {' '.join(op[:12])}", flush=True)
print("(legal>95% = learned chess; lower centipawn loss = stronger; this is the new MULTI-HEAD net)", flush=True)
