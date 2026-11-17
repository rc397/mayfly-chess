"""Compare the base network with the repertoire fine-tuned one on held-out repertoire positions.

Loads ckpt_chess_net.pt and ckpt_chess_net_rep.pt and scores both on repertoire_fens.txt positions
that were not in the annotated training slice (index % 7 != 0, since annotation took fens[::7]). For
each position it records top-1 agreement with Stockfish at depth 12 and a clamped centipawn loss,
skipping decided positions.

    python chess/eval_repertoire.py [N_POS=400]
    run after train_chess_finetune.py has written ckpt_chess_net_rep.pt
"""
import os, sys, random
import torch, chess, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet
random.seed(11)
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
device = "cuda" if torch.cuda.is_available() else "cpu"
SF = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
FENS = os.path.join(HERE, "repertoire_fens.txt")

def load(path):
    ck = torch.load(path, map_location=device); cfg = ck["cfg"]
    m = ChessNet(d=cfg["d"], nhead=max(4, cfg["d"] // 64), layers=cfg["layers"], dim_ff=4 * cfg["d"]).to(device).eval()
    m.load_state_dict(ck["model"]); return m

base = load(os.path.join(HERE, "ckpt_chess_net.pt"))
rep_path = os.path.join(HERE, "ckpt_chess_net_rep.pt")
rep = load(rep_path) if os.path.exists(rep_path) else None
print(f"base vs rep={'loaded' if rep else 'MISSING'} on {device}", flush=True)

all_fens = [l.strip() for l in open(FENS, encoding="utf-8") if l.strip()]
held = [f for i, f in enumerate(all_fens) if i % 7 != 0]      # held out, not in the annotated fens[::7] slice
random.shuffle(held); held = held[:N_POS * 3]                  # oversample, since many get skipped as decided

eng = chess.engine.SimpleEngine.popen_uci(SF)
def sf(board):
    if board.is_game_over():
        return None, (-100000 if board.is_checkmate() else 0)
    info = eng.analyse(board, chess.engine.Limit(depth=12))
    pv = info.get("pv"); sc = info["score"].pov(board.turn).score(mate_score=100000)
    return (pv[0] if pv else None), (sc if sc is not None else 0)

def cpl_of(board, mv, sc_best):
    b2 = board.copy(); b2.push(mv); _, sc_opp = sf(b2)
    return min(1000, max(0, sc_best - (-sc_opp)))     # cap per-move loss (ACPL convention) so mates do not dominate

def measure(net, board, best, sc_best):
    mv = net.best_move(board, device=device)
    if mv is None or mv not in board.legal_moves:
        return 0, 0
    return int(mv == best), cpl_of(board, mv, sc_best)

bt = rt = 0; bc = rc = 0; nc = 0
for fen in held:
    if nc >= N_POS:
        break
    try:
        board = chess.Board(fen)
    except Exception:
        continue
    best, sc_best = sf(board)
    if best is None or abs(sc_best) > 800:                     # skip decided positions for a clean CPL

        continue
    m, c = measure(base, board, best, sc_best); bt += m; bc += c
    if rep:
        m2, c2 = measure(rep, board, best, sc_best); rt += m2; rc += c2
    nc += 1
eng.quit()
n = max(1, nc)
print(f"held-out repertoire positions scored: {nc}", flush=True)
print(f"BASE engine: top-1 {bt/n:.1%} | avg CPL {bc/n:.0f}cp", flush=True)
if rep:
    print(f"REP  engine: top-1 {rt/n:.1%} | avg CPL {rc/n:.0f}cp", flush=True)
    print(f"DELTA: top-1 {(rt-bt)/n:+.1%} | CPL {(rc-bc)/n:+.0f}cp  (higher top-1 / lower CPL = stronger in repertoire)", flush=True)
