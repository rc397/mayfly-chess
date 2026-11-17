"""One engine move for a game played one call at a time, using search.SearchPlayer.

SearchPlayer is policy-filtered negamax alpha-beta with the engine's value head, plus an optional
style net. This replays the move history (UCI or SAN tokens), makes the engine's move, and prints
machine-parseable lines (MOVE, FEN, STATUS, RESULT) for a wrapper to read. A GPU makes it take about
two seconds a move, a CPU minutes.

  python chess/play_step.py "<space-separated moves so far>" [DEPTH=3] [TOPK=5]
"""
import sys, os, torch, chess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from search import SearchPlayer

tokens = sys.argv[1].split() if len(sys.argv) > 1 and sys.argv[1].strip() else []
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
topk = int(sys.argv[3]) if len(sys.argv) > 3 else 5
device = "cuda" if torch.cuda.is_available() else "cpu"
sp = SearchPlayer(device=device, depth=depth, topk=topk)

board = chess.Board()
for tok in tokens:
    ok = False
    for push in (board.push_uci, board.push_san):
        try:
            push(tok); ok = True; break
        except Exception:
            continue
    if not ok:
        print("ILLEGAL " + tok)
        print("LEGAL " + " ".join(mv.uci() for mv in list(board.legal_moves)[:40]))
        sys.exit()

if board.is_game_over():
    print("FEN " + board.fen()); print("RESULT " + board.result()); sys.exit()

mv = sp.best_move(board)
san = board.san(mv); board.push(mv)
print("MOVE " + mv.uci() + " " + san)
print("FEN " + board.fen())
print("STATUS " + ("MATE" if board.is_checkmate() else "STALEMATE" if board.is_stalemate()
                   else "CHECK" if board.is_check() else "ok"))
print("RESULT " + (board.result() if board.is_game_over() else "*"))
