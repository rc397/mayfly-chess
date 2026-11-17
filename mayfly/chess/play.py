"""
Play the early alpha-beta engine (search.SearchPlayer) interactively in a terminal.

  python play.py                 # you are White
  python play.py black           # you are Black
  python play.py white 4 6       # you White, search depth 4, top-K 6 (stronger, slower)

SearchPlayer is value-head alpha-beta with a policy filter and an optional style net, with Syzygy
tablebases for exact endgames. While you think, a background thread computes the reply to your most
likely moves, so a hit comes back instantly.

Enter moves in SAN (Nf3, exd5, O-O) or UCI (g1f3).
Commands are undo, fen, board, new, resign and quit.
"""
import os, sys, time, threading, torch, chess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from search import SearchPlayer
try:
    from repertoire import book_move
except Exception:
    def book_move(b):
        return None

# ---- args -------------------------------------------------------------------
args = [a.lower() for a in sys.argv[1:]]
human_white = not any(a in ("black", "b") for a in args)
nums = [a for a in args if a.isdigit()]
depth = int(nums[0]) if len(nums) > 0 else 3
topk = int(nums[1]) if len(nums) > 1 else 5

# ---- load -------------------------------------------------------------------
eng_ckpt = os.path.join(HERE, "ckpt_chess_net.pt")
if not os.path.exists(eng_ckpt):
    print("engine checkpoint not found:\n  " + eng_ckpt +
          "\nput ckpt_chess_net.pt in " + HERE + " first")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading 256M engine on {device} (first move warms up the GPU, ~10s) ...", flush=True)
sp = SearchPlayer(device=device, depth=depth, topk=topk)
print(f"you are {'White' if human_white else 'Black'} | search depth {depth} topk {topk} | "
      f"style {'on' if sp.human else 'off'} | tablebases {'on' if sp.tb else 'off'}")
print("moves: SAN (Nf3, O-O, exd5) or UCI (g1f3).  commands: undo, fen, board, new, resign, quit\n")

# ---- pondering on the human's clock -----------------------------------------
# While input() blocks the main thread (releasing the GIL), a daemon thread computes the engine's
# reply to the human's 3 most likely moves, so a ponder hit is instant. The thread is joined before
# any main-thread search, so the two never use the GPU at once.
_pond = {"map": {}, "stop": threading.Event(), "thr": None}

def start_ponder(board):
    if board.is_game_over():
        return
    _pond["map"] = {}; _pond["stop"].clear()
    snap = board.copy()
    def work():
        try:
            _, logits, flip = sp._eval(snap)
            cands = sp._ordered(snap, logits, flip, 3)
        except Exception:
            cands = list(snap.legal_moves)[:3]
        for hm in cands:
            if _pond["stop"].is_set():
                return
            snap.push(hm)
            try:
                reply = sp.best_move(snap)
            except Exception:
                reply = None
            snap.pop()
            if reply is not None:
                _pond["map"][hm.uci()] = reply
    _pond["thr"] = threading.Thread(target=work, daemon=True); _pond["thr"].start()

def stop_ponder():
    _pond["stop"].set()
    if _pond["thr"] is not None:
        _pond["thr"].join()                  # wait out any in-flight search so the GPU is ours
        _pond["thr"] = None

# ---- rendering --------------------------------------------------------------
def render(board):
    rows = str(board).split("\n")            # rank 8 .. rank 1, White's POV
    ranks = list(range(8, 0, -1)); files = "a b c d e f g h"
    if not human_white:                      # flip to Black's POV
        rows = [" ".join(r.split()[::-1]) for r in rows[::-1]]
        ranks = list(range(1, 9)); files = "h g f e d c b a"
    print()
    for rk, r in zip(ranks, rows):
        print(f"  {rk}  {r}")
    print(f"     {files}\n")

# ---- engine move ------------------------------------------------------------
def engine_move(board, human_uci=None):
    bm = book_move(board)
    if bm is not None and bm in board.legal_moves:
        san = board.san(bm); board.push(bm)
        print(f"  engine (book): {san}")
        return
    hit = _pond["map"].get(human_uci) if human_uci else None
    if hit is not None and hit in board.legal_moves:
        san = board.san(hit); board.push(hit)
        print(f"  engine: {san}   [pondered, instant]")
        return
    t = time.time(); mv = sp.best_move(board); dt = time.time() - t
    san = board.san(mv); board.push(mv)
    print(f"  engine: {san}   [{sp.nodes} nodes, {dt:.1f}s]")

# ---- game loop --------------------------------------------------------------
board = chess.Board()
render(board)
if not human_white:                          # engine (White) moves first
    engine_move(board); render(board)
start_ponder(board)

while True:
    if board.is_game_over():
        stop_ponder()
        end = ("checkmate -- " + ("you win!" if (board.turn == chess.BLACK) == human_white else "engine wins")
               if board.is_checkmate() else "draw (" + board.result() + ")")
        print(f"game over: {board.result()}  {end}")
        break
    try:
        cmd = input("your move > ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); stop_ponder(); break
    stop_ponder()                            # you acted, so stop pondering and keep what was computed
    if not cmd:
        start_ponder(board); continue
    low = cmd.lower()
    if low in ("quit", "exit", "q"):
        break
    if low == "resign":
        print("you resigned. gg."); break
    if low == "fen":
        print(" ", board.fen()); start_ponder(board); continue
    if low == "board":
        render(board); start_ponder(board); continue
    if low == "new":
        board = chess.Board(); render(board)
        if not human_white:
            engine_move(board); render(board)
        start_ponder(board); continue
    if low == "undo":
        if len(board.move_stack) >= 2:
            board.pop(); board.pop(); render(board)
        else:
            print("  nothing to undo")
        start_ponder(board); continue
    mv = None
    for parse in (board.parse_san, board.parse_uci):
        try:
            mv = parse(cmd); break
        except Exception:
            continue
    if mv is None or mv not in board.legal_moves:
        print("  illegal or unreadable -- use SAN (Nf3) or UCI (g1f3)")
        start_ponder(board); continue
    human_uci = mv.uci()
    board.push(mv); render(board)
    if board.is_game_over():
        continue
    print("  thinking ...", flush=True)
    engine_move(board, human_uci=human_uci); render(board)
    start_ponder(board)                      # ponder during your next think
