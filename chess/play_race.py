"""
Play the two small experiment models (tropical and transformer) in a terminal and see what they think.

These are policy-head engines with no search, so you are playing the raw network. Both are weak, and
the point is to find where they break, which a loss curve cannot show.

  python chess/play_race.py                  you are White against tropical
  python chess/play_race.py ctrl black       you are Black against the transformer
  python chess/play_race.py both             play tropical, but see both models every move

Moves in SAN (Nf3, exd5, O-O) or UCI (g1f3).

  board   redraw            fen     show the FEN         undo   take back a full move
  hint    your top moves    eval    both models on this position
  swap    change opponent   temp N  0 is greedy, 0.5 is loose
  new     restart           quit
"""
import os, sys, json, math
import torch, chess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet, board_to_ids, move_to_idx, clock_and_ep
from mayfly_chess import MayflyChessNet, matched_dff_h

device = "cuda" if torch.cuda.is_available() else "cpu"

# The Windows console defaults to cp1252, which has no chess glyphs. Ask for UTF-8, then test that
# it took.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
_FIGURINE = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕",
             "K": "♔", "p": "♟", "n": "♞", "b": "♝", "r": "♜",
             "q": "♛", "k": "♚"}
try:
    "".join(_FIGURINE.values()).encode(sys.stdout.encoding or "ascii")
    GLYPH, EMPTY = _FIGURINE, "·"
except Exception:
    GLYPH, EMPTY = {k: k for k in _FIGURINE}, "."   # uppercase White, lowercase Black


def tuned_cfg(arch):
    p = os.path.join(HERE, f"sweep_{arch}.json")
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p, encoding="utf-8"))["final"]
    except Exception:
        return {}


def load(arch):
    """Rebuild at the configuration the convergence run actually used, then load strictly."""
    path = os.path.join(HERE, f"ckpt_long_{arch}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"missing {os.path.basename(path)}, put it in {HERE} first")
    ck = torch.load(path, map_location=device)
    sd, cfg, tuned = ck["model"], ck.get("cfg", {}), tuned_cfg(arch)
    d, L = cfg.get("d", 256), cfg.get("layers", 4)
    if arch == "trop":
        H = int(tuned.get("H", cfg.get("H", 8)))
        m = MayflyChessNet(d=d, layers=L, dff_h=matched_dff_h(d, H), H=H,
                           T=float(tuned.get("T", 0.66)), sc=float(tuned.get("SC", 1.5)))
        name = "tropical"
    else:
        m = ChessNet(d=d, nhead=int(tuned.get("NHEAD", 0)) or max(4, d // 64), layers=L,
                     dim_ff=int(sd["enc.layers.0.linear1.weight"].shape[0]))
        name = "transformer"
    m.load_state_dict(sd, strict=True)
    m.to(device).eval()
    return m, name, float(ck.get("val", {}).get("policy", float("nan")))


@torch.no_grad()
def think(model, board, topn=5, castle="king"):
    """Legal-masked move distribution plus a value, from the side to move.

    castle is the version's castle_prior, where its policy head keeps castling (mcts.policy_index)."""
    ids, cast, flip = board_to_ids(board)
    hm, ep = clock_and_ep(board, flip)
    out = model(torch.tensor([ids], device=device), torch.tensor([cast], device=device),
                torch.tensor([hm], device=device), torch.tensor([ep], device=device))
    legal = list(board.legal_moves)
    if not legal:
        return [], 0.5
    idxs = torch.tensor([move_to_idx(m, flip) for m in legal], device=device)
    # The policy reads castling where the version keeps it (v4 at e1h1 and e1a1). The q head names
    # castling by the king's destination, so idxs stays.
    pidx = idxs
    if castle != "king":
        from mcts import policy_index
        pidx = torch.from_numpy(policy_index(board, legal, flip, castle)).to(device)
    p = torch.softmax(out["policy"][0][pidx], dim=-1)
    order = torch.argsort(p, descending=True)[:topn]
    # ChessBench barely trains the value head, so the action-value head is read instead when there
    # is one.
    if out.get("qvalue") is not None:
        # Policy weighted, not the maximum, since the maximum of noisy action values is biased
        # upward. This asks what the position is worth if the model plays as it actually plays.
        q = torch.sigmoid(out["qvalue"][0][idxs].float())
        wp = float((p.float() * q).sum() / p.float().sum())
    else:
        wp = float(torch.sigmoid(out["value"])[0])
    return [(legal[int(i)], float(p[int(i)])) for i in order], wp


def cp_of(wp):
    wp = min(max(wp, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / wp - 1.0)


def show_thinking(model, name, board):
    moves, wp = think(model, board)
    if not moves:
        return
    who = "White" if board.turn == chess.WHITE else "Black"
    line = "  ".join(f"{board.san(m)} {100*p:.0f}%" for m, p in moves)
    print(f"  {name:11s} {line}")
    print(f"  {'':11s} rates {who} at {100*wp:.0f}%  ({cp_of(wp):+.0f}cp)")


def draw(board, human_white):
    print()
    ranks = range(7, -1, -1) if human_white else range(8)
    files = range(8) if human_white else range(7, -1, -1)
    for r in ranks:
        row = []
        for f in files:
            pc = board.piece_at(chess.square(f, r))
            row.append(GLYPH[pc.symbol()] if pc else EMPTY)
        print(f"  {r+1}  " + " ".join(row))
    print("     " + " ".join("abcdefgh"[f] for f in files))
    if board.is_check():
        print("     check")
    print()


def parse(board, s):
    for fn in (board.parse_san, board.parse_uci):
        try:
            return fn(s)
        except Exception:
            continue
    return None


def pick(moves, temp):
    if temp <= 0 or len(moves) == 1:
        return moves[0][0]
    w = torch.tensor([p for _, p in moves]) ** (1.0 / temp)
    return moves[int(torch.multinomial(w / w.sum(), 1))][0]


def outcome(board):
    if not board.is_game_over():
        return None
    r = board.result()
    if r == "1-0":
        return "White wins"
    if r == "0-1":
        return "Black wins"
    return f"draw ({'stalemate' if board.is_stalemate() else 'material or repetition'})"


if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]]
    human_white = not any(a in ("black", "b") for a in args)
    both = "both" in args
    which = "ctrl" if "ctrl" in args else "trop"

    models = {}
    for a in (("trop", "ctrl") if both or True else (which,)):
        try:
            models[a] = load(a)
        except SystemExit as e:
            print(e)
    if which not in models:
        raise SystemExit("cannot continue without the opponent checkpoint")

    print(f"\ndevice {device}")
    for a, (m, nm, v) in models.items():
        print(f"  {nm:11s} {m.num_params()/1e6:.3f}M params, held-out policy loss {v:.4f}")
    print(f"\nyou are {'White' if human_white else 'Black'} against {models[which][1]}."
          f"{'  both models shown each move.' if both else ''}")
    print("type help for commands\n")

    board = chess.Board()
    temp = 0.0
    draw(board, human_white)

    while True:
        over = outcome(board)
        if over:
            print(f"\n  {over}\n")
            board = chess.Board(); draw(board, human_white); continue

        if board.turn == (chess.WHITE if human_white else chess.BLACK):
            try:
                s = input("your move> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not s:
                continue
            low = s.lower()
            if low in ("quit", "q", "exit"):
                break
            if low == "help":
                print(__doc__); continue
            if low == "board":
                draw(board, human_white); continue
            if low == "fen":
                print(f"  {board.fen()}"); continue
            if low == "new":
                board = chess.Board(); draw(board, human_white); continue
            if low == "undo":
                for _ in range(2):
                    if board.move_stack:
                        board.pop()
                draw(board, human_white); continue
            if low == "swap":
                which = "ctrl" if which == "trop" else "trop"
                print(f"  opponent is now {models[which][1]}"); continue
            if low.startswith("temp"):
                try:
                    temp = float(low.split()[1]); print(f"  temperature {temp}")
                except Exception:
                    print("  usage: temp 0.3")
                continue
            if low in ("hint", "eval"):
                for a in (models if low == "eval" else (which,)):
                    show_thinking(models[a][0], models[a][1], board)
                continue
            mv = parse(board, s)
            if mv is None or mv not in board.legal_moves:
                print("  not a legal move here"); continue
            board.push(mv)
            draw(board, human_white)
            continue

        model, name, _ = models[which]
        moves, wp = think(model, board)
        if not moves:
            continue
        if both:
            for a in models:
                show_thinking(models[a][0], models[a][1], board)
        mv = pick(moves, temp)
        san = board.san(mv)
        board.push(mv)
        print(f"  {name} plays {san}   ({100*dict(moves)[mv]:.0f}% of its mass, "
              f"eval {cp_of(wp):+.0f}cp)")
        draw(board, human_white)
