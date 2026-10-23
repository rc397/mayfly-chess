r"""Measure the model's rook-pawn habit, and whether it is specific to Black.

How often the model pushes rook pawns is compared with two references on the same positions.

  Stockfish  what a strong engine would play there
  the games  the move a titled player actually played from that exact position

Positions come from real master games (TWIC), not from the model's own self-play.

It first checks a simpler explanation. The board is mirrored to the side to move's frame before the
network sees it, and so is the move index. If the two used different conventions (ranks only against
ranks and files), the a-file and h-file would swap for Black only, which would look like the same
habit.

  python chess/quirk.py [N]
  env: VERSION=mayfly_v1.pt  SF_MS=100  VISITS=0  PGN=<data>/twic
"""
import os, sys, glob, random, collections
import chess, chess.pgn, chess.engine
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SF_MS = int(E("SF_MS", 100))
PGN_DIR = E("PGN", paths.under("twic"))


def sample_positions(n, seed=0):
    """Real positions from real games, one per game, past the opening book."""
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
    rng.shuffle(files)
    out = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while len(out) < n:
                try:
                    g = chess.pgn.read_game(fh)
                except Exception:
                    break
                if g is None:
                    break
                moves = list(g.mainline_moves())
                if len(moves) < 24:
                    continue
                # somewhere in the middlegame, past the opening book
                k = rng.randrange(12, min(len(moves) - 1, 70))
                b = g.board()
                for m in moves[:k]:
                    b.push(m)
                if b.is_game_over():
                    continue
                out.append((b.fen(), moves[k].uci()))
        if len(out) >= n:
            break
    return out[:n]


def is_rook_pawn(board, mv):
    p = board.piece_at(mv.from_square)
    return bool(p and p.piece_type == chess.PAWN and chess.square_file(mv.from_square) in (0, 7))


def is_pawn(board, mv):
    p = board.piece_at(mv.from_square)
    return bool(p and p.piece_type == chess.PAWN)


def edge_file(board, mv):
    """a-file or h-file, any piece, to separate liking the edge from liking rook pawns."""
    return chess.square_file(mv.to_square) in (0, 7)


def rate(rows, pred):
    if not rows:
        return float("nan")
    return 100.0 * sum(1 for r in rows if pred(r)) / len(rows)


if __name__ == "__main__":
    import torch
    from play_race import think
    from play_web import load_version
    from moves import move_to_idx

    print("=== first, do the board flip and the move flip agree? ===")
    # If they disagreed by mirroring files as well as ranks, a-file and h-file would swap for
    # Black. Play the same position from both sides and check the model's ranking of a move is
    # the same move, not its file-mirror.
    from chess_net import board_to_ids
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    ids_w, _, flip_w = board_to_ids(b)
    b2 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    ids_b, _, flip_b = board_to_ids(b2)
    same = ids_w == ids_b
    h_push_w = move_to_idx(chess.Move.from_uci("h2h4"), False)
    h_push_b = move_to_idx(chess.Move.from_uci("h7h5"), True)
    a_push_b = move_to_idx(chess.Move.from_uci("a7a5"), True)
    print(f"  the start position looks identical from both sides: {same}")
    print(f"  White h2h4 index {h_push_w}, Black h7h5 index {h_push_b}  -> same index: "
          f"{h_push_w == h_push_b}")
    print(f"  Black a7a5 index {a_push_b}, which must NOT equal the h index: "
          f"{a_push_b != h_push_b}")
    print(f"  verdict: the flip is consistent, files are preserved, "
          f"{'no file swap' if h_push_w == h_push_b and a_push_b != h_push_b else 'FILE SWAP BUG'}")
    print("")

    print(f"sampling {N} middlegame positions from real games", flush=True)
    pos = sample_positions(N)
    print(f"  {len(pos)} positions", flush=True)

    model, label, _ = load_version(E("VERSION", "mayfly_v1.pt"))
    sf_path = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 2, "Hash": 256})

    # VISITS=0 measures the bare policy head, which gives the tree its priors. VISITS>0 measures the
    # engine with search, which may overrule a bad prior.
    VISITS = int(E("VISITS", 0))
    tree = None
    if VISITS:
        from mcts import MCTS
        tree = MCTS(model, str(next(model.parameters()).device), batch=48, cache_size=20000)

    rows = []
    for i, (fen, played) in enumerate(pos):
        b = chess.Board(fen)
        if tree is not None:
            tree.reset()
            mine, _ = tree.search(b, visits=VISITS, reuse=False)
            if mine is None:
                continue
        else:
            moves, _ = think(model, b)
            if not moves:
                continue
            mine = moves[0][0]
        try:
            sf = eng.play(b, chess.engine.Limit(time=SF_MS / 1000.0)).move
        except Exception:
            continue
        rows.append({"board": b, "model": mine, "sf": sf,
                     "human": chess.Move.from_uci(played), "black": b.turn == chess.BLACK})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pos)}", flush=True)
    eng.quit()

    print("")
    print(f"{len(rows)} positions compared, Stockfish at {SF_MS} ms, "
          f"MayFly at {VISITS} visits" if VISITS else
          f"{len(rows)} positions compared, Stockfish at {SF_MS} ms, MayFly policy head only")
    print("")
    for who in ("model", "sf", "human"):
        name = {"model": label, "sf": "Stockfish", "human": "the player"}[who]
        for subset, rs in (("all", rows),
                           ("white to move", [r for r in rows if not r["black"]]),
                           ("black to move", [r for r in rows if r["black"]])):
            rp = rate(rs, lambda r: is_rook_pawn(r["board"], r[who]))
            pw = rate(rs, lambda r: is_pawn(r["board"], r[who]))
            ef = rate(rs, lambda r: edge_file(r["board"], r[who]))
            print(f"  {name:<12} {subset:<14} n={len(rs):<4} "
                  f"rook pawn {rp:5.1f}%   any pawn {pw:5.1f}%   to a/h file {ef:5.1f}%")
        print("")

    # where it differs from Stockfish, what does it reach for
    diff = [r for r in rows if r["model"] != r["sf"]]
    print(f"  it disagrees with Stockfish on {100*len(diff)/max(len(rows),1):.0f}% of positions")
    kinds = collections.Counter()
    for r in diff:
        p = r["board"].piece_at(r["model"].from_square)
        kinds[chess.piece_name(p.piece_type) if p else "?"] += 1
    print("  when it disagrees, the piece it moves:")
    for k, v in kinds.most_common():
        print(f"    {k:<8} {100*v/len(diff):5.1f}%")
