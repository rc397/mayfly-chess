"""Turn per-player PGN files into whole-game sequences for training the game memory.

Test-time training needs whole games, since the memory should know more about an opponent on move 40
than on move 10. Games stay grouped by player (gm_pgn holds one file per player), so there is a
particular opponent to adapt to.

    python chess/gen_ttt_games.py [OUT=ttt_games] [MAX_GAMES_PER_PLAYER=400]
    env SRC=gm_pgn, MIN_PLIES=30, MAX_PLIES=120

Writes one .npz per player with these arrays.
  ids   (G, P, 64) uint8    board as piece ids, already canonicalised to side-to-move
  cast  (G, P, 4)  float32  castling rights
  move  (G, P)     int16    the move actually played from that position, -1 where padded
  stm   (G, P)     uint8    1 if the side to move is the player this file is named for
  plies (G,)       int16    real length before padding
"""
import os, sys, glob, time
import numpy as np
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import board_to_ids, move_to_idx

E = os.environ.get
OUT = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "ttt_games")
MAX_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SRC = os.path.join(HERE, E("SRC", "gm_pgn"))
MIN_PLIES = int(E("MIN_PLIES", 30))
MAX_PLIES = int(E("MAX_PLIES", 120))


def player_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def walk(game, who):
    """One game to per-ply arrays. Returns None if it is too short to be useful.

    `who` is the surname this file is named for, to mark which plies are theirs. The memory learns
    from every transition, but only their plies test opponent adaptation.

    """
    board = game.board()
    white = (game.headers.get("White", "") or "").lower()
    black = (game.headers.get("Black", "") or "").lower()
    w = who.lower()
    theirs_white = w in white
    theirs_black = w in black
    if not (theirs_white or theirs_black):
        return None

    ids_l, cast_l, mv_l, stm_l = [], [], [], []
    for mv in game.mainline_moves():
        if len(ids_l) >= MAX_PLIES:
            break
        try:
            ids, cast, flip = board_to_ids(board)
            idx = move_to_idx(mv, flip)
        except Exception:
            break
        is_theirs = (board.turn == chess.WHITE and theirs_white) or \
                    (board.turn == chess.BLACK and theirs_black)
        ids_l.append(ids); cast_l.append(cast); mv_l.append(idx)
        stm_l.append(1 if is_theirs else 0)
        board.push(mv)

    if len(ids_l) < MIN_PLIES:
        return None
    return (np.asarray(ids_l, dtype=np.uint8), np.asarray(cast_l, dtype=np.float32),
            np.asarray(mv_l, dtype=np.int16), np.asarray(stm_l, dtype=np.uint8))


def pack(rows):
    """Pad a list of variable-length games into one rectangular block."""
    G, P = len(rows), max(len(r[0]) for r in rows)
    ids = np.zeros((G, P, 64), dtype=np.uint8)
    cast = np.zeros((G, P, 4), dtype=np.float32)
    move = np.full((G, P), -1, dtype=np.int16)
    stm = np.zeros((G, P), dtype=np.uint8)
    plies = np.zeros(G, dtype=np.int16)
    for i, (a, c, m, s) in enumerate(rows):
        n = len(a)
        ids[i, :n] = a; cast[i, :n] = c; move[i, :n] = m; stm[i, :n] = s
        plies[i] = n
    return ids, cast, move, stm, plies


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.pgn")))
    if not files:
        raise SystemExit(f"no pgn files in {SRC}")
    print(f"{len(files)} player files in {os.path.basename(SRC)} -> {os.path.basename(OUT)}",
          flush=True)

    t0, tot_g, tot_p = time.time(), 0, 0
    for fp in files:
        who = player_name(fp)
        dest = os.path.join(OUT, who + ".npz")
        if os.path.exists(dest):
            print(f"  {who:22s} already done", flush=True)
            continue
        rows = []
        with open(fp, encoding="utf-8", errors="ignore") as fh:
            while len(rows) < MAX_GAMES:
                try:
                    g = chess.pgn.read_game(fh)
                except Exception:
                    continue
                if g is None:
                    break
                r = walk(g, who)
                if r is not None:
                    rows.append(r)
        if len(rows) < 20:
            print(f"  {who:22s} only {len(rows)} usable games, skipped", flush=True)
            continue
        ids, cast, move, stm, plies = pack(rows)
        np.savez_compressed(dest, ids=ids, cast=cast, move=move, stm=stm, plies=plies)
        tot_g += len(rows); tot_p += int(plies.sum())
        print(f"  {who:22s} {len(rows):4d} games  {int(plies.mean()):3d} plies avg  "
              f"{os.path.getsize(dest)/1e6:5.1f} MB", flush=True)

    print(f"\n{tot_g} games, {tot_p} plies, {time.time()-t0:.0f}s", flush=True)
