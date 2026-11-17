"""
Check whether individual players differ enough for opponent adaptation to have anything to learn.

If every strong player plays the same move in the same position, opponent adaptation is learning a
constant. This is model free and compares, over positions that several players reached,

  within-player agreement   two games by the same player from the same position
  across-player agreement   two games by different players from the same position

Higher within than across means players are individually predictable. Equal means they are not.

  python chess/probe_opponent_signal.py [MAX_GAMES_PER_PLAYER]
  env: SRC=gm_pgn  MIN_PLY=8  MAX_PLY=40
"""
import os, sys, glob, random
from collections import defaultdict
import chess, chess.pgn

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
SRC = os.path.join(HERE, E("SRC", "gm_pgn"))
MIN_PLY = int(E("MIN_PLY", 8))
MAX_PLY = int(E("MAX_PLY", 40))
MAXG = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def key(board):
    return " ".join(board.fen().split()[:4])


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(SRC, "*.pgn")))
    if not files:
        raise SystemExit(f"no pgn in {SRC}")

    # position -> player -> list of moves played from it
    table = defaultdict(lambda: defaultdict(list))
    print(f"reading up to {MAXG} games from each of {len(files)} players", flush=True)
    for fp in files:
        who = os.path.splitext(os.path.basename(fp))[0]
        n = 0
        with open(fp, encoding="utf-8", errors="ignore") as fh:
            while n < MAXG:
                try:
                    g = chess.pgn.read_game(fh)
                except Exception:
                    continue
                if g is None:
                    break
                h = g.headers
                w, b = (h.get("White", "") or "").lower(), (h.get("Black", "") or "").lower()
                low = who.lower()
                side = chess.WHITE if low in w else chess.BLACK if low in b else None
                if side is None:
                    continue
                n += 1
                board = g.board()
                for ply, mv in enumerate(g.mainline_moves()):
                    if ply > MAX_PLY:
                        break
                    if ply >= MIN_PLY and board.turn == side:
                        table[key(board)][who].append(mv.uci())
                    board.push(mv)

    within_same = within_tot = 0
    across_same = across_tot = 0
    rng = random.Random(0)
    shared = 0
    for pos, byplayer in table.items():
        players = [p for p, mv in byplayer.items() if mv]
        if len(players) < 2:
            # still usable for the within-player half if one player was there twice
            for p in players:
                mv = byplayer[p]
                if len(mv) >= 2:
                    for _ in range(min(len(mv), 6)):
                        a, b = rng.sample(mv, 2)
                        within_same += (a == b); within_tot += 1
            continue
        shared += 1
        for p in players:
            mv = byplayer[p]
            if len(mv) >= 2:
                for _ in range(min(len(mv), 6)):
                    a, b = rng.sample(mv, 2)
                    within_same += (a == b); within_tot += 1
        for _ in range(min(len(players) * 3, 18)):
            p, q = rng.sample(players, 2)
            a, b = rng.choice(byplayer[p]), rng.choice(byplayer[q])
            across_same += (a == b); across_tot += 1

    print(f"\npositions indexed            {len(table):,}")
    print(f"reached by 2+ players        {shared:,}")
    print(f"within-player comparisons    {within_tot:,}")
    print(f"across-player comparisons    {across_tot:,}")
    print()
    w = within_same / max(within_tot, 1)
    a = across_same / max(across_tot, 1)
    print(f"same move, SAME player       {w*100:.1f}%")
    print(f"same move, DIFFERENT players {a*100:.1f}%")
    print(f"difference                   {(w-a)*100:+.1f} points")
    print()
    if w - a > 0.05:
        print("Players are individually predictable. There IS something to adapt to.")
    else:
        print("Players are interchangeable here. Opponent adaptation has nothing to learn.")
