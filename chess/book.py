"""An opening book built from the TWIC master games in chess/twic.

A book saves clock time in the opening, gives variety (the search picks moves by argmax, so without
a book every game from the start position is the same), and avoids dubious moves in well-known
positions. Moves are sampled by how often masters played them. A book is borrowed strength and says
nothing about the network, so it is on for live play and off in every test harness.

    python chess/book.py build [PLIES] [MIN_ELO]     writes book.json
    python chess/book.py show                        what it knows about the start position
    env BOOK_FILE (a .json built here, or a Polyglot .bin), PGN,
        BOOK_MIN_GAMES=5, BOOK_MIN_SHARE=0.03, BOOK_MIN_SCORE=0.40
"""
import os, sys, glob, json, random, collections
import chess, chess.pgn, chess.polyglot

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
BOOK_FILE = E("BOOK_FILE", os.path.join(HERE, "book.json"))
PGN_DIR = E("PGN", os.path.join(HERE, "twic"))
MIN_GAMES = int(E("BOOK_MIN_GAMES", 5))       # a position nobody has played is not knowledge
MIN_SHARE = float(E("BOOK_MIN_SHARE", 0.03))  # nor is a move played once in a hundred
MIN_SCORE = float(E("BOOK_MIN_SCORE", 0.40))  # nor one that scores badly for the side to move

_BOOK = None
_BOOK_AT = None


def load(path=None):
    """The book, or an empty one. Cached, and re-read when the file on disk changes, so a rebuilt
    book reaches a running board without a restart."""
    global _BOOK, _BOOK_AT
    src = path or BOOK_FILE
    try:
        stamp = os.stat(src).st_mtime_ns
    except OSError:
        stamp = None
    if path:
        try:
            with open(src, encoding="utf-8") as fh:
                return {int(k): v for k, v in json.load(fh).get("positions", {}).items()}
        except (OSError, ValueError, KeyError):
            return {}
    if _BOOK is None or stamp != _BOOK_AT:
        try:
            with open(src, encoding="utf-8") as fh:
                _BOOK = {int(k): v for k, v in json.load(fh).get("positions", {}).items()}
        except (OSError, ValueError, KeyError):
            _BOOK = {}
        _BOOK_AT = stamp
    return _BOOK


def _polyglot(board, rng, path):
    """A move from a Polyglot .bin book, weighted by its entries.

    The built book uses Polyglot Zobrist keys too, so the two are interchangeable. Deep engine books
    can leave the network in positions it does not understand, so a deeper book is not always better.
    """
    import chess.polyglot
    try:
        with chess.polyglot.open_reader(path) as reader:
            entries = [e for e in reader.find_all(board) if e.move in board.legal_moves]
    except (OSError, ValueError, IndexError):
        return None
    if not entries:
        return None
    total = sum(max(e.weight, 1) for e in entries)
    r = (rng or random).random() * total
    for e in entries:
        r -= max(e.weight, 1)
        if r <= 0:
            return e.move
    return entries[-1].move


def pick(board, rng=None, plies=16, path=None):
    """A move for this position, sampled by how often masters chose it, or None.

    Sampled rather than argmax so that two games from the same position differ.
    """
    if len(board.move_stack) >= plies:
        return None
    src = path or BOOK_FILE
    if str(src).lower().endswith(".bin"):
        return _polyglot(board, rng, src)
    entry = (load(path) if path else load()).get(chess.polyglot.zobrist_hash(board))
    if not entry:
        return None
    legal = {m.uci() for m in board.legal_moves}
    moves = [(u, c) for u, (c, _s) in entry.items() if u in legal]
    if not moves:
        return None
    total = sum(c for _u, c in moves)
    r = (rng or random).random() * total
    for u, c in moves:
        r -= c
        if r <= 0:
            return chess.Move.from_uci(u)
    return chess.Move.from_uci(moves[-1][0])


def build(plies=16, min_elo=2400):
    counts = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
    games = kept = 0
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while True:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                games += 1
                h = g.headers
                try:
                    if min(int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))) < min_elo:
                        continue
                except ValueError:
                    continue
                res = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(h.get("Result", ""))
                if res is None:
                    continue
                kept += 1
                b = g.board()
                for i, mv in enumerate(g.mainline_moves()):
                    if i >= plies:
                        break
                    key = chess.polyglot.zobrist_hash(b)
                    slot = counts[key][mv.uci()]
                    slot[0] += 1
                    slot[1] += res if b.turn == chess.WHITE else 1.0 - res
                    b.push(mv)
        print(f"  {os.path.basename(path)}: {games:,} games seen, {kept:,} kept, "
              f"{len(counts):,} positions", flush=True)

    # prune hard, a book entry is only worth having where many strong games agree
    out = {}
    for key, moves in counts.items():
        total = sum(c for c, _s in moves.values())
        if total < MIN_GAMES:
            continue
        keep = {u: [c, round(s / c, 3)] for u, (c, s) in moves.items()
                if c >= 3 and c / total >= MIN_SHARE and s / c >= MIN_SCORE}
        if keep:
            out[str(key)] = keep
    doc = {"plies": plies, "min_elo": min_elo, "games": kept, "positions": out}
    tmp = BOOK_FILE + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    os.replace(tmp, BOOK_FILE)
    print(f"{len(out):,} positions from {kept:,} master games at {min_elo}+, {plies} plies deep "
          f"-> {BOOK_FILE} ({os.path.getsize(BOOK_FILE)/1e6:.1f} MB)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 16,
              int(sys.argv[3]) if len(sys.argv) > 3 else 2400)
    else:
        bk = load()
        b = chess.Board()
        entry = bk.get(chess.polyglot.zobrist_hash(b), {})
        print(f"{len(bk):,} positions in {BOOK_FILE}")
        for u, (c, s) in sorted(entry.items(), key=lambda kv: -kv[1][0]):
            print(f"  {b.san(chess.Move.from_uci(u)):<5} {c:>6,} games, scores {s:.3f}")
        rng = random.Random(0)
        print("  ten openings it would pick: "
              + ", ".join(b.san(pick(b, rng) or chess.Move.null()) for _ in range(10)))
