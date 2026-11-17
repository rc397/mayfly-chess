"""Append shipped self-play position chunks to the labeller's input file.

The labeller reads positions.txt and keeps its place by line, so chunks are only ever appended, as whole
lines, in name order. Each chunk's name carries its source file and byte range, and appended names are
recorded, so a chunk sent twice is appended once. Processed chunks move to incoming/done. The bot's
finished Lichess games are folded in too.

    python append_incoming.py       run after each chunk is copied in
"""
import os, glob, re, sys, time, atexit
import paths
IN = os.path.join(paths.SELFPLAY, "incoming")
POS = os.path.join(paths.SELFPLAY, "positions.txt")
DONE = os.path.join(IN, "appended.txt")
LOCK = os.path.join(IN, ".lock")
os.makedirs(os.path.join(IN, "done"), exist_ok=True)
# One run at a time, since two runs could both append a chunk before either recorded it. A run that finds
# the lock exits 1 and its caller retries later. A lock ten minutes old was left by a run that died.
try:
    os.close(os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
except FileExistsError:
    if time.time() - os.path.getmtime(LOCK) < 600:
        print("another append_incoming is running")
        sys.exit(1)
    os.utime(LOCK, None)
atexit.register(lambda: os.path.exists(LOCK) and os.remove(LOCK))
# A file shorter than the byte range in its name is still being copied. It waits for the next run.
RANGE = re.compile(r"_(\d{12})_(\d{12})\.txt$")
done = set(open(DONE, encoding="utf-8").read().split()) if os.path.exists(DONE) else set()
# Bookkeeping files are not chunks. games_read.txt once lived here and was appended as positions,
# which made every game get re-read. It now lives outside this folder, and both names are skipped.
BOOKKEEPING = {"appended.txt", "games_read.txt"}
added = 0
for p in sorted(glob.glob(os.path.join(IN, "*.txt"))):
    name = os.path.basename(p)
    if name in BOOKKEEPING:
        continue
    m = RANGE.search(name)
    if m and os.path.getsize(p) < int(m.group(2)) - int(m.group(1)):
        continue
    if name not in done:
        data = open(p, encoding="utf-8").read()
        if data and not data.endswith("\n"):
            data += "\n"
        with open(POS, "a", encoding="utf-8") as fh:
            fh.write(data)
        with open(DONE, "a", encoding="utf-8") as fh:
            fh.write(name + "\n")
        added += data.count("\n")
    try:
        os.replace(p, os.path.join(IN, "done", name))
    except OSError:
        pass                    # still held by its copy, already recorded, so the next run only moves it

# The bot's own Lichess games are positions it really reached, mistakes included, which is what route 1
# wants. lichess-bot writes one PGN per finished game (its pgn_directory). Every position from ply 8 on
# goes to the labeller. Only games with a result, written more than two minutes ago, are read, once each.
import time
try:
    import chess.pgn
    GAMES = os.environ.get("BOT_GAMES", paths.under("game_records"))   # lichess-bot's pgn_directory
    SEEN = os.path.join(os.path.dirname(IN), "games_read.txt")      # outside incoming, see above
    if not os.path.exists(SEEN):
        # carry over the list from its old location so those games are not appended again
        old = set()
        for q in (os.path.join(IN, "games_read.txt"), os.path.join(IN, "done", "games_read.txt")):
            if os.path.exists(q):
                old |= set(open(q, encoding="utf-8").read().splitlines())
        with open(SEEN, "w", encoding="utf-8") as fh:
            fh.write("".join(n + chr(10) for n in sorted(old) if n))
    seen = set(open(SEEN, encoding="utf-8").read().splitlines()) if os.path.exists(SEEN) else set()
    from_games = n_games = 0
    for p in sorted(glob.glob(os.path.join(GAMES, "*.pgn"))):
        name = os.path.basename(p)
        if name in seen or time.time() - os.path.getmtime(p) < 120:
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            game = chess.pgn.read_game(fh)
        if game is None or game.headers.get("Result", "*") == "*":
            continue
        board, fens = game.board(), []
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= 8 and not board.is_game_over():
                fens.append(board.fen())
            board.push(move)
        if fens:
            with open(POS, "a", encoding="utf-8") as fh:
                fh.write("\n".join(fens) + "\n")
        with open(SEEN, "a", encoding="utf-8") as fh:
            fh.write(name + "\n")
        from_games += len(fens)
        n_games += 1
    games_note = f", {from_games} from {n_games} Lichess game(s)"
except Exception as e:
    games_note = f", Lichess games not read: {type(e).__name__}: {e}"
print(f"appended {added} positions{games_note}")
