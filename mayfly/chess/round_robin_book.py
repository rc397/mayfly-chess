r"""round_robin_book builds the round robin's fifty openings into round_robin_book.json.

Every version's search is deterministic, so two versions from the same position and colours play the same
game every time. A pairing of 100 games therefore needs fifty openings, each played once from each side.
The openings come from TWIC master games where both players are rated MIN_ELO or more. For each of the most
played ECO codes it takes the position reached most often after PLIES plies, one per ECO code and no
position twice. A position is kept only if Stockfish puts it within BAL_CP of level, so the games are about
the players more than the opening.

  python chess/round_robin_book.py        env N (50) PLIES (10) MIN_ELO (2500) BAL_CP (60) SF_SECONDS (2.0)
                                              PGN (chess/twic) OUT (chess/round_robin_book.json)
"""
import collections
import glob
import json
import os
import re
import time

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.environ.get
N = int(E("N", 50))
PLIES = int(E("PLIES", 10))
MIN_ELO = int(E("MIN_ELO", 2500))
BAL_CP = int(E("BAL_CP", 60))
SF_SECONDS = float(E("SF_SECONDS", 2.0))
PGN_DIR = E("PGN", os.path.join(HERE, "twic"))
OUT = E("OUT", os.path.join(HERE, "round_robin_book.json"))
HEADER = re.compile(r'^\[(\w+) "(.*)"\]\s*$', re.M)


def first_moves(movetext, n):
    """The first n moves of a game's movetext, comments, variations and move numbers taken out."""
    text = re.sub(r"\{[^}]*\}", " ", movetext)
    while "(" in text:
        stripped = re.sub(r"\([^()]*\)", " ", text)
        if stripped == text:
            break
        text = stripped
    out = []
    for tok in text.split():
        tok = re.sub(r"^\d+\.(\.\.)?", "", tok)
        if not tok or tok.startswith("$") or tok in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        out.append(tok)
        if len(out) == n:
            break
    return out


def games():
    """(headers, first PLIES moves in SAN) for every game in the TWIC files."""
    for path in sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn"))):
        text = open(path, encoding="utf-8", errors="replace").read()
        for chunk in re.split(r"\n(?=\[Event )", text):
            heads = dict(HEADER.findall(chunk))
            body = HEADER.sub("", chunk)
            yield heads, first_moves(body, PLIES)


def main():
    by_eco = collections.defaultdict(collections.Counter)
    names = collections.defaultdict(collections.Counter)
    sans = {}
    seen = 0
    for heads, moves in games():
        try:
            if min(int(heads.get("WhiteElo", 0)), int(heads.get("BlackElo", 0))) < MIN_ELO:
                continue
        except ValueError:
            continue
        eco = heads.get("ECO", "")
        if len(moves) < PLIES or not re.fullmatch(r"[A-E]\d\d", eco):
            continue
        board = chess.Board()
        try:
            for san in moves:
                board.push_san(san)
        except ValueError:
            continue
        seen += 1
        key = board.fen()
        by_eco[eco][key] += 1
        names[(eco, key)][", ".join(x for x in (heads.get("Opening", ""), heads.get("Variation", "")) if x)] += 1
        sans.setdefault(key, moves)
    print(f"{seen:,} master games reach {PLIES} plies under {len(by_eco)} ECO codes", flush=True)

    candidates, used = [], set()
    for eco, counts in sorted(by_eco.items(), key=lambda kv: -sum(kv[1].values())):
        fen, n = counts.most_common(1)[0]
        if fen in used or n < 3:
            continue
        used.add(fen)
        candidates.append((eco, fen, n, sum(counts.values())))

    sf_path = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 4, "Hash": 512})
    sf_name = eng.id.get("name", "Stockfish")
    book = []
    for eco, fen, n, total in candidates:
        board = chess.Board(fen)
        cp = eng.analyse(board, chess.engine.Limit(time=SF_SECONDS))["score"].white().score(mate_score=100000)
        keep = abs(cp) <= BAL_CP
        name = names[(eco, fen)].most_common(1)[0][0] or eco
        print(f"  {eco} {name[:40]:<40} {n:>5} of {total:>5} games  {cp:+5d} cp  {'kept' if keep else 'too far from level'}",
              flush=True)
        if keep:
            moves = sans[fen]
            text = " ".join((f"{i // 2 + 1}. " if i % 2 == 0 else "") + m for i, m in enumerate(moves))
            book.append({"name": f"{eco} {name}", "eco": eco, "moves": text, "fen": fen, "games": n, "sf_cp": cp})
        if len(book) == N:
            break
    eng.quit()
    if len(book) < N:
        raise SystemExit(f"only {len(book)} openings within {BAL_CP} cp; widen BAL_CP or lower MIN_ELO")
    out = {"made": time.strftime("%Y-%m-%d %H:%M"), "source": "TWIC master games, both players rated "
           f"{MIN_ELO} or more", "plies": PLIES, "balanced_within_cp": BAL_CP,
           "judged_by": f"{sf_name}, {SF_SECONDS:g} s a position", "openings": book}
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print(f"{len(book)} openings -> {OUT}")


if __name__ == "__main__":
    main()
