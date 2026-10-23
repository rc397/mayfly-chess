"""
An opening repertoire for the early models, with two roles.
  1. book_move(board) plays from a French book (french_book.json, position EPD to moves) when that file
     is present, and returns None everywhere else, so other openings are left to the engine.
  2. seed_lines_uci() gives opening lines to seed data generation, the families below (French, Catalan,
     Queen's Indian, Caro-Kann, Queen's Gambit, English) plus french_seeds.json when present.
"""
import os, json, random, chess

HERE = os.path.dirname(os.path.abspath(__file__))

# Family seed lines (SAN), used only to seed data generation, never forced at play time.
WHITE_LINES = [
    "d4 Nf6 c4 e6 g3 d5 Nf3 Be7 Bg2 O-O O-O",          # Catalan, closed
    "d4 Nf6 c4 e6 g3 d5 Nf3 Be7 Bg2 O-O O-O Qc2",      # Closed Catalan main
    "d4 d5 c4 e6 Nf3 Nf6 g3 Be7 Bg2",                  # Catalan via QGD
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3",          # Queen's Gambit Declined
    "d4 d5 c4 c6 Nf3 Nf6 g3",                          # Slav, g3
    "d4 d5 c4 dxc4 Nf3 Nf6 g3 e6 Bg2",                 # Queen's Gambit Accepted
    "c4 e6 Nc3 d5 d4 Nf6 Nf3 Be7 g3",                  # English -> Catalan/QGD
    "c4 Nf6 Nc3 e6 Nf3 d5 d4 Be7 g3",                  # English
    "c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2",               # English, reversed Sicilian
]
BLACK_LINES = [
    "e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7 f4 c5",               # French, Classical/Steinitz
    "e4 e6 d4 d5 e5 c5 c3 Nc6",                        # French, Advance
    "e4 e6 d4 d5 Nd2 c5",                              # French, Tarrasch
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5",                   # Caro-Kann, Classical
    "e4 c6 d4 d5 e5 Bf5",                              # Caro-Kann, Advance
    "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7",             # QID
    "d4 Nf6 c4 e6 Nc3 d5 Bg5 Be7 e3 O-O",             # QGD as Black
    "d4 Nf6 Nf3 e6 c4 b6 g3 Bb7",                      # QID via Nf3
]

def _san_to_uci(san_line):
    b = chess.Board(); out = []
    for tok in san_line.split():
        mv = b.parse_san(tok); out.append(mv.uci()); b.push(mv)
    return out

def seed_lines_uci():
    """The family lines plus french_seeds.json, as UCI move lists, for seeding data generation."""
    lines = []
    for ln in WHITE_LINES + BLACK_LINES:
        try:
            lines.append(_san_to_uci(ln))
        except Exception:
            pass
    fp = os.path.join(HERE, "french_seeds.json")
    if os.path.exists(fp):
        lines += [l for l in json.load(open(fp)) if l]
    return lines

_FB = None
def book_move(board, rng=random):
    """A move from the French book (french_book.json), or None outside it.

    Other openings are left to the model's own learned choices."""
    global _FB
    if _FB is None:
        fp = os.path.join(HERE, "french_book.json")
        _FB = json.load(open(fp)) if os.path.exists(fp) else {}
    opts = _FB.get(board.epd())
    return chess.Move.from_uci(rng.choice(opts)) if opts else None

if __name__ == "__main__":
    sl = seed_lines_uci()
    print(f"seed lines: {len(sl)} (families + French seeds)")
    b = chess.Board(); b.push_san("e4")
    bm = book_move(b)
    print("French book vs 1.e4:", b.san(bm) if bm else "(none)")
    b.push_san("e6"); b.push_san("d4")
    bm = book_move(b)
    print("French book after 1.e4 e6 2.d4:", b.san(bm) if bm else "(none)")
