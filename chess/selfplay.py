r"""selfplay lets the model play itself and keeps the positions it reaches, with the search's targets.

The labelled corpus comes from human games, and the model does not play like a human. The positions it
reaches itself are where a strong opponent can find a repeatable hole, so the FENs are kept for labelling
by a stronger teacher, and the search's own visit and value targets are kept beside them.

Three things keep the positions varied.
  temperature   for the first TEMP_PLIES plies the move is sampled from the visit counts. After that
                it plays its best move.
  root noise    AlphaZero's Dirichlet noise at the root. ROOT_NOISE of the prior is replaced by noise
                whose total concentration NOISE_ALPHA is spread over the legal moves (about 0.3 a move
                in a typical middlegame). It lives only in this process's tree, so no sealed search
                ever sees it.
  mixed starts  some games start from a random corpus position, since games from move one rarely
                reach the sharp endgames the shallow teacher is worst at.

  python chess/selfplay.py [POSITIONS]
  env: TAG=mayfly VERSION=  (a file in the versions folder, else the live checkpoint)
       VISITS=400 TEMP_PLIES=20 OPENING_FRAC=0.6 OUT (default paths.under("selfplay", "positions.txt"))
       MAX_PLIES=200 BATCH=48 ROOT_NOISE=0.25 NOISE_ALPHA=10.83 (ROOT_NOISE=0 turns it off)
"""
import os, sys, time, json, random, glob
import numpy as np
import chess
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mcts import MCTS
from chess_net import ChessNet
import paths

E = os.environ.get
TAG = E("TAG", "mayfly")
VERSION = E("VERSION", "")
VISITS = int(E("VISITS", 400))
TEMP_PLIES = int(E("TEMP_PLIES", 20))
OPENING_FRAC = float(E("OPENING_FRAC", 0.6))
MAX_PLIES = int(E("MAX_PLIES", 200))
BATCH = int(E("BATCH", 48))
ROOT_NOISE = float(E("ROOT_NOISE", 0.25))
NOISE_ALPHA = float(E("NOISE_ALPHA", 10.83))
# Keep the search's targets beside the FENs. They are cheap and are the one thing here that can
# outgrow the teacher.
TARGETS = E("TARGETS", "1") == "1"
TARGETS_OUT = E("TARGETS_OUT", paths.under("selfplay", "targets.jsonl"))
# How often to look for a newer checkpoint, in games. Between games only, so a game is never
# played by two different models.
RELOAD_GAMES = int(E("RELOAD_GAMES", 25))
OUT = E("OUT", paths.under("selfplay", "positions.txt"))
SHARDS = E("SHARDS", paths.under("shards"))
TARGET = int(float(sys.argv[1])) if len(sys.argv) > 1 and sys.argv[1].replace(".", "").isdigit() \
    else int(float(E("POSITIONS", 20000)))


def ckpt_dir():
    """Where the trainer writes. The launched configuration (autostart.json) names CKPT_DIR, so a
    restart by the supervisor finds the same weights as a start by hand. Falls back to CKPT_DIR
    from the environment, then to this folder.
    """
    try:
        with open(os.path.join(HERE, "autostart.json"), encoding="utf-8-sig") as fh:
            got = json.load(fh).get("env", {}).get("CKPT_DIR")
        if got:
            return got
    except (OSError, ValueError, AttributeError):
        pass
    return E("CKPT_DIR", HERE)


def model_path():
    """Where the weights come from, a named frozen version or the current training checkpoint.

    `ckpt_{TAG}.pt` is the best-validation checkpoint and only changes when the score improves.
    `ckpt_{TAG}_last.pt` is what the trainer has right now. For DAgger the positions should come from
    the weights currently training, so the live file is preferred.
    """
    if VERSION:
        return VERSION if os.path.isabs(VERSION) else os.path.join(
            E("VERSION_DIR", paths.under("versions")), VERSION)
    d = ckpt_dir()
    live = os.path.join(d, f"ckpt_{TAG}_last.pt")
    return live if os.path.exists(live) else os.path.join(d, f"ckpt_{TAG}.pt")


def resolve_cfg(ck):
    """The architecture for a checkpoint, its own `cfg`, else the launched configuration.

    The live `_last` checkpoint carries no `cfg`. Falling back to hard-coded defaults would silently
    build the wrong architecture if D or LAYERS changed, and `strict=False` would hide it. Kept apart
    from `load_model` so it can be tested without weights on disk.
    """
    c = dict(ck.get("cfg") or {})
    if c:
        return c
    try:
        with open(os.path.join(HERE, "autostart.json"), encoding="utf-8-sig") as fh:
            env = json.load(fh).get("env", {})
        for key, cast, name in (("D", int, "d"), ("LAYERS", int, "layers"),
                                ("NHEAD", int, "nhead"), ("FF_MULT", float, "ff_mult")):
            if env.get(key):
                c[name] = cast(env[key])
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return c


def load_model(device):
    """A frozen version if one is named, otherwise whatever the live run has written.

    The path comes from `model_path` and is not recomputed here, so the reload check and the loader
    always look at the same file.
    """
    p = model_path()
    ck = torch.load(p, map_location=device)
    c = resolve_cfg(ck)
    m = ChessNet(d=c.get("d", 1024), nhead=c.get("nhead", 16), layers=c.get("layers", 16),
                 dim_ff=int(c.get("ff_mult", 2.0) * c.get("d", 1024)),
                 qhead=bool(c.get("qhead", "qvalue.weight" in ck["model"])),
                 att=bool(c.get("att", "att.q.weight" in ck["model"])),
                 dist=int(c.get("dist", 64 if "vdist.weight" in ck["model"] else 0)),
                 mlh=bool(c.get("mlh", "mleft.0.weight" in ck["model"]))).to(device)
    missing, _ = m.load_state_dict(ck["model"], strict=False)
    m.eval()
    label = os.path.basename(p)
    print(f"{label}: {m.num_params()/1e6:.1f}M params, iteration {ck.get('iter', '?'):,}"
          + (f", version {ck['version']}" if "version" in ck else ""), flush=True)
    if missing:
        print(f"  {len(missing)} tensor(s) at their init: {', '.join(missing)}", flush=True)
    return m


def random_corpus_board(rng):
    """A position from the training corpus, so games do not all begin in the opening.

    The board is rebuilt in the mover's frame, as the corpus stores it, so it is always white to move.
    The network only sees that frame anyway.

    The row's castling rights, en passant square and 50-move clock are restored too, in the same frame
    (pack_av.encode_av stores castling as bits for mine short, mine long, theirs short, theirs long, en
    passant as the square plus one, and the clock as hm). Without them every corpus start would lose
    its castling rights.
    """
    from pack_av import open_shard, unpack_ids
    paths = random_corpus_board.paths
    if paths is None:
        paths = random_corpus_board.paths = sorted(glob.glob(os.path.join(SHARDS, "av_*.bin")))
    if not paths:
        return chess.Board()
    PIECE = {1: chess.PAWN, 2: chess.KNIGHT, 3: chess.BISHOP, 4: chess.ROOK,
             5: chess.QUEEN, 6: chess.KING}
    for _ in range(30):
        mm = open_shard(rng.choice(paths))
        if not len(mm):
            continue
        rec = mm[rng.randrange(len(mm))]
        row = unpack_ids(rec["ids4"])[0]
        b = chess.Board(None)
        for sq in range(64):
            v = int(row[sq])
            if v:
                b.set_piece_at(sq, chess.Piece(PIECE[v if v <= 6 else v - 6],
                                               chess.WHITE if v <= 6 else chess.BLACK))
        b.turn = chess.WHITE
        cast = int(rec["cast"])
        b.castling_rights = ((chess.BB_H1 if cast & 1 else 0) | (chess.BB_A1 if cast & 2 else 0)
                             | (chess.BB_H8 if cast & 4 else 0) | (chess.BB_A8 if cast & 8 else 0))
        b.castling_rights = b.clean_castling_rights()
        ep = int(rec["ep"])
        b.ep_square = ep - 1 if ep else None
        b.halfmove_clock = int(rec["hm"])
        if not b.is_valid():
            b.ep_square = None          # a square that does not fit the board costs the square, not the row
        if b.is_valid() and not b.is_game_over() and b.legal_moves.count() > 2:
            return b
    return chess.Board()


random_corpus_board.paths = None


def with_noise(tree, rng):
    """This process's tree with AlphaZero's root noise switched on (ROOT_NOISE, and 0 leaves it off).

    The noise has its own generator, seeded from the game stream's, so a run is repeatable from its
    seed and two generators never share a noise sequence. It is set here and not in any search table,
    so a sealed search applied with tree.apply leaves it alone.
    """
    tree.root_noise = ROOT_NOISE
    tree.noise_alpha = NOISE_ALPHA
    tree.noise_rng = np.random.default_rng(rng.randrange(1 << 62))
    return tree


def play(model, device, rng, tree, rows=None):
    """One game. Returns the positions it passed through, as FENs.

    With `rows` it also keeps what the search thought, as in AlphaZero and Leela. The policy target
    is the root visit distribution and the value target is the game outcome, which lets a network
    exceed its teacher, whereas labelling FENs with a teacher (DAgger) cannot.

    Kept per position are the visit counts over legal moves, the per-move values, the root value,
    the side to move and the ply. The game result is stamped onto every row from that side's view.
    """
    b = chess.Board() if rng.random() < OPENING_FRAC else random_corpus_board(rng)
    seen = []
    mine = []
    tree.reset()
    for ply in range(MAX_PLIES):
        if b.is_game_over() or not list(b.legal_moves):
            break
        seen.append(b.fen())
        mv, info = tree.search(b, visits=VISITS)
        if mv is None:
            break
        if rows is not None:
            root = getattr(tree, "root", None)
            n = root.N.astype(np.float64) if root is not None and root.N is not None else None
            if n is not None and n.sum() > 0:
                # The visit share is the policy target and the per-move mean value is the
                # search's own target for the action-value head.
                w = root.W / np.maximum(root.N, 1)
                live = np.flatnonzero(n > 0)
                mine.append({"fen": b.fen(), "ply": ply,
                             "stm": "w" if b.turn == chess.WHITE else "b",
                             "q": round(float(info.get("q", 0.0)), 4),
                             "n": {root.moves[i].uci(): int(n[i]) for i in live},
                             "qa": {root.moves[i].uci(): round(float(w[i]), 4) for i in live}})
        if ply < TEMP_PLIES and info.get("top"):
            # sample from the visit counts, which is what makes a thousand games different
            names = [t[0] for t in info["top"]]
            counts = np.array([t[1] for t in info["top"]], dtype=np.float64)
            if counts.sum() > 0:
                pick = rng.choices(names, weights=(counts / counts.sum()).tolist())[0]
                mv = chess.Move.from_uci(pick)
        # Only this ply's own row. A proven move (mate in one, a tablebase win) plays without a
        # row, and writing `played` onto the previous row would give it a move that is illegal there.
        if mine and mine[-1]["fen"] == b.fen():
            mine[-1]["played"] = mv.uci()
        b.push(mv)
    if rows is not None:
        # The outcome from each position's own side. A game cut off at MAX_PLIES has no result, so
        # z stays null and the row still carries the search's value, leaving the choice to the trainer.
        res = b.result(claim_draw=True) if b.is_game_over(claim_draw=True) else "*"
        z = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(res)
        gid = format(int(time.time() * 1000), "x") + format(rng.randrange(1 << 16), "04x")
        for r in mine:
            r["z"] = None if z is None else (z if r["stm"] == "w" else 1.0 - z)
            r["result"] = res
            r["game"] = gid
        rows.extend(mine)
    return seen


if __name__ == "__main__":
    from pidfile import claim
    claim("selfplay")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model = load_model(device)
    loaded_at = os.path.getmtime(model_path()) if os.path.exists(model_path()) else 0.0
    tree = MCTS(model, device, batch=BATCH)
    if VERSION:
        # a named frozen version plays the search it was sealed with (castle_prior "rook" for v4)
        from mcts import pinned_search
        tree.apply(pinned_search(os.path.basename(VERSION)))
    have = 0
    if os.path.exists(OUT):
        have = sum(1 for _ in open(OUT, encoding="utf-8"))
        print(f"{have:,} positions already written, continuing", flush=True)
    at_start = have
    # A restart must not replay what is already written. A sealed search on a frozen network is
    # deterministic, so the seed includes the count on disk, which only grows.
    rng = random.Random(f"{int(E('SEED', 0))}/{have}")
    with_noise(tree, rng)
    print(f"target {TARGET:,} positions at {VISITS} visits a move, "
          f"{100*OPENING_FRAC:.0f}% from the opening, root noise {ROOT_NOISE:g} "
          f"(alpha {NOISE_ALPHA:g} over the legal moves)\n", flush=True)

    t0 = time.time()
    games = 0
    f = open(OUT, "a", encoding="utf-8", buffering=1)
    try:
        while have < TARGET:
            rows = [] if TARGETS else None
            fens = play(model, device, rng, tree, rows)
            for fen in fens:
                f.write(fen + "\n")
            if rows:
                with open(TARGETS_OUT, "a", encoding="utf-8") as tf:
                    for r in rows:
                        tf.write(json.dumps(r) + chr(10))
            have += len(fens)
            games += 1
            # Follow the trainer, so the positions come from the current model. Checked between
            # games only, so a game is played by one model. A named VERSION is never reloaded.
            if not VERSION and games % RELOAD_GAMES == 0:
                try:
                    mt = os.path.getmtime(model_path())
                    if mt > loaded_at + 1.0:
                        model = load_model(device)
                        tree = with_noise(MCTS(model, device, batch=BATCH), rng)
                        loaded_at = mt
                        print(f"  picked up a newer checkpoint after {games:,} games", flush=True)
                except OSError as e:
                    print(f"  could not check the checkpoint: {type(e).__name__}: {e}", flush=True)
            if games % 10 == 0:
                el = time.time() - t0
                # this run's positions over this run's seconds, not the cumulative total
                made = have - at_start
                print(f"  {games:,} games, {have:,} positions "
                      f"({100*have/TARGET:.1f}%), {made/max(el,1):.1f} a second, "
                      f"{el/3600:.2f}h", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        f.close()
    print(f"\n{have:,} positions from {games:,} games in {(time.time()-t0)/3600:.2f}h")
    print(f"written to {OUT}")
