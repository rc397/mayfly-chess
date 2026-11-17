r"""Batched PUCT tree search, MayFly's main search.

The algorithm is AlphaZero's PUCT, written from the published description.

    pick the child maximising   Q(s,a) + c * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

  batching      descents are collected until there are BATCH leaves, then evaluated in one forward
                pass, since one position at a time leaves the GPU almost idle.
  virtual loss  each descent subtracts a notional loss on the way down, pushing the next descent
                elsewhere, and it is undone during backup.
  collisions    a descent that lands on a position already queued is abandoned and its virtual loss
                unwound rather than evaluated twice.
  reuse         the subtree under the move played becomes the next root.
  repetition    each node keeps its position key and the game's history is walked once per search,
                so a line that returns to a position already seen scores a draw.

A transposition cache sits in front of the network, so a revisited position costs a dictionary
lookup instead of a forward pass.

  from mcts import MCTS
  s = MCTS(model, device="cuda", visits=800)
  move, info = s.search(board)

  python chess/mcts.py bench     nodes per second against batch size
  python chess/mcts.py tactics   mate in one, which the raw policy head fails
"""
import gc, os, sys, math, time, json, threading
import numpy as np
import chess
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import board_to_ids, move_to_idx, clock_and_ep
from encode import encode_keys
import paths

E = os.environ.get

# The compiled pick. native/pick.c does _pick_fast's arithmetic in one pass with the same float32
# numbers bit for bit (harness.py checks it). It is built as mayfly_pick.<abi>.pyd (or .so) beside
# this file. Without it, or with MCTS_NATIVE=0, the numpy lines run. Which versions use it is the
# sealed setting native_pick.
NATIVE_PICK = None
if E("MCTS_NATIVE", "1") == "1":
    try:
        from mayfly_pick import pick as NATIVE_PICK
    except ImportError:
        NATIVE_PICK = None

# Whether this GPU has room for the autocast graphs (see autocast_graph in the constructor). There is
# one graph per batch size, each holding GPU memory. Which versions may use them is the sealed setting.
AUTOCAST_GRAPH_OK = E("AUTOCAST_GRAPH", "0") == "1"

# The cyclic garbage collector is kept off during a search (timing_fixes). The tree has no reference
# cycles (Node has no parent pointer), so reference counting frees everything and the collector only
# scans, costing 1 to 12 percent of tree time and pauses of up to a second on a big tree. A search
# holds it off and freezes what is alive at its start and end, and reset() collects once between
# games. It is a count, so several trees searching in threads keep it off until the last finishes.
# Garbage collection never changes a number, so the moves are the same.
_GC_LOCK = threading.Lock()
_GC_HELD = [0, True]      # searches holding it off, and whether it was on before the first


def _gc_hold():
    with _GC_LOCK:
        if _GC_HELD[0] == 0:
            _GC_HELD[1] = gc.isenabled()
            gc.disable()
        _GC_HELD[0] += 1
        gc.freeze()


def _gc_release():
    with _GC_LOCK:
        gc.freeze()
        _GC_HELD[0] -= 1
        if _GC_HELD[0] == 0 and _GC_HELD[1]:
            gc.enable()


def _bf16_copies(model):
    """bf16 copies of exactly the weights autocast casts for a matmul.

    That is every nn.Linear weight and bias (the heads, MultiheadAttention's out_proj, cast) and
    MultiheadAttention's packed in_proj weight and bias, each rounded once with .to(bfloat16) as
    autocast does. LayerNorm, Embedding, cls and ep_mark stay fp32, as autocast leaves them."""
    out = {}
    for mname, mod in model.named_modules():
        names = (("weight", "bias") if isinstance(mod, torch.nn.Linear) else
                 ("in_proj_weight", "in_proj_bias") if isinstance(mod, torch.nn.MultiheadAttention) else ())
        for pn in names:
            p = getattr(mod, pn, None)
            if p is not None:
                out[f"{mname}.{pn}" if mname else pn] = p.detach().to(torch.bfloat16)
    return out


# The compiled move generator (native/movegen.c, mayfly_movegen, under timing_fixes). It gives legal
# moves, policy index rows, check and insufficient-material flags, the batch encoder and the forcing
# mask, each matching python-chess and encode_keys. It loads the first time a timing_fixes search asks
# and is checked against python-chess on the positions below, their children and a seeded random game.
# Any disagreement leaves the search on python-chess. MCTS_NATIVE=0 switches it off.
_NATIVE_MOVES = [None, False]         # the module once checked, and whether the check has run
_NATIVE_LOCK = threading.Lock()
_NATIVE_CHECK_FENS = (
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/8/3p4/KPp4r/1R2Pp1k/8/6P1/8 b - e3 0 1",
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
)


def _native_agrees(mod):
    """Every move, its order, index row, flags, forcing bit and encoding, against python-chess."""
    import random
    boards = []
    for fen in _NATIVE_CHECK_FENS:
        b = chess.Board(fen)
        boards.append(b)
        for m in list(b.legal_moves):
            c = b.copy(stack=False)
            c.push(m)
            boards.append(c)
    rng, b = random.Random(20261002), chess.Board()
    for _ in range(400):
        moves = list(b.legal_moves)
        if not moves or b.halfmove_clock >= 100:
            b = chess.Board()
            continue
        b.push(rng.choice(moves))
        boards.append(b.copy(stack=False))
    for b in boards:
        key = b._transposition_key()
        mv, ix, fl = mod.legal(key)
        ref = list(b.legal_moves)
        flip = b.turn == chess.BLACK
        if (mv != ref or ix.dtype != np.int64 or list(ix) != [move_to_idx(m, flip) for m in ref]
                or bool(fl & 1) != b.is_check() or bool(fl & 2) != b.is_insufficient_material()
                or list(mod.forcing(key)) != [int(b.is_capture(m) or b.gives_check(m)) for m in ref]):
            return False
        ids, cast, hm, ep = mod.encode([key], [b.halfmove_clock])
        rids, rcast, rhm, rep = encode_keys([key], [b.halfmove_clock])
        if not (np.array_equal(ids, rids) and ids.dtype == np.asarray(rids).dtype
                and np.array_equal(cast, rcast) and cast.dtype == np.asarray(rcast).dtype
                and list(hm) == list(rhm) and list(ep) == list(rep)):
            return False
    return True


def _native_moves():
    """mayfly_movegen ready to use, or None (missing, MCTS_NATIVE=0, or it disagreed with python-chess)."""
    if _NATIVE_MOVES[1]:
        return _NATIVE_MOVES[0]
    with _NATIVE_LOCK:
        if _NATIVE_MOVES[1]:
            return _NATIVE_MOVES[0]
        mod = None
        if E("MCTS_NATIVE", "1") == "1":
            try:
                import mayfly_movegen as mod
                promos = (None, None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
                tab, iw, ib = [], [], []
                for p in range(6):                       # code = promotion * 4096 + from * 64 + to
                    for f in range(64):
                        for t in range(64):
                            m = chess.Move(f, t, promos[p])
                            tab.append(m)
                            iw.append(move_to_idx(m, False))
                            ib.append(move_to_idx(m, True))
                mod.init(tab, iw, ib, list(chess.BB_KNIGHT_ATTACKS), list(chess.BB_KING_ATTACKS),
                         list(chess.BB_PAWN_ATTACKS[0]), list(chess.BB_PAWN_ATTACKS[1]),
                         [chess.BB_RAYS[a][b] for a in range(64) for b in range(64)])
                if not _native_agrees(mod):
                    print("[mcts] the compiled move generator disagrees with python-chess; python-chess it is",
                          file=sys.stderr, flush=True)
                    mod = None
            except Exception as e:
                print(f"[mcts] no compiled move generator ({type(e).__name__}: {e}); python-chess it is",
                      file=sys.stderr, flush=True)
                mod = None
        _NATIVE_MOVES[0], _NATIVE_MOVES[1] = mod, True
        return mod


# The compiled walk (native/walk.c, mayfly_walk, under timing_fixes). It runs _descend, _unwind and
# _backup in C over the same Node objects and float32 rows, each number the one numpy computes. It is
# used only where it is exact, the descent for _pick_fast's selection with a virtual mean and no graph
# search, and the backup at power_p 1 with proofs off and no moves-left sample. MCTS_NATIVE=0 switches
# it off. Its expand_rows does _expand's softmax prior and five zero rows. That relies on float64 exp
# matching numpy's (true without AVX-512) and numpy's pairwise sum, so it is compared with numpy when
# the module loads and left unused (_NATIVE_PRIOR None) if anything differs.
_NATIVE_WALK = [None, False]
_NATIVE_PRIOR = [None]


def _prior_agrees(mod):
    """expand_rows against _expand's numpy lines on random rows, and its exp and sum on their own."""
    rng = np.random.default_rng(20261002)
    x = -np.abs(rng.standard_normal(20000)) * rng.choice([0.5, 4.0, 16.0, 60.0], 20000)
    if not np.array_equal(np.exp(x), mod.check_exp(x)):
        return False
    for n in list(range(1, 300)) + [511, 1024, 4162]:
        a = rng.standard_normal(n) * 5.0
        e = np.exp(a - a.max())
        if float(np.add.reduce(a)) != mod.check_sum(a) or float(e.sum()) != mod.check_sum(e):
            return False
    for _ in range(3000):
        k = int(rng.integers(1, 120))
        lg = (rng.standard_normal(4162) * float(rng.choice([0.5, 2.0, 6.0, 14.0]))).astype(np.float32)
        idx = rng.choice(4162, k, replace=False).astype(np.int64)
        for temp in (1.0, 0.889688):
            z = lg[idx].astype(np.float64)
            if temp != 1.0:
                z /= temp
            z -= z.max()
            p = np.exp(z)
            got = mod.expand_rows(lg, idx, temp, k)
            if (got is None or got[0].dtype != np.float32
                    or not np.array_equal(got[0], (p / p.sum()).astype(np.float32))
                    or any(r.dtype != np.float32 or r.shape != (k,) or r.any() for r in got[1:])):
                return False
    return True


def _native_walk():
    """mayfly_walk ready to use (Node's slots read), or None (missing, or MCTS_NATIVE=0)."""
    if _NATIVE_WALK[1]:
        return _NATIVE_WALK[0]
    with _NATIVE_LOCK:
        if _NATIVE_WALK[1]:
            return _NATIVE_WALK[0]
        mod = None
        if E("MCTS_NATIVE", "1") == "1":
            try:
                import mayfly_walk as mod
                mod.setup(Node)
                if _prior_agrees(mod):
                    _NATIVE_PRIOR[0] = mod.expand_rows
                else:
                    print("[mcts] the compiled prior disagrees with numpy on this machine; numpy's it is",
                          file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[mcts] no compiled walk ({type(e).__name__}: {e}); the Python walk it is",
                      file=sys.stderr, flush=True)
                mod = None
                _NATIVE_PRIOR[0] = None
        _NATIVE_WALK[0], _NATIVE_WALK[1] = mod, True
        return mod

# Virtual loss, from a sweep over c_puct, vl and fpu (24 positions at 400 visits). Raising vl from
# 1.0 to 2.0 halves the collisions (0.8 to 1.3 an eval down to about 0.4) and makes them insensitive
# to fpu. Move quality was flat across the sweep, so this buys a tidier search, not better moves.
VIRTUAL_LOSS = 2.0

# Material only, side to move first. Deliberately not a real evaluation. It only has to notice that
# a leaf hangs something, which is where the value head is weakest.
_PIECE_VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.25,
              chess.ROOK: 5.0, chess.QUEEN: 9.0}


def _material(board):
    t = board.turn
    s = 0.0
    for pt, v in _PIECE_VAL.items():
        s += v * (len(board.pieces(pt, t)) - len(board.pieces(pt, not t)))
    return s


def policy_index(board, moves, flip, castle="king"):
    """Where the policy head keeps each of moves, in the side to move's frame.

    Every move is move_to_idx(m, flip) except, under castle="rook", a legal castle, which is read
    at the king takes rook index its training corpus used, O-O at e1h1 (263) and O-O-O at e1a1
    (256), for either colour since the frame is flipped for black. "king" reads python-chess's
    king destination, e1g1 (262) and e1c1 (258), as versions before v4 did. The action-value head
    is never read through this, since ChessBench names castling by the king's destination."""
    if castle not in ("king", "rook"):
        raise ValueError(f"unknown castle_prior {castle!r}")
    idx = np.fromiter((move_to_idx(m, flip) for m in moves), np.int64, len(moves))
    if castle == "rook" and board.castling_rights:
        for i, m in enumerate(moves):
            if board.is_castling(m):
                file = 7 if board.is_kingside_castling(m) else 0
                rook = chess.square(file, chess.square_rank(m.from_square))
                idx[i] = move_to_idx(chess.Move(m.from_square, rook), flip)
    return idx


# Monte-Carlo graph search (Czech, Korus and Kersting, "Monte-Carlo Graph Search for AlphaZero").
# Most of the saving comes from the early stop below. A transposition whose edge already agrees with
# the shared node to within Q_EPS is left alone.
Q_EPS = float(E("Q_EPS", 0.01))
# Power mean backup (Dam et al., "Generalized Mean Estimation in Monte-Carlo Tree Search"). p = 1 is
# the arithmetic mean, standard MCTS, and p towards infinity is the maximum.
POWER_P = float(E("POWER_P", 1.0))

# Every released version is sealed with the search settings it had at release and never changes, so
# old versions stay fixed opponents and improvements go into the next version. Only settings that
# change which move comes back are sealed. How descents are packed into batches does change moves
# (it decides which descents see virtual loss), so fill_once is sealed too.
SEARCH_KEYS = ("c_puct", "fpu", "vl", "policy_temp", "value_from_q",
               "power_p", "mcgs", "qdepth", "qweight", "adaptive", "early_stop", "calibrate",
               "tree_policy", "c1", "c2", "fpu_root", "fpu_dynamic", "fpu_gain", "time_scale",
               "gumbel_m", "gumbel_c_visit", "gumbel_c_scale",
               "root_mate", "tb_dtz", "ponder_discount",
               # from v4 onwards, each sealed off for v1 to v3
               "lcb", "lcb_floor", "lcb_alpha", "virtual_mean", "proofs", "tb_rule50",
               "draw_plies", "q_init", "q_n0", "fpu_base", "root_check", "root_check_margin",
               "root_check_share",
               # Which heads the tree reads. From v4 the network also has heads that score a move
               # from its two square tokens, which changes every prior and leaf value.
               "policy_head", "q_head",
               # Time management, under a clock only.
               "tm_extend", "tm_stop", "tm_margin", "tm_extend_max", "tm_parity", "tm_deep",
               "tm_open_spend", "tm_open_plies",
               # the opening book
               "book", "book_plies",
               # varied openings, sampling the root by visit share for the first few plies
               "open_temp", "open_plies", "open_floor", "open_margin", "collide",
               # whether a pass served entirely from the eval cache ends the search
               "cache_progress",
               # Correctness fixes that change moves, each sealed at what a version was released with.
               # See the constructor for what each one does.
               "castle_prior", "discount_q0", "cache_clock", "prior_fallback", "fill_once",
               "use_tb",
               # each off by default and sealed off for v1 to v4
               "tm_catchup", "r50_damp", "root_temp",
               # the moves-left head in the tree (Lc0's MLH utility), off unless the network has the
               # head and mlh is on, gated on this network's value scale (mlh_threshold)
               "mlh", "mlh_threshold", "mlh_slope", "mlh_cap", "mlh_const", "mlh_lin", "mlh_sq",
               # a mate is worth a little less the further away it is
               "mate_dist",
               # two batches in flight, the GPU running one while the CPU gathers the next
               "pipeline",
               # growth of the exploration term with the parent's visits (lc0's CPuctFactor)
               "explore_growth", "explore_base",
               # puct_v's spread from completed visits only, sealed off for v1 to v4
               "sigma_done",
               # Guards for when the q head undervalues winning captures and checks. Sealed off for
               # v1 to v4.
               "q_floor", "forcing", "forcing_after",
               # forced visits kept out of the root's statistics, sealed off for v1 to v4
               "forcing_apart",
               # The compiled pick. The same move at any visit count, but on a clock it means more
               # visits and so different moves. Sealed off for v1 to v4.
               "native_pick",
               # The autocast graph. The same numbers with fewer launches, so more visits on a
               # clock. Sealed off for v1 to v4.
               "autocast_graph",
               # move-neutral timing fixes, one switch for all, each noted where it acts. Sealed off
               # for v1 to v4.
               "timing_fixes",
               # 1+0 planned at its own network price, under a clock with no increment only. Sealed
               # off for v1 to v5.
               "tm_short_sudden",
               # One host-to-GPU copy in and one GPU-to-host copy out a pass, the legal-move slice
               # only. The same numbers, so speed only. Sealed off for v1 to v5, on for v6.
               "sw25_one_copy",
               # Which search runs, "python" (this file) or "cpp" (mcts_cpp.py, C++ tree and network).
               # v6 runs on "cpp", v1 to v5 are sealed "python".
               "engine",
               # With engine "cpp", run the network on chess/cudanet (a fingerprinted .cnw beside the
               # version) instead of the LibTorch trace. Not the same numbers (fp16 products), so
               # sealed off for v1 to v6.
               "cuda_net")

class _Cut:
    """A policy row already cut to one node's legal moves on the GPU (sw25_one_copy).

    _collect hands it to _expand and the cache in place of a (4,162) row. It owns its float32 data."""
    __slots__ = ("row",)

    def __init__(self, row):
        self.row = row


def _oc_layout(B, L, W, sep):
    """Offsets (in int64 words) of one sw25_one_copy input blob, and the total.

    The blob holds ids (B, L), cast (B, 4) float32 as B*2 words, hm (B), ep (B), I (B, W), Ip (B, W)
    only if it differs from I, and M (B, W) as 0 or 1."""
    o_ids = 0
    o_cast = o_ids + B * L
    o_hm = o_cast + B * 2
    o_ep = o_hm + B
    o_I = o_ep + B
    o_Ip = o_I + B * W
    o_M = o_Ip + (B * W if sep else 0)
    return o_ids, o_cast, o_hm, o_ep, o_I, o_Ip, o_M, o_M + B * W


# The calibration stretch. value_calib.py compares the leaf value with Stockfish on 3,000 positions.
# The quiet band was compressed into a small part of the model's range, so Q barely moved against U
# and the search re-sampled its own prior. The map is quantile matching rather than a regression,
# which would compress that band further. It is monotone, so it never reorders siblings, it only
# changes how strongly Q argues against U.
CALIB_FILE = os.path.join(HERE, "value_calib.json")
_CALIB = None


def calibration(path=None):
    """The (xs, ys) knots of the stretch, or () if there is no map on disk."""
    global _CALIB
    if _CALIB is None or path:
        try:
            with open(path or CALIB_FILE, encoding="utf-8-sig") as fh:
                d = json.load(fh)
            xs = np.asarray(d["xs"], np.float64)
            ys = np.asarray(d["ys"], np.float64)
            got = (xs, ys) if len(xs) >= 2 and bool(np.all(np.diff(xs) > 0)) else ()
        except (OSError, ValueError, KeyError):
            got = ()
        if path:
            return got
        _CALIB = got
    return _CALIB

# The search for versions frozen before settings were recorded automatically. v1 predates the
# virtual loss change. Later versions record their own and never consult this table.
PINNED = {
    1: {"vl": 1.0},
}


_T_ISF = None


def _t_quantile(alpha, df):
    """Student-t upper quantile, P(T > t) = alpha, elementwise over df.

    scipy is imported on first use, so a tree that never asks for a lower confidence bound never
    loads it."""
    global _T_ISF
    if _T_ISF is None:
        from scipy.stats import t as _student
        _T_ISF = _student.isf
    return np.asarray(_T_ISF(alpha, df), np.float64)


def versions_on_disk(vdir=None):
    """Every frozen version's manifest row.

    Only OSError and bad JSON are tolerated, so a real error is never mistaken for "no sealed
    search"."""
    vdir = vdir or E("VERSION_DIR", paths.under("versions"))
    try:
        with open(os.path.join(vdir, "versions.json"), encoding="utf-8-sig") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return []


def pinned_search(version, vdir=None):
    """The search a version plays with, the block it was sealed with at release.

    A version is sealed when it is frozen and never changes afterwards. Improvements go into the
    next version. A version on disk with no sealed block is a failed freeze. It gets today's
    defaults with a warning, and unit_every_sealed_version_pins_every_setting fails on it.
    """
    if version is None:
        return {}
    for r in versions_on_disk(vdir):
        if r.get("file") == version or r.get("version") == version:
            got = r.get("search")
            if got:
                return {k: v for k, v in got.items() if k in SEARCH_KEYS}
            num = int(r.get("version", 0))
            if num in PINNED:
                return dict(PINNED[num])
            print(f"  WARNING: MayFly v{num} has no sealed search and is playing today's "
                  f"defaults", file=sys.stderr, flush=True)
            return {}
    return {}


def default_search():
    """Today's defaults, as a dict, for recording into a version at freeze time.

    Read off a real tree rather than retyped, so the record always has every key. apply leaves a
    missing key at today's default, so a missing key would let a sealed version drift.
    """
    return MCTS(None, "cpu").settings()


# Syzygy tablebases. With few enough pieces the result is known, so those nodes become terminal
# with the true result instead of a value head guess. One handle for the whole process, behind a
# lock, because python-chess's reader is not documented as thread safe and the board runs a search
# per player. Probes are cached, since an endgame probes the same positions many times a search.
TB_PATH = E("SYZYGY_DIR", paths.under("syzygy"))
TB_PIECES = int(E("TB_PIECES", 6))
_TB = None
_TB_TRIED = False
_TB_LOCK = threading.Lock()
# The caches hold raw table results (wdl, and dtz when needed), not verdicts. A version that reads
# the fifty-move counter values a position differently from one that does not, and both can run in
# one process, so each tree turns the raw result into a value under its own rule.
_TB_CACHE = {}
_DTZ_CACHE = {}


def tablebase():
    """The shared reader, opened on first use. Returns None if there are no tables."""
    global _TB, _TB_TRIED
    if _TB_TRIED:
        return _TB
    with _TB_LOCK:
        if not _TB_TRIED:
            try:
                import chess.syzygy
                _TB = chess.syzygy.open_tablebase(TB_PATH)
                # stderr, never stdout. The tables open lazily inside a search, and under uci.py
                # stdout is the UCI protocol.
                print(f"[mcts] syzygy open at {TB_PATH}, exact play under {TB_PIECES+1} pieces",
                      file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[mcts] no syzygy ({type(e).__name__}: {e}), the network plays endgames",
                      file=sys.stderr, flush=True)
                _TB = None
            # Set last, so a thread that reads _TB_TRIED outside the lock finds the tables already
            # open, not the None _TB holds while they are being opened.
            _TB_TRIED = True
    return _TB


class Node:
    __slots__ = ("board", "moves", "idx", "P", "N", "W", "V2", "kids", "term", "sum_n",
                 "pending", "value", "key", "lo", "hi", "Q0", "F", "X", "q0w", "tempered",
                 "m0", "ML", "MN", "FM", "IX")

    def __init__(self, board, draw_plies=150, native=None):
        self.board = board
        self.key = board._transposition_key()
        # Generating the legal moves once and reading the game state off them halves move
        # generation, the largest CPU cost in the tree once the network is batched. With the
        # compiled generator (native is mayfly_movegen, else None) one call gives the same moves in
        # python-chess's order, each move's policy index (IX), and the check and insufficient
        # material flags.
        flags = None
        if native is not None:
            try:
                self.moves, self.IX, flags = native.legal(self.key)
            except (OverflowError, TypeError):
                flags = None
        if flags is None:
            self.moves = list(board.legal_moves)
            self.IX = None
        self.idx = self.P = self.N = self.W = self.kids = None
        # which moves are captures or checks, worked out the first time `forcing` asks
        self.FM = None
        # sum of v^2 backed up through each action, for the variance-aware policies. Allocated
        # with the rest at expansion, None until then, like N and W.
        self.V2 = None
        # In flight, per edge, how many descents are waiting on the network below it (F) and the
        # provisional values they added to W (X). Selection never reads them. They let a provisional
        # value be taken back exactly, and let a proof tell backed-up visits from waiting ones.
        self.F = self.X = None
        # The q head's value of every move at expansion, 2*sigmoid-1 from this node's side, kept for
        # q_init and root_check.
        self.Q0 = None
        # How many of q_n0's visits Q0 is worth at this node, as a multiple of q_n0. 1.0 except at
        # a root the ponder discount has scaled, where it is scaled with N and W (discount_q0).
        self.q0w = 1.0
        # the root temperature this node's priors were sharpened or flattened with (root_temp), so
        # a root searched twice, phase one and phase two or a ponder carried over, is not done twice
        self.tempered = 1.0
        # Moves left (mlh). The head's estimate of plies to the end at expansion, and per edge the
        # sum and count of moves left backed up through it, measured from this node (so a child's
        # own figure is ML/MN - 1).
        self.m0 = self.ML = self.MN = None
        self.sum_n = 0
        self.pending = False
        self.value = 0.0
        if not self.moves:
            self.term = -1.0 if ((flags & 1) if flags is not None else board.is_check()) else 0.0
        # 150 plies is the seventy-five-move rule, which v1 to v3 were sealed with. v4 draws at 100,
        # where a draw is normally claimed.
        elif (((flags & 2) if flags is not None else board.is_insufficient_material())
              or board.halfmove_clock >= draw_plies):
            self.term = 0.0
        else:
            self.term = None
        # Proven bounds on this node's true value from its side to move, lo <= value <= hi. Only a
        # game the rules have ended is exact at birth. A repetition is scored 0 by convention and
        # stays unproven, and a tablebase hit is made exact where it is created, only when the
        # probe respected the fifty-move counter.
        self.lo, self.hi = (self.term, self.term) if self.term is not None else (-1.0, 1.0)

    @property
    def expanded(self):
        return self.P is not None


class MCTS:
    def __init__(self, model, device=None, visits=800, batch=48, c_puct=1.75, fpu=0.33,
                 cache_size=400_000, policy_temp=0.889688, value_from_q=True):
        self.model = model
        self.device = str(device or next(model.parameters()).device)
        self.visits, self.batch = visits, batch
        self.c_puct, self.fpu, self.policy_temp = c_puct, fpu, policy_temp
        self.value_from_q = value_from_q
        self.power_p = float(E("POWER_P", POWER_P))
        # The graph, key -> node, where the key carries the ply as well as the position. Without
        # the ply a repetition would make the graph cyclic. With it the graph is a DAG, as in the
        # paper. mcgs is an integer, 0 off, 1 the paper's merge with the early stop, 2 the merge
        # that keeps descending. False and True in versions.json read as 0 and 1.
        self.mcgs = int(E("MCGS", "0"))
        self.tt = {}
        self.tt_cap = int(E("TT_CAP", 400_000))
        # A short captures-only search at each leaf, material only. Off by default, since it costs
        # real Python time per leaf.
        self.qdepth = int(E("LEAF_QSEARCH", 0))
        self.qweight = float(E("LEAF_QWEIGHT", 0.5))
        # Spend visits where the position is unclear, and stop early when the answer is settled.
        self.adaptive = E("ADAPTIVE_VISITS", "0") == "1"
        self.early_stop = E("EARLY_STOP", "0") == "1"
        self.calibrate = E("CALIBRATE", "0") == "1"
        # The tree policy. "puct" is AlphaZero's. "puct_v" and "uct_v_p" are the variance-aware
        # pair from Inverse-RPO (arXiv 2512.21648), which derives prior-based policies from UCB-V.
        # puct_v beat plain PUCT by about 90 Elo at 200 visits and is used from v3. c1 and c2 are
        # the published constants, and the outer 2.0 in _terms converts units, which makes c2
        # effectively 6 on the [-1, 1] scale. v1 and v2 are sealed with "puct".
        self.tree_policy = E("TREE_POLICY", "puct_v")
        # v5's search comes from a joint tuning of v4's search (about +60 Elo at 400 and 1,600
        # visits). The values marked v5 are that trial's, and v1 to v4 are sealed with their own.
        #   policy_temp 0.889688 (the constructor's default, v4 1.0), c1 1.0236 (v4 sqrt 2), c2 1.70513 (v4 3),
        #   lcb_alpha 0.00184487 (v4 1e-5), lcb_floor 0.0928919 (v4 0.10), q_n0 1.45229 (v4 2.0),
        #   explore_growth 0.205868 (v4 0) and explore_base 207.255 (v4 500).
        self.c1 = float(E("C1", 1.0236))
        # c2 1.07 for v6. v5's q head spreads a position's top moves about 37 percent less than v4's,
        # so 1.70513 explores as if it were about 2.7, and 1.07 = 1.70513 x 0.629 corrects for it.
        # v5 keeps 1.70513 and v1 to v4 keep 3.0, since every sealed version pins its own c2.
        self.c2 = float(E("C2", 1.07))
        # FPU at the root, separately. KataGo uses c_FPU 0.2 inside the tree and 0.0 at the root,
        # where every child should be looked at. Negative means use the same value everywhere.
        self.fpu_root = float(E("FPU_ROOT", 0.0))
        # KataGo's dynamic first-play urgency. If the moves already tried turned out worse than the
        # parent expected, lower the bar for untried ones, and if better, raise it.
        self.fpu_dynamic = E("FPU_DYNAMIC", "0") == "1"
        self.fpu_gain = float(E("FPU_GAIN", 1.0))
        # A constant multiplier on the deadline, the control for adaptive. Part of adaptive's gain
        # is simply using more of the clock, and running this at adaptive's mean stretch separates
        # that from allocating time well.
        self.time_scale = float(E("TIME_SCALE", 1.0))
        # Gumbel AlphaZero root selection. Samples a shortlist by g(a) + log P(a), gives every
        # survivor equal visits and halves the shortlist each phase, which suits small budgets.
        # Below the root PUCT still guides the descent.
        self.gumbel_m = int(E("GUMBEL_M", 16))
        self.gumbel_c_visit = float(E("GUMBEL_C_VISIT", 50.0))
        self.gumbel_c_scale = float(E("GUMBEL_C_SCALE", 1.0))
        self.rng = np.random.default_rng(int(E("GUMBEL_SEED", 0)) or None)
        # Three fixes from v4 onwards, sealed off for v1 to v3.
        #
        # root_mate plays a mate in one without searching for it, since the policy can rank a mate
        #   very low.
        # tb_dtz plays, in a tablebase win, the move that zeroes the fifty-move counter soonest. WDL
        #   alone rates every winning move equally, so a won ending could shuffle towards a draw.
        # ponder_discount scales visits carried over from pondering at the new root, values kept.
        #   The move is chosen by visit count, so a reused tree can arrive with the expected reply
        #   far ahead and outvote a better move found later. 1.0 is the old behaviour.
        self.root_mate = E("ROOT_MATE", "1") == "1"
        self.tb_dtz = E("TB_DTZ", "1") == "1"
        self.ponder_discount = float(E("PONDER_DISCOUNT", 0.25))
        # From v4 onwards, each sealed off for v1 to v3.
        #
        # lcb plays the root move with the highest lower confidence bound on its value, not the
        #   most visits, among moves with at least lcb_floor of the leader's visits. A Student-t
        #   bound at one-sided level lcb_alpha on the spread V2 already keeps, as Leela Zero does.
        #   On from v4, where it was worth well over 100 Elo.
        self.lcb = E("LCB", "1") == "1"
        self.lcb_floor = float(E("LCB_FLOOR", 0.0928919))       # v5, v4 0.10
        self.lcb_alpha = float(E("LCB_ALPHA", 0.00184487))      # v5, v4 1e-5
        # virtual_mean makes a pending visit count in N but add the value selection gave the edge
        #   to W, so it narrows U without pretending the move lost (lc0's unscored virtual visit).
        #   On from v4.
        self.virtual_mean = E("VIRTUAL_MEAN", "1") == "1"
        # proofs lets exact results climb the tree. A node with a lost child is won, a node whose
        #   children are all won for the opponent is lost. A proven node becomes terminal, the
        #   values its ancestors averaged are corrected, and the root plays a proven win first.
        self.proofs = E("PROOFS", "0") == "1"
        # tb_rule50 and draw_plies. A tablebase WDL result assumes a fresh fifty-move counter, so a
        #   win only counts if it zeroes in time (|dtz| + clock <= 100), and the tree draws at 100
        #   plies, where a draw is claimed, not at the seventy-five-move 150.
        self.tb_rule50 = E("TB_RULE50", "1") == "1"
        self.draw_plies = int(E("DRAW_PLIES", 100))
        # q_init scores an unvisited edge by the q head's own value for that move, and a visited one
        #   by (q_n0 * q0 + W) / (q_n0 + N), in place of first-play urgency. Without it a refutation
        #   with a tiny prior can get no visits at all because first-play urgency marks it down at
        #   every internal node. On from v4.
        self.q_init = E("Q_INIT", "1") == "1"
        self.q_n0 = float(E("Q_N0", 1.45229))                   # v5, v4 2.0
        # q_floor lets the q head lift an edge above first-play urgency but never push it below,
        #   so the prior in the blend is max(Q0, the fpu value). The q head is sometimes confidently
        #   wrong about the winning reply, which then gets no visits. Below the root only, where
        #   the refutations are. Off, and sealed off for v1 to v4.
        self.q_floor = E("Q_FLOOR", "0") == "1"
        # forcing and forcing_after. One ply below the root, once the search has put forcing_after
        #   visits into a candidate, every capture and check the opponent has in reply gets forcing
        #   real visits before PUCT takes over, whatever its prior and q value. Many missed
        #   refutations are captures or checks. 0 is off, sealed off for v1 to v4. On for v5 at 4
        #   visits from 8, which fixed many tactical blunders for about 5 percent of the visits.
        self.forcing = int(E("FORCING", 4))
        self.forcing_after = int(E("FORCING_AFTER", 8))
        # forcing_apart. A forced reply is usually a bad capture or check for the opponent, so a
        #   forced visit backs up a value better for us than the reply the opponent would play, and
        #   at the root those visits would vote for the candidate that drew them. With forcing_apart
        #   the forced visit still teaches the reply node and everything below it, but is taken
        #   back out of the root edge's N, W, V2 and in-flight counters. It stays in the root's
        #   sum_n, which the budget counts, so a search spends exactly the visits it would have.
        #   Sealed off for v1 to v4, on for v5 (about +70 Elo against Stockfish).
        self.forcing_apart = E("FORCING_APART", "1") == "1"
        self._forced_pick = False
        # native_pick hands _pick_fast's arithmetic to the compiled pick (native/pick.c) when it
        #   loaded. Bit for bit the numpy answer, so at a fixed visit count nothing changes, but on
        #   a clock it fits more visits into the same time, which wins games. On for v5, sealed off
        #   for v1 to v4.
        self.native_pick = True
        # fpu_base, what first-play urgency is subtracted from. "value" is the network's score at
        #   expansion, as v1 to v3 play, "mean" is the node's running mean, as lc0 and KataGo use.
        self.fpu_base = E("FPU_BASE", "value")
        # root_check holds back root_check_share of the visits. If the q head's best reply to the
        #   chosen move says it is worse than the tree's Q by root_check_margin and the tree has
        #   barely looked at that reply, the reserve is spent there, otherwise as usual.
        self.root_check = E("ROOT_CHECK", "0") == "1"
        self.root_check_margin = float(E("ROOT_CHECK_MARGIN", 0.3))
        self.root_check_share = float(E("ROOT_CHECK_SHARE", 0.2))
        # policy_head and q_head. "cls" reads the heads on the summary token, all v1 to v3 have.
        #   "att" reads the attention heads added for v4, which score a move from its two square
        #   tokens. A network without them falls back to "cls".
        self.policy_head = E("POLICY_HEAD", "cls")
        self.q_head = E("Q_HEAD", "cls")
        # Time management only applies when a search is given seconds. At fixed visits these are
        # never read, so v1 to v3 are unaffected however they are set.
        #
        # tm_extend keeps going at the deadline while the top two moves are too close to separate,
        #   up to tm_extend_max of the allocation again.
        # tm_stop stops early and hands the time back when the leader is clearly ahead on both
        #   value and visits.
        # book plays master practice for the first book_plies instead of searching. Off by default
        #   and in every measurement, since a book is not the network's strength. The board turns
        #   it on.
        self.book = E("BOOK", "0") == "1"
        self.book_plies = int(E("BOOK_PLIES", 16))
        # open_temp samples the move from the root's visit counts, N^(1/temp), for the first
        #   open_plies, so openings vary while the distribution stays the model's own. 0 turns it
        #   off, as every measurement does. Above 1 flattens and below 1 sharpens, since the
        #   exponent is 1/temp. 1.0 is visit-proportional.
        # open_floor and open_margin keep sampling sound. The floor drops moves the search only
        #   glanced at, where Q is noise, and the margin drops anything rated clearly worse than
        #   the move it would have played.
        self.open_temp = float(E("OPEN_TEMP", 0.0))
        self.open_plies = int(E("OPEN_PLIES", 12))
        # cache_progress, whether a collection pass that produced no leaf but did advance the root
        #   counts as progress. Off is the old reading, where such a pass ended the search, which
        #   could cut a search short after pondering filled the cache. It changes moves, so it is
        #   sealed, off for v1 to v3.
        self.cache_progress = E("CACHE_PROGRESS", "1") == "1"
        # collide, what a walk does when it reaches a leaf another walk in this batch is already
        #   waiting on. 0 unwinds and walks again, which can repeat the same path many times and
        #   send the batch out short. 1 keeps the virtual loss and backs the collided walks up with
        #   the leaf's value when it arrives, which counts visits with no new evaluation behind them.
        #   2 holds instead, as lc0 does. The collided walk keeps its virtual visit while the batch
        #   gathers and gives it back when the batch returns, with nothing backed up and nothing
        #   charged to the budget. That matters under virtual_mean, where a pending visit barely
        #   discourages its path. Sealed off (0) for v1 to v3, 2 from v4. On a clock it gives far
        #   more visits a move, so fixed-visit measurements understate it.
        self.collide = int(E("COLLIDE", "2"))
        self._collided = None
        self.open_floor = float(E("OPEN_FLOOR", 0.02))
        self.open_margin = float(E("OPEN_MARGIN", 0.05))
        self.tm_extend = E("TM_EXTEND", "1") == "1"
        self.tm_stop = E("TM_STOP", "1") == "1"
        self.tm_margin = float(E("TM_MARGIN", 0.05))
        # tm_parity and tm_deep size the ceiling the caller hands the search for a long think.
        #   tm_parity is how much of its lead over the opponent's clock one move may spend, 1.0
        #   meaning down to level, and tm_deep caps a single think as a share of its own clock.
        #   tm_parity 0 is the old behaviour (only tm_extend_max of this move's allocation), which
        #   v1 to v3 are sealed with.
        # tm_open_spend and tm_open_plies give move one this share of a normal allocation, ramped
        #   back to full by tm_open_plies, so the opening does not take more than the middlegame.
        #   1.0 is a no-op, as v1 to v3 are sealed. The caller reads these, the tree does not.
        self.tm_open_spend = float(E("TM_OPEN_SPEND", 0.4))
        self.tm_open_plies = int(E("TM_OPEN_PLIES", 16))
        self.tm_parity = float(E("TM_PARITY", 1.0))
        self.tm_deep = float(E("TM_DEEP", 0.5))
        self.tm_extend_max = float(E("TM_EXTEND_MAX", 0.6))
        # tm_short_sudden, the network price uci.py plans each move at in a no-increment game that
        #   started with a minute or less (1+0), in place of its flat NET_SUDDEN of 1.25 s. 0 keeps
        #   the flat price. At 1+0 the flat price reserves most of the clock, so almost every move
        #   falls under uci.INSTANT and only pondered moves get a search. 2+0 and longer keep the
        #   flat price. The caller reads it, the tree does not. On for v6, sealed off (0) for v1 to v5.
        self.tm_short_sudden = float(E("TM_SHORT_SUDDEN", 0.8))
        self._force = None             # {id(node): edge} the root check is verifying, or None
        self._root_allowed = None      # per search, not per tree, reset at the top of search()
        self._gumbel = None
        self.cache, self.cache_size = {}, cache_size
        self.root = None
        self.history = set()
        # Set to stop a search early. Pondering searches on the opponent's time and is abandoned
        # when they move, and its visits are kept because the tree persists.
        self.abort = threading.Event()
        # FAST_EVAL, off unless asked for. It removes two costs that are not arithmetic, autocast
        # re-casting the trunk every pass and a batch shape that changes every pass. It changes no
        # sealed setting, but it does change how much a version searches per second.
        self.fast_eval = E("FAST_EVAL", "0") == "1"
        # Kernel launches are the third cost. Under FAST_EVAL a batch of 48 spends most of its time
        # launching a few hundred kernels one by one, and a CUDA graph replays the whole pass as one
        # launch. Only with FAST_EVAL, whose padding gives the fixed shape a graph needs, and only
        # on a GPU. Not a sealed setting, the graph runs the same kernels on the same numbers.
        self.cuda_graph = E("CUDA_GRAPH", "0") == "1"
        # The same launch cost in the pass the sealed versions play. Without FAST_EVAL the pass is
        # fp32 weights under bfloat16 autocast, and about half of it is Python launching kernels.
        # FAST_EVAL casts the weights to bfloat16 for good, which changes the numbers, so sealed
        # versions cannot use it. This captures the autocast pass itself, one graph per batch size,
        # with the autocast cache off inside the graph so every replay casts the weights to the
        # values the cache would hold. Checked bit for bit against the eager pass at batch sizes 1
        # to 48. On for v5, sealed off for v1 to v4. It runs only where the machine allows it
        # (AUTOCAST_GRAPH=1, AUTOCAST_GRAPH_OK above), since each graph holds GPU memory.
        self.autocast_graph = True
        # timing_fixes, the move-neutral timing fixes, all behind one sealed switch. None changes a
        #   number or a move at a given visit count (harness.py checks each), so on a clock they only
        #   add visits. They are the garbage collector held off during a search (_gc_hold), the
        #   autocast graph run on bf16 copies of the weights with only the needed outputs copied out
        #   (_replay_autocast_fast), the pass and wait timers (stats pass_s, wait_s), the compiled
        #   move generator (mayfly_movegen), the compiled walk (mayfly_walk), and the finiteness
        #   checks in _expand done with one sum. On for v5, sealed off for v1 to v4.
        self.timing_fixes = True
        # sw25_one_copy, see _prepare, _dispatch and _collect. Only the autocast forward pass uses
        # it, any other pass runs as before. Speed only, the same moves, checked bit for bit.
        # On for v6, sealed off for v1 to v5.
        self.sw25_one_copy = True
        # engine "cpp" runs the search in C++ (mcts_cpp.py, the tree, and the network too when a
        # fingerprinted TorchScript trace of this network sits beside its checkpoint), with this
        # file's clock rules between batches. It returns the same moves as this file's search and
        # runs about 1.3 to 1.5 times the visits a second. A setting it does not implement, or a
        # missing build, runs this file's search instead. Released versions before v6 pin "python".
        # ENGINE=python forces this file's search.
        self.engine = E("ENGINE", "cpp")
        # cuda_net, with engine "cpp", runs the network on chess/cudanet (mcts_cpp.trace_for picks
        # <version>.cnw when its fingerprint matches this network). Off by default, sealed off for
        # v1 to v6.
        self.cuda_net = E("CUDA_NET", "0") == "1"
        self._cpp = None
        self._cpp_cfg = None
        self._cpp_live = False
        self._search_ptrs = None
        self._nm = None                 # the compiled move generator for this search, under timing_fixes
        # The compiled walk for this search, under timing_fixes. The descent (or None where it would
        # not be exact), the backup (likewise), the unwind, the descent's fixed settings and its three
        # callbacks, and the compiled prior (None where it disagreed with numpy).
        self._wk = self._wkb = self._wku = self._wk_params = self._wk_cb = self._wkp = None
        self.model_dtype = torch.float32
        if self.fast_eval and self.model is not None and self.device != "cpu":
            self.model.to(torch.bfloat16)
            self.model_dtype = torch.bfloat16
        self._warm()
        # Set by the board during a game, an OpponentModel and the colour of the player being
        # modelled. None everywhere else, so self-play and benchmarks are unaffected.
        self.opponent = None
        self.opp_colour = None
        # TB=0 turns the tablebases off, to measure what they are worth. Sealed, since older
        # versions were released before the tree used tablebases.
        self.use_tb = E("TB", "1") == "1"
        # Correctness fixes that change which move comes back, so each is sealed. The default is
        # the fixed behaviour, used from v5, and v1 to v4 are sealed at what they were released with.
        #
        # castle_prior, where the policy head keeps castling. The Lichess and Leela policy corpora
        #   store O-O as king takes rook, e1h1, index 263 in the side to move's frame, and O-O-O as
        #   e1a1, 256, while python-chess's legal moves are e1g1 (262) and e1c1 (258). A network
        #   trained on them puts castling's prior on 263 and 256, and a tree reading 262 and 258
        #   almost never castles. "rook" reads a legal castle's prior from 263 or 256, as v4 was
        #   trained, "king" reads 262 and 258, as v1 to v3 and networks trained after the trainer's
        #   fix want. Only the policy is affected. ChessBench's action values name castling e1g1,
        #   so the q head is always read at 262 and 258.
        self.castle_prior = E("CASTLE_PRIOR", "king")
        # discount_q0. Under q_init an edge is scored (q_n0*Q0 + W)/(q_n0 + N), which changes when
        #   the ponder discount scales N and W, so the q head's prior gains weight and the new search
        #   reopens moves already refuted. On, the discounted root's prior weight is scaled too, so
        #   every Q stays what it was.
        self.discount_q0 = E("DISCOUNT_Q0", "1") == "1"
        # cache_clock. The network reads the halfmove clock, so a cache keyed only on the position
        #   serves a reversible ending the answer it got many plies earlier. On, the clock the
        #   network reads is part of the key.
        self.cache_clock = E("CACHE_CLOCK", "1") == "1"
        # prior_fallback. A search whose deadline passes before its first visit would play argmax
        #   of an all-zero N, which is root.moves[0]. On, it plays the prior's favourite, which the
        #   root's forward pass has already paid for.
        self.prior_fallback = E("PRIOR_FALLBACK", "1") == "1"
        # fill_once. The batch-fill loop counts a pending leaf once, on the root's own count. Off
        #   is v1's double count (root.sum_n + len(leaves) < target), which packs the same visits
        #   into more, smaller batches and so changes some moves.
        self.fill_once = E("FILL_ONCE", "1") == "1"
        # Each of the next three is sealed off for v1 to v4.
        # tm_catchup (Baier and Winands' STOP rule). Under a clock, stop once the runner-up could
        #   not catch the leader in visits even with tm_catchup of every visit left in the base
        #   allowance. Unlike tm_stop it reads visits, not a value margin, so it also fires in flat
        #   positions. 0 is off.
        self.tm_catchup = float(E("TM_CATCHUP", 0.0))
        # r50_damp (as Stockfish's evaluate.cpp, v -= v * rule50 / 199). A leaf's value, and each
        #   move's q price, shrink towards the draw as the fifty-move counter climbs, so a shuffling
        #   line loses value and a pawn move or capture keeps it. The network already sees the
        #   counter, so this may count it twice. 1.0 is Stockfish's rate, 0 is off.
        self.r50_damp = float(E("R50_DAMP", 0.0))
        # root_temp sharpens (<1) or flattens (>1) the root's priors only, P^(1/T). 1.0 is off.
        self.root_temp = float(E("ROOT_TEMP", 1.0))
        # Root noise, self-play only and not a search setting. AlphaZero's Dirichlet noise on the
        #   root's priors, fresh every search, P = (1 - root_noise) * P + root_noise * Dir(noise_alpha
        #   / the number of legal moves). selfplay.py switches it on for its own tree. No environment
        #   variable reaches it and it is not in SEARCH_KEYS, so nothing else picks it up. 0.0 is off.
        self.root_noise = 0.0
        self.noise_alpha = 10.83
        self.noise_rng = None
        self._noised = None             # (the root carrying noise, its clean priors)
        # mlh, Lc0's moves-left utility (GetMUtility in lc0's search.cc). Where the node's own value
        #   is past mlh_threshold, a child that shortens the game earns up to mlh_cap when winning,
        #   and one that lengthens it when losing, clamp(slope * (M_child - M_parent), -cap, cap) *
        #   sign(-q), scaled by const + lin*|q| + sq*q^2. Lc0's defaults except the gate, since Lc0's
        #   0.8 never fires on this network's value scale. Needs a network with the head.
        self.mlh = E("MLH", "0") == "1"
        self.mlh_threshold = float(E("MLH_THRESHOLD", 0.4))
        self.mlh_slope = float(E("MLH_SLOPE", 0.0027))
        self.mlh_cap = float(E("MLH_CAP", 0.0345))
        self.mlh_const = float(E("MLH_CONST", 0.0))
        self.mlh_lin = float(E("MLH_LIN", 1.6521))
        self.mlh_sq = float(E("MLH_SQ", -0.6521))
        self._ml = None
        # mate_dist. Every checkmate used to be worth exactly 1, so a mate in 2 and a mate in 9 read
        #   the same and the tree could dither. A mate found this many plies below the root is backed
        #   up as 1 - mate_dist * plies, and under proofs each proven level multiplies a proven win
        #   or loss by (1 - mate_dist), so the root plays the shortest proven mate and the side being
        #   mated takes the longest defence. 0 is off, as v1 to v4 are sealed.
        self.mate_dist = float(E("MATE_DIST", 0.0))
        # Pipeline. A search alone spends more than half its time on the CPU (selection, expansion,
        # backup) while the GPU waits. With pipeline on, two batches are in flight and the GPU runs
        # one while the CPU backs up the one before and gathers the one after. 1 is two batches of
        # batch leaves, 2 is two of half that. A pending leaf is seen only through its virtual
        # visit, so a visit is worth a little less and the gain is on a clock. 0 is off, as v1 to v4
        # are sealed.
        self.pipeline = int(E("PIPELINE", "0"))
        # Exploration that grows with the tree. lc0 scales c_puct by
        # 1 + CPuctFactor * log((N + CPuctBase) / CPuctBase), and here the same factor multiplies
        # the whole exploration term of whichever tree policy is on. lc0's base does nothing below
        # tens of thousands of visits, so the base is tuned for this engine. 0 is off, as v1 to v4
        # are sealed.
        self.explore_growth = float(E("EXPLORE_GROWTH", 0.205868))    # v5, v1 to v4 0
        self.explore_base = float(E("EXPLORE_BASE", 207.255))        # v5, v4 500
        # sigma_done. puct_v and uct_v_p should read the spread of completed visits. V2 holds only
        #   those, but N and W also hold walks still in flight, which pulled an edge's spread
        #   towards 0. Sealed off for v1 to v4.
        self.sigma_done = E("SIGMA_DONE", "0") == "1"
        # Per instance so it can be swept. One pending visit counts as this many losses, the
        # standard formulation, N goes up by one and W goes down by vl. At vl = 1.0 this equals
        # moving both by vl. Above 1.0, vl means how bad the pending visit is assumed to be.
        self.vl = float(E("VIRTUAL_LOSS", VIRTUAL_LOSS))
        self.stats = {"evals": 0, "cache_hits": 0, "collisions": 0, "batches": 0,
                      "repetitions": 0, "tablebase": 0, "transpositions": 0, "early_stop": 0,
                      "timed_out": 0, "proven": 0, "discounted": 0, "solved": 0,
                      "lcb_overrode": 0, "root_checks": 0, "extended": 0, "banked": 0,
                      "book": 0}

    # ---------------------------------------------------------------- the network
    @torch.inference_mode()
    def _evaluate(self, nodes):
        """Priors and values for a batch of unexpanded nodes, in one forward pass."""
        return self._collect(self._dispatch(self._prepare(nodes)))

    # The forward pass in three parts, so the pipelined search can do CPU work between sending a
    # batch and reading it back. _evaluate runs the three in a row. The legal-move indices for the
    # value gather are copied to the GPU before the network runs, since a copy queued after the
    # forward pass would hold the CPU until the network finished. The opponent adapter's test reads
    # the boards instead of synchronising on a GPU tensor.
    def _prepare(self, nodes):
        """The CPU half, every array the batch needs, before anything goes to the GPU."""
        # One numpy decode of the nodes' own keys. The transposition key already holds every bit
        # board_to_ids reads (bitboards, turn, castling rights, the legal en passant square), so no
        # board is touched. The arrays match board_to_ids and clock_and_ep.
        if self._nm is not None:
            # the compiled encoder (timing_fixes), the same arrays written straight from the keys
            ids, cast, hm, ep = self._nm.encode([n.key for n in nodes], [n.board.halfmove_clock for n in nodes])
        else:
            ids, cast, hm, ep = encode_keys([n.key for n in nodes],
                                            [n.board.halfmove_clock for n in nodes])
        prep = {"nodes": nodes, "ids": ids, "cast": cast, "hm": hm, "ep": ep}
        if self.value_from_q:
            # Positions have different numbers of legal moves, so the indices are padded to the
            # widest and the padding is masked to minus infinity before the softmax. Same
            # transformation as ttt._stack.
            widths = [len(n.moves) for n in nodes]
            wide = max(widths)
            I = np.zeros((len(nodes), wide), np.int64)
            M = np.zeros((len(nodes), wide), bool)
            # The policy's own indices, which differ from I only at a legal castle under
            # castle_prior "rook", so the weights below match the priors _expand gives the node.
            Ip = I if self.castle_prior == "king" else np.zeros_like(I)
            for i, n in enumerate(nodes):
                flip = n.board.turn == chess.BLACK
                k = widths[i]
                I[i, :k] = (n.IX if n.IX is not None
                            else np.fromiter((move_to_idx(m, flip) for m in n.moves), np.int64, k))
                if Ip is not I:
                    Ip[i, :k] = policy_index(n.board, n.moves, flip, self.castle_prior)
                M[i, :k] = True
            prep.update(widths=widths, I=I, Ip=Ip, M=M)
        if self.opponent is not None and self.opp_colour is not None:
            prep["theirs"] = [float(n.board.turn == self.opp_colour) for n in nodes]
        if self._oc_ok() and "theirs" not in prep:
            # sw25_one_copy puts every host array of the batch in one pinned buffer, so _dispatch sends
            # it with one asynchronous copy instead of six blocking ones. torch's caching host allocator
            # does not reuse a block until the copy that read it has finished, so a pipelined batch's
            # inputs are never overwritten.
            B, L = ids.shape[0], ids.shape[1]
            W = prep["I"].shape[1]
            sep = prep["Ip"] is not prep["I"]
            o = _oc_layout(B, L, W, sep)
            blob = torch.empty(o[7], dtype=torch.long, pin_memory=True)
            a = blob.numpy()
            a[o[0]:o[1]].reshape(B, L)[:] = ids
            a[o[1]:o[2]].view(np.float32).reshape(B, 4)[:] = cast
            a[o[2]:o[3]] = hm
            a[o[3]:o[4]] = ep
            a[o[4]:o[5]].reshape(B, W)[:] = prep["I"]
            if sep:
                a[o[5]:o[6]].reshape(B, W)[:] = prep["Ip"]
            a[o[6]:o[7]].reshape(B, W)[:] = prep["M"]
            prep["oc"] = (blob, B, L, W, sep)
        return prep

    def _oc_ok(self):
        """Which forward pass sw25_one_copy rewrites, or None.

        "graph" is the autocast graph under timing_fixes, "eager" the autocast pass a machine without
        AUTOCAST_GRAPH runs. Both read the q head's leaf values, with no opponent adapter and no
        moves-left head. Anything else runs as before."""
        if not (self.sw25_one_copy and self.value_from_q and not self.fast_eval and self.device != "cpu"
                and not self.mlh and (self.opponent is None or self.opp_colour is None)):
            return None
        if self.autocast_graph and AUTOCAST_GRAPH_OK:
            return "graph" if self.timing_fixes else None
        return "eager"

    @torch.inference_mode()
    def _dispatch(self, prep):
        """Send the batch to the GPU and queue everything computed there.

        Nothing here waits for the GPU, so the CPU is free as soon as this returns."""
        nodes = prep["nodes"]
        t_pass = time.perf_counter() if self.timing_fixes else None
        # FAST_EVAL set after construction (a match's OPTS, applied as attributes) takes effect here
        # on the first batch. Only this tree's network is cast, so a match gives that side its own
        # copy (clock_match's B_VERSION).
        if self.fast_eval and self.model_dtype is not torch.bfloat16 and self.device != "cpu":
            self.model.to(torch.bfloat16)
            self.model_dtype = torch.bfloat16
            self.model._mayfly_cast = True
        elif (not self.fast_eval and self.device != "cpu"
              and getattr(next(self.model.parameters(), None), "dtype", None) is torch.bfloat16):
            # Another tree's fast_eval cast this shared network to bfloat16, which would silently
            # play different numbers. That tree needs its own copy (B_VERSION).
            raise RuntimeError("this tree's network was cast to bfloat16 by another tree's "
                               "fast_eval; give that tree its own copy of the network")
        if "oc" in prep and self._oc_ok() and "theirs" not in prep:
            return self._dispatch_one_copy(prep, t_pass)
        L = lambda a: torch.tensor(a, dtype=torch.long, device=self.device)
        ids = torch.tensor(prep["ids"], dtype=torch.long, device=self.device)
        cast = torch.tensor(prep["cast"], device=self.device)
        hm, ep = L(prep["hm"]), L(prep["ep"])
        if self.value_from_q:
            It = torch.from_numpy(prep["I"]).to(self.device)
            Ipt = It if prep["Ip"] is prep["I"] else torch.from_numpy(prep["Ip"]).to(self.device)
            Mt = torch.from_numpy(prep["M"]).to(self.device)
        theirs = (torch.tensor(prep["theirs"], device=self.device).unsqueeze(1)
                  if "theirs" in prep else None)
        if self.fast_eval:
            # The weights are already bfloat16 (cast once in __init__), so there is no autocast and
            # no re-casting a pass, and the batch is padded to a fixed size so the kernels see one
            # shape and can be captured. Padding rows copy the last board and are thrown away.
            k = ids.shape[0]
            if k < self.batch:
                pad = self.batch - k
                ids = torch.cat([ids, ids[-1:].expand(pad, -1)], 0)
                cast = torch.cat([cast, cast[-1:].expand(pad, -1)], 0)
                hm = torch.cat([hm, hm[-1:].expand(pad)], 0)
                ep = torch.cat([ep, ep[-1:].expand(pad)], 0)
            if self.cuda_graph and self.device != "cpu":
                out = self._replay(ids, cast.to(self.model_dtype), hm, ep)
            else:
                out = self.model(ids, cast.to(self.model_dtype), hm, ep)
            if k < self.batch:
                out = {name: v[:k] for name, v in out.items()}
        elif self.autocast_graph and AUTOCAST_GRAPH_OK and self.device != "cpu":
            if self.timing_fixes:
                need = {"policy", "qvalue"}
                need |= {"policy_att"} if self.policy_head == "att" else set()
                need |= {"qvalue_att"} if self.q_head == "att" else set()
                need |= {"h"} if self.opponent is not None else set()
                need |= {"mleft"} if self.mlh else set()
                need |= set() if self.value_from_q else {"value"}
                out = self._replay_autocast_fast(ids, cast, hm, ep, need)
            else:
                out = self._replay_autocast(ids, cast, hm, ep)
        else:
            with torch.autocast("cuda", torch.bfloat16, enabled=(self.device != "cpu")):
                out = self.model(ids, cast, hm, ep)
        # The opponent adapter, during a game. It shifts the policy only at nodes where the modelled
        # player is to move, so the tree expects what this player plays rather than what the model
        # would. The model's view of its own moves is untouched, and expanded nodes keep their
        # priors, so new knowledge reaches the tree through new leaves.
        # The head is chosen once here so the priors, the adapter and the leaf value all read the
        # same one. A checkpoint without the attention heads falls back to the summary-token ones.
        pk = "policy_att" if (self.policy_head == "att" and "policy_att" in out) else "policy"
        qk = "qvalue_att" if (self.q_head == "att" and "qvalue_att" in out) else "qvalue"
        if theirs is not None and any(prep["theirs"]):
            out[pk] = (out[pk].float()
                       + theirs * self.opponent.delta(out["h"].float()))
        h = {"nodes": nodes, "pol": out[pk].float()}
        # Which head scores a leaf. ChessBench barely trains the value head (only the small policy
        # slice carries a position value), so it stayed near chance. The action-value head is
        # trained by every ChessBench record, and a position's value follows from its moves' values,
        # as in minimax, at no extra cost.
        if self.value_from_q and out.get(qk) is not None:
            q = torch.sigmoid(out[qk].float())
            pl = out[pk].float()
            # One gather for the whole batch, since a per-node loop synchronised with the GPU once
            # per node. The value is policy weighted, not the maximum. The maximum of noisy action
            # values is biased upward (a level opening read as well over a pawn up), while this asks
            # what the position is worth if the model plays as it actually plays.
            w = torch.softmax(torch.gather(pl, 1, Ipt).masked_fill(~Mt, float("-inf")), -1)
            qg = torch.gather(q, 1, It)
            h["val"] = (w * qg).sum(1) * 2.0 - 1.0
            # every child's own q
            h["qg"] = qg * 2.0 - 1.0
            h["widths"] = prep["widths"]
        else:
            h["value"] = torch.sigmoid(out["value"].float())
        # plies left from each position, if the network has the head and the tree reads it
        h["ml"] = ((out["mleft"].float() * 100.0).clamp(min=0.0)
                   if (self.mlh and out.get("mleft") is not None) else None)
        if t_pass is not None:
            # time spent sending the batch (with the graph, mostly the replay and its copies)
            self.stats["pass_s"] = self.stats.get("pass_s", 0.0) + (time.perf_counter() - t_pass)
        return h

    def _dispatch_one_copy(self, prep, t_pass):
        """_dispatch for the autocast pass with one copy in and one copy out (sw25_one_copy).

        The inputs go over as one pinned blob, asynchronously, and are split into views on the GPU.
        The graph's static inputs are filled from them on the GPU, the network replays, and the tail
        (the legal-move policy slice, every child's q and the policy-weighted leaf value, the same
        operations _dispatch runs) is queued inside the graph lock, reading the graph's static outputs
        directly, so nothing is cloned and another tree's replay cannot overwrite them first. The tail
        is packed into one (batch, 2 W + 1) tensor, copied back to pinned memory without waiting, and
        marked with one event that _collect waits on."""
        nodes = prep["nodes"]
        blob, B, L, W, sep = prep["oc"]
        o = _oc_layout(B, L, W, sep)
        dev = blob.to(self.device, non_blocking=True)
        ids = dev[o[0]:o[1]].view(B, L)
        cast = dev[o[1]:o[2]].view(torch.float32).view(B, 4)
        hm = dev[o[2]:o[3]]
        ep = dev[o[3]:o[4]]
        It = dev[o[4]:o[5]].view(B, W)
        Ipt = dev[o[5]:o[6]].view(B, W) if sep else It
        Mt = dev[o[6]:o[7]].view(B, W) != 0
        need = {"policy", "qvalue"}
        need |= {"policy_att"} if self.policy_head == "att" else set()
        need |= {"qvalue_att"} if self.q_head == "att" else set()

        def tail(out):
            pk = "policy_att" if (self.policy_head == "att" and "policy_att" in out) else "policy"
            qk = "qvalue_att" if (self.q_head == "att" and "qvalue_att" in out) else "qvalue"
            q = torch.sigmoid(out[qk].float())
            pg = torch.gather(out[pk].float(), 1, Ipt)
            w = torch.softmax(pg.masked_fill(~Mt, float("-inf")), -1)
            qg = torch.gather(q, 1, It)
            val = (w * qg).sum(1) * 2.0 - 1.0
            packed = torch.cat([pg, qg * 2.0 - 1.0, val.unsqueeze(1)], 1)
            host = torch.empty(packed.shape, dtype=torch.float32, pin_memory=True)
            host.copy_(packed, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            return host, ev

        if self._oc_ok() == "graph":
            res = self._replay_autocast_fast(ids, cast, hm, ep, need, tail=tail)
            host, ev = res[1] if isinstance(res, tuple) else tail(res)
        else:
            with torch.autocast("cuda", torch.bfloat16):
                host, ev = tail(self.model(ids, cast, hm, ep))
        h = {"nodes": nodes, "oc": (host, ev), "widths": prep["widths"], "ml": None}
        if t_pass is not None:
            self.stats["pass_s"] = self.stats.get("pass_s", 0.0) + (time.perf_counter() - t_pass)
        return h

    def _replay(self, ids, cast, hm, ep):
        """The FAST_EVAL forward pass from a CUDA graph, captured on first use and shared by every
        tree on this network and batch size.

        The graph writes into the same output tensors every time, so they are copied out at once,
        since a batch still waiting to be read (the pipeline keeps one) must not be overwritten."""
        # Several trees in several threads can share one network and replay through the same static
        # tensors, so copying in, replaying and copying out happen under one lock. The graph also
        # keeps the addresses of the weights it was captured on, so if the network has moved (the
        # board moves it to system memory while idle) the graph is captured again.
        lock = self.model.__dict__.setdefault("_mayfly_graph_lock", threading.Lock())
        with lock:
            graphs = self.model.__dict__.setdefault("_mayfly_graphs", {})
            ptrs = (tuple(t.data_ptr() for t in self.model.parameters())
                    + tuple(t.data_ptr() for t in self.model.buffers()))
            key = (tuple(ids.shape), tuple(cast.shape), str(self.device))
            g = graphs.get(key)
            if g is not None and g["ptrs"] != ptrs:
                del graphs[key]
                g = None
            if g is None:
                g = {"ids": ids.clone(), "cast": cast.clone(), "hm": hm.clone(), "ep": ep.clone(),
                     "ptrs": ptrs}
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):              # warm-up off the capture, for the allocator and cuBLAS
                        self.model(g["ids"], g["cast"], g["hm"], g["ep"])
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    g["out"] = self.model(g["ids"], g["cast"], g["hm"], g["ep"])
                g["graph"] = graph
                graphs[key] = g
            g["ids"].copy_(ids)
            g["cast"].copy_(cast)
            g["hm"].copy_(hm)
            g["ep"].copy_(ep)
            g["graph"].replay()
            return {name: v.clone() for name, v in g["out"].items()}

    def _replay_autocast(self, ids, cast, hm, ep):
        """The pass without FAST_EVAL, fp32 weights under bfloat16 autocast, from a CUDA graph per
        batch size (autocast_graph).

        The batch is never padded, since another shape may get another kernel that rounds
        differently. Each size gets its own graph, captured on first use, all in one memory pool.
        They never run at once (the lock), and every output is copied out before the lock is
        released, so a later graph reusing a buffer cannot overwrite an answer still wanted. The
        graphs are captured again if the weights move."""
        lock = self.model.__dict__.setdefault("_mayfly_graph_lock", threading.Lock())
        with lock:
            graphs = self.model.__dict__.setdefault("_mayfly_ac_graphs", {})
            ptrs = (tuple(t.data_ptr() for t in self.model.parameters())
                    + tuple(t.data_ptr() for t in self.model.buffers()))
            if graphs and next(iter(graphs.values()))["ptrs"] != ptrs:
                graphs.clear()                      # the weights moved, so every graph points at the old place
                self.model.__dict__.pop("_mayfly_ac_pool", None)
            key = (tuple(ids.shape), tuple(cast.shape), str(self.device))
            g = graphs.get(key)
            if g is None and "_mayfly_ac_failed" not in self.model.__dict__:
                try:
                    pool = self.model.__dict__.get("_mayfly_ac_pool")
                    if pool is None:
                        pool = self.model.__dict__["_mayfly_ac_pool"] = torch.cuda.graph_pool_handle()
                    g = {"ids": ids.clone(), "cast": cast.clone(), "hm": hm.clone(), "ep": ep.clone(),
                         "ptrs": ptrs}
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
                            for _ in range(3 if not graphs else 1):   # set-up off the capture
                                self.model(g["ids"], g["cast"], g["hm"], g["ep"])
                    torch.cuda.current_stream().wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
                            g["out"] = self.model(g["ids"], g["cast"], g["hm"], g["ep"])
                    g["graph"] = graph
                    graphs[key] = g
                except Exception as e:
                    # A failed capture costs speed, never a move. This network goes back to the
                    # eager pass for good, which gives the same numbers, and the reason is kept.
                    self.model.__dict__["_mayfly_ac_failed"] = f"{type(e).__name__}: {e}"
                    g = None
            if g is None:
                with torch.autocast("cuda", torch.bfloat16):
                    return self.model(ids, cast, hm, ep)
            g["ids"].copy_(ids)
            g["cast"].copy_(cast)
            g["hm"].copy_(hm)
            g["ep"].copy_(ep)
            g["graph"].replay()
            return {name: v.clone() for name, v in g["out"].items()}

    def _replay_autocast_fast(self, ids, cast, hm, ep, need, tail=None):
        """_replay_autocast with three move-neutral savings (timing_fixes).

          * The weights are cast once, not on every replay. With the autocast cache off inside the
            graph, every replay re-cast all the fp32 weights to bfloat16. Here the graph is captured
            with bf16 copies of exactly those weights (_bf16_copies) swapped in by
            torch.func.functional_call, so autocast finds them already bfloat16 and the same matmul
            kernels read the same values. The copies are remade if the weights move.
          * The weights' addresses are read once a search (search() fills _search_ptrs), not by
            walking every tensor on every replay.
          * Only what _dispatch reads is copied out (need).

        A failed capture falls back to _replay_autocast, which gives the same numbers."""
        lock = self.model.__dict__.setdefault("_mayfly_graph_lock", threading.Lock())
        with lock:
            ptrs = self._search_ptrs or (tuple(t.data_ptr() for t in self.model.parameters())
                                         + tuple(t.data_ptr() for t in self.model.buffers()))
            graphs = self.model.__dict__.setdefault("_mayfly_pc_graphs", {})
            pc = self.model.__dict__.get("_mayfly_pc_weights")
            if pc is None or pc["ptrs"] != ptrs:
                graphs.clear()                      # new weights or new addresses make every graph stale
                self.model.__dict__.pop("_mayfly_pc_pool", None)
                pc = self.model.__dict__["_mayfly_pc_weights"] = {"ptrs": ptrs, "w": _bf16_copies(self.model)}
            key = (tuple(ids.shape), tuple(cast.shape), str(self.device))
            g = graphs.get(key)
            if g is None and "_mayfly_pc_failed" not in self.model.__dict__:
                try:
                    pool = self.model.__dict__.get("_mayfly_pc_pool")
                    if pool is None:
                        pool = self.model.__dict__["_mayfly_pc_pool"] = torch.cuda.graph_pool_handle()
                    g = {"ids": ids.clone(), "cast": cast.clone(), "hm": hm.clone(), "ep": ep.clone()}
                    w = pc["w"]
                    run = lambda: torch.func.functional_call(self.model, w, (g["ids"], g["cast"], g["hm"], g["ep"]))  # noqa: E731
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
                            for _ in range(3 if not graphs else 1):   # set-up off the capture
                                run()
                    torch.cuda.current_stream().wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
                            g["out"] = run()
                    g["graph"] = graph
                    graphs[key] = g
                except Exception as e:
                    self.model.__dict__["_mayfly_pc_failed"] = f"{type(e).__name__}: {e}"
                    g = None
            if g is not None:
                g["ids"].copy_(ids)
                g["cast"].copy_(cast)
                g["hm"].copy_(hm)
                g["ep"].copy_(ep)
                g["graph"].replay()
                if tail is not None:
                    # sw25_one_copy, the caller's tail reads the static outputs here, under the lock
                    return ("tail", tail(g["out"]))
                return {name: v.clone() for name, v in g["out"].items() if name in need}
        # outside the lock, which is not re-entrant and which _replay_autocast takes itself
        return self._replay_autocast(ids, cast, hm, ep)

    @torch.inference_mode()
    def _collect(self, h):
        """Read a dispatched batch back. This is where the CPU waits for the GPU, if it has to."""
        nodes = h["nodes"]
        t_wait = time.perf_counter() if self.timing_fixes else None
        if "oc" in h:
            # sw25_one_copy waits once, on the event the copy back was recorded with. Every array handed
            # on owns its data, since a view of the pinned buffer must never reach a node or the cache.
            host, ev = h["oc"]
            ev.synchronize()
            if t_wait is not None:
                self.stats["wait_s"] = self.stats.get("wait_s", 0.0) + (time.perf_counter() - t_wait)
            arr = host.numpy()
            widths = h["widths"]
            W = (arr.shape[1] - 1) // 2
            pol = [_Cut(arr[i, :widths[i]].copy()) for i in range(len(nodes))]
            val = arr[:, 2 * W].copy()
            qv = [arr[i, W:W + widths[i]].astype(np.float32) for i in range(len(nodes))]
            self._ml = None
            self.stats["evals"] += len(nodes)
            self.stats["batches"] += 1
            return pol, val, qv
        pol = h["pol"].cpu().numpy()
        if t_wait is not None:
            # the first read back is where the CPU waits for the GPU
            self.stats["wait_s"] = self.stats.get("wait_s", 0.0) + (time.perf_counter() - t_wait)
        if "val" in h:
            val = h["val"].cpu().numpy()
            qg = h["qg"].cpu().numpy()
            widths = h["widths"]
            qv = [qg[i, :widths[i]].astype(np.float32) for i in range(len(nodes))]
        else:
            val = h["value"].cpu().numpy() * 2.0 - 1.0
            qv = [None] * len(nodes)
        # kept on the tree for the caller, so the evaluator's return shape does not change
        self._ml = h["ml"].cpu().numpy() if h["ml"] is not None else None
        self.stats["evals"] += len(nodes)
        self.stats["batches"] += 1
        return pol, val, qv

    def _proven_move(self, board):
        """(move, why) when the best move is proven without a search, otherwise None."""
        b = board.copy(stack=False)
        if self.root_mate:
            for mv in b.legal_moves:
                b.push(mv)
                mate = b.is_checkmate()
                b.pop()
                if mate:
                    return mv, "mate in one"
        if self.tb_dtz and self.use_tb:
            mv = self._tb_root_move(b)
            if mv is not None:
                return mv, "tablebase win, fastest to zero"
        return None

    def _tb_root_move(self, b):
        """In a tablebase WIN, the winning move that reaches a capture or pawn move soonest.

        Only a won root is taken over. A drawn or lost one goes to the search, which can still
        play for a mistake, and anything the tables cannot answer returns None rather than a guess.
        A cursed win (wdl +1) is treated as a draw, because under the fifty-move rule it is one.
        """
        if chess.popcount(b.occupied) > TB_PIECES or b.castling_rights:
            return None
        tb = tablebase()
        if tb is None:
            return None
        try:
            with _TB_LOCK:
                if tb.probe_wdl(b) < 2:
                    return None
                best, best_d = None, None
                for mv in b.legal_moves:
                    zeroing = b.is_zeroing(mv)
                    b.push(mv)
                    if b.is_checkmate():
                        b.pop()
                        return mv
                    theirs = tb.probe_wdl(b)
                    d = None
                    if theirs <= -2:
                        # our move zeroes the counter itself, nothing is faster
                        d = 1 if zeroing else 1 + abs(tb.probe_dtz(b))
                    b.pop()
                    if d is not None and (best_d is None or d < best_d):
                        best, best_d = mv, d
                return best
        except Exception:
            return None

    def _tb_dtz(self, board):
        """|DTZ| of a tablebase position, in plies to the next zeroing move of the right result, or
        None when the probe fails. Cached with the fifty-move check's probes."""
        tb = tablebase()
        if tb is None:
            return None
        key = board._transposition_key()
        dtz = _DTZ_CACHE.get(key)
        if dtz is None:
            try:
                with _TB_LOCK:
                    dtz = tb.probe_dtz(board)
            except Exception:
                return None
            if len(_DTZ_CACHE) < 200_000:
                _DTZ_CACHE[key] = dtz
        return abs(int(dtz))

    def _tb_value(self, board):
        """The true result from the side to move, or None if this position is not in the tables.

        A cursed win, wdl +1, is a win the fifty move rule takes away, so it is reported as a draw.
        """
        if not self.use_tb or chess.popcount(board.occupied) > TB_PIECES:
            return None
        if board.castling_rights:          # syzygy positions have no castling rights
            return None
        tb = tablebase()
        if tb is None:
            return None
        key = board._transposition_key()
        wdl = _TB_CACHE.get(key)
        if wdl is None:
            try:
                with _TB_LOCK:
                    wdl = tb.probe_wdl(board)
            except Exception:
                return None
            if len(_TB_CACHE) < 200_000:
                _TB_CACHE[key] = wdl
        if abs(wdl) < 2:
            return 0.0                     # a draw, a cursed win or a blessed loss
        clock = board.halfmove_clock
        if not self.tb_rule50 or clock == 0:
            return 1.0 if wdl > 0 else -1.0
        # The counter has already started. WDL assumes a position just reached by a capture or pawn
        # move, so a win whose next zeroing move comes too late is a draw. The same test Stockfish
        # applies at its root, |dtz| + clock <= 100.
        dtz = _DTZ_CACHE.get(key)
        if dtz is None:
            try:
                with _TB_LOCK:
                    dtz = tb.probe_dtz(board)
            except Exception:
                return None
            if len(_DTZ_CACHE) < 200_000:
                _DTZ_CACHE[key] = dtz
        if abs(dtz) + clock > 100:
            return 0.0
        return 1.0 if wdl > 0 else -1.0

    def _expand(self, node, logits, value, q0=None, sliced=False, m0=None):
        """Attach the legal moves with their priors. Softmax over the legal moves only.

        sliced says the row has already been cut down to this node's legal moves, which is what the
        evaluation cache stores. A numpy row of the batch array would be a view that keeps the whole
        batch alive, while the legal slice is a few dozen floats that own their data.
        """
        if isinstance(logits, _Cut):
            logits, sliced = logits.row, True       # sw25_one_copy, the row was cut on the GPU
        board = node.board
        moves = node.moves
        flip = board.turn == chess.BLACK
        # The policy's indices (castle_prior), which the cache also slices the row with. Under
        # castle_prior "king" they are the move indices, which a compiled node already holds.
        idx = (node.IX if (node.IX is not None and self.castle_prior == "king")
               else policy_index(board, moves, flip, self.castle_prior))
        # The prior and the five rows in C (timing_fixes, mayfly_walk.expand_rows), the same numbers
        # as the numpy lines below, or None at a row with a NaN or an infinity, which the numpy handles.
        rows = (self._wkp(logits, None if sliced else idx, self.policy_temp, len(moves))
                if self._wkp is not None else None)
        if rows is None:
            z = (logits if sliced else logits[idx]).astype(np.float64)
            # A finite sum proves there is no NaN and no infinity, so the two scans below can be
            # skipped (timing_fixes). A sum that is not finite runs them as before.
            if (not (self.timing_fixes and math.isfinite(float(np.add.reduce(z))))
                    and (np.isnan(z).any() or np.isposinf(z).any())):
                # One NaN or overflowed logit would make every prior here NaN, and argmax reads NaN as
                # the best move. An unreadable logit gets no prior, and a row with nothing readable is
                # uniform.
                z = np.where(np.isfinite(z), z, -np.inf)
                if not np.isfinite(z).any():
                    z = np.zeros_like(z)
        if not math.isfinite(float(value)):
            value = 0.0                    # a NaN leaf value would be every ancestor's best move
        if (q0 is not None and not (self.timing_fixes and math.isfinite(float(np.add.reduce(q0))))
                and not np.isfinite(q0).all()):
            q0 = np.nan_to_num(np.asarray(q0, np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        if m0 is not None and not math.isfinite(float(m0)):
            m0 = None
        node.idx = idx
        if rows is None:
            if self.policy_temp != 1.0:
                z /= self.policy_temp
            z -= z.max()
            p = np.exp(z)
            node.P = (p / p.sum()).astype(np.float32)
            node.N = np.zeros(len(moves), np.float32)
            node.W = np.zeros(len(moves), np.float32)
            node.V2 = np.zeros(len(moves), np.float32)
            node.F = np.zeros(len(moves), np.float32)
            node.X = np.zeros(len(moves), np.float32)
        else:
            node.P, node.N, node.W, node.V2, node.F, node.X = rows
        if self.r50_damp > 0.0:
            h = board.halfmove_clock
            value = float(value) * max(0.0, 1.0 - self.r50_damp * h / 199.0)
            if q0 is not None:
                # each move's own counter afterwards, a pawn move or a capture sets it back to 0
                hk = np.array([0 if (board.is_capture(m)
                                     or board.piece_type_at(m.from_square) == chess.PAWN)
                               else h + 1 for m in moves], np.float32)
                q0 = (np.asarray(q0, np.float32)
                      * np.maximum(0.0, 1.0 - self.r50_damp * hk / 199.0)).astype(np.float32)
        node.Q0 = q0
        node.kids = [None] * len(moves)
        node.value = float(value)
        if self.mlh:
            node.m0 = None if m0 is None else float(m0)
            node.ML = np.zeros(len(moves), np.float32)
            node.MN = np.zeros(len(moves), np.float32)
        return float(value)

    # ---------------------------------------------------------------- the walk
    def _sigma(self, node, n, w, visited):
        """Standard deviation of the values backed up through each action.

        Unvisited actions get 1.0, the largest value possible on a [-1, 1] scale, since an action
        never looked at is maximally uncertain.
        """
        sd = np.ones_like(w)
        if node.V2 is not None and visited.any():
            nn = n[visited]
            mean = w[visited] / nn
            sd[visited] = np.sqrt(np.maximum(node.V2[visited] / nn - mean * mean, 0.0))
        return sd

    def _select(self, node, parent_q, is_root=False):
        return self._pick(node, parent_q, is_root)[0]

    def _pick(self, node, parent_q, is_root=False, ply1=False):
        """(edge, the q that selection read for it). The q is what a virtual mean adds to W.

        ply1 says this node is one of the root's children, the opponent's reply to a candidate."""
        self._forced_pick = False
        if (ply1 and self.forcing > 0 and node.sum_n >= self.forcing_after and node.moves
                and (self._force is None or id(node) not in self._force)):
            need = self._forcing_mask(node) & (node.N < self.forcing)
            if need.any():
                cand = np.flatnonzero(need)
                a = int(cand[int(np.argmin(node.N[cand]))])
                q, _ = self._terms(node, parent_q, is_root)
                self.stats["forced"] = self.stats.get("forced", 0) + 1
                self._forced_pick = True
                return a, float(q[a])
        if self._force is not None and id(node) in self._force:
            # the root check verifying one reply, so every descent through here takes that line
            a = int(self._force[id(node)])
            q, _ = self._terms(node, parent_q, is_root)
            return a, float(q[a])
        if is_root and self._root_allowed is not None:
            # Sequential Halving visits the survivors equally, so the least visited candidate goes
            # next. This is the whole root policy, PUCT does not run here.
            cand = self._root_allowed
            a = int(cand[int(np.argmin(node.N[cand]))])
            q, _ = self._terms(node, parent_q, is_root)
            return a, float(q[a])
        q, u = self._terms(node, parent_q, is_root)
        a = int(np.argmax(q + u))
        return a, float(q[a])

    def _forcing_mask(self, node):
        """node.FM, which of node's moves are captures or checks, worked out the first time forcing asks."""
        if node.FM is None:
            if self._nm is not None:
                try:            # the same bits from C (timing_fixes), in node.moves' order
                    node.FM = np.frombuffer(self._nm.forcing(node.key), np.bool_)
                except (OverflowError, TypeError):
                    node.FM = None
            if node.FM is None:
                b = node.board
                node.FM = np.array([b.is_capture(m) or b.gives_check(m) for m in node.moves], bool)
        return node.FM

    # The compiled walk's three callbacks (timing_fixes), the parts that stay in Python, as in _descend.
    def _walk_term(self, path, node):
        """_descend's terminal step."""
        tv = node.term
        if self.mate_dist > 0.0 and tv <= -1.0 and not node.moves and path:
            tv = -max(0.5, 1.0 - self.mate_dist * len(path))
        self._backup(path, tv, m=0.0 if (self.mlh and node.board.is_game_over()) else None)

    def _walk_fm(self, node):
        self._forcing_mask(node)

    def _walk_pick(self, node, is_root, ply1):
        """_pick_fast for a node the compiled pick cannot serve, and whether it was a forced pick."""
        a, q = self._pick_fast(node, is_root, ply1)
        return a, q, self._forced_pick

    def _pick_fast(self, node, is_root, ply1):
        """_pick for the search v4 and v5 play, in fewer and cheaper numpy calls, with the same numbers.

        It serves trees whose selection is q_init + puct_v with sigma_done off, mlh off, no Gumbel
        shortlist and no root check (_descend tests that once a descent), and falls back to _pick for
        a node that does not fit (no q head, a q_n0 of zero). The differences from _pick + _terms +
        _sigma, and why each is exact:
          * The spread is computed over the whole row and masked with where, not gathered with
            boolean indexing. Every element goes through the same float32 operations, an unvisited
            edge's divisor is 1 (never 0, and not a tiny floor, which a W left at 1e-8 by a
            take-back would overflow) and its result is masked out, so visited entries match.
          * The two power-of-two factors around the variance term (* 0.5 on sigma, 2.0 * on the sum)
            are folded into c2 (2.0 * c2) and dropped from c1's term. Scaling by 2 commutes with
            float32 rounding, so every intermediate is the reference's times an exact power of two
            (they differ only below 1e-37, where u cannot move an argmax against a q).
          * 1.0 + n once, .argmax() on the array, .item(a) for the float, and no _fpu_base (q_init
            never reads the parent's value).
        The forcing branch is the same as _pick's."""
        n, w, p = node.N, node.W, node.P
        n0 = self.q_n0 * node.q0w
        q0 = node.Q0
        if q0 is None or n0 <= 0.0 or node.V2 is None:
            return self._pick(node, self._fpu_base(node), is_root, ply1)
        self._forced_pick = False                    # as _pick does, for forcing_apart
        if ply1 and self.forcing > 0 and node.sum_n >= self.forcing_after and node.moves:
            need = self._forcing_mask(node) & (n < self.forcing)
            if need.any():
                cand = np.flatnonzero(need)
                a = int(cand[int(np.argmin(n[cand]))])
                q = (n0 * q0 + w) / (n0 + n)
                self.stats["forced"] = self.stats.get("forced", 0) + 1
                self._forced_pick = True
                return a, q.item(a)
        if NATIVE_PICK is not None and self.native_pick:
            # the same arithmetic in C (native/pick.c), given exactly the scalars the lines below use
            N = max(node.sum_n, 1)
            grow = (1.0 + self.explore_growth * math.log((N + self.explore_base) / self.explore_base)
                    if self.explore_growth > 0.0 else 0.0)
            try:
                return NATIVE_PICK(n, w, p, q0, node.V2, n0, self.c1, 2.0 * self.c2, math.sqrt(N),
                                   math.log(N + 1.0), grow)
            except TypeError:
                pass                            # a row that is not contiguous float32 uses numpy below
        q = (n0 * q0 + w) / (n0 + n)
        seen = n > 0
        nn = np.where(seen, n, 1.0)
        mean = w / nn
        t = node.V2 / nn
        t -= mean * mean
        np.maximum(t, 0.0, out=t)
        np.sqrt(t, out=t)
        sd = np.where(seen, t, 1.0)
        N = max(node.sum_n, 1)
        d = 1.0 + n
        u = (self.c1 * p * sd * math.sqrt(N)) / d + (2.0 * self.c2 * p * math.log(N + 1.0)) / d
        if self.explore_growth > 0.0:
            u = u * (1.0 + self.explore_growth * math.log((N + self.explore_base) / self.explore_base))
        u += q
        a = int(u.argmax())
        return a, q.item(a)

    def _fpu_base(self, node):
        """What first-play urgency is measured down from, decided in one place for _descend and explain.

        "value" is the network's score at expansion, which v1 to v3 play. "mean" is the running mean
        of everything backed up through the node, its own evaluation counted as the first visit,
        in-flight visits left out."""
        if self.fpu_base == "mean" and node.N is not None:
            done = float((node.N - node.F).sum())
            if done > 0:
                return (float(node.value) + float((node.W - node.X).sum())) / (1.0 + done)
        return float(node.value)

    def _gumbel_score(self, root, cand):
        """g(a) + log P(a) + sigma(q(a)), the quantity Gumbel AlphaZero ranks candidates by."""
        g, logits = self._gumbel
        q = np.divide(root.W[cand], np.maximum(root.N[cand], 1.0))
        sigma = (self.gumbel_c_visit + float(root.N.max())) * self.gumbel_c_scale * q
        return g[cand] + logits[cand] + sigma

    def _terms(self, node, parent_q, is_root=False):
        """The two arrays argmax is taken over.

        explain shows these too, so there is only one implementation of the formula."""
        n, w, p = node.N, node.W, node.P
        visited = n > 0
        q_init = self.q_init and node.Q0 is not None
        if q_init:
            # The move's own value, not its parent's less a penalty. The q head priced every child at
            # expansion. An unvisited edge is that price, and a visited one starts from it as q_n0
            # visits' worth of evidence that the search then outvotes. q0w is 1.0 except at a root
            # the ponder discount scaled under discount_q0, so this ratio is what it was.
            n0 = self.q_n0 * node.q0w
            q0 = node.Q0
            if self.q_floor and not is_root:
                # the q head may lift an edge above first-play urgency, never push it below
                f = self.fpu
                q0 = np.maximum(q0, np.float32(parent_q - f * math.sqrt(float(p[visited].sum()) + 1e-8)))
            if n0 > 0.0:
                q = (n0 * q0 + w) / (n0 + n)
            else:
                # No weight left on the prior (q_n0 0, or a root the ponder discount scaled to 0).
                # An unvisited edge is still its own q0, the limit of the ratio above, not 0/0,
                # which argmax would read as the best move.
                q = np.where(visited, w / np.maximum(n, 1e-12), q0).astype(np.float32)
        else:
            q = np.zeros_like(w)
            np.divide(w, n, out=q, where=visited)
        # First play urgency. An unvisited child is scored below the parent's own value, or the
        # search would try every move once before deepening anything. Leela discounts it by how
        # much prior mass has been explored, so a node whose good moves are all tried stops widening.
        if not q_init and not visited.all():
            f = self.fpu_root if (is_root and self.fpu_root >= 0.0) else self.fpu
            if self.fpu_dynamic and visited.any():
                # Worse than expected below means look wider, better than expected means look harder
                # at what is working, so a negative surprise lowers the bar. Completed visits only,
                # or the batch's virtual losses in flight would read as a bad surprise.
                nd = n - node.F
                done = nd > 0
                if done.any():
                    mq = float((w - node.X)[done].sum() / nd[done].sum())
                    f = max(0.0, f * (1.0 + self.fpu_gain * (mq - parent_q)))
            q[~visited] = parent_q - f * math.sqrt(float(p[visited].sum()) + 1e-8)
        N = max(node.sum_n, 1)
        # "gumbel" is a root policy only, and the root never reaches here while a shortlist is
        # active. Below the root Gumbel AlphaZero is ordinary PUCT.
        if self.tree_policy in ("puct", "gumbel"):
            u = self.c_puct * p * math.sqrt(N) / (1.0 + n)
        else:
            # Done in the paper's units, so c1 and c2 are the published values. UCB-V assumes
            # rewards in [0, 1] and Q here is in [-1, 1]. Mapping z = (q+1)/2 makes sigma_z =
            # sigma/2, and since argmax is invariant under a monotone affine map,
            # argmax(z + u_z) == argmax(q + 2*u_z). Expanding,
            #
            #   2*u_z = c1*pi*sigma*sqrt(N)/(1+n)  +  2*c2*pi*logN/(1+n)
            #
            # so the variance term needs no rescaling and it is the logarithmic term, with no sigma
            # in it, that doubles. Halving c1 instead would be wrong.
            if self.sigma_done:
                # The spread of completed visits. V2 holds only those, so N and W are read without
                # the walks still waiting on the network. An edge whose only visit is still waiting
                # reads 0 as before.
                nd = n - node.F
                sd_z = self._sigma(node, nd, w - node.X, nd > 0) * 0.5
                sd_z[(nd <= 0) & visited] = 0.0
            else:
                sd_z = self._sigma(node, n, w, visited) * 0.5
            lg = math.log(N + 1.0)
            if self.tree_policy == "puct_v":
                u = 2.0 * (self.c1 * p * sd_z * math.sqrt(N) / (1.0 + n)
                           + self.c2 * p * lg / (1.0 + n))
            elif self.tree_policy == "uct_v_p":
                u = 2.0 * (self.c1 * sd_z * np.sqrt(p * lg / (1.0 + n))
                           + self.c2 * p * lg / (1.0 + n))
            else:
                raise ValueError(f"unknown tree policy {self.tree_policy!r}")
        if self.explore_growth > 0.0:
            u = u * (1.0 + self.explore_growth * math.log((N + self.explore_base) / self.explore_base))
        if self.mlh and node.ML is not None and visited.any():
            u = u + self._m_utility(node, q, n)
        return q, u

    def _m_utility(self, node, q, n):
        """Lc0's moves-left utility for every child, zero for one never visited or with no estimate.

        It is added to the exploration term, so explain() shows it where it acts."""
        tot_n = float(node.MN.sum())
        if tot_n <= 0:
            return 0.0
        parent_m = float(node.ML.sum()) / tot_n                 # plies left from here, backed up
        nq = float(node.W[n > 0].sum() / max(float(n[n > 0].sum()), 1.0))
        if abs(nq) <= self.mlh_threshold:
            return 0.0
        has = node.MN > 0
        child_m = np.where(has, node.ML / np.maximum(node.MN, 1.0) - 1.0, parent_m)
        mu = np.clip(self.mlh_slope * (child_m - parent_m), -self.mlh_cap, self.mlh_cap)
        mu = mu * -np.sign(q)
        aq = np.abs(q)
        mu = mu * (self.mlh_const + self.mlh_lin * aq + self.mlh_sq * q * q)
        return np.where(has, mu, 0.0)

    def explain(self, node=None, top=0):
        """Every root move with the numbers PUCT actually used to choose between them.

        The Q, the first play urgency fallback and the U term come from the same _terms call the
        search makes, so what is shown is what the search did. q + u is the whole decision.
        """
        node = node or self.root
        if node is None or not node.expanded:
            return []
        n, w, p = node.N, node.W, node.P
        visited = n > 0
        # the same base and the same call the search uses
        parent_q = self._fpu_base(node)
        q, u = self._terms(node, parent_q, is_root=(node is self.root))
        rows = []
        for i, mv in enumerate(node.moves):
            rows.append({
                "move": mv.uci(),
                "san": node.board.san(mv),
                "prior": float(p[i]),          # what the policy head wanted, over legal moves only
                "visits": int(n[i]),           # how much of the budget it got
                "q": float(q[i]),              # its mean value, or the fpu fallback if unvisited
                "u": float(u[i]),              # the exploration bonus, c_puct * P * sqrt(N) / (1+n)
                "puct": float(q[i] + u[i]),    # the number argmax is taken over
                "unvisited": not bool(visited[i]),
            })
        rows.sort(key=lambda r: (-r["visits"], -r["puct"]))
        return rows[:top] if top else rows

    def explain_text(self, node=None, top=12):
        rows = self.explain(node, top)
        if not rows:
            return "nothing searched yet"
        out = [f"  c_puct {self.c_puct}   fpu {self.fpu}   policy_temp {self.policy_temp}   "
               f"root visits {int((node or self.root).sum_n)}",
               f"  {'move':<7} {'prior':>7} {'visits':>7} {'share':>6} {'Q':>8} {'U':>8} {'Q+U':>8}"]
        total = sum(r["visits"] for r in rows) or 1
        for r in rows:
            out.append(f"  {r['san']:<7} {100*r['prior']:>6.1f}% {r['visits']:>7,} "
                       f"{100*r['visits']/total:>5.1f}% {r['q']:>8.3f} {r['u']:>8.3f} "
                       f"{r['puct']:>8.3f}" + ("   unvisited" if r["unvisited"] else ""))
        return chr(10).join(out)

    def _descend(self, root):
        """One walk to a leaf, applying a virtual visit. Returns (path, parent, action) or None.

        Each step of the path is (node, edge, x), where x is exactly what this visit added to
        W[edge] while pending, minus vl under virtual loss, or the q selection read under a
        virtual mean. Backup replaces x with the real value and _unwind takes x back, so neither
        needs to know which rule put it there."""
        wk = self._wk
        if wk is not None and self._force is None and self._root_allowed is None:
            # The same walk in C (timing_fixes, mayfly_walk). It is this loop for fast below with a
            # virtual mean and no graph search, the same numbers in the same order, calling back here
            # for a terminal (_walk_term), a forcing mask (_walk_fm) and a node the fast pick hands to
            # _pick (_walk_pick).
            return wk.descend(root, self._wk_params, self.stats, self._collided if self.collide else None,
                              *self._wk_cb)
        path = []
        node = root
        # the selection v4 and v5 play gets the fast pick, every other setting keeps _pick (v1 to v3)
        fast = (self.q_init and self.tree_policy == "puct_v" and not self.q_floor and not self.sigma_done
                and not self.mlh and self._force is None and self._root_allowed is None)
        while True:
            if node.term is not None:
                tv = node.term
                if self.mate_dist > 0.0 and tv <= -1.0 and not node.moves and path:
                    # a checkmate is worth less the further down it is on every visit, not only on
                    # the first one, which _gather scores
                    tv = -max(0.5, 1.0 - self.mate_dist * len(path))
                # A game over is zero plies from its end on every visit, as _gather backs it up the
                # first time. Otherwise a mating edge would keep one moves-left sample however often
                # it was visited.
                self._backup(path, tv, m=0.0 if (self.mlh and node.board.is_game_over()) else None)
                return None
            if node.pending:
                self.stats["collisions"] += 1
                # Without collide a collision repeats until the gather gives up, since _pick is
                # deterministic and _unwind restores the exact state, so the batch goes out short.
                # With collide on, the virtual visit stays on the path, the next walk goes elsewhere,
                # and the path is remembered and settled when the batch returns, as lc0 does with
                # MaxCollisionVisits.
                if self.collide and self._collided is not None:
                    self._collided.append((path, node))
                    return None
                self._unwind(path)
                return None
            if fast:
                a, qa = self._pick_fast(node, node is root, len(path) == 1)
            else:
                a, qa = self._pick(node, self._fpu_base(node), is_root=(node is root),
                                   ply1=(len(path) == 1))
            x = qa if self.virtual_mean else -self.vl
            node.N[a] += 1.0
            node.W[a] += x
            node.sum_n += 1.0
            node.F[a] += 1.0
            node.X[a] += x
            path.append((node, a, x))
            if self.forcing_apart and self._forced_pick and len(path) == 2:
                # Kept apart from the root. Take this visit back out of the root edge exactly as
                # _unwind would, leaving only root.sum_n (the budget). x None marks the step for
                # _backup, _unwind and _held_edges.
                r, ra, rx = path[0]
                r.N[ra] -= 1.0
                r.W[ra] -= rx
                r.F[ra] -= 1.0
                r.X[ra] -= rx
                path[0] = (r, ra, None)
                self.stats["forced_apart"] = self.stats.get("forced_apart", 0) + 1
            kid = node.kids[a]
            if self.mcgs:
                # Under the graph a repetition belongs to the line, not the node. A node below a
                # merge is reached through different positions, so a draw stored on it at birth
                # would be the verdict of whichever line built it. The walk checks its own line
                # before it steps in, and a repeating edge makes no node.
                if not path[1:]:
                    self._line = {root.key}
                if kid is not None:
                    ck = kid.key
                else:
                    cb = node.board.copy(stack=False)
                    cb.push(node.moves[a])
                    ck = cb._transposition_key()
                if ck in self._line or ck in self.history:
                    self.stats["repetitions"] += 1
                    self._backup(path, 0.0)
                    return None
                self._line.add(ck)
            if kid is None:
                return path, node, a
            if (self.mcgs == 1 and self.power_p == 1.0 and self.collide != 1 and kid.expanded
                    and not kid.pending and kid.term is None):
                # The paper's check, on every visit into a node, not only the one that made the link.
                # When this edge's completed visits disagree with the node's own value (its evaluation
                # and everything completed below it, from every parent), the descent stops and backs
                # up the correction. With one parent the two are the same number (except under
                # collide 1 or a power mean, which are left out), so this only fires on a
                # transposition.
                n = float(node.N[a] - node.F[a])
                if n >= 1.0:
                    dn = float((kid.N - kid.F).sum())
                    v = (float(kid.value) + float((kid.W - kid.X).sum())) / (1.0 + dn)
                    q_edge = -float(node.W[a] - node.X[a]) / n
                    if abs(q_edge - v) > Q_EPS:
                        self.stats["corrected"] = self.stats.get("corrected", 0) + 1
                        self._backup(path, max(-1.0, min(1.0, n * (v - q_edge) + v)))
                        return None
            node = kid

    def _history_keys(self, board):
        """Every position the real game has already been in.

        Boards inside the tree are copied without their move stack to keep them cheap, so the game's
        own history is walked once per search and kept as keys.
        """
        keys = set()
        tmp = board.copy()
        while tmp.move_stack:
            tmp.pop()
            keys.add(tmp._transposition_key())
        return keys

    def _repeats(self, key, path):
        """Has this exact position already occurred, in the game or higher up this line?

        A position cannot recur across a pawn move or a capture, because either changes the
        position key, so a key match is enough on its own and no fifty move window is needed.

        The first repetition inside the search counts as a draw, as engines usually do, since a side
        that can reach a position twice can reach it a third time.
        """
        if key in self.history:
            return True
        for step in path:
            if step[0].key == key:
                return True
        return False

    def _unwind(self, path):
        """Take back a descent that hit a pending node.

        It must mirror exactly what selection applied, one visit and its provisional value, or the
        counters drift over thousands of collisions."""
        if self._wku is not None and type(path) is list:
            self._wku.unwind(path)              # the same in C (timing_fixes)
            return
        for node, a, x in path:
            if x is None:
                # a forced visit kept apart from the root (forcing_apart), only sum_n holds it
                node.sum_n -= 1.0
                continue
            node.N[a] -= 1.0
            node.W[a] -= x
            node.sum_n -= 1.0
            node.F[a] -= 1.0
            node.X[a] -= x

    def settings(self):
        """Every setting that can change which move comes back."""
        return {k: getattr(self, k) for k in SEARCH_KEYS}

    def apply(self, opts):
        """Pin this tree to a recorded search.

        Unknown keys are refused rather than ignored, so a typo in a version's manifest cannot
        silently give it today's search instead of its own."""
        for k, v in (opts or {}).items():
            if k not in SEARCH_KEYS:
                raise KeyError(f"{k!r} is not a pinned search setting")
            cur = getattr(self, k)
            if isinstance(cur, bool) and isinstance(v, str):
                # A switch written as text. bool("0") is True, so text is read as the word it is.
                # The manifests carry real booleans and are not affected.
                w = v.strip().lower()
                if w not in ("1", "0", "true", "false", "yes", "no", "on", "off"):
                    raise ValueError(f"{k}={v!r} is not an on/off value")
                v = w in ("1", "true", "yes", "on")
            setattr(self, k, type(cur)(v))
        self._warm()
        return self

    def _warm(self):
        """Import everything a search needs here, on the thread that builds the tree, never inside
        the search itself.

        uci.py searches in a background thread, and an import from that thread (scipy's Student-t
        for the lower confidence bound) once hung until the game was lost on time. One throwaway
        call loads it up front, where a slow or stuck import is visible."""
        if getattr(self, "lcb", False):
            _t_quantile(1e-5, np.array([2.0], np.float64))
        if getattr(self, "timing_fixes", False):
            # the compiled move generator and walk, loaded and checked here rather than in the first search
            _native_moves()
            _native_walk()

    def _node_q(self, node):
        """What this node is worth from its own point of view, read off its children.

        The power mean sits between the average of the children and the best of them. At p = 1 it
        is the arithmetic mean, exactly standard MCTS. Q lives in [-1, 1] and a power mean needs a
        non-negative base, so it is mapped to [0, 1], averaged, and mapped back.
        """
        if node is None or node.N is None:
            return None
        # Completed visits only. A walk still waiting on the network holds a virtual visit here,
        # which is no evidence of the value, and what this returns is written into a parent's W
        # for good (the power mean backup, a graph merge).
        n, w = node.N, node.W
        if getattr(node, "F", None) is not None:
            n, w = n - node.F, w - node.X
        m = n > 0
        if not m.any():
            return None
        nn = n[m]
        q = w[m] / nn
        if self.power_p == 1.0:
            return float(w[m].sum() / nn.sum())
        z = np.clip((q + 1.0) * 0.5, 0.0, 1.0)
        w = nn / nn.sum()
        pm = float(np.power(float((w * np.power(z, self.power_p)).sum()), 1.0 / self.power_p))
        return pm * 2.0 - 1.0

    def _merge_value(self, parent, a, shared):
        """The correction a transposition hands back, in the shared node's own frame.

        A node with several parents cannot simply return its own value. The edge pointing at it
        already carries N(s,a) visits of a different estimate, and overwriting that in one step
        would let two parents' statistics leak into each other.

        The sign is derived here. The paper's correction, Qf = N*Qd + V* with Qd = Q - V*, expands
        to N*Q - (N-1)*V* and moves the edge mean away from the shared node. The value x that makes
        the edge's mean equal V* after one more visit is (W + x)/(N + 1) = V*, so x = N*(V* - Q) + V*,
        the same magnitude with the opposite sign, which is what is used. Clipping to [-1, 1] stops
        a large N from swinging the edge across in one update.
        """
        v = self._node_q(shared)
        if v is None:
            v = float(shared.value)
        # Completed visits only. _descend has already put this walk's pending visit and its
        # provisional value on the edge, which are not evidence.
        n = float(parent.N[a] - parent.F[a])
        if n < 1.0:
            return v
        q_edge = -float(parent.W[a] - parent.X[a]) / n    # parent frame to the shared node's frame
        if abs(q_edge - v) <= Q_EPS:
            return v
        return max(-1.0, min(1.0, n * (v - q_edge) + v))

    def _qsearch(self, board, alpha, beta, depth):
        """Captures only, material only, so a leaf that hangs a queen is not scored as quiet.

        Full-width alpha-beta at every leaf is too expensive, while captures only has a branching
        factor of about 5.
        """
        stand = _material(board)
        if depth <= 0 or stand >= beta:
            return stand if depth <= 0 else beta
        if stand > alpha:
            alpha = stand
        for m in board.generate_legal_captures():
            board.push(m)
            try:
                sc = -self._qsearch(board, -beta, -alpha, depth - 1)
            finally:
                board.pop()
            if sc >= beta:
                return beta
            if sc > alpha:
                alpha = sc
        return alpha

    def _stretch(self, v):
        """Put a leaf value on Stockfish's scale. Monotone, so it cannot reorder siblings."""
        c = calibration()
        if not c:
            return v
        return float(np.interp(v, c[0], c[1]))

    def _tactical(self, board, v):
        """Pull the network's leaf value toward what a capture sequence says the material is."""
        swing = self._qsearch(board, -99.0, 99.0, self.qdepth) - _material(board)
        if abs(swing) < 0.5:
            return v
        # a pawn is worth roughly 0.1 of the [-1, 1] value scale at this model's calibration
        tv = max(-1.0, min(1.0, v + 0.1 * swing))
        return v + self.qweight * (tv - v)

    def _backup(self, path, leaf_value, m=None):
        """Undo the virtual loss and apply the real value, flipping sign at every ply.

        leaf_value is from the point of view of the side to move at the leaf, and the edge directly
        above it belongs to the other side, so the first step is a negation.
        """
        if (self._wkb is not None and type(leaf_value) is float and type(path) is list
                and (m is None or not self.mlh)):
            # The same in C (timing_fixes). With power_p 1 and proofs off only the loop's first four
            # lines run, and v is a double here because leaf_value is a Python float.
            self._wkb.backup(path, leaf_value)
            return
        v = -leaf_value
        if m is not None and self.mlh:
            # plies left, one more for every step back up the line, counted from each node
            mm = float(m)
            for node, a, x in reversed(path):
                mm += 1.0
                if node.ML is not None and x is not None:
                    node.ML[a] += mm
                    node.MN[a] += 1.0
        for node, a, x in reversed(path):
            if x is None:
                # A forced visit kept apart from the root (forcing_apart). Nothing of it is in this
                # edge, so nothing is taken back or added, but the sign still turns for the ply.
                v = -v
                continue
            # N and sum_n were already incremented on the way down, so only the provisional value
            # is taken back here and the real value added.
            node.W[a] += v - x
            node.V2[a] += v * v
            node.F[a] -= 1.0
            node.X[a] -= x
            # Power mean. Standard MCTS carries the leaf value unchanged to the root. The power mean
            # re-reads this node's children after the update and sends up a value leaning towards
            # its best child, so a refutation reaches the root in fewer visits. At p = 1 _node_q
            # returns the plain average and this is a no-op.
            if self.power_p != 1.0:
                nq = self._node_q(node)
                if nq is not None:
                    v = nq
            # A proven node passes its exact value upward, whatever this visit found beneath it.
            # It can sit mid-path if it was proven while this visit was waiting on the network.
            if self.proofs and node.term is not None:
                v = node.term
            v = -v

    # ---------------------------------------------------------------- proofs
    def _bounds(self, node):
        """(lo, hi) for a node from its children's.

        A node can choose, so it is worth the best of them, and a child's [lo, hi] is [-hi, -lo]
        from here. An unexpanded edge could be anything, so it holds hi at +1."""
        lo, hi = -1.0, -1.0
        for kid in node.kids:
            if kid is None:
                hi = 1.0
                continue
            if -kid.hi > lo:
                lo = -kid.hi
            if -kid.lo > hi:
                hi = -kid.lo
        return lo, hi

    def _propagate(self, path):
        """Tighten the bounds above a proven result just backed up through path.

        Each node that becomes exact turns terminal, until one stops changing. The root's bounds
        are kept but it is never made terminal, since it still has to produce a move."""
        if not self.proofs or self.mcgs:
            return
        for i in range(len(path) - 1, -1, -1):
            node = path[i][0]
            if node.term is not None:
                return
            lo, hi = self._bounds(node)
            if self.mate_dist > 0.0 and lo > 0.5:
                # A proven win is exact at the mate it has found. A discounted win never reaches the
                # 1.0 an open sibling holds hi at, so without this the proof would stop one level
                # above the first mate.
                hi = lo
            if lo == node.lo and hi == node.hi:
                return
            if lo == hi and i != 0 and self.mate_dist > 0.0 and abs(lo) > 0.5:
                # One level further from the mate, floored just above the 0.5 every proof test reads
                # as decided, so a long proven mate still counts as a win or a loss.
                lo = hi = math.copysign(max(0.501, abs(lo) * (1.0 - self.mate_dist)), lo)
            node.lo, node.hi = lo, hi
            if lo != hi or i == 0:
                continue
            node.term = lo
            self.stats["solved"] += 1
            self._correct(path, i, lo)

    def _correct(self, path, i, t):
        """path[i]'s node has just been proven worth t, so rewrite the visits backed up through it.

        Every visit already backed up through it carried a guess. They are rewritten as t in its
        parent's edge and every ancestor's. Only completed visits are rewritten, since one still
        waiting gets t when it comes back (_backup passes a proven node's value upward). The same
        visits went through every ancestor with alternating sign, which is the whole correction for
        a plain mean. A power mean does not pass values up unchanged, so under one this is skipped.
        Clipped, because the root's statistics may carry the ponder discount and a correction in
        unscaled visits must not push a mean outside [-1, 1]."""
        if self.power_p != 1.0 or i == 0:
            return
        parent, a = path[i - 1][0], path[i - 1][1]
        done = float(parent.N[a] - parent.F[a])
        if done <= 0:
            return
        delta = done * -t - float(parent.W[a] - parent.X[a])
        dv2 = done * t * t - float(parent.V2[a])
        parent.W[a] += delta
        parent.V2[a] += dv2
        sign = -1.0
        for j in range(i - 2, -1, -1):
            node, b = path[j][0], path[j][1]
            n = float(node.N[b] - node.F[b])
            lo_w, hi_w = -n + float(node.X[b]), n + float(node.X[b])
            # The root may carry the ponder discount, which scales its N, W and V2 but not its
            # children's, so a correction counted in the visits below is weighted by the root edge's
            # share of them. Undiscounted the share is exactly one.
            k = 1.0
            if j == 0:
                below = path[1][0]
                seen = float(below.sum_n) - float(below.F.sum()) + 1.0
                k = min(1.0, n / seen) if seen > 0.0 else 1.0
            node.W[b] = min(hi_w, max(lo_w, float(node.W[b]) + sign * delta * k))
            node.V2[b] = min(n, max(0.0, float(node.V2[b]) + dv2 * k))
            sign = -sign

    # ---------------------------------------------------------------- the search
    def search(self, board, visits=None, reuse=True, clear_abort=True, seconds=None,
               hard_seconds=None, max_nodes=None, crit_seconds=None):
        """One autocast context for the whole search, then the search itself.

        autocast caches the bfloat16 copies of the weights and drops the cache when the outermost
        context exits, so one outer context lets every forward pass of the search reuse it. The
        inner contexts nest inside it. Same kernels, same numbers, same moves.

        Under timing_fixes the garbage collector is held off for the search (_gc_hold) and the
        weights' addresses are read once for _replay_autocast_fast. Neither changes a number.
        """
        if not self.timing_fixes:
            with torch.autocast("cuda", torch.bfloat16, enabled=(self.device != "cpu")):
                return self._run_search(board, visits, reuse, clear_abort, seconds, hard_seconds,
                                        max_nodes, crit_seconds)
        _gc_hold()
        try:
            self._nm = _native_moves()
            wk = _native_walk()
            if wk is not None:
                self._wku = wk
                self._wkp = _NATIVE_PRIOR[0]
                # The descent for _pick_fast's selection (the parts of fast in _descend fixed for a whole
                # search) with a virtual mean and no graph search, and the backup at power_p 1 with
                # proofs off.
                if (self.q_init and self.tree_policy == "puct_v" and not self.q_floor and not self.sigma_done
                        and not self.mlh and self.virtual_mean and not self.mcgs):
                    self._wk = wk
                    self._wk_params = (float(self.q_n0), float(self.c1), float(self.c2),
                                       float(self.explore_growth), float(self.explore_base),
                                       int(self.forcing), float(self.forcing_after), bool(self.forcing_apart))
                    self._wk_cb = (self._walk_term, self._walk_fm, self._walk_pick)
                if self.power_p == 1.0 and not self.proofs:
                    self._wkb = wk
            if self.model is not None and self.device != "cpu":
                self._search_ptrs = (tuple(t.data_ptr() for t in self.model.parameters())
                                     + tuple(t.data_ptr() for t in self.model.buffers()))
            with torch.autocast("cuda", torch.bfloat16, enabled=(self.device != "cpu")):
                return self._run_search(board, visits, reuse, clear_abort, seconds, hard_seconds,
                                        max_nodes, crit_seconds)
        finally:
            # nothing compiled outlives the search, so a tree later set without timing_fixes is pure Python
            self._search_ptrs = self._nm = None
            self._wk = self._wkb = self._wku = self._wk_params = self._wk_cb = self._wkp = None
            _gc_release()

    def _run_search(self, board, visits=None, reuse=True, clear_abort=True, seconds=None,
                    hard_seconds=None, max_nodes=None, crit_seconds=None):
        """The search. seconds is a deadline, which a real time control needs.

        Under engine "cpp" the search runs in mcts_cpp.run_search, the same control flow over the
        C++ tree, unless mcts_cpp.covers() names a setting the C++ search does not implement.

        A visit count alone does not control time, since the time per visit varies with the
        position, the cache and other load on the GPU. The deadline is checked in both loops, so a
        search stops between batches. MCTS always holds a usable best move, so stopping early is
        safe.
        """
        if self.engine == "cpp":
            import mcts_cpp
            why = mcts_cpp.covers(self)
            if why is None:
                return mcts_cpp.run_search(self, board, visits, reuse, clear_abort, seconds, hard_seconds,
                                           max_nodes, crit_seconds)
            mcts_cpp._warn(f"engine cpp: this search runs in Python ({why})")
            self._cpp_live = False
        # stop() sets the abort flag to end a ponder, so a new search clears it, or it would do no
        # visits and play root.moves[0]. ponder() passes clear_abort=False, so a background search
        # never clears a stop the foreground has asked for.
        if clear_abort:
            self.abort.clear()
        visits = visits or self.visits
        self.history = self._history_keys(board)
        # A snapshot, so the counters reported are this search's own. stats is zeroed once a game
        # by reset(). Taken before the book and proven move returns, which report it too.
        before_stats = dict(self.stats)
        # The book answers first, then a proven move is played without a search. Both drop the
        # tree, since nothing in it describes the position that follows.
        if self.book and len(board.move_stack) < self.book_plies:
            from book import pick as book_pick
            bm = book_pick(board, self.rng, self.book_plies)
            if bm is not None and bm in board.legal_moves:
                self.root = None
                self.stats["book"] += 1
                return bm, {**self._delta(before_stats), "visits": 0, "carried": 0, "move": bm,
                            "q": 0.0, "pv": bm.uci(), "top": [(bm.uci(), 0, 0.0)],
                            "proof": "opening book", "rows": [], "totals": dict(self.stats)}
        proven = self._proven_move(board)
        if proven is not None:
            mv, why = proven
            self.root = None
            self.stats["proven"] += 1
            # stats first, so a counter can never overwrite a field of the same name (the counter
            # is proven, the reason is proof)
            return mv, {**self._delta(before_stats), "visits": 0, "carried": 0, "move": mv,
                        "q": 1.0, "pv": mv.uci(), "top": [(mv.uci(), 0, 1.0)], "proof": why,
                        "rows": [], "totals": dict(self.stats)}
        # The clock starts here, before the setup (tree reuse, the ponder rescale, the history walk
        # and the root's own forward pass), so the setup counts against the deadline. At a fixed
        # visit count nothing below reads this.
        t_start = time.time()
        root = self._root_for(board) if reuse else None
        # how many visits carried over from the ponder, 0 when the move played was not in its tree
        carried = int(root.sum_n) if root is not None else 0
        # The ponder gets a vote, not a veto. Visit counts and value sums at the new root are scaled
        # together, so every W/N and variance estimate is unchanged and only the expected reply's
        # lead is cut down. Under q_init selection reads (q_n0*Q0 + W)/(q_n0 + N), so discount_q0
        # scales the root's prior weight too, keeping the ratio. v4 is sealed without it.
        if (carried and root.expanded and self.ponder_discount < 1.0
                and root.N is not None and root.N.sum() > 0):
            d = max(0.0, float(self.ponder_discount))
            root.N *= d
            root.W *= d
            root.V2 *= d
            root.sum_n = int(round(root.sum_n * d))
            # a zero discount keeps the prior's weight, since with N and W at zero it would read 0/0
            if self.discount_q0 and d > 0.0:
                root.q0w *= d
            self.stats["discounted"] += 1
        if root is None:
            root = Node(board.copy(stack=False), self.draw_plies, self._nm)
        # The root is always searched. The game's position can be a terminal of the previous tree
        # (a repetition or tablebase hit met as a grandchild, a claimable draw, a proven node), and
        # a root with legal moves must still return a move. This can only turn a missing move into
        # a move, so it applies to every version.
        if root.term is not None and root.moves:
            root.term = None
            root.lo, root.hi = self._bounds(root) if root.expanded else (-1.0, 1.0)
        self.root = root
        # Root noise comes off first. The root the last search noised gets its own priors back,
        # whether it is still the root (noised afresh below, never on top of the old noise) or has
        # dropped out of play (with the graph on, a transposition could bring it back).
        if self._noised is not None:
            node, p0 = self._noised
            node.P = p0
            self._noised = None
        if not root.expanded and root.term is None:
            pol, val, qv = self._evaluate([root])
            # The root's own value gets what every other leaf's gets, or its first-play urgency base
            # would sit on a different scale from its children.
            v0 = float(val[0])
            if self.calibrate:
                v0 = self._stretch(v0)
            if self.qdepth > 0:
                v0 = self._tactical(root.board, v0)
            self._expand(root, pol[0], v0, qv[0],
                         m0=None if self._ml is None else float(self._ml[0]))
        if (self.root_temp != 1.0 and root.P is not None and len(root.P) > 1
                and root.tempered != self.root_temp):
            # from whatever it was tempered with before to root_temp, so it never compounds
            pw = root.tempered / self.root_temp
            p = np.power(np.maximum(root.P.astype(np.float64), 1e-12), pw)
            root.P = (p / p.sum()).astype(np.float32)
            root.tempered = self.root_temp
        # The graph table is rebuilt for each root, since its keys carry the ply counted from the
        # root. Re-reading the carried tree keeps each merge target still reachable, at its depth
        # from this root, and drops everything else.
        self.tt = self._graph_table(root) if self.mcgs else {}

        # Entropy as a budget dial. A spread prior gets more budget than a confident one. Normalised
        # so a uniform prior is 1.0 and a certain one 0.0, then mapped to between half and twice the
        # budget. It scales time when there is a clock and visits otherwise, though it only really
        # pays under a clock, where saved time is spent on harder positions later.
        stretch = 1.0
        if self.adaptive and root.P is not None and len(root.P) > 1:
            pr = np.clip(root.P.astype(np.float64), 1e-9, 1.0)
            h = float(-(pr * np.log(pr)).sum() / math.log(len(pr)))
            stretch = 0.5 + 1.5 * h
            if seconds is None:
                visits = max(1, int(visits * stretch))
        # Root noise goes on after the budget is set from the clean prior, so noise never buys a
        # search more visits. Off (0.0) in every sealed search, selfplay.py turns it on.
        if self.root_noise > 0.0 and root.P is not None and len(root.P) > 1:
            p0 = root.P
            d = self.noise_rng.dirichlet(np.full(len(p0), self.noise_alpha / len(p0)))
            root.P = ((1.0 - self.root_noise) * p0.astype(np.float64)
                      + self.root_noise * d).astype(np.float32)
            self._noised = (root, p0)
        # A ceiling on the tree in nodes, for a caller that must stay inside a memory budget (uci.py
        # pondering with no visit limit). A visit makes at most one node, so holding visits to the
        # room left is enough. The room is read with tree_nodes(), not root.sum_n, because a
        # discounted root understates its tree. A budget like visits and seconds, not a sealed
        # setting, and None changes nothing.
        capped = False
        if max_nodes is not None:
            room = max(0, int(max_nodes) - self.tree_nodes(root))
            if room < visits:
                visits, capped = room, True
        target = root.sum_n + visits
        # A zero allowance is a deadline of now, never "no clock". Only None means there is no clock.
        budget = (seconds * stretch * self.time_scale) if seconds is not None else None
        deadline = (t_start + budget) if budget is not None else None
        # The furthest the deadline may ever be pushed, fixed here so an extension cannot extend
        # itself. The reserve is a fraction of this move's share of the clock, not of what is left.
        hard = ((t_start + budget * (1.0 + max(0.0, self.tm_extend_max)))
                if budget is not None else None)
        # A ceiling the caller worked out from both clocks. tm_extend_max only buys a little more
        # than this move's allocation, while hard_seconds, from a caller that knows the opponent's
        # clock, can give an unresolved position much longer, down to parity. It is only a ceiling,
        # the search still stops once the top two moves separate.
        if budget and hard_seconds:
            hard = max(hard or 0.0, t_start + float(hard_seconds))
        # A longer run for an unstable decision, never for a merely close one. crit_seconds is a
        # budget like seconds (not a sealed setting, None changes nothing), passed by the Lichess
        # bridge. Past hard the search keeps going only while the decision is still moving, that is
        # the most visited move is not the best valued one (as in Lc0 and Baier and Winands' UNST),
        # or the leader changed since the last deadline (Stockfish's best-move instability). "The
        # top two are close" fires too often with this value head, so it keeps the sealed cap.
        crit = (t_start + float(crit_seconds)) if (budget and crit_seconds) else None
        self._last_lead = None

        # The Gumbel shortlist, drawn once per search. Taking the top m of g(a) + log P(a) is
        # sampling m actions from the policy without replacement.
        self._root_allowed = None
        self._gumbel = None
        phase_end = float("inf")
        per_phase = 0.0
        # At a fixed visit count only. Its phases are shares of visits, which a clock caller sets
        # very high, and its survivors have equal visits by design, which tm_extend reads as close.
        if (self.tree_policy == "gumbel" and root.P is not None and len(root.moves) > 1
                and budget is None):
            m = int(min(self.gumbel_m, len(root.moves)))
            logits = np.log(np.maximum(root.P.astype(np.float64), 1e-9))
            g = self.rng.gumbel(size=len(root.moves))
            self._gumbel = (g, logits)
            self._root_allowed = np.argsort(-(g + logits))[:m].astype(np.int64)
            phases = max(1, int(math.ceil(math.log2(max(m, 2)))))
            per_phase = max(1.0, visits / float(phases))
            phase_end = root.sum_n + per_phase

        # The root check's reserve. The main search runs to (1 - share) of the budget, in visits or
        # in time, and the check then aims the rest at one reply or leaves it to PUCT.
        self._force = None
        checked = False
        check_at = check_time = None
        if (self.root_check and self._root_allowed is None
                and 0.0 < self.root_check_share < 1.0 and root.P is not None):
            if deadline is not None:
                now = time.time()
                check_time = now + (deadline - now) * (1.0 - self.root_check_share)
            else:
                check_at = root.sum_n + visits * (1.0 - self.root_check_share)

        # why the search ended, reported in info["stop"] and the engine log
        stop, ext_kind = "visits", None
        n_start = root.sum_n
        fly = None                        # the batch on the GPU, under pipeline
        base_end = deadline
        while root.sum_n - self._held(fly) < target and not self.abort.is_set():
            if deadline is not None and time.time() >= deadline:
                # Think longer while the decision is close. The allocation is up, so if the top two
                # moves still cannot be told apart, take some of the reserve and keep going. Not in
                # the opening, where several moves are close because they are all fine, and the
                # opening ramp's allocation is the ceiling. The game's ply is read off the position,
                # so a game set up from a FEN counts correctly.
                opening = (self.tm_open_spend < 1.0
                           and 2 * (board.fullmove_number - 1) + int(board.turn == chess.BLACK)
                           < self.tm_open_plies)
                # Nor while the root check holds the search on one line, since an extension spent
                # there gives the runner-up nothing and the top two could never separate.
                opening = opening or self._force is not None
                # the batch on the GPU holds collisions that are not visits (see _held)
                seen = root
                if fly is not None:
                    # Completed statistics only. The batch on the GPU is not evidence yet, its held
                    # collisions are not visits and its leaves carry provisional values. With nothing
                    # on the GPU this is the root itself.
                    seen = Node.__new__(Node)
                    seen.N = root.N - root.F
                    seen.W = root.W - root.X
                    seen.Q0 = root.Q0              # _settled prices an untried runner-up with it
                if (self.tm_extend and not opening and hard is not None and time.time() < hard
                        and not self._settled(seen, self.tm_margin, 1.2)):
                    # From now, not from the old deadline. The batch that crossed it can return late,
                    # and a deadline already in the past lets each gather make only one descent.
                    deadline = min(hard, max(deadline, time.time()) + budget * 0.25)
                    self.stats["extended"] += 1
                    ext_kind = ext_kind or "close"
                    if crit is not None:
                        self._unstable(seen)                # keeps the leader it last saw current
                elif (crit is not None and self.tm_extend and not opening and time.time() < crit
                        and self._unstable(seen)):
                    deadline = min(crit, max(deadline, time.time()) + budget * 0.25)
                    self.stats["extended"] += 1
                    self.stats["unstable"] = self.stats.get("unstable", 0) + 1
                    ext_kind = "unstable"
                else:
                    self.stats["timed_out"] += 1
                    stop = "time" if ext_kind is None else f"time after {ext_kind} extensions"
                    break
            if not checked and ((check_at is not None and root.sum_n >= check_at)
                                or (check_time is not None and time.time() >= check_time)):
                checked = True
                if fly is not None:
                    # The check reads the root's values and spreads, so the batch on the GPU is
                    # finished first.
                    self._pipe_finish(fly)
                    fly = None
                self._arm_root_check(root)
            if self.pipeline:
                fly, saturated = self._pipe_step(root, target, deadline, fly)
                if saturated:
                    break
            else:
                pending, leaves = [], []
                collided = []
                self._collided = collided if self.collide else None
                tries = 0
                spent_before = root.sum_n
                served_before = self.stats["cache_hits"]
                # _descend already counts each pending leaf in root.sum_n, so the fill condition
                # reads root.sum_n alone. Adding len(leaves) on top (fill_once off, as v1 is sealed)
                # counts them twice and sends out more, smaller batches, which changes some moves.
                # A held collision sits on the root's count until the batch returns and is not a
                # visit, so the budget reads past it.
                hold = self.collide == 2
                twice = not self.fill_once
                # Where a failure lands decides what is left of the tree. If the forward pass raises
                # (out of memory, a driver reset), the batch is taken back exactly and the tree
                # survives, so later searches do not collide with stuck pending nodes. A failure
                # anywhere else in the gather is a bug whose half-applied walk cannot be located, so
                # that tree is dropped and the next search starts clean. A lost tree costs visits, a
                # corrupt one costs moves.
                stage, done = "collect", 0
                try:
                    while (len(leaves) < self.batch and tries < 4 * self.batch
                           and root.sum_n + (len(leaves) if twice else 0)
                           - (len(collided) if hold else 0) < target):
                        if deadline is not None and tries and time.time() >= deadline:
                            break
                        tries += 1
                        got = self._descend(root)
                        if got is None:
                            continue
                        path, parent, a = got
                        self._gather(path, parent, a, pending, leaves)
                    if not leaves:
                        # Nothing was gathered, so nothing can be pending, and a held collision here
                        # can only be with a node an earlier failure left behind, so give it back.
                        for cpath, _cnode in collided:
                            self._unwind(cpath)
                        collided = []
                        self._collided = None
                        # A cached leaf is work, not saturation. This break is for a tree that cannot
                        # be descended at all, where every walk collides and unwinds. A pass served
                        # entirely from the cache collected no leaves but did advance the root, so it
                        # must not stop the search (cache_progress). A pass that ended only at
                        # terminals is still saturated, since once the root is proven every walk ends
                        # at a solved node. The two are told apart by whether the cache answered
                        # anything, since a terminal backup also advances the root.
                        if tries >= 4 * self.batch and (not self.cache_progress
                                                        or self.stats["cache_hits"] == served_before
                                                        or root.sum_n <= spent_before):
                            break
                        continue
                    stage = "eval"
                    pol, val, qv = self._evaluate(leaves)
                    mls = self._ml if self._ml is not None else [None] * len(pending)
                    for (path, kid, key), lg, v, q0, ml in zip(pending, pol, val, qv, mls):
                        kid.pending = False
                        # Applied here and not inside _backup, which also carries exact terminal and
                        # tablebase results that must not be stretched.
                        v = raw = float(v)
                        if self.calibrate:
                            v = self._stretch(v)
                        if self.qdepth > 0:
                            v = self._tactical(kid.board, v)
                        ml = None if ml is None else float(ml)
                        self._backup(path, self._expand(kid, lg, v, q0, m0=ml), m=ml)
                        done += 1
                        # Walks that collided with this leaf get the value they were waiting for.
                        # Held collisions (collide 2) are not visits and wait for the unwind below.
                        if not hold:
                            for cpath, cnode in collided:
                                if cnode is kid:
                                    self._backup(cpath, kid.value)
                            collided = [c for c in collided if c[1] is not kid]
                        # After the expansion, which built this node's legal-move indices. The cache
                        # keeps the network's raw value, and a hit applies the stretch and tactical
                        # correction to it as a fresh leaf gets them, so nothing is corrected twice.
                        if len(self.cache) < self.cache_size:
                            self.cache[key] = (self._own_row(lg, kid), raw, q0, ml)
                except BaseException:
                    if stage == "eval":
                        self._abandon(pending[done:], collided)
                    else:
                        self.root = None
                        self.tt = {}
                    self._collided = None
                    raise

                # Anything still waiting on a leaf this gather never evaluated gives its visit back,
                # so a phantom visit cannot outlive its batch. Under collide 2 that is every
                # collision, by design.
                for cpath, _cnode in collided:
                    self._unwind(cpath)
                collided = []
                self._collided = None

            # Halve the shortlist. Every survivor has had the same number of visits, so the
            # comparison between them is fair.
            if self._root_allowed is not None and root.sum_n >= phase_end:
                if fly is not None:
                    self._pipe_finish(fly)         # the survivors are ranked on completed visits
                    fly = None
                if len(self._root_allowed) > 1:
                    cand = self._root_allowed
                    keep = max(1, len(cand) // 2)
                    order = np.argsort(-self._gumbel_score(root, cand))
                    self._root_allowed = cand[order[:keep]]
                phase_end = root.sum_n + per_phase

            # Early stopping compares visit counts, so it cannot run under Sequential Halving, where
            # every survivor has the same count. Nor under lcb, where the lower bound picks the move
            # and more visits can still move it.
            if (self.early_stop and self._root_allowed is None and not self.lcb
                    and root.N is not None and len(root.N) >= 2):
                # If the runner-up cannot catch the leader even with every visit left, no further
                # work can change the move. Leela does the same, and it costs two array reads.
                rn = root.N if fly is None else root.N - self._held_edges(root, fly)
                order = np.argsort(rn)
                left = target - (root.sum_n - self._held(fly))
                got = root.sum_n - self._held(fly) - n_start
                if budget and got >= 64:
                    # Under a clock the visits still to come are what the time can buy, up to the
                    # furthest the deadline may be pushed, since clock callers pass a huge visit
                    # count. An estimate, not the fixed-visit proof.
                    now = time.time()
                    end = max(x for x in (deadline, hard, crit) if x is not None)
                    left = min(left, got / max(now - t_start, 1e-6) * max(0.0, end - now))
                if float(rn[order[-1]] - rn[order[-2]]) > left:
                    self.stats["early_stop"] += 1
                    break

            # A proven win at the root is the answer. More visits only reshuffle the proven wins.
            if self.proofs and root.lo > 0.5:
                break

            # Think less when it is obvious, only under a clock, since at fixed visits the saved time
            # is wasted. A third of the allocation is spent first. The clock rules read the root
            # without the batch still on the GPU.
            seen = root
            if fly is not None:
                # Completed statistics only. The batch on the GPU is not evidence yet, its held
                # collisions are not visits and its leaves carry provisional values.
                seen = Node.__new__(Node)
                seen.N = root.N - root.F
                seen.W = root.W - root.X
                seen.Q0 = root.Q0              # _settled prices an untried runner-up with it
            if (self.tm_stop and budget and time.time() - t_start > 0.33 * budget
                    and self._settled(seen, 2.0 * self.tm_margin, 2.0)):
                self.stats["banked"] += 1
                stop = "tm_stop"
                break
            if self.tm_catchup > 0.0 and budget and base_end is not None and ext_kind is None:
                now = time.time()
                done = root.sum_n - self._held(fly) - n_start
                # Before the base deadline only. Past it nothing of the base allowance is left, so
                # any lead would read as uncatchable before tm_extend or crit had their say.
                if (now - t_start > 0.1 * budget and now < base_end and done >= 64
                        and len(root.moves) > 1):
                    left = done / max(now - t_start, 1e-6) * max(0.0, base_end - now)
                    top2 = np.sort(seen.N)[-2:]
                    if float(top2[1] - top2[0]) > self.tm_catchup * left:
                        self.stats["caught"] = self.stats.get("caught", 0) + 1
                        stop = "catchup"
                        break

        if fly is not None:
            self._pipe_finish(fly)
        if self.abort.is_set() and stop == "visits":
            stop = "abort"
        elif capped and root.sum_n >= target:
            stop = "memory ceiling"
        elif stop == "visits" and root.sum_n < target:
            # the loop broke before the budget, from a proven root, the early stop or a saturated tree
            stop = ("proven" if (self.proofs and root.lo > 0.5)
                    else "early stop" if self.stats["early_stop"] > before_stats.get("early_stop", 0)
                    else "saturated")
        forced = self._force
        self._force = None
        if root.term is not None or not root.moves:
            return None, {"visits": 0, "term": root.term}
        if self._root_allowed is not None and len(self._root_allowed):
            # The surviving candidate with the best g + log P + sigma(q), not the most visited,
            # since under equal visits argmax(N) is a coin flip.
            cand = self._root_allowed
            best = int(cand[int(np.argmax(self._gumbel_score(root, cand)))])
        else:
            best = self._choose(root)
            if forced is not None:
                best = self._after_root_check(root, forced, best)
            # the game's ply, not the moves pushed on this board, since a FEN has no move stack
            if self.open_temp > 0 and board.ply() < self.open_plies:
                best = self._sample_opening(root, best)
        info = {
            **self._delta(before_stats),
            "visits": int(root.sum_n),
            "carried": carried,
            # The tree's size and whether the node ceiling ended the search. A ponder that stops
            # by itself has either filled its memory or proved the position.
            "tree_nodes": self.tree_nodes(root),
            "at_ceiling": bool(capped and root.sum_n >= target),
            # what it actually took against what it was allowed, so an overrun is on record
            "spent": round(time.time() - t_start, 3),
            "allowed": round(budget, 3) if budget is not None else None,
            "stop": stop,
            "move": root.moves[best],
            "q": float(root.W[best] / max(root.N[best], 1)),
            "pv": self._pv(root, first=best),
            "totals": dict(self.stats),          # the game-long counters, for anyone who wants them
            "top": sorted(((root.moves[i].uci(), int(root.N[i]),
                            float(root.W[i] / max(root.N[i], 1)))
                           for i in range(len(root.moves))), key=lambda t: -t[1])[:5],
        }
        info["rows"] = self.explain(root)
        return root.moves[best], info

    def _pipe_step(self, root, target, deadline, fly):
        """One cycle of the pipelined search (the pipeline key).

        Batch A is on the GPU. Gather batch B meanwhile, read A back, send B, then expand and back
        up A while the GPU runs B. Returns the batch now on the GPU and whether the tree is
        saturated, by the plain loop's own test. Held collisions (collide 2) are given back when
        their batch returns. A B walk that collided with one of A's leaves under collide 1 is backed
        up when A returns. A failure while gathering drops the tree, as the plain loop does, and one
        after it gives back every visit still waiting on the GPU, A's and B's."""
        hold = self.collide == 2
        twice = not self.fill_once
        bsz = self.batch if self.pipeline == 1 else max(1, self.batch // 2)
        pending, leaves, collided = [], [], []
        self._collided = collided if self.collide else None
        tries = 0
        spent_before = root.sum_n
        served_before = self.stats["cache_hits"]
        # A's held collisions sit on the root's count as B's would, and neither is a visit
        held = len(fly["collided"]) if (fly is not None and hold) else 0
        stage, a_open, h = "collect", fly is not None, None
        try:
            while (len(leaves) < bsz and tries < 4 * bsz
                   and root.sum_n + (len(leaves) if twice else 0)
                   - ((len(collided) + held) if hold else 0) < target):
                if deadline is not None and tries and time.time() >= deadline:
                    break
                tries += 1
                got = self._descend(root)
                if got is None:
                    continue
                path, parent, a = got
                self._gather(path, parent, a, pending, leaves)
            self._collided = None
            stage = "eval"
            prep = self._prepare(leaves) if leaves else None
            res = None
            if fly is not None:
                pol, val, qv = self._collect(fly["h"])
                res = (pol, val, qv, self._ml if self._ml is not None else [None] * len(pol))
            if prep is not None:
                h = self._dispatch(prep)
            if fly is not None:
                a_open = False              # _pipe_complete gives back whatever it cannot finish
                self._pipe_complete(fly, res, collided)
        except BaseException:
            self._collided = None
            if stage == "eval":
                if a_open:
                    self._abandon(fly["pending"], fly["collided"])
                self._abandon(pending, collided)
            else:
                self.root = None
                self.tt = {}
            raise
        if h is not None:
            return {"h": h, "pending": pending, "collided": collided}, False
        # nothing went to the GPU, so whatever collided has nothing left to wait for
        for cpath, _cnode in collided:
            self._unwind(cpath)
        if fly is not None:
            return None, False                  # A's leaves were backed up, which is progress
        # a cached leaf is work, not saturation, the plain loop's test
        return None, (tries >= 4 * bsz and (not self.cache_progress
                                            or self.stats["cache_hits"] == served_before
                                            or root.sum_n <= spent_before))

    # Held collisions are not visits, and under the pipeline the batch on the GPU still holds some
    # when the loop looks at the root. Read as visits they would end a search early and inflate the
    # leader for the early stop, so both reads discount them. With nothing on the GPU the discount
    # is zero.
    def _held(self, fly):
        if fly is None or self.collide != 2:
            return 0.0
        return float(len(fly["collided"]))

    def _held_edges(self, root, fly):
        held = np.zeros_like(root.N)
        if self.collide == 2:
            for cpath, _cnode in fly["collided"]:
                if cpath and cpath[0][0] is root and cpath[0][2] is not None:
                    held[cpath[0][1]] += 1.0
        return held

    def _pipe_complete(self, fly, res, later):
        """Expand and back up a batch the GPU has answered, exactly as the plain loop does.

        later is the collided list of the batch gathered after it, whose walks may be waiting on
        this batch's leaves."""
        pending, collided = fly["pending"], fly["collided"]
        pol, val, qv, mls = res
        hold = self.collide == 2
        done = 0
        try:
            for (path, kid, key), lg, v, q0, ml in zip(pending, pol, val, qv, mls):
                kid.pending = False
                v = raw = float(v)
                if self.calibrate:
                    v = self._stretch(v)
                if self.qdepth > 0:
                    v = self._tactical(kid.board, v)
                ml = None if ml is None else float(ml)
                self._backup(path, self._expand(kid, lg, v, q0, m0=ml), m=ml)
                done += 1
                if not hold:
                    for lst in (collided, later):
                        for cpath, cnode in lst:
                            if cnode is kid:
                                self._backup(cpath, kid.value)
                        lst[:] = [c for c in lst if c[1] is not kid]
                if len(self.cache) < self.cache_size:
                    self.cache[key] = (self._own_row(lg, kid), raw, q0, ml)
        except BaseException:
            self._abandon(pending[done:], collided)
            collided.clear()
            raise
        for cpath, _cnode in collided:
            self._unwind(cpath)
        collided.clear()

    def _pipe_finish(self, fly):
        """Finish the batch still on the GPU when the search ends.

        Its visits are already counted, so it is read back and backed up rather than thrown away."""
        try:
            pol, val, qv = self._collect(fly["h"])
        except BaseException:
            self._abandon(fly["pending"], fly["collided"])
            raise
        mls = self._ml if self._ml is not None else [None] * len(pol)
        self._pipe_complete(fly, (pol, val, qv, mls), [])

    def _gather(self, path, parent, a, pending, leaves):
        """Handle one walk's leaf, reached by _descend through parent's edge a.

        Merge it into the graph, score it as a terminal, serve it from the cache, or queue it for
        the network."""
        b = parent.board.copy(stack=False)
        b.push(parent.moves[a])
        gkey = None
        if self.mcgs:
            # The merge. The same position at the same ply reached by another move order points at
            # the existing node rather than splitting statistics across two subtrees. The traversal
            # stops here, which is where the paper's saving in network evaluations comes from. A
            # pending node is left alone, since it has no statistics yet.
            gkey = (b._transposition_key(), len(path))
            shared = self.tt.get(gkey)
            # Never into a repetition, since the shared node was built on a path that did not repeat
            # and the _repeats test below is only reached without a merge. Never across fifty-move
            # counts either, since the key has no clock and the network, the cache_clock key,
            # draw_plies and the tablebase's rule 50 all read it.
            if (shared is not None and shared.expanded and not shared.pending
                    and shared.board.halfmove_clock == b.halfmove_clock
                    and not self._repeats(gkey[0], path)):
                parent.kids[a] = shared
                self.stats["transpositions"] += 1
                if self.mcgs == 2:
                    # Merge without the early stop. mcgs=1 lost Elo here, because the early stop
                    # saves a forward pass the eval cache already saves, and costs tree depth. Here
                    # the link is made and the descent abandoned, so the next descent walks through
                    # the shared node and keeps going deeper.
                    self._unwind(path)
                    return
                self._backup(path, self._merge_value(parent, a, shared))
                return
        kid = Node(b, self.draw_plies, self._nm)
        parent.kids[a] = kid
        if gkey is not None and len(self.tt) < self.tt_cap:
            # a live merge target stays, a later duplicate (a terminal above all) must not replace it
            old = self.tt.get(gkey)
            if old is None or not (old.expanded or old.pending):
                self.tt[gkey] = kid
        if kid.term is None and self._repeats(kid.key, path):
            kid.term = 0.0                 # by convention, so its bounds stay open
            self.stats["repetitions"] += 1
        if kid.term is None:
            tbv = self._tb_value(kid.board)
            if tbv is not None:
                if self.mate_dist > 0.0 and tbv != 0.0:
                    # A tablebase result under mate_dist is a mate past the horizon. At +-1 it would
                    # outrank every mate the tree finds. It is worth what a mate |dtz| plies below is
                    # proven at, (1 - mate_dist) a ply (the distance to the next zeroing move bounds
                    # the distance to mate from below), floored at 0.501 like every long mate.
                    d = self._tb_dtz(kid.board)
                    tbv = math.copysign(max(0.501, (1.0 - self.mate_dist) ** (100 if d is None else d)),
                                        tbv)
                kid.term = tbv
                # exact only when the probe read the fifty-move counter
                if self.tb_rule50:
                    kid.lo = kid.hi = tbv
                self.stats["tablebase"] += 1
        if kid.term is not None:
            tv = kid.term
            if self.mate_dist > 0.0 and tv <= -1.0 and not kid.moves:
                tv = -max(0.5, 1.0 - self.mate_dist * len(path))   # checkmate, len(path) plies down
            self._backup(path, tv, m=0.0 if kid.board.is_game_over() else None)
            if kid.lo == kid.hi:
                self._propagate(path)
            return
        key = self._cache_key(kid)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cache_hits"] += 1
            hv = float(hit[1])
            if self.calibrate:
                hv = self._stretch(hv)
            if self.qdepth > 0:
                hv = self._tactical(kid.board, hv)
            hm = hit[3] if len(hit) > 3 else None
            self._backup(path, self._expand(kid, hit[0], hv, hit[2], sliced=True, m0=hm), m=hm)
            return
        kid.pending = True
        pending.append((path, kid, key))
        leaves.append(kid)

    def _cache_key(self, kid):
        """What the evaluation cache files a leaf under.

        v1 to v4 used the position alone. The network also reads the halfmove clock, capped at 100
        as clock_and_ep feeds it, so under cache_clock the clock is part of the key. En passant needs
        nothing extra, since the position key carries the square only when a capture is legal,
        which is when clock_and_ep feeds it."""
        if self.cache_clock:
            return (kid.key, min(int(kid.board.halfmove_clock), 100))
        return kid.key

    @staticmethod
    def _own_row(lg, kid):
        """The legal-move slice a cache entry keeps.

        A row _collect cut on the GPU (sw25_one_copy, which owns its data) as it is, else a copy of
        the full row's legal slice."""
        return lg.row if isinstance(lg, _Cut) else lg[kid.idx].copy()

    def _abandon(self, pending, collided):
        """A forward pass failed. Give back every visit the batch was holding.

        The leaves it never evaluated are detached, so the tree is what it was before the batch
        plus the walks that finished without the network (cached, terminal, merged). A detached
        leaf is rebuilt by the next walk that reaches it, while one left attached, unexpanded and
        not pending would crash that walk."""
        for cpath, _cnode in collided:
            self._unwind(cpath)
        for path, kid, _key in pending:
            kid.pending = False
            self._unwind(path)
            parent, a = path[-1][0], path[-1][1]
            if parent.kids[a] is kid:
                parent.kids[a] = None

    def _graph_table(self, root):
        """The MCGS table for this root.

        Every expanded node still reachable from it, keyed as the merge keys it, by position and by
        depth below this root. Every link goes from depth k to depth k+1, so each node has one depth
        and the graph stays layered and acyclic. Anything no longer reachable is left out."""
        tt = {}
        frontier, depth, seen = [root], 0, {id(root)}
        while frontier:
            depth += 1
            nxt = []
            for n in frontier:
                for k in (n.kids or ()):
                    if k is None or id(k) in seen:
                        continue
                    seen.add(id(k))
                    if k.expanded:
                        if len(tt) >= self.tt_cap:
                            return tt
                        tt.setdefault((k.key, depth), k)
                        nxt.append(k)
            frontier = nxt
        return tt

    def _delta(self, before):
        """This search's counters, the game-long ones less a snapshot taken at its start."""
        return {k: v - before.get(k, 0) for k, v in self.stats.items()}

    def _pv(self, root, depth=12, first=None):
        """The line the tree expects, the move actually chosen, then the most visited below it."""
        if type(root).__name__ == "RootView":
            import mcts_cpp
            return mcts_cpp.pv(self, depth, first)
        out, node = [], root
        while node is not None and node.expanded and node.sum_n > 0 and len(out) < depth:
            a = first if (node is root and first is not None) else int(np.argmax(node.N))
            if node.N[a] < 1:
                break
            out.append(node.moves[a].uci())
            node = node.kids[a]
        return " ".join(out)

    # ---------------------------------------------------------------- which move is played
    def _choose(self, root):
        """The root move. With everything off this is argmax N exactly, as v1 to v3 play.

        Proofs come first. A move proven to win is played, the most visited of them if there are
        several, and a move whose best case cannot reach what another move guarantees is never
        played. Then LCB or the visit count decides among what is left."""
        n = root.N
        cand = np.arange(len(root.moves))
        if self.proofs:
            lo = np.array([-1.0 if k is None else -k.hi for k in root.kids])
            hi = np.array([1.0 if k is None else -k.lo for k in root.kids])
            if lo.max() > 0.5:
                win = np.flatnonzero(lo > 0.5)
                if self.mate_dist > 0.0:
                    # the shortest proven win has the largest bound, and visits break a tie
                    best = lo[win].max()
                    win = win[lo[win] >= best - 1e-9]
                return int(win[int(np.argmax(n[win]))])
            g = float(lo.max())
            cand = np.flatnonzero(~((hi < g) | ((hi == g) & (lo < g))))
            # A proven loss is never played while another move might not lose. An unproven move's
            # lower bound is also -1, so the rule above would keep a proven loss that had most of
            # the visits before its refutation was proven.
            alive = cand[hi[cand] > -0.5]
            if len(alive):
                cand = alive
        # No visits at all is not a tie between every move. argmax of zeros is index 0, the first
        # move python-chess generates, so under prior_fallback the root's prior favourite plays.
        # v1 to v4 are sealed without it.
        if self.prior_fallback and root.P is not None and float(n[cand].sum()) <= 0.0:
            return int(cand[int(np.argmax(root.P[cand]))])
        if self.lcb:
            return self._lcb_pick(root, cand)
        return int(cand[int(np.argmax(n[cand]))])

    def _sample_opening(self, root, best):
        """Vary the opening while keeping the model's own distribution.

        A move is drawn in proportion to the visits the search gave it, N^(1/open_temp), so what
        comes out is what this network likes. Only the argmax is given up, and only for the first
        open_plies. Two guards keep it sound. A visit floor drops moves the search barely looked
        at, whose average may flatter them, and a value margin drops moves rated clearly worse.
        """
        n = root.N.astype(np.float64)
        if n.max() < 2:
            return best
        q = root.W / np.maximum(root.N, 1)
        ok = np.flatnonzero((n >= max(2.0, self.open_floor * n.max()))
                            & (q >= q[best] - self.open_margin))
        # The move it would have played is always a candidate. With lcb on, best is not argmax N,
        # and a floor in visits could otherwise drop it.
        ok = np.union1d(ok, [best])
        if len(ok) < 2:
            return best
        # In log space, since N^(1/temp) overflows at a small temperature, and the board accepts
        # any temperature in a request.
        lg = np.log(n[ok]) / max(self.open_temp, 1e-3)
        w = np.exp(lg - lg.max())
        w = w / w.sum()
        pick = int(ok[int(self.rng.choice(len(ok), p=w))])
        if pick != best:
            self.stats["varied"] = self.stats.get("varied", 0) + 1
        return pick

    def _lcb_pick(self, root, cand):
        """The candidate whose value is best at its lower bound, Leela Zero's rule.

        Eligible only with at least lcb_floor of the leader's visits and two of its own, so a move
        seen a handful of times cannot win on a lucky average. The bound is Student-t on the sample
        variance of the values backed up through the move, one-sided at lcb_alpha. A proven result
        has no spread, so it stands at its value."""
        n = root.N[cand].astype(np.float64)
        top = int(cand[int(np.argmax(n))])
        ok = n >= max(2.0, self.lcb_floor * float(n.max()))
        if int(ok.sum()) < 2:
            return top
        c, nn = cand[ok], n[ok]
        mean = root.W[c].astype(np.float64) / nn
        var = np.maximum(root.V2[c].astype(np.float64) - nn * mean * mean, 0.0) / (nn - 1.0)
        lcb = mean - _t_quantile(self.lcb_alpha, nn - 1.0) * np.sqrt(var / nn)
        best = int(c[int(np.argmax(lcb))])
        if best != top:
            self.stats["lcb_overrode"] += 1
        return best

    def _unstable(self, root):
        """Whether the decision is still moving.

        True when the most visited move is not the best valued of the moves with a real share of the
        visits (5 percent, and at least 2), or when the most visited move changed since the last
        call. Remembers the leader either way."""
        n = root.N
        if n is None or len(n) < 2 or float(n.sum()) <= 0:
            return False
        lead = int(np.argmax(n))
        last = getattr(self, "_last_lead", None)
        moved = last is not None and lead != last
        self._last_lead = lead
        floor = max(2.0, 0.05 * float(n.sum()))
        seen = np.flatnonzero(n >= floor)
        if len(seen) < 2:
            return moved
        q = root.W[seen] / n[seen]
        return moved or int(seen[int(np.argmax(q))]) != lead

    def _settled(self, root, margin, ratio):
        """Whether the decision is made.

        The leader has to be ahead of the runner-up in value by margin and in visits by ratio. Value
        alone is not enough at a few hundred visits, where a move seen twenty times can hold a
        flattering average."""
        n = root.N
        if n is None or len(n) < 2:
            return True
        order = np.argsort(-n)
        a, b = int(order[0]), int(order[1])
        if n[a] < 2:
            return False
        qa = float(root.W[a]) / float(n[a])
        if n[b] < 1:
            # The leader has every visit, as with a forced recapture. The runner-up is then the best
            # move the tree never tried, at the q head's price when the root has one, so a move the
            # prior buried but the q head likes still keeps the search going.
            q0 = getattr(root, "Q0", None)
            if q0 is None:
                return True
            return (qa - float(np.max(q0[n < 1]))) >= margin
        qb = float(root.W[b]) / float(n[b])
        return (qa - qb) >= margin and float(n[a]) >= ratio * float(n[b])

    def _arm_root_check(self, root):
        """Before the reserve is spent, check whether the q head's best reply refutes the chosen move.

        The q head priced every reply when the chosen move's node was expanded, so the check costs
        no forward pass. If the reply the q head likes best for the opponent leaves this side worse
        than the tree's Q by the margin, and the tree has given that reply almost nothing, the
        reserve is aimed down that one line."""
        seen = self.stats["lcb_overrode"]
        best = self._choose(root)
        self.stats["lcb_overrode"] = seen        # a look before the reserve, not a move played
        kid = root.kids[best]
        if kid is None or not kid.expanded or kid.term is not None or kid.Q0 is None:
            return
        r = int(np.argmax(kid.Q0))
        after = -float(kid.Q0[r])
        q_tree = float(root.W[best] / max(float(root.N[best]), 1.0))
        if after >= q_tree - self.root_check_margin:
            return
        if float(kid.N[r]) >= max(4.0, 0.1 * float(kid.N.sum())):
            return                             # the tree has already weighed that reply
        self._force = {id(root): best, id(kid): r}
        self.stats["root_checks"] += 1

    def _after_root_check(self, root, forced, best):
        """After the root check, keep the chosen move only if it is still worth more than the runner-up.

        The reserve was aimed at the chosen move, so its visit count is inflated by design."""
        chosen = forced.get(id(root))
        if chosen is None or chosen != best or len(root.moves) < 2:
            return best
        n = root.N.astype(np.float64).copy()
        n[chosen] = -1.0
        if self.proofs:
            # The proofs outrank the check (a kid's bounds are the opponent's), so a proven win
            # stays and a proven loss never comes in.
            if root.kids[chosen] is not None and root.kids[chosen].hi < -0.5:
                return best
            for i, k in enumerate(root.kids):
                if k is not None and k.lo > 0.5:
                    n[i] = -1.0
        alt = int(np.argmax(n))
        if n[alt] < 1.0:
            return best
        q = lambda i: float(root.W[i]) / max(float(root.N[i]), 1.0)
        return alt if q(chosen) < q(alt) else best

    def tree_nodes(self, root=None):
        """An upper bound on the nodes in the tree below root, in O(branching).

        Every node is made by one visit, and every visit below the root passes through exactly one
        child, whose own sum_n counts it. Only the root's counts are ever rescaled (the ponder
        discount), so the children's are exact. A visit that ends on a terminal or proven node makes
        no node, which errs on the safe side. Checked against a walk of the tree in selftest."""
        r = self.root if root is None else root
        if r is None:
            return 0
        if type(r).__name__ == "RootView":
            import mcts_cpp
            return mcts_cpp.tree_nodes(self)
        if not r.kids:
            return 1
        return 1 + sum(1 + int(k.sum_n) for k in r.kids if k is not None)

    def _root_for(self, board):
        """The subtree for this position if the last search already built it."""
        r = self.root
        if r is None or not r.expanded:
            return None
        key = board._transposition_key()
        if r.board._transposition_key() == key:
            return r
        for kid in r.kids:
            if kid is None:
                continue
            if kid.board._transposition_key() == key:
                return kid
            if not kid.expanded or kid.kids is None:
                continue
            for g in kid.kids:
                if g is not None and g.board._transposition_key() == key:
                    return g
        return None

    def ponder(self, board, visits=1_000_000):
        """Search the current position until told to stop. The tree keeps whatever it found.

        The caller clears the abort flag before starting this thread. Clearing it here would race
        with a stop() that arrives while the thread is still being scheduled.
        """
        try:
            self.search(board, visits=visits, reuse=True, clear_abort=False)
        except Exception as e:
            # stderr, since a pondering engine's stdout is the UCI channel
            print(f"[ponder] {type(e).__name__}: {e}", file=sys.stderr, flush=True)

    def stop(self):
        self.abort.set()

    def reset(self):
        self.abort.clear()
        self.root = None
        if self._cpp is not None:
            import mcts_cpp
            mcts_cpp.reset(self)
        self._noised = None
        self.cache.clear()
        self.tt.clear()
        for k in self.stats:
            self.stats[k] = 0
        if self.timing_fixes:
            # a new game, so what the searches froze is collected once here (_gc_hold)
            with _GC_LOCK:
                if _GC_HELD[0] == 0:
                    gc.unfreeze()
                    gc.collect()


# -------------------------------------------------------------------- raw head, for comparison
@torch.inference_mode()
def policy_move(model, board, device, head="cls", castle="king"):
    """The bare network's move, from the head a version is sealed with.

    Castling is read where castle_prior says the head keeps it ("rook" for v4)."""
    ids, cast, flip = board_to_ids(board)
    hm, ep = clock_and_ep(board, flip)
    # The castling input in the network's own dtype, since a FAST_EVAL tree may have cast a shared
    # network to bfloat16.
    wd = getattr(next(model.parameters(), None), "dtype", torch.float32)
    out = model(torch.tensor([ids], device=device), torch.tensor([cast], device=device, dtype=wd),
                torch.tensor([hm], device=device), torch.tensor([ep], device=device))
    legal = list(board.legal_moves)
    if not legal:
        return None
    key = "policy_att" if (head == "att" and out.get("policy_att") is not None) else "policy"
    idx = torch.from_numpy(policy_index(board, legal, flip, castle)).to(device)
    return legal[int(out[key][0][idx].argmax())]


def load(tag="engine", device="cuda"):
    from chess_net import ChessNet
    p = os.path.join(HERE, f"ckpt_{tag}.pt")
    ck = torch.load(p, map_location=device)
    c = ck.get("cfg", {})
    m = ChessNet(d=c.get("d", 1024), nhead=c.get("nhead", 16), layers=c.get("layers", 16),
                 dim_ff=int(c.get("ff_mult", 2.0) * c.get("d", 1024)),
                 qhead=bool(c.get("qhead", "qvalue.weight" in ck["model"])),
                 att=bool(c.get("att", "att.q.weight" in ck["model"])),
                 dist=int(c.get("dist", 64 if "vdist.weight" in ck["model"] else 0)),
                 mlh=bool(c.get("mlh", "mleft.0.weight" in ck["model"]))).to(device)
    # The move space grew from 4,096 to 4,162 for underpromotions, and the clock and en passant
    # inputs added tensors older checkpoints lack. Every old index still means the same move, so the
    # old weights are copied into the overlap and the new rows keep their initial values.
    sd = dict(ck["model"])
    cur = m.state_dict()
    grown = []
    for k, v in list(sd.items()):
        if k in cur and cur[k].shape != v.shape:
            t = cur[k].clone()
            cut = tuple(slice(0, min(a, b)) for a, b in zip(t.shape, v.shape))
            t[cut] = v[cut]
            sd[k] = t
            grown.append(f"{k} {tuple(v.shape)}->{tuple(t.shape)}")
    if grown:
        print(f"  widened from an older move space: {', '.join(grown)}")
    missing, _ = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"  {len(missing)} tensor(s) not in this checkpoint, left at their "
              f"zero init: {', '.join(missing)}")
    m.eval()
    return m, ck


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load(E("TAG", "engine"), dev)
    print(f"{model.num_params()/1e6:.1f}M params, checkpoint at iter {ck.get('iter', '?')}, {dev}\n")

    if cmd == "bench":
        b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        print(f"{'batch':>6} {'visits':>8} {'seconds':>8} {'nodes/s':>9} {'evals':>8} "
              f"{'cached':>7} {'collide':>8}")
        for bs in (1, 8, 16, 32, 48, 64, 128, 256):
            s = MCTS(model, dev, batch=bs)
            s.search(b, visits=64)                       # warm the kernels
            s.reset()
            t0 = time.time()
            _, info = s.search(b, visits=1600, reuse=False)
            el = time.time() - t0
            print(f"{bs:>6} {info['visits']:>8,} {el:>8.2f} {info['visits']/el:>9,.0f} "
                  f"{info['evals']:>8,} {info['cache_hits']:>7,} {info['collisions']:>8,}")

    elif cmd == "repetition":
        # After 1.Nf3 Nf6 2.Ng1 Ng8 the position after 1.Nf3 has already occurred, so Nf3 from here
        # repeats and the tree must score it a draw rather than ask the value head.
        b = chess.Board()
        for u in ("g1f3", "g8f6", "f3g1", "f6g8"):
            b.push(chess.Move.from_uci(u))
        tree = MCTS(model, dev, batch=32)
        mv, info = tree.search(b, visits=600, reuse=False)
        uci = [m.uci() for m in tree.root.moves]
        rep, new = tree.root.kids[uci.index("g1f3")], tree.root.kids[uci.index("b1c3")]
        print(f"  the game has been through {len(tree.history)} earlier positions")
        print(f"  {info['repetitions']} repetition(s) found in {info['visits']} visits, "
              f"best move {mv.uci()}")
        print(f"  Nf3, which repeats: terminal={rep is not None and rep.term is not None}, "
              f"value={None if rep is None else rep.term}")
        print(f"  Nc3, which does not: terminal={new is not None and new.term is not None}")

        fresh = chess.Board(b.fen())          # the same position with no history at all
        t2 = MCTS(model, dev, batch=32)
        _, i2 = t2.search(fresh, visits=600, reuse=False)
        u2 = [m.uci() for m in t2.root.moves]
        rep2 = t2.root.kids[u2.index("g1f3")]
        print(f"  same position, history thrown away: Nf3 terminal="
              f"{rep2 is not None and rep2.term is not None}, "
              f"{i2['repetitions']} repetition(s) found deeper in the tree")

        ok = (rep is not None and rep.term == 0.0
              and (new is None or new.term is None)
              and (rep2 is None or rep2.term is None))
        print(f"  {'PASS' if ok else 'FAIL'}  a repeated position scores a draw, a new one does not")

    elif cmd == "tactics":
        # Mate in one. python-chess finds every mating move in each position, so a badly chosen
        # test fails loudly instead of quietly scoring a miss.
        FENS = [
            ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "back rank"),
            ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "scholar"),
            ("rnbqkbnr/ppp2ppp/8/3pp3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 3", "fool"),
            ("7k/8/6KQ/8/8/8/8/8 w - - 0 1", "queen and king"),
            ("2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", "double knight"),
        ]
        tests = []
        for fen, name in FENS:
            b = chess.Board(fen)
            mates = set()
            for m in b.legal_moves:
                b.push(m)
                if b.is_checkmate():
                    mates.add(m.uci())
                b.pop()
            if not mates:
                print(f"  SKIP {name}: no mate in one in this position")
                continue
            tests.append((fen, mates, name))
        print(f"{len(tests)} mate in one positions\n")
        for visits in (1, 100, 400, 1600):
            hit, t0 = 0, time.time()
            got = []
            for fen, mates, name in tests:
                b = chess.Board(fen)
                if visits == 1:
                    mv = policy_move(model, b, dev)
                else:
                    s = MCTS(model, dev, batch=32)
                    mv, _ = s.search(b, visits=visits, reuse=False)
                ok = mv is not None and mv.uci() in mates
                hit += bool(ok)
                got.append(f"{name} {mv.uci() if mv else '-'}{'' if ok else ' MISS'}")
            label = "raw policy head" if visits == 1 else f"{visits} visits"
            print(f"  {label:>16}: {hit}/{len(tests)}  {time.time()-t0:5.1f}s   "
                  + " | ".join(got))
