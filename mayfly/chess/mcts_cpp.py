r"""The C++ search behind MCTS's sealed engine key "cpp". v6 runs on it, v1 to v5 are sealed "python".

It is a port of the Python search that returns the same moves, and a tree runs it when its engine is "cpp".

  the tree      mayfly_cppsearch (chess/cpp) does selection, collisions, forcing, backup, the evaluation
                cache, repetitions, the LCB pick, tree reuse and the ponder discount, and the tablebases
                (asked through this tree's own _tb_value).
  the network   runs in C++ too when a TorchScript trace of this network is found (cpp/export_ts.py "cuda",
                made by export_trace() below, beside the checkpoint as <name>.cuda.ts with <name>.cuda.ts.json
                holding the weights' fingerprint). That is LibTorch with a CUDA graph per batch size. Without a
                trace, the tree calls back into this tree's own _dispatch and _collect, with the same numbers,
                slower.
  the clock     run_search is MCTS._run_search's control flow (deadline, tm_extend, crit, tm_stop, tm_catchup,
                the abort flag, the node ceiling, the stop reasons), reading the C++ root between batches.
  unchanged     the opening book, the proven move, and everything the bot calls (search, abort, ponder,
                reset, _pv).

It runs about 1.3 to 1.5 times the visits a second of the Python search. A setting the C++ tree does not
implement (covers() names it), a missing extension or a trace of another network falls back to the Python
search or the Python network, never to a different search.

MCTS's pipeline key (two batches in flight) runs on the C++ tree in a build that has Tree.finish, with
MCTS._pipe_step's order of tree operations and the next batch queued on the GPU before the last is read.
run_search then reads the root as the Python does under the pipeline, where held collisions are not visits
and the clock rules see completed visits only.
"""
import hashlib
import json
import os
import sys
import time
import weakref

import chess
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.environ.get("CPPSEARCH_DIR", HERE)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
try:
    import mayfly_cppsearch  # noqa: E402  (the build links one LibTorch version and loads only beside the matching torch)
    IMPORT_ERROR = None
except Exception as e:                       # another torch, or no build at all, so the Python search runs
    mayfly_cppsearch = None
    IMPORT_ERROR = f"{type(e).__name__}: {e}"

# the tree's settings, by the MCTS attribute that holds each one
_KEYS = ("batch", "policy_temp", "q_n0", "c1", "c2", "explore_growth", "explore_base", "forcing", "forcing_after",
         "forcing_apart", "collide", "cache_progress", "fill_once", "draw_plies", "lcb", "lcb_alpha", "lcb_floor",
         "prior_fallback", "cache_size")
_WARNED = set()


def _warn(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[mcts_cpp] {msg}", file=sys.stderr, flush=True)


# -------------------------------------------------------------------- what the C++ tree covers
def covers(tree):
    """None when the C++ search can run this tree exactly, else the first reason it cannot."""
    if mayfly_cppsearch is None:
        return f"no mayfly_cppsearch ({IMPORT_ERROR})"
    if not tree.timing_fixes:
        return "timing_fixes"
    checks = (
        ("tree_policy", tree.tree_policy == "puct_v"),
        ("q_init", bool(tree.q_init)),
        ("virtual_mean", bool(tree.virtual_mean)),
        ("value_from_q", bool(tree.value_from_q)),
        ("mcgs", not tree.mcgs),
        ("pipeline", not tree.pipeline or hasattr(mayfly_cppsearch.Tree, "finish")),
        ("proofs", not tree.proofs),
        ("root_check", not tree.root_check),
        ("root_noise", not tree.root_noise),
        ("root_temp", tree.root_temp == 1.0),
        ("adaptive", not tree.adaptive),
        ("early_stop", not tree.early_stop),
        ("calibrate", not tree.calibrate),
        ("qdepth", not tree.qdepth),
        ("mlh", not tree.mlh),
        ("q_floor", not tree.q_floor),
        ("sigma_done", not tree.sigma_done),
        ("power_p", float(tree.power_p) == 1.0),
        ("castle_prior", tree.castle_prior == "king"),
        ("opponent", tree.opponent is None),
        ("open_temp", not tree.open_temp),
        ("mate_dist", not tree.mate_dist),
        ("fast_eval", not tree.fast_eval),
    )
    for name, ok in checks:
        if not ok:
            return name
    return None


def _tb_live(tree):
    if not tree.use_tb:
        return False
    from mcts import tablebase
    return tablebase() is not None


def _settings(tree):
    from mcts import TB_PIECES
    cfg = {k: float(getattr(tree, k)) for k in _KEYS}
    cfg["use_tb"] = float(_tb_live(tree))
    cfg["tb_pieces"] = float(TB_PIECES)
    if tree.pipeline:                # only when on, so a tree without it is made exactly as before
        cfg["pipeline"] = float(tree.pipeline)
    return cfg


# -------------------------------------------------------------------- the network's trace
def fingerprint(model):
    """A cheap identity for a network's weights.

    Hashes the first 32 values of the first and last four tensors of its state dict, as float32. Written
    beside a trace when it is exported, compared before the trace is used."""
    sd = model.state_dict()
    keys = [k for k in sd if sd[k].is_floating_point()]
    h = hashlib.sha1()
    for k in keys[:4] + keys[-4:]:
        h.update(k.replace("_orig_mod.", "").encode())
        h.update(sd[k].detach().flatten()[:32].float().cpu().numpy().tobytes())
    return h.hexdigest()


def _trace_candidates(tree):
    import paths
    out = []
    if getattr(tree, "cpp_ts", None):
        out.append(tree.cpp_ts)
    if os.environ.get("CPP_NET"):
        out.append(os.environ["CPP_NET"])
    ver = os.environ.get("VERSION")
    if ver:
        stem = os.path.splitext(os.path.basename(ver))[0] + ".cuda.ts"
        for d in (os.environ.get("VERSION_DIR"), paths.under("versions")):
            if d:
                out.append(os.path.join(d, stem))
    return out


def _cnw_candidates(tree):
    """Where a hand-tuned CUDA net's weight file for this tree's version may be (cuda_net)."""
    import paths
    out = []
    if getattr(tree, "cpp_cnw", None):
        out.append(tree.cpp_cnw)
    if os.environ.get("CPP_CNW"):
        out.append(os.environ["CPP_CNW"])
    ver = os.environ.get("VERSION")
    if ver:
        stem = os.path.splitext(os.path.basename(ver))[0] + ".cnw"
        for d in (os.environ.get("VERSION_DIR"), paths.under("versions"), paths.under("cudanet")):
            if d:
                out.append(os.path.join(d, stem))
    return out


def cnw_fingerprint(path):
    """The fingerprint a .cnw file carries in its header (chess/cudanet/cudanet_spec.h), read without CUDA."""
    with open(path, "rb") as fh:
        if fh.read(8) != b"MFCNW001":
            return None
        n = int.from_bytes(fh.read(4), "little")
        return json.loads(fh.read(n).decode("utf-8")).get("fingerprint")


def trace_for(tree):
    """The trace of this tree's network, or None, in which case the network runs through the Python.

    A trace is used only when its sidecar fingerprint matches the weights the tree holds, and the answer is
    remembered per tree. With cuda_net on, a .cnw of this network comes first (pyext builds the tree on
    chess/cudanet from it), and if none is found it warns and uses the trace."""
    if tree.device == "cpu" or tree.model is None:
        return None
    want_cnw = bool(getattr(tree, "cuda_net", False))
    if getattr(tree, "_cpp_ts_checked", None) == want_cnw:
        return tree._cpp_ts_ok
    tree._cpp_ts_checked = want_cnw
    tree._cpp_ts_ok = None
    mine = None
    if want_cnw:
        for cnw in _cnw_candidates(tree):
            if not os.path.exists(cnw):
                continue
            mine = mine or fingerprint(tree.model)
            try:
                got = cnw_fingerprint(cnw)
            except (OSError, ValueError) as e:
                _warn(f"{cnw} unreadable ({type(e).__name__}); not used")
                continue
            if got == mine:
                _warn(f"cuda_net: the hand-tuned CUDA net from {cnw}")      # once, so a match's logs show which net ran
                tree._cpp_ts_ok = cnw
                return cnw
            _warn(f"{cnw} is another network's weight file; not used")
        _warn("cuda_net is on but no .cnw of this network was found: the LibTorch trace instead")
    for ts in _trace_candidates(tree):
        if not os.path.exists(ts):
            continue
        try:
            side = json.load(open(ts + ".json", encoding="utf-8"))
        except (OSError, ValueError):
            _warn(f"{ts} has no fingerprint (.json beside it); not used")
            continue
        mine = mine or fingerprint(tree.model)
        if side.get("fingerprint") == mine:
            tree._cpp_ts_ok = ts
            return ts
        _warn(f"{ts} is another network's trace; not used")
    return None


# -------------------------------------------------------------------- the tree and its callbacks
def _tree(tree):
    """The C++ tree for this MCTS, made again when a setting or the trace changes."""
    cfg = _settings(tree)
    ts = trace_for(tree)
    cfg["_ts"] = ts or ""
    if getattr(tree, "_cpp", None) is None or cfg != tree._cpp_cfg:
        ref = weakref.ref(tree)       # the C++ tree is not garbage-collected, so no strong cycle back to the MCTS
        ev = (lambda *a: _evaluate(ref(), *a))
        tb = (lambda fen: ref()._tb_value(chess.Board(fen)))
        if ts:
            tree._cpp = mayfly_cppsearch.Tree(cfg, None, tb, ts, "cuda", True)
        else:
            tree._cpp = mayfly_cppsearch.Tree(cfg, ev, tb)
        tree._cpp_cfg = cfg
        tree._cpp_live = False
    return tree._cpp


def _evaluate(tree, B, wide, ids, cast, hm, ep, I, M, widths):
    """The network through the tree's own _dispatch and _collect, from the C++ encoding.

    Builds MCTS._prepare's arrays (and sw25_one_copy's single pinned buffer, as _prepare fills it). Returns the
    logits cut to each row's legal moves, the value and the per-move q, as float32 bytes."""
    from mcts import _Cut, _oc_layout
    ids = np.frombuffer(ids, np.int64).reshape(B, 64).copy()
    cast = np.frombuffer(cast, np.float32).reshape(B, 4).copy()
    hm = np.frombuffer(hm, np.int64).copy()
    ep = np.frombuffer(ep, np.int64).copy()
    I = np.frombuffer(I, np.int64).reshape(B, wide).copy()
    M = np.frombuffer(M, np.uint8).reshape(B, wide).astype(bool)
    widths = [int(x) for x in np.frombuffer(widths, np.int32)]
    prep = {"nodes": [None] * B, "ids": ids, "cast": cast, "hm": hm, "ep": ep,
            "widths": widths, "I": I, "Ip": I, "M": M}
    if tree._oc_ok():
        L = ids.shape[1]
        o = _oc_layout(B, L, wide, False)
        blob = torch.empty(o[7], dtype=torch.long, pin_memory=True)
        a = blob.numpy()
        a[o[0]:o[1]].reshape(B, L)[:] = ids
        a[o[1]:o[2]].view(np.float32).reshape(B, 4)[:] = cast
        a[o[2]:o[3]] = hm
        a[o[3]:o[4]] = ep
        a[o[4]:o[5]].reshape(B, wide)[:] = I
        a[o[6]:o[7]].reshape(B, wide)[:] = M
        prep["oc"] = (blob, B, L, wide, False)
    pol, val, qv = tree._collect(tree._dispatch(prep))
    P = np.zeros((B, wide), np.float32)
    Q = np.zeros((B, wide), np.float32)
    for i in range(B):
        k = widths[i]
        row = pol[i]
        P[i, :k] = row.row if isinstance(row, _Cut) else np.asarray(row, np.float32)[I[i, :k]]
        Q[i, :k] = qv[i]
    return P.tobytes(), np.asarray(val, np.float32).tobytes(), Q.tobytes()


def _game(board):
    """The game as the C++ tree takes it, the starting FEN and the moves played from it."""
    return board.root().fen(), [m.uci() for m in board.move_stack]


class RootView:
    """The C++ root's statistics with mcts.Node's attribute names.

    The clock rules, explain(), the bot's _pv and the final info read it."""
    __slots__ = ("N", "W", "V2", "F", "X", "Q0", "P", "sum_n", "q0w", "value", "expanded", "term", "moves",
                 "kids_n", "board")


def view(tree, board, moves=None):
    """A RootView of the C++ root.

    moves is the root's chess.Move list from an earlier view of the same root (run_search between batches,
    where a step never changes the root), so it is not parsed again."""
    r = tree._cpp.root() if getattr(tree, "_cpp", None) is not None else None
    if r is None:
        return None
    v = RootView()
    f = lambda k: np.frombuffer(r[k], np.float32).copy()
    v.N, v.W, v.V2, v.F, v.X, v.Q0, v.P = f("N"), f("W"), f("V2"), f("F"), f("X"), f("Q0"), f("P")
    v.sum_n = r["sum_n"]
    v.q0w = r["q0w"]
    v.value = r["value"]
    v.expanded = bool(r["expanded"])
    v.term = r["term"]
    if moves is not None and len(moves) == len(r["moves"]):
        v.moves = moves
    else:
        v.moves = [chess.Move.from_uci(u) for u in r["moves"]]
    v.kids_n = r["kids_n"]
    v.board = board
    if not v.expanded:
        v.N = v.W = v.V2 = v.F = v.X = v.Q0 = v.P = None
    return v


def _completed(t, root):
    """The root as the clock rules may read it while a pipeline batch is on the GPU (MCTS._run_search's seen).

    Completed visits only, N - F and W - X. With nothing on the GPU, the root itself."""
    if root is None or root.N is None or not t.in_flight():
        return root
    seen = RootView()
    for k in RootView.__slots__:
        setattr(seen, k, getattr(root, k))
    seen.N = root.N - root.F
    seen.W = root.W - root.X
    return seen


def pv(tree, depth=12, first=None):
    return " ".join(tree._cpp.pv(-1 if first is None else int(first), int(depth)))


def tree_nodes(tree):
    return int(tree._cpp.tree_nodes())


def reset(tree):
    if getattr(tree, "_cpp", None) is not None:
        tree._cpp.reset()
    tree._cpp_live = False


def _sync(tree, before):
    """The C++ counters since before into tree.stats, under the names the Python uses."""
    now = tree._cpp.stats()
    for k in ("evals", "batches", "cache_hits", "collisions", "repetitions", "lcb_overrode", "tablebase"):
        tree.stats[k] = tree.stats.get(k, 0) + int(now[k] - before[k])


# -------------------------------------------------------------------- the search, with the clock
def run_search(tree, board, visits=None, reuse=True, clear_abort=True, seconds=None, hard_seconds=None,
               max_nodes=None, crit_seconds=None):
    """MCTS._run_search's control flow with the batches done by the C++ tree.

    Runs inside MCTS.search's autocast and timing_fixes setup. Every rule between the batches is the Python's,
    reading the C++ root."""
    if clear_abort:
        tree.abort.clear()
    visits = visits or tree.visits
    before_stats = dict(tree.stats)
    if tree.book and len(board.move_stack) < tree.book_plies:
        from book import pick as book_pick
        bm = book_pick(board, tree.rng, tree.book_plies)
        if bm is not None and bm in board.legal_moves:
            tree.root = None
            tree._cpp_live = False
            tree.stats["book"] += 1
            return bm, {**tree._delta(before_stats), "visits": 0, "carried": 0, "move": bm,
                        "q": 0.0, "pv": bm.uci(), "top": [(bm.uci(), 0, 0.0)],
                        "proof": "opening book", "rows": [], "totals": dict(tree.stats)}
    proven = tree._proven_move(board)
    if proven is not None:
        mv, why = proven
        tree.root = None
        tree._cpp_live = False
        tree.stats["proven"] += 1
        return mv, {**tree._delta(before_stats), "visits": 0, "carried": 0, "move": mv,
                    "q": 1.0, "pv": mv.uci(), "top": [(mv.uci(), 0, 1.0)], "proof": why,
                    "rows": [], "totals": dict(tree.stats)}
    t_start = time.time()
    t = _tree(tree)
    cpp_before = t.stats()
    fen, moves = _game(board)
    carried = int(t.begin(fen, moves, bool(reuse) and getattr(tree, "_cpp_live", False)))
    tree._cpp_live = True
    rb = board.copy(stack=False)
    root = view(tree, rb)
    if (carried and root.expanded and tree.ponder_discount < 1.0
            and root.N is not None and root.N.sum() > 0):
        t.discount(max(0.0, float(tree.ponder_discount)), bool(tree.discount_q0))
        tree.stats["discounted"] += 1
    if t.needs_eval():
        t.expand_root()
    root = view(tree, rb)
    tree.root = root
    tree.tt = {}
    capped = False
    if max_nodes is not None:
        room = max(0, int(max_nodes) - int(t.tree_nodes()))
        if room < visits:
            visits, capped = room, True
    target = root.sum_n + visits
    budget = (seconds * tree.time_scale) if seconds is not None else None
    deadline = (t_start + budget) if budget is not None else None
    hard = ((t_start + budget * (1.0 + max(0.0, tree.tm_extend_max)))
            if budget is not None else None)
    if budget and hard_seconds:
        hard = max(hard or 0.0, t_start + float(hard_seconds))
    crit = (t_start + float(crit_seconds)) if (budget and crit_seconds) else None
    tree._last_lead = None
    tree._root_allowed = None
    tree._gumbel = None
    tree._force = None
    stop, ext_kind = "visits", None
    n_start = root.sum_n
    base_end = deadline
    piped = bool(tree.pipeline)
    held = (lambda: t.held()) if piped else (lambda: 0.0)       # held collisions on the GPU are not visits
    seen = (lambda r: _completed(t, r)) if piped else (lambda r: r)
    try:
        while root.sum_n - held() < target and not tree.abort.is_set():
            if deadline is not None and time.time() >= deadline:
                opening = (tree.tm_open_spend < 1.0
                           and 2 * (board.fullmove_number - 1) + int(board.turn == chess.BLACK)
                           < tree.tm_open_plies)
                if (tree.tm_extend and not opening and hard is not None and time.time() < hard
                        and not tree._settled(seen(root), tree.tm_margin, 1.2)):
                    deadline = min(hard, max(deadline, time.time()) + budget * 0.25)
                    tree.stats["extended"] += 1
                    ext_kind = ext_kind or "close"
                    if crit is not None:
                        tree._unstable(seen(root))
                elif (crit is not None and tree.tm_extend and not opening and time.time() < crit
                        and tree._unstable(seen(root))):
                    deadline = min(crit, max(deadline, time.time()) + budget * 0.25)
                    tree.stats["extended"] += 1
                    tree.stats["unstable"] = tree.stats.get("unstable", 0) + 1
                    ext_kind = "unstable"
                else:
                    tree.stats["timed_out"] += 1
                    stop = "time" if ext_kind is None else f"time after {ext_kind} extensions"
                    break
            code = t.step(float(target), float(deadline) if deadline is not None else -1.0)
            if code == 2:
                break
            root = view(tree, rb, root.moves)
            tree.root = root
            if code == 1 and not piped:
                continue
            if (tree.tm_stop and budget and time.time() - t_start > 0.33 * budget
                    and tree._settled(seen(root), 2.0 * tree.tm_margin, 2.0)):
                tree.stats["banked"] += 1
                stop = "tm_stop"
                break
            if tree.tm_catchup > 0.0 and budget and base_end is not None and ext_kind is None:
                now = time.time()
                done = root.sum_n - held() - n_start
                if (now - t_start > 0.1 * budget and now < base_end and done >= 64
                        and len(root.moves) > 1):
                    left = done / max(now - t_start, 1e-6) * max(0.0, base_end - now)
                    top2 = np.sort(seen(root).N)[-2:]
                    if float(top2[1] - top2[0]) > tree.tm_catchup * left:
                        tree.stats["caught"] = tree.stats.get("caught", 0) + 1
                        stop = "catchup"
                        break
        if piped:
            t.finish()                    # as MCTS._pipe_finish, the batch on the GPU is backed up, not thrown away
    except BaseException:
        if t.root() is None:
            tree.root = None              # a failure inside the walk dropped the tree there
            tree._cpp_live = False
        _sync(tree, cpp_before)
        raise
    root = view(tree, rb, root.moves)
    tree.root = root
    if tree.abort.is_set() and stop == "visits":
        stop = "abort"
    elif capped and root.sum_n >= target:
        stop = "memory ceiling"
    elif stop == "visits" and root.sum_n < target:
        stop = "saturated"
    if root.term is not None or not root.moves:
        _sync(tree, cpp_before)
        return None, {"visits": 0, "term": root.term}
    best = int(t.choose())
    _sync(tree, cpp_before)
    info = {
        **tree._delta(before_stats),
        "visits": int(root.sum_n),
        "carried": carried,
        "tree_nodes": int(t.tree_nodes()),
        "at_ceiling": bool(capped and root.sum_n >= target),
        "spent": round(time.time() - t_start, 3),
        "allowed": round(budget, 3) if budget is not None else None,
        "stop": stop,
        "move": root.moves[best],
        "q": float(root.W[best] / max(root.N[best], 1)),
        "pv": pv(tree, first=best),
        "totals": dict(tree.stats),
        "top": sorted(((root.moves[i].uci(), int(root.N[i]),
                        float(root.W[i] / max(root.N[i], 1)))
                       for i in range(len(root.moves))), key=lambda t_: -t_[1])[:5],
        "engine": "cpp" + ("+net" if trace_for(tree) else ""),
    }
    info["rows"] = tree.explain(root)
    return root.moves[best], info


def export_trace(ckpt, out=None):
    """Export the trace and its fingerprint for a checkpoint, on a machine with a GPU.

    Runs export_ts.py's "cuda" flavour, then writes <out>.json with the weights' fingerprint as this module
    computes it from a fresh load. Returns the trace path."""
    out = out or os.path.splitext(ckpt)[0] + ".cuda.ts"
    net_dir = os.environ.get("CPP_NET_DIR", os.path.join(HERE, "cpp"))
    if net_dir not in sys.path:
        sys.path.insert(0, net_dir)
    import export_ts
    if not os.path.exists(out):
        export_ts.trace("cuda", out, ckpt)
    m = export_ts.load_v5(ckpt, "cpu")
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump({"fingerprint": fingerprint(m), "ckpt": os.path.abspath(ckpt),
                   "made": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=1)
    return out


def _cpp_mcts():
    from mcts import MCTS

    class CppMCTS(MCTS):
        """MCTS with engine "cpp" from the start, for the tests. A sealed manifest sets the same key."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.engine = "cpp"

    return CppMCTS


def __getattr__(name):           # `from mcts_cpp import CppMCTS` without importing mcts at module import
    if name == "CppMCTS":
        return _cpp_mcts()
    raise AttributeError(name)
