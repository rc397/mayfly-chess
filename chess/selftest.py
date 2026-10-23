r"""selftest exercises the real code paths rather than proxies for them.

A log line that says "listening", a live process or an open port does not prove the board can play. So
besides unit checks of the search, trainer and tools, this starts a real server on a spare port, talks
to it over real HTTP as several browsers at once, and asserts on what comes back. Nothing here reads a
log.

  python chess/selftest.py           everything
  python chess/selftest.py fast      skip the ones that need a search
  env: SELFTEST_PORT=8099  VERSION=mayfly_v1.pt
"""
import os, sys, json, time, threading, subprocess, urllib.request, urllib.error, http.cookiejar
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

E = os.environ.get
PORT = int(E("SELFTEST_PORT", 8099))
BASE = f"http://127.0.0.1:{PORT}"
FAST = len(sys.argv) > 1 and sys.argv[1] == "fast"

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  {'pass' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""), flush=True)
    return ok


class Browser:
    """One visitor, with its own cookie jar, which is the whole point of the exercise."""

    def __init__(self, name):
        self.name = name
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))

    def get(self, path="/", timeout=60):
        return self.opener.open(BASE + path, timeout=timeout).read()

    def post(self, what, timeout=180, **kw):
        req = urllib.request.Request(f"{BASE}/{what}", data=json.dumps(kw).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(self.opener.open(req, timeout=timeout).read())

    @property
    def sid(self):
        for c in self.jar:
            if c.name == "mfs":
                return c.value
        return None


# ---------------------------------------------------------------- unit checks, no server needed
def unit_the_visit_counters_balance():
    """Selection adds a visit and a virtual loss on the way down, backup takes back the virtual
    loss, and _unwind takes back both when a descent collides. If those three disagree the counters
    drift silently in proportion to the collision count, which is high in this search. Checked at
    two virtual loss settings, since at 1.0 they could agree by accident."""
    import torch, chess
    from mcts import MCTS

    class Stub(torch.nn.Module):
        def forward(self, ids, cast, hm, ep):
            n = ids.shape[0]
            return {"h": torch.zeros(n, 4), "policy": torch.randn(n, 4162),
                    "value": torch.zeros(n), "qvalue": torch.randn(n, 4162)}

    # Each rule by name. From v4 virtual_mean is the default and `vl` is not read at all, so the two
    # loss settings are pinned to the loss, and the mean and held collisions (collide 2) are checked
    # beside them.
    rules = (("vl 1.0", {"virtual_mean": False, "vl": 1.0, "collide": 0}),
             ("vl 2.5", {"virtual_mean": False, "vl": 2.5, "collide": 0}),
             ("virtual mean", {"virtual_mean": True, "collide": 0}),
             ("virtual mean, held collisions", {"virtual_mean": True, "collide": 2}))
    bad = []
    for name, opts in rules:
        t = MCTS(Stub(), "cpu", batch=16)
        for k, v in opts.items():
            setattr(t, k, v)
        t.search(chess.Board(), visits=300, reuse=False)
        r = t.root
        if abs(float(r.N.sum()) - r.sum_n) > 1e-6:
            bad.append(f"{name}: children sum to {float(r.N.sum()):.1f}, root says {r.sum_n:.1f}")
        if float(r.N.min()) < 0:
            bad.append(f"{name}: a child has {float(r.N.min()):.1f} visits")
        # nothing may still be in flight anywhere in the tree once the search has returned
        # counts must be exactly zero, while the pending value is float32 and keeps rounding, about 1e-6
        f_left, x_left, stack = 0.0, 0.0, [r]
        while stack:
            n = stack.pop()
            if n.F is not None:
                f_left += float(abs(n.F).sum())
                x_left = max(x_left, float(abs(n.X).max()))
                stack.extend(k for k in n.kids if k is not None)
        if f_left > 0 or x_left > 1e-4:
            bad.append(f"{name}: {f_left:.0f} visits and {x_left:.2g} of value still in flight")
    check("the visit counters balance after collisions", not bad, "; ".join(bad))


def unit_the_batch_is_filled():
    """A forward pass costs the same at batch 8 as at batch 48, so a half-empty batch is a
    half-wasted pass. `_descend` already counts each pending leaf at the root, so the collection
    loop must not count them again against the visit budget, or it stops at half a batch. The move
    returned would still be right, so only this check notices.

    The floor is visits/batch. Allowed slack covers one tail batch a search plus the saturation
    cap firing on a crowded tree."""
    import torch, chess
    from mcts import MCTS

    class Stub(torch.nn.Module):
        def forward(self, ids, cast, hm, ep):
            n = ids.shape[0]
            return {"h": torch.zeros(n, 4), "policy": torch.randn(n, 4162),
                    "value": torch.zeros(n), "qvalue": torch.randn(n, 4162)}

    # At production settings. The double count only bites once the remaining budget drops below
    # 2 x batch, so at 300 visits and batch 16 the check would pass even with the bug. With the bug
    # it takes 15 passes and without it 10 (floor 8.33), so the limit sits between them at 11.25.
    #
    # Under virtual loss, where it was calibrated. virtual_mean (the v4 default) fills a batch less by
    # design, since a pending visit barely discourages its path. The accounting guarded here is shared
    # by both.
    torch.manual_seed(0)
    visits, batch = 400, 48
    t = MCTS(Stub(), "cpu", batch=batch)
    t.virtual_mean = False
    t.collide = 0                       # and without held collisions, v4's default
    t.search(chess.Board(), visits=visits, reuse=False)
    floor = visits / batch
    got = t.stats["batches"]
    check("the batch is actually filled", got <= floor * 1.35,
          f"{got} forward passes for {visits} visits at batch {batch}, "
          f"floor is {floor:.1f}, mean fill {t.stats['evals']/max(got,1):.1f} of {batch}")


def unit_a_stale_source_is_reported():
    """A running board holds the modules it imported at startup, which is invisible from outside
    (the port answers, /api/health returns 200, a forward pass succeeds) while the engine is
    whatever was on disk when the process began.

    Exercised against the fault, not just the happy path. The mtimes are forced rather than the
    files touched, so running the suite does not mark the live board stale."""
    import play_web

    play_web.SOURCE_MTIMES.clear()
    play_web.record_sources()
    check("a fresh process reports no stale sources", not play_web.stale_sources(),
          f"{len(play_web.SOURCE_MTIMES)} source file(s) tracked")
    if play_web.SOURCE_MTIMES:
        f = sorted(play_web.SOURCE_MTIMES)[0]
        play_web.SOURCE_MTIMES[f] = 0.0
        got = play_web.stale_sources()
        check("a source edited since startup is reported", got == [os.path.basename(f)],
              f"expected [{os.path.basename(f)}], got {got}")
        play_web.SOURCE_MTIMES.clear()
        play_web.record_sources()


def unit_batched_leaf_values_match_the_loop():
    """_evaluate computes leaf values with one gather for the whole batch instead of a loop with a
    GPU sync per node. A speedup must not change the values, so this recomputes them the old way
    and demands they agree."""
    import torch, chess, numpy as np
    from mcts import MCTS, Node, _native_moves
    from moves import move_to_idx

    torch.manual_seed(0)
    POL = torch.randn(1, 4162)
    QV = torch.randn(1, 4162)

    class Stub(torch.nn.Module):
        def forward(self, ids, cast, hm, ep):
            n = ids.shape[0]
            return {"h": torch.zeros(n, 4), "policy": POL.repeat(n, 1),
                    "value": torch.zeros(n), "qvalue": QV.repeat(n, 1)}

    boards = [chess.Board()]
    b = chess.Board()
    for u in ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"):
        b.push(chess.Move.from_uci(u))
        boards.append(b.copy(stack=False))

    t = MCTS(Stub(), "cpu", batch=8)
    # Real nodes, built the way the tree builds them, since _prepare reads the key and the compiled
    # generator's move indices (IX, under timing_fixes). Where the generator is not built, IX is None
    # and _prepare indexes the moves in Python, which is compared just the same.
    t._nm = _native_moves() if t.timing_fixes else None
    nodes = [Node(x, t.draw_plies, t._nm) for x in boards]
    _, got, _ = t._evaluate(nodes)

    # the old way, one node at a time
    q = torch.sigmoid(QV[0].float())
    pl = POL[0].float()
    want = []
    for n in nodes:
        flip = n.board.turn == chess.BLACK
        idx = torch.tensor([move_to_idx(m, flip) for m in n.moves])
        w = torch.softmax(pl[idx], -1)
        want.append(float((w * q[idx]).sum()) * 2.0 - 1.0)
    want = np.array(want, np.float32)
    err = float(np.abs(got - want).max())
    widths = sorted({len(n.moves) for n in nodes})
    check("the batched leaf values equal the per-node loop", err < 2e-6,
          f"largest difference {err:.2e} over {len(nodes)} nodes, "
          f"move counts {widths[0]} to {widths[-1]}")


def _det_stub():
    """A stub whose output depends on the POSITION and not on call order, so two searches of the
    same position are comparable. The randn stub used elsewhere cannot be used for anything that
    compares two runs."""
    import torch

    class Det(torch.nn.Module):
        def forward(self, ids, cast, hm, ep):
            n = ids.shape[0]
            h = (ids.float().sum(1) * 0.37).unsqueeze(1)
            base = torch.arange(4162, dtype=torch.float32).unsqueeze(0)
            return {"h": torch.zeros(n, 4), "policy": torch.sin(base * 0.017 + h),
                    "value": torch.zeros(n), "qvalue": torch.cos(base * 0.011 + h)}

    return Det()


def unit_the_power_mean_is_a_mean():
    """p = 1 must reproduce the arithmetic mean exactly, or turning the feature off does not turn it
    off. For p > 1 the power mean must sit between the mean and the maximum, closer to the best child
    than an average without being the max.

    Checked on hand-built statistics rather than a search, because the property is arithmetic."""
    import numpy as np, torch, chess
    from mcts import MCTS

    t = MCTS(_det_stub(), "cpu", batch=8)

    class N:
        pass

    node = N()
    node.N = np.array([10.0, 5.0, 1.0, 0.0], np.float32)
    node.W = np.array([2.0, -1.5, 0.9, 0.0], np.float32)
    mean = float(node.W[:3].sum() / node.N[:3].sum())
    q = node.W[:3] / node.N[:3]

    t.power_p = 1.0
    got1 = t._node_q(node)
    bad = []
    if got1 is None or abs(got1 - mean) > 1e-6:
        bad.append(f"p=1 gave {got1} not the mean {mean:.6f}")
    for pp in (1.5, 2.0, 3.0):
        t.power_p = pp
        g = t._node_q(node)
        if g is None or g < mean - 1e-6:
            bad.append(f"p={pp} gave {g}, below the mean {mean:.4f}")
        if g is not None and g > float(q.max()) + 1e-6:
            bad.append(f"p={pp} gave {g}, above the best child {float(q.max()):.4f}")
    check("the power mean sits between the average and the best child", not bad,
          "; ".join(bad) or f"mean {mean:.4f}, p2 {t._node_q(node):.4f}, "
                            f"max {float(q.max()):.4f}")


def unit_the_graph_merges_transpositions():
    """MCGS has to actually merge, and must not build a cycle. A cycle would make the principal
    variation walk forever, and a repetition is how one would appear. The ply is in the key to
    prevent it, and this asserts the PV still terminates.

    The evaluation saving is measured with the cache off, which is not a production claim. With the
    position cache a transposition is already served without a forward pass, so in real games MCGS
    saves almost no evaluations and its value is the shared statistics. `cache_size=0` isolates the
    merging mechanism.

    1,000 visits, because a merge also needs the same fifty-move count (the key carries no clock).
    At 300 visits this position only meets transpositions with different clocks, so nothing merges,
    correctly."""
    import chess
    from mcts import MCTS

    # a position reached by transposable move orders, so there is something to merge
    b = chess.Board("rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4")
    out = {}
    for on in (False, True):
        t = MCTS(_det_stub(), "cpu", batch=8, cache_size=0)
        t.mcgs = on
        mv, info = t.search(b.copy(), visits=1000, reuse=False)
        out[on] = (mv, info["evals"], t.stats["transpositions"], info["pv"])

    bad = []
    if out[True][2] <= 0:
        bad.append("MCGS merged nothing")
    if out[True][1] >= out[False][1]:
        bad.append(f"MCGS did not save evaluations: {out[True][1]} against {out[False][1]}")
    if len(out[True][3].split()) > 12:
        bad.append("the principal variation did not terminate, the graph may be cyclic")
    check("the graph merges transpositions without building a cycle", not bad,
          "; ".join(bad) or f"{out[True][2]} merges, {out[True][1]} evals against "
                            f"{out[False][1]} as a tree, pv {len(out[True][3].split())} plies")


def unit_early_stop_never_changes_the_move():
    """Stopping early is only allowed when no remaining visit could change the answer. If the move
    ever differs, the condition is wrong and the feature is trading strength for time silently."""
    import chess
    from mcts import MCTS

    bad = []
    stopped = 0
    for fen in ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"):
        got = {}
        for on in (False, True):
            t = MCTS(_det_stub(), "cpu", batch=8, cache_size=0)
            t.early_stop = on
            t.lcb = False          # under lcb the rule stands down (the bound picks the move), so off
            mv, info = t.search(chess.Board(fen), visits=240, reuse=False)
            got[on] = (mv, info["visits"])
            stopped += t.stats["early_stop"]
        if got[False][0] != got[True][0]:
            bad.append(f"{fen.split()[0][:16]}: {got[False][0]} became {got[True][0]}")
    check("stopping early never changes the move", not bad,
          "; ".join(bad) or f"fired {stopped} time(s) across 3 positions, same move every time")


def unit_the_leaf_search_sees_a_hanging_piece():
    """The captures-only leaf search exists for one case, a leaf where the side to move can simply
    take something. If it cannot see a queen en prise it is not worth its cost."""
    import chess
    from mcts import MCTS

    t = MCTS(_det_stub(), "cpu", batch=8)
    t.qdepth, t.qweight = 2, 1.0
    quiet = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    hanging = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")   # exd5 wins the queen
    q0, h0 = t._tactical(quiet, 0.0), t._tactical(hanging, 0.0)
    check("the leaf search leaves a quiet position alone", abs(q0) < 1e-9, f"{q0:+.3f}")
    check("the leaf search sees a queen it can take", h0 > 0.5,
          f"quiet {q0:+.3f}, queen en prise {h0:+.3f}")


def unit_each_version_keeps_its_own_search():
    """A frozen version plays with the search it was frozen against. v1 predates the virtual loss
    change and must still get 1.0, and v2 must get 2.0. If search improvements leaked backwards onto
    old versions, every rung of the ladder would be on today's tree and v1 against v2 would measure
    nothing.

    The refusal is checked too. A typo in a version's manifest must raise, not silently hand that
    version today's defaults."""
    import chess
    from mcts import MCTS, pinned_search, SEARCH_KEYS

    from mcts import versions_on_disk
    rows = versions_on_disk()
    newest = max((int(r.get("version", 0)) for r in rows), default=0)
    v1, live = pinned_search("mayfly_v1.pt"), pinned_search(f"mayfly_v{newest}.pt")
    bad = []
    if newest < 2:
        bad.append(f"only {newest} version(s) on disk, nothing to compare")
    if not v1:
        bad.append("v1 has no sealed search, so it would silently play today's tree")
    if v1.get("vl") != 1.0:
        bad.append(f"v1 was sealed before virtual loss moved and should play 1.0, "
                   f"got {v1.get('vl')}")
    if not live:
        bad.append(f"v{newest} is the newest and has no sealed search: under frozen-at-release "
                   f"it would silently play whatever today's defaults become")
    for k in v1:
        if k not in SEARCH_KEYS:
            bad.append(f"{k} is recorded but not a pinned setting")

    t = MCTS(_det_stub(), "cpu", batch=8).apply(v1)
    if t.vl != 1.0:
        bad.append(f"apply did not take: tree has vl {t.vl}")
    try:
        MCTS(_det_stub(), "cpu", batch=8).apply({"not_a_setting": 1})
        bad.append("apply accepted a setting that does not exist")
    except KeyError:
        pass
    check("every version keeps the search it was released with", not bad,
          "; ".join(bad) or f"v1 sealed at vl={v1.get('vl')} with {len(v1)} settings, "
                            f"v{newest} sealed at release as {live.get('tree_policy')}")


def unit_the_calibration_stretches_the_middle():
    """The calibration map widens the band the search cannot resolve. The leaf value spans only 0.068
    across the quiet middlegame where Stockfish spans 0.270, so Q loses to U before it starts.

    Four things must hold. With no map on disk it is the exact identity. With a map it is monotone, so
    it never reorders siblings. It widens the middle rather than merely shifting it. And a forced mate
    is still found, because an exact terminal must not be stretched onto an estimated scale.

    Built on a synthetic map so it does not depend on a file that only a GPU run produces."""
    import numpy as np, chess
    import mcts as M
    from mcts import MCTS

    t = MCTS(_det_stub(), "cpu", batch=8)
    saved = M._CALIB
    bad = []
    try:
        M._CALIB = ()
        for v in (-1.0, -0.4, 0.0, 0.17, 1.0):
            if t._stretch(v) != v:
                bad.append(f"no map, but {v} became {t._stretch(v)}")

        # the measured shape, the model uses about [-0.6, 0.3] where Stockfish uses [-1, 1]
        xs = np.linspace(-0.6, 0.3, 21)
        ys = np.linspace(-1.0, 1.0, 21)
        M._CALIB = (xs, ys)
        sweep = [t._stretch(v) for v in np.linspace(-0.8, 0.5, 40)]
        if any(b < a - 1e-9 for a, b in zip(sweep, sweep[1:])):
            bad.append("the map is not monotone")
        narrow = t._stretch(0.05) - t._stretch(-0.05)
        if narrow <= 0.1 * 1.5:
            bad.append(f"the middle was not widened: 0.10 became {narrow:.3f}")

        # A terminal must reach backup unstretched, and a network value must be stretched. Checking
        # the move played would only test the stub's arbitrary priors, so this watches the values
        # entering _backup and runs the same search twice, once each way.
        #
        # Thirty pieces on the board, so tablebases (where installed) cannot answer every leaf and
        # the result does not depend on the machine.
        def watched(cal):
            seen = []

            class Watch(MCTS):
                def _backup(self, path, leaf_value, **kw):     # m is the moves-left estimate
                    seen.append(round(float(leaf_value), 9))
                    return super()._backup(path, leaf_value, **kw)

            w = Watch(_det_stub(), "cpu", batch=8, cache_size=0)
            w.calibrate = cal
            # The stub's arbitrary priors need v5's wider exploration to reach the mate at 600 visits.
            # With v6's c2 of 1.07 Qxf7 is never visited and the precondition below fails.
            w.c2 = 1.70513
            # This measures a mate reached inside the tree. v4's root_mate plays a mate in one before
            # any tree exists, which is right for play and leaves nothing here to measure.
            w.root_mate = False
            w.tb_dtz = False
            # 3.Qf3 against 2...Nc6, so Qxf7 is mate in one with everything still on the board
            w.search(chess.Board(
                "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"),
                visits=600, reuse=False)
            hit = sum(int(n) for mv, n in zip(w.root.moves, w.root.N) if mv.uci() == "f3f7")
            return seen, hit

        off, _ = watched(False)
        on, hit = watched(True)
        # The precondition, asserted. If the terminal is never reached the test proves nothing, and it
        # should say so rather than blame the code.
        if hit == 0:
            bad.append("the mating move was never visited, so this proves nothing")
        elif not any(abs(abs(v) - 1.0) < 1e-9 for v in on):
            bad.append("no exact +-1 reached backup, so the checkmate was stretched")
        if sorted(off) == sorted(on):
            bad.append("the values were identical with the map on and off: it was not applied")
    finally:
        M._CALIB = saved
        M.calibration()
    check("the calibration widens the middle without reordering or breaking mates", not bad,
          "; ".join(bad) or f"identity with no map, monotone with one, 0.10 in the middle "
                            f"becomes {narrow:.2f}, checkmate still arrives as exactly -1")


def unit_importing_the_board_does_not_read_our_arguments():
    """Many analysis tools import play_web for `load_version`, so play_web must not read argv[1] as
    its port at import time, or a tool with its own non-numeric first argument dies on import."""
    import subprocess, sys as _s
    code = ("import sys; sys.path.insert(0, r'%s'); sys.argv = ['x', 'mcgs=1', '60']; "
            "import play_web; print(play_web.PORT)" % HERE)
    r = subprocess.run([_s.executable, "-c", code], capture_output=True, text=True, timeout=180)
    ok = r.returncode == 0 and r.stdout.strip().endswith("8000")
    check("importing the board does not try to read our arguments", ok,
          (r.stderr.strip().splitlines() or ["port " + r.stdout.strip()])[-1][:120])


def unit_the_tree_policies_agree_with_what_the_page_shows():
    """`explain` shows prior, Q, U and Q+U to a person. Both it and `_select` call `_terms`, and this
    asserts they cannot drift apart by checking that the move the rows rank first is the move
    selection returns.

    Also checks the other policies run and that the default is unchanged."""
    import numpy as np, chess
    from mcts import MCTS

    bad = []
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    base = None
    # the policy a tree gets when nobody sets one
    default_pol = MCTS(None, "cpu").tree_policy
    for pol in ("puct", "puct_v", "uct_v_p"):
        t = MCTS(_det_stub(), "cpu", batch=8, cache_size=0)
        t.tree_policy = pol
        mv, info = t.search(b.copy(), visits=300, reuse=False)
        if mv is None:
            bad.append(f"{pol}: returned no move")
            continue
        if pol == default_pol:
            base = mv.uci()
        rows = t.explain(t.root)
        if not rows:
            bad.append(f"{pol}: explain returned nothing")
            continue
        # the row with the highest Q+U must be the one _select would take right now, fed the same
        # FPU base `_descend` feeds it, so the comparison is with the search itself
        top = max(rows, key=lambda r: r["puct"])["move"]
        chosen = t.root.moves[t._select(t.root, t._fpu_base(t.root), is_root=True)].uci()
        if top != chosen:
            bad.append(f"{pol}: the page ranks {top} first, the search takes {chosen}")

    # Setting the default must be the same as not setting it, so a tree left alone plays exactly the
    # move of one told the default policy by name.
    t = MCTS(_det_stub(), "cpu", batch=8, cache_size=0)
    mv2, _ = t.search(b.copy(), visits=300, reuse=False)
    if base is not None and mv2 is not None and mv2.uci() != base:
        bad.append(f"the default changed: {base} became {mv2.uci()}")
    check("the page shows the formula the search actually used", not bad,
          "; ".join(bad) or f"puct, puct_v and uct_v_p all agree with explain; default plays {base}")


def unit_the_variance_estimate_is_a_variance():
    """The variance-aware policies are only as good as sigma. An action whose backed-up values are all
    identical has variance zero, and one whose values disagree has variance above zero. An unvisited
    action reports 1.0, the largest value on a [-1, 1] scale, as maximal uncertainty."""
    import numpy as np
    from mcts import MCTS

    t = MCTS(_det_stub(), "cpu", batch=8)

    class N:
        pass

    node = N()
    node.N = np.array([4.0, 4.0, 0.0], np.float32)
    node.W = np.array([2.0, 0.0, 0.0], np.float32)      # a is four 0.5s, b is +1,-1,+1,-1
    node.V2 = np.array([1.0, 4.0, 0.0], np.float32)
    sd = t._sigma(node, node.N, node.W, node.N > 0)
    bad = []
    if abs(float(sd[0])) > 1e-6:
        bad.append(f"identical values gave sd {float(sd[0]):.4f}, not 0")
    if abs(float(sd[1]) - 1.0) > 1e-6:
        bad.append(f"values of +-1 gave sd {float(sd[1]):.4f}, not 1")
    if abs(float(sd[2]) - 1.0) > 1e-6:
        bad.append(f"an unvisited action gave sd {float(sd[2]):.4f}, not 1")
    check("the variance estimate is a variance", not bad,
          "; ".join(bad) or f"settled {float(sd[0]):.2f}, split {float(sd[1]):.2f}, "
                            f"unseen {float(sd[2]):.2f}")


def unit_a_job_can_be_restarted_from_its_marker():
    """A supervised job that only works when started by hand is a job that stays down.

    Self-play must find the trainer's checkpoint from the launched configuration, not beside the
    source, so a supervisor restart without the shell's environment still works. With a temporary
    autostart.json naming a CKPT_DIR, `ckpt_dir` must return that and not the source directory."""
    import json as _json, tempfile, shutil
    import selfplay

    real = os.path.join(HERE, "autostart.json")
    keep = real + ".selftest-bak"
    bad = []
    had = os.path.exists(real)
    try:
        if had:
            shutil.copy2(real, keep)
        want = tempfile.gettempdir()
        with open(real, "w", encoding="utf-8") as fh:
            _json.dump({"env": {"CKPT_DIR": want, "TAG": "mayfly"}}, fh)
        got = selfplay.ckpt_dir()
        if got != want:
            bad.append(f"ckpt_dir returned {got!r}, not the launched CKPT_DIR {want!r}")
        if os.path.dirname(selfplay.model_path()) != want:
            bad.append(f"model_path is not under CKPT_DIR: {selfplay.model_path()!r}")
    finally:
        if had:
            shutil.move(keep, real)
        elif os.path.exists(real):
            os.remove(real)
    # The architecture, from a checkpoint that carries none. The live checkpoint has no cfg, and
    # hard-coded defaults would build the wrong shapes silently under strict=False. Exercised with a
    # D nothing would ever default to.
    try:
        if had:
            shutil.copy2(real, keep)
        with open(real, "w", encoding="utf-8") as fh:
            _json.dump({"env": {"D": "777", "LAYERS": "5", "NHEAD": "7", "FF_MULT": "3.5"}}, fh)
        got = selfplay.resolve_cfg({"model": {}})
        if got.get("d") != 777 or got.get("layers") != 5:
            bad.append(f"a checkpoint with no cfg resolved to {got}, not the launched 777/5")
        keeps = selfplay.resolve_cfg({"cfg": {"d": 1024}, "model": {}})
        if keeps.get("d") != 1024:
            bad.append(f"a checkpoint WITH cfg was overridden: {keeps}")
    finally:
        if had:
            shutil.move(keep, real)
        elif os.path.exists(real):
            os.remove(real)

    # And against the real configuration. The name is checked everywhere, but the file only exists on
    # a machine that trains, so that part is gated.
    where = selfplay.model_path()
    # "this machine trains" means a checkpoint directory was configured, since `ckpt_dir` falls back
    # to the source folder, which always exists
    d = selfplay.ckpt_dir()
    trains = os.path.isdir(d) and os.path.abspath(d) != os.path.abspath(HERE)
    if trains and not os.path.exists(where):
        bad.append(f"the checkpoint self-play would load does not exist: {where}")
    if trains and not where.endswith("_last.pt"):
        bad.append(f"self-play is reading the best-of checkpoint, not the live one: "
                   f"{os.path.basename(where)}")
    check("self-play can be restarted from its marker alone", not bad,
          "; ".join(bad) or f"loads {os.path.basename(where)} from {os.path.dirname(where)}")


def unit_gumbel_concentrates_the_budget():
    """Gumbel AlphaZero exists because PUCT spreads a few hundred visits across every root move
    and then compares its favourite against moves it barely looked at. Sequential Halving instead
    shortlists, gives every survivor EQUAL visits, and halves, so the comparison at the end is
    between candidates that got the same look.

    Four properties, each a way the implementation could be wrong and still run.
      the shortlist is honoured        no more than gumbel_m root moves ever touched
      the budget concentrates          top two hold far more than PUCT's top two
      the move comes from the shortlist, not from argmax(N)
      early stopping is off            it compares visit counts, and these are equal by design
    """
    import numpy as np, chess
    from mcts import MCTS

    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    out = {}
    for pol in ("puct", "gumbel"):
        t = MCTS(_det_stub(), "cpu", batch=8, cache_size=0)
        t.tree_policy = pol
        t.gumbel_m = 16
        t.early_stop = True          # must be ignored under gumbel, not obeyed
        mv, _ = t.search(b.copy(), visits=400, reuse=False)
        N = t.root.N.copy()
        share = float(np.sort(N)[-2:].sum() / max(N.sum(), 1))
        out[pol] = (mv, int((N > 0).sum()), share, t, N)

    bad = []
    mv_g, touched, share_g, tg, Ng = out["gumbel"]
    _, touched_p, share_p, _, _ = out["puct"]
    if touched > 16:
        bad.append(f"touched {touched} root moves with a shortlist of 16")
    if share_g <= share_p:
        bad.append(f"did not concentrate: top two hold {100*share_g:.0f}% against "
                   f"puct's {100*share_p:.0f}%")
    if mv_g is not None and tg._root_allowed is not None:
        survivors = {tg.root.moves[i].uci() for i in tg._root_allowed}
        if mv_g.uci() not in survivors:
            bad.append(f"played {mv_g.uci()}, which is not among the survivors {survivors}")
    if tg.stats.get("early_stop", 0):
        bad.append("early stopping fired under gumbel, where visits are equal by design")
    check("gumbel concentrates the budget on a shortlist", not bad,
          "; ".join(bad) or f"{touched}/{len(tg.root.moves)} moves touched, top two hold "
                            f"{100*share_g:.0f}% against puct's {100*share_p:.0f}%")


def unit_the_tablebases_are_used():
    """Under seven pieces the answer is a fact, not an estimate, and the value head is the weakest
    thing the network has. The search must actually probe the tablebases."""
    import torch, chess
    from mcts import MCTS, tablebase

    if tablebase() is None:
        # skipped like the other two tablebase checks, since a machine without the tables cannot say
        # anything about the wiring either way
        print("  syzygy probes: no tables on this machine, skipped", flush=True)
        return

    class Stub(torch.nn.Module):
        def forward(self, ids, cast, hm, ep):
            n = ids.shape[0]
            return {"h": torch.zeros(n, 4), "policy": torch.randn(n, 4162),
                    "value": torch.zeros(n), "qvalue": None}

    t = MCTS(Stub(), "cpu", batch=8)
    # The test positions are checked for legality first, so an impossible position (probe_wdl refuses
    # it) is not mistaken for a bug in the tablebase wiring.
    cases = [("k7/8/2K5/8/8/8/8/7Q w - - 0 1", 1.0, "KQ v K, the side with the queen to move"),
             ("k7/8/2K5/8/8/8/8/7Q b - - 0 1", -1.0, "the same position from the bare king"),
             ("k7/8/2K5/8/8/8/8/7B w - - 0 1", 0.0, "KB v K, drawn")]
    illegal = [why for fen, _, why in cases if not chess.Board(fen).is_valid()]
    if illegal:
        check("the tablebase test positions are legal", False, "; ".join(illegal))
        return
    bad = []
    for fen, want, why in cases:
        got = t._tb_value(chess.Board(fen))
        if got != want:
            bad.append(f"{why}: got {got}, wanted {want}")
    check("syzygy probes return the true result", not bad, "; ".join(bad))

    # and the search must actually consult them, not merely be able to
    t2 = MCTS(Stub(), "cpu", batch=8)
    t2.search(chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"), visits=64, reuse=False)
    check("the search consults the tablebases", t2.stats["tablebase"] > 0,
          f"{t2.stats['tablebase']} probes hit in 64 visits")


def unit_a_dead_loader_does_not_hang_the_run():
    """If a loader's filler thread dies (for example from a transient out of memory), the main loop
    would wait forever on a queue nothing fills, with the process alive and idle. next() must raise
    instead.

    The loaders are not constructed here, which would want the corpus and a minute of shard
    scanning. The method is called against a stand-in with the same three attributes it reads.
    """
    import queue as _q, threading, time as _t
    from data_av import AVLoader
    from data_packed import PackedLoader

    class Stub:
        POLL = 0.2

        def __init__(self, err):
            self.q = _q.Queue()
            self.error = err
            self.stop = threading.Event()
            self.t = threading.Thread(target=lambda: None)
            self.t.start()
            self.t.join()

    out = []
    for name, cls in (("action value", AVLoader), ("policy", PackedLoader)):
        s = Stub(RuntimeError("the filler died"))
        t0 = _t.time()
        try:
            cls.next(s)
            out.append(f"{name}: RETURNED, it should have raised")
        except RuntimeError:
            out.append(f"{name}: raised in {_t.time()-t0:.1f}s")
        except Exception as e:
            out.append(f"{name}: raised {type(e).__name__}")
    check("a dead loader raises instead of waiting forever",
          all("raised" in o for o in out), "; ".join(out))


def unit_the_process_list_is_not_truncated():
    """The supervisor decides what to restart from a list of running processes. PowerShell wraps
    string output at the host width, only eighty columns under a scheduled task, which would cut the
    script names off and make every job look dead.

    The check that holds anywhere is that this process, running selftest.py, appears in its own
    answer. In a wide console this passes even with the bug, which is why supervise.procs() also
    refuses to return a list that does not contain itself.
    """
    import supervise
    rows = supervise.procs()
    if rows is None:
        check("the process list contains the process asking for it", False,
              "procs() refused the list it got back")
        return
    ok = any("selftest.py" in c for _, _, c in rows)
    longest = max((len(c) for _, _, c in rows), default=0)
    check("the process list contains the process asking for it", ok,
          f"{len(rows)} rows, longest command line {longest} chars")


def unit_a_stop_does_not_kill_the_next_search():
    """stop() ends a ponder by setting a flag. If nothing cleared it, the next real search would fail
    its first loop check, do zero visits and return the first legal move python-chess generates. After
    an unexpected reply, when tactics happen, that would be the move played."""
    import torch, chess
    from mcts import MCTS

    class Stub(torch.nn.Module):
        def forward(self, ids, cast, hm, ep):
            n = ids.shape[0]
            return {"h": torch.zeros(n, 4), "policy": torch.randn(n, 4162),
                    "value": torch.zeros(n), "qvalue": None}

    t = MCTS(Stub(), "cpu", batch=8)
    t.stop()                                   # exactly what stop_pondering does
    mv, info = t.search(chess.Board(), visits=32)
    check("a stopped ponder does not kill the next search", info.get("visits", 0) >= 32,
          f"{info.get('visits', 0)} visits after a stop")


def unit_move_space():
    """Every move index must survive a round trip, or a move the model wants is a move it cannot
    play. Underpromotions are the ones that break."""
    import chess
    from moves import move_to_idx, idx_to_move, N_MOVES
    bad = []
    for f in range(64):
        for t in range(64):
            for promo in (None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                m = chess.Move(f, t, promotion=promo)
                try:
                    i = move_to_idx(m, False)
                except Exception:
                    continue
                if not (0 <= i < N_MOVES):
                    bad.append((m.uci(), i))
    check("every move index is inside the move space", not bad, f"{len(bad)} outside" if bad else "")

    # the four promotion pieces must give four DIFFERENT indices from the same square pair
    seen = {move_to_idx(chess.Move(52, 60, promotion=p), False)
            for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)}
    check("the four promotions are four different moves", len(seen) == 4, f"{len(seen)} distinct")


def unit_clock_and_ep():
    """python-chess disagrees with itself about en passant. ep_square is set whenever a pawn just
    moved two squares, while the FEN only shows it when the capture is actually legal. The corpus
    uses the FEN convention, so this does."""
    import chess
    from chess_net import clock_and_ep
    b = chess.Board()
    b.push_san("e4")                    # ep_square set, no black pawn can take, so the FEN says none
    _, ep = clock_and_ep(b, False)
    ok_a = ep == 0
    b2 = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    _, ep2 = clock_and_ep(b2, False)
    check("en passant follows the FEN, not ep_square", ok_a and ep2 != 0,
          f"quiet {ep}, real {ep2}")


def unit_the_batch_encoder_equals_board_to_ids():
    """_prepare decodes the nodes' transposition keys with numpy instead of walking each board (about
    8 times faster). It must give what board_to_ids and clock_and_ep give, to the last bit, on every
    kind of position, both colours, en passant that is legal and en passant that is only set, every
    castling combination, promotions, clocks past 100, and castling flags a hostile FEN contradicts,
    with and without a move stack."""
    import random, chess
    from chess_net import board_to_ids, clock_and_ep
    from encode import encode_keys

    rng = random.Random(5)
    boards = [chess.Board(f) for f in (
        chess.STARTING_FEN, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 99 60",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
        "4k3/8/8/8/1r1pP2K/8/8/8 b - e3 0 1", "8/P7/8/8/8/8/7p/k6K b - - 250 1",
        "r3k2r/8/8/8/8/4K3/8/R6R w KQkq - 0 1", "4k3/8/8/8/8/8/8/4K3 w KQkq - 0 1")]
    for _ in range(40):                                  # random games, with the move stack
        b = chess.Board()
        for _ in range(rng.randrange(1, 200)):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
            boards.append(b.copy(stack=True))
    for _ in range(600):                                 # piece soup with contradicting castling flags
        b = chess.Board(None)
        sq = rng.sample(range(64), rng.randrange(2, 24))
        b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sq[1], chess.Piece(chess.KING, chess.BLACK))
        for s in sq[2:]:
            b.set_piece_at(s, chess.Piece(rng.choice([1, 2, 3, 4, 5]), rng.random() < 0.5))
        b.castling_rights = sum(chess.BB_SQUARES[s] for s in (chess.A1, chess.H1, chess.A8, chess.H8)
                                if rng.random() < 0.6)
        b.turn = rng.random() < 0.5
        b.halfmove_clock = rng.choice([0, 99, 100, 101, 300])
        if rng.random() < 0.3:
            b.ep_square = rng.choice(list(range(16, 24)) + list(range(40, 48)))
        boards.append(b)
        ms = [m for m in b.legal_moves if b.piece_type_at(m.from_square) != chess.KING]
        if ms:                                           # with a move stack the raw flags are trusted
            c = b.copy(stack=False)
            c.push(rng.choice(ms))
            boards.append(c)
    bad = 0
    for i in range(0, len(boards), 37):                  # an odd batch size on purpose
        chunk = boards[i:i + 37]
        ids, cast, hm, ep = encode_keys([b._transposition_key() for b in chunk],
                                        [b.halfmove_clock for b in chunk])
        for j, b in enumerate(chunk):
            i0, c0, flip = board_to_ids(b)
            h0, e0 = clock_and_ep(b, flip)
            if not (list(ids[j]) == i0 and list(cast[j]) == c0 and hm[j] == h0 and ep[j] == e0):
                bad += 1
    check("the batch encoder gives what board_to_ids gives", bad == 0,
          f"{len(boards)} boards, {bad} differ")


def unit_adapter_is_identity_at_birth():
    """The opponent adapter must start as exactly the frozen model, or every game begins with a
    different engine than the one that was measured."""
    import torch, chess
    from ttt import OpponentModel

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = torch.nn.Linear(8, 4162)
    om = OpponentModel(Stub(), "cpu")
    d = om.delta(torch.randn(3, 8))
    check("a fresh adapter changes nothing", float(d.abs().max()) == 0.0,
          f"max delta {float(d.abs().max()):.3e}")


def unit_adapter_only_touches_their_nodes():
    """The adapter predicts the person. If it leaked into nodes where the engine is to move it would
    change the engine's own judgement."""
    import torch, chess, numpy as np
    from mcts import MCTS, Node

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = torch.nn.Linear(4, 4162)
            self.qvalue = None

        def forward(self, ids, cast, hm, ep):
            n = ids.shape[0]
            return {"h": torch.zeros(n, 4), "policy": torch.zeros(n, 4162),
                    "value": torch.zeros(n), "qvalue": None}
        def parameters(self, *a, **k):
            return iter([torch.zeros(1)])

    class Adapter:
        def delta(self, h):
            return torch.full((h.shape[0], 4162), 7.0)

    m = Stub()
    t = MCTS(m, "cpu")
    t.opponent, t.opp_colour = Adapter(), chess.BLACK

    # real nodes, as the tree builds them, since _prepare reads IX
    # (see unit_batched_leaf_values_match_the_loop)
    white = chess.Board()
    black = chess.Board(); black.push_san("e4")
    pol, _, _ = t._evaluate([Node(white, t.draw_plies, t._nm), Node(black, t.draw_plies, t._nm)])
    check("the adapter touches their nodes only",
          abs(float(pol[0].max())) < 1e-6 and abs(float(pol[1].max()) - 7.0) < 1e-6,
          f"engine-to-move {float(pol[0].max()):.2f}, person-to-move {float(pol[1].max()):.2f}")


# ---------------------------------------------------------------- the server, over real HTTP
def server_up(proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(BASE + "/", timeout=5).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def http_checks():
    import chess
    a, b = Browser("a"), Browser("b")
    a.get(); b.get()
    check("each browser gets its own session", a.sid and b.sid and a.sid != b.sid,
          f"{a.sid} against {b.sid}")

    sa = a.post("new")
    sb = b.post("new")
    check("a new game starts on move one", not sa["movelist"].count("1."), "")

    # Two people, two boards. Player b must not see the move player a just made.
    sa = a.post("move", uci="e2e4")
    sb = b.post("state")
    check("one player's move does not appear on another's board",
          "e4" in sa["movelist"] and "e4" not in sb["movelist"],
          "a sees e4, b sees " + ("e4" if "e4" in sb["movelist"] else "nothing"))

    sb = b.post("move", uci="d2d4")
    sa = a.post("state")
    check("and the same in the other direction",
          "d4" in sb["movelist"] and "d4" not in sa["movelist"])

    # the engine must actually reply, not leave it as your move forever. Not via
    # chess.Board(fen).move_stack, since a FEN has no history and that is always zero.
    check("the engine replies", sa["plies"] == 2 and sa["last"] is not None,
          f"{sa['plies']} plies, engine played {sa['last']}")
    fen_a = chess.Board(sa["fen"])
    check("the position is legal after the engine moved", fen_a.is_valid(), sa["fen"])

    # legal moves the page draws must be the legal moves of the position it is showing
    board = chess.Board(sa["fen"])
    real = {}
    for m in board.legal_moves:
        real.setdefault(chess.square_name(m.from_square), set()).add(chess.square_name(m.to_square))
    shown = {k: set(v) for k, v in sa["legal"].items()}
    check("the highlighted squares are the legal moves", shown == real,
          f"{len(shown)} squares shown, {len(real)} legal")

    # an illegal move must be refused without changing anything
    before = sa["fen"]
    after = a.post("move", uci="a1a8")["fen"]
    check("an illegal move changes nothing", before == after)

    # rubbish must not take the server down
    junk = a.post("move", uci="not a move")
    check("rubbish input is survived", isinstance(junk, dict) and "fen" in junk)

    # undo takes back a pair
    n_before = a.post("state")["plies"]
    n_after = a.post("undo")["plies"]
    check("undo takes back your move and its reply", n_after == max(0, n_before - 2),
          f"{n_before} to {n_after}")

    # the visit dropdown must reach the engine
    s = a.post("depth", d=1600)
    check("the visit setting is the one the engine gets", s["depth"] == 1600, str(s["depth"]))
    s = a.post("depth", d=400)

    # Choosing an engine must reach a real handler. The version is per session, so one player
    # switching must not move anyone else's engine.
    s = a.post("new")
    menu = s.get("opponents") or []
    check("the board offers every frozen version", len(menu) >= 2,
          ", ".join(m["label"] for m in menu) or "nothing on the menu")
    if len(menu) >= 2:
        other = next(m["key"] for m in menu if m["key"] != s["opponent"])
        was = s["opponent"]
        s2 = a.post("opponent", which=other)
        check("picking a version switches the engine", s2["opponent"] == other,
              f"asked for {other}, got {s2['opponent']}")
        check("the eval bar names the version that judged it",
              other.replace("mayfly_v", "v") in s2["evaltext"].replace("MayFly ", ""),
              s2["evaltext"])
        sb = b.post("new")
        check("one player's choice does not move another's engine", sb["opponent"] == was,
              f"a chose {other}, b is on {sb['opponent']}")
        s3 = a.post("opponent", which=was)
        check("and it switches back", s3["opponent"] == was, s3["opponent"])

    # swapping sides must make the engine move first
    s = a.post("new")
    s = a.post("swapsides")
    check("swapping sides makes the engine open",
          s["you_are"] == "Black" and s["plies"] == 1 and s["last"] is not None,
          f"you are {s['you_are']}, {s['plies']} plies, opened with {s['last']}")
    a.post("swapsides"); a.post("new")

    # underpromotion has to survive the round trip to a knight
    u = Browser("u")
    u.get()
    s = u.post("new")
    # walk a pawn to the eighth rank by force is slow, so check the parse path directly instead
    s = u.post("move", uci="e2e4")
    check("a five character move is parsed", "fen" in s)


def concurrency_check(n=5):
    """Five people at once. The question is not speed, it is whether anything deadlocks or leaks
    from one game into another."""
    import chess
    brs = [Browser(f"p{i}") for i in range(n)]
    for b in brs:
        b.get()
        b.post("new")
    openings = ["e2e4", "d2d4", "g1f3", "c2c4", "b1c3"]
    out, errs = {}, []

    def play(i):
        try:
            s = brs[i].post("move", uci=openings[i], timeout=300)
            out[i] = s
        except Exception as e:
            errs.append(f"p{i}: {type(e).__name__}: {e}")

    t0 = time.time()
    ts = [threading.Thread(target=play, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=400)
    el = time.time() - t0
    check(f"{n} people move at once without an error", not errs, "; ".join(errs[:2]))
    check(f"{n} people get {n} different games",
          len({json.dumps(s["pieces"], sort_keys=True) for s in out.values()}) == len(out),
          f"{len(out)} replies, {el:.1f}s for all of them")
    sids = {b.sid for b in brs}
    check(f"{n} people have {n} sessions", len(sids) == n, f"{len(sids)} distinct cookies")
    return el


def search_report_check():
    """The page must report what the tree concluded, not only the bare network's opinion, so the
    number on screen is the one the engine played on."""
    if FAST:
        # `fast` starts the board at VISITS 0, the bare network, so there is no tree to report
        print("  search report: fast mode plays the bare network, no tree to report, skipped", flush=True)
        return
    r = Browser("r")
    r.get()
    r.post("new")
    s = r.post("move", uci="e2e4", timeout=300)
    sr = s.get("searched")
    check("the page reports what the tree concluded",
          isinstance(sr, dict) and sr.get("visits", 0) >= 1 and bool(sr.get("move")),
          f"{sr['visits']} visits, played {sr['move']}, {sr['cp']:+.0f}cp, line {sr['pv']}"
          if isinstance(sr, dict) else "nothing reported")


def clock_checks():
    """A real clock over real HTTP. Setting one restarts the game with both clocks full, the engine
    spends its own time and gets some back through the increment, and the page is told enough to
    draw a countdown. The engine must never spend more than the allocator allows."""
    b = Browser("clock")
    b.get()
    st = b.post("tc", v="60+2", timeout=300)
    ok = check("a time control starts a fresh game with both clocks full",
               st.get("tc") == "60+2" and st.get("clock") and abs(st["clock"]["you"] - 60) < 1.5
               and abs(st["clock"]["them"] - 60) < 1.5,
               f"tc {st.get('tc')}, clocks {st.get('clock')}")
    if not ok:
        return
    before = st["clock"]["them"]
    st = b.post("move", uci="e2e4", timeout=300)
    them, you = st["clock"]["them"], st["clock"]["you"]
    # the engine thought, so its clock moved, and 60+2 at ply 1 allocates well under ten seconds
    check("the engine pays for its own thinking", before - them > 0.02,
          f"{before:.1f}s to {them:.1f}s")
    check("one move cannot eat the clock", before - them < 10.0, f"spent {before-them:.2f}s")
    check("the increment comes back", you > 58.0, f"your clock reads {you:.1f}s")
    check("the page can draw a countdown", st.get("tcname") and st.get("flagged") is None
          and "turn_is_you" in st, f"tcname {st.get('tcname')!r}")
    b.post("tc", v="", timeout=300)          # back to fixed visits for the rest of the suite


def game_log_checks():
    """Games against people reach positions self-play cannot generate, since the model does not play
    like a human. They are only useful if they are kept and attributed."""
    import glob
    b = Browser("logger")
    b.get()
    b.post("name", n="selftest player")
    got = b.post("serious", on=True, elo=2090)
    check("a serious game and its player's rating are taken", got.get("serious") is True
          and got.get("elo") == 2090, f"serious {got.get('serious')}, elo {got.get('elo')}")
    b.post("new")
    for uci in ("e2e4", "g1f3", "f1c4", "b1c3"):
        b.post("move", uci=uci, timeout=300)
    before = len(glob.glob(os.path.join(HERE, "selftest_games", "*.pgn")))
    b.post("new", timeout=300)          # abandoning a game still keeps it
    pgns = glob.glob(os.path.join(HERE, "selftest_games", "*.pgn"))
    check("an abandoned game is still written", len(pgns) > before,
          f"{len(pgns)} pgn file(s)")
    if pgns:
        newest = max(pgns, key=os.path.getmtime)
        body = open(newest, encoding="utf-8").read()
        check("the game is attributed to the player",
              "selftest player" in body and "MayFly" in body,
              os.path.basename(newest))
        check("a serious game is saved as serious, with the player's rating",
              '[Serious "yes"]' in body and '2090' in body, os.path.basename(newest))
    # The name must outlive the process. Kept only on the Session object, a restart, an eviction or
    # a long idle spell would file the next game under "anonymous".
    names = os.path.join(HERE, "player_names.json")
    saved = json.load(open(names, encoding="utf-8-sig")) if os.path.exists(names) else {}
    check("the name you typed is kept where a restart cannot lose it",
          any("selftest player" in v for v in saved.values()),
          f"{len(saved)} name(s) on disk: " + ", ".join(sorted(set(saved.values()))[:4]))

    q = os.path.join(HERE, "selftest_games", "positions.txt")
    n = sum(1 for _ in open(q, encoding="utf-8")) if os.path.exists(q) else 0
    check("its positions are queued for labelling", n >= 8, f"{n} positions queued")


def idle_checks():
    """The board gives the GPU back when nobody is playing, and the health check must not count as
    somebody playing. The supervisor asks every five minutes whether the board can think, and if
    that counted as activity the board would never go idle.
    """
    import urllib.request as U

    def health():
        r = U.Request(f"{BASE}/api/health", data=b"{}",
                      headers={"Content-Type": "application/json"})
        return json.loads(U.urlopen(r, timeout=120).read())

    h = health()
    check("the health check answers and proves it can think",
          h.get("status") in ("awake", "sleeping"), str(h)[:90])

    print("  waiting out the idle timeout, polling health the whole time", flush=True)
    t0 = time.time()
    polls = 0
    while time.time() - t0 < 100:
        time.sleep(10)
        polls += 1
        if health().get("status") == "sleeping":
            break
    h = health()
    check("it goes to sleep with nobody playing",
          h.get("status") == "sleeping",
          f"{h.get('status')} after {time.time()-t0:.0f}s and {polls} health checks")

    # and the health checks did not keep it awake, which is the point
    h2 = health()
    check("the health check does not count as playing",
          h2.get("status") == "sleeping", str(h2)[:90])

    b = Browser("idle")
    b.get()
    s = b.post("state", timeout=300)
    check("a real request wakes it", "status" in s and health().get("status") == "awake",
          f"woke to: {health().get('status')}")


def tactics_checks():
    """Two positions with one right answer each. Not a strength test but a check that the search runs
    at all, since an engine doing zero visits fails both."""
    import chess
    t = Browser("t")
    t.get()
    t.post("depth", d=200)
    cases = [
        ("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", "e4d5", "takes a hanging queen"),
        ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8", "finds mate in one"),
    ]
    for fen, want, label in cases:
        s = t.post("fen", fen=fen, timeout=300)
        if "error" in s:
            check(f"the engine {label}", False, "setting the position failed: " + s["error"])
            continue
        # the person to move is the human, so swap sides to make the engine play this position
        s = t.post("swapsides", timeout=300)
        if "error" in s:
            check(f"the engine {label}", False, "swapping sides failed: " + s["error"])
            continue
        check(f"the engine {label}", s["last"] == want,
              f"played {s['last']}, wanted {want}")
    t.post("depth", d=400)


def reap_strays(port):
    """Kill any board left listening on this suite's port by an interrupted earlier run."""
    me = os.getpid()
    killed = []
    try:
        import subprocess as _sp
        ps = ("Get-CimInstance Win32_Process -Filter \"Name LIKE 'py%'\" | "
              "Where-Object { $_.CommandLine -like '*play_web.py " + str(port) + "*' } | "
              "ForEach-Object { $_.ProcessId }")
        out = _sp.run(["powershell", "-NoProfile", "-Command", ps],
                      capture_output=True, text=True, timeout=60).stdout
        for line in out.split():
            if line.strip().isdigit() and int(line) != me:
                _sp.run(["taskkill", "/PID", line.strip(), "/F"],
                        capture_output=True, timeout=30)
                killed.append(line.strip())
    except Exception as e:
        print(f"  could not check for strays: {type(e).__name__}: {e}", flush=True)
    if killed:
        print(f"  reaped {len(killed)} stray board(s) from an earlier run: "
              f"{', '.join(killed)}", flush=True)


def unit_the_legality_mask_never_hides_the_target():
    """The mask must reproduce python-chess exactly, and must never hide a move being trained on.

    A target move the mask calls illegal gets zero probability, which is an infinite cross-entropy,
    and the decode is not perfect (about one record in 250). So batch_tensors adds the target's own
    moves to the mask, and this checks that on a record deliberately built to need it.

    Exactness matters the other way too. A mask that wrongly hides a legal move teaches the head to
    put no prior on it, while `_expand` still offers it to PUCT at play time. So castling rights and
    the en passant square are reconstructed and checked here, not just the piece placement."""
    import numpy as np, torch, chess
    import torch.nn.functional as F
    from pack import encode_record, batch_tensors, legal_mask, DT
    from moves import move_to_idx, N_MOVES
    from chess_net import board_to_ids

    feat = {"material": [0, 0], "stm_mobility": 0, "king_safety": [0, 0],
            "pawns": [[0, 0, 0], [0, 0, 0]], "bishop_pair": [0, 0]}

    def rec(fen, ucis):
        b = chess.Board(fen)
        ids, cast, flip = board_to_ids(b)
        idx = [move_to_idx(chess.Move.from_uci(m), flip) for m in ucis]
        ep, epv = 0, False
        if b.ep_square is not None:
            ep, epv = (b.ep_square ^ 56 if flip else b.ep_square) + 1, True
        return encode_record(ids, cast, idx, [0] * len(idx), 10, feat, 0, True,
                             b.halfmove_clock, ep, True, epv), b, flip

    # Castling for both colours, en passant and a promotion, the things a decode that only restores
    # piece placement gets wrong, and each a move the head must be able to learn a prior for.
    cases = [
        ("start position",            chess.STARTING_FEN, ["e2e4"]),
        ("white castling both ways",  "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", ["e1g1"]),
        ("black to move, castling",   "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", ["e8c8"]),
        ("en passant on the board",   "8/8/8/3pP3/8/8/4K1k1/8 w - d6 0 2", ["e5d6"]),
        ("a promotion",               "8/P6k/8/8/8/8/8/K7 w - - 0 1", ["a7a8q"]),
        ("rights the rook has lost",  "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/4K2R w Kkq - 0 1", ["e1g1"]),
    ]
    rows = np.zeros(len(cases), DT)
    meta = []
    for i, (name, fen, mv) in enumerate(cases):
        rows[i], b, flip = rec(fen, mv)
        meta.append((name, b, flip))

    m = legal_mask(rows, N_MOVES)
    for i, (name, b, flip) in enumerate(meta):
        want = {j for j in (move_to_idx(mv, flip) for mv in b.legal_moves) if 0 <= j < N_MOVES}
        got = set(np.where(m[i])[0])
        assert want == got, (f"{name}: the mask disagrees with python-chess on "
                             f"{sorted(want ^ got)}")
        assert 0 < len(got) < 60, f"{name}: {len(got)} legal moves is not a chess position"
    # not vacuous, since if this ever reads 4,162 the mask is masking nothing and every other
    # assertion in here passes trivially
    assert m.sum(1).mean() < 60, f"the mask hides nothing: {m.sum(1).mean():.0f} moves a position"

    # The union, on a record built to need it. The start position with its target at a1a8, a rook
    # move through its own pieces, is what a decode failure looks like.
    bad = np.zeros(1, DT)
    b = chess.Board()
    ids, cast, flip = board_to_ids(b)
    j = move_to_idx(chess.Move.from_uci("a1a8"), flip)
    bad[0] = encode_record(ids, cast, [j], [0], 10, feat, 0, True, 0, 0, True, False)
    raw = legal_mask(bad, N_MOVES)
    assert not raw[0, j], "a1a8 is legal at the start position, so this check proves nothing"
    out = batch_tensors(bad, N_MOVES, 100.0, legal=True)
    assert out["legal"][0, j], "batch_tensors let a target move stay masked: the run would die"

    # and the loss the trainer will actually compute is finite, and lower, on that same record
    pol = torch.as_tensor(out["policy"])
    lg = torch.as_tensor(out["legal"])
    logits = torch.zeros(1, N_MOVES)
    logits[0, j] = 2.0
    plain = float(-(pol * F.log_softmax(logits, -1)).sum(-1).mean())
    masked = float(-(pol * F.log_softmax(logits.masked_fill(~lg, -1e9), -1)).sum(-1).mean())
    assert masked == masked and masked < float("inf"), f"masked loss is not finite: {masked}"
    assert masked < plain, f"masking did not concentrate the distribution: {masked} vs {plain}"
    print(f"  legality mask: {len(cases)} positions exact, "
          f"{m.sum(1).mean():.0f} moves a position, target survives a bad decode, "
          f"loss {plain:.2f} -> {masked:.2f}", flush=True)


def unit_the_rate_halves_on_the_version_boundaries():
    """The learning rate must step exactly where the versions freeze, and a restart must find the
    same rate without being told.

    The schedule is driven by `versions_owed(hours)` with an offset, and an offset wrong by one would
    silently leave v4 at v3's rate or skip to v5's.

    It also pins what makes the schedule safe to restart. `lr_at` reads hours and nothing else, so the
    rate comes back from the checkpoint's trained_hours rather than from a counter a crash could lose."""
    import train_av as T

    # the ladder, against the table it was designed from
    want = [(10.0, 1.2e-4), (30.0, 1.2e-4), (60.0, 1.2e-4),      # v1, v2, v3 all ran flat
            (96.0, 6.0e-5), (102.7, 6.0e-5), (191.9, 6.0e-5),    # v4's window
            (192.0, 3.0e-5), (384.0, 1.5e-5), (768.0, 7.5e-6)]
    for h, lr in want:
        got = T.lr_at(1_720_000, h)
        assert abs(got - lr) < 1e-12, f"at {h}h the rate is {got:.3g}, not {lr:.3g}"

    # each boundary is a halving and not a drift
    for lo, hi in ((95.9, 96.0), (191.9, 192.0), (383.9, 384.0)):
        a, b = T.lr_at(1_720_000, lo), T.lr_at(1_720_000, hi)
        assert abs(b * 2 - a) < 1e-12, f"the step at {hi}h is {a:.3g} -> {b:.3g}, not a halving"

    # warmup still owns the first WARMUP steps, and is not cancelled by the version factor
    assert T.lr_at(0, 102.7) < T.lr_at(T.WARMUP, 102.7), "warmup was lost"
    assert abs(T.lr_at(T.WARMUP * 10, 102.7) - T.lr_at(T.WARMUP, 102.7)) < 1e-12

    # and the whole thing is a function of hours, not of hidden state, so the same answer twice
    assert T.lr_at(1_720_000, 102.7) == T.lr_at(1_720_000, 102.7)
    print(f"  learning rate: flat at {T.LR:.3g} through v1-v3, "
          f"{T.lr_at(1_720_000, 102.7):.3g} in v4's window, halving at every freeze after",
          flush=True)


def unit_every_sealed_version_pins_every_setting():
    """Every version on disk must pin every search setting, the newest included.

    A version is frozen at release and never changes, and improvements go into the next version. No
    version may be missing a key, because `apply` leaves an unnamed key at today's default, so a
    missing key lets the next change leak backwards.

    This reads the live manifest, not a fixture. A new key added to SEARCH_KEYS fails it until every
    version is given the value it actually played with."""
    from mcts import MCTS, SEARCH_KEYS, default_search, pinned_search, versions_on_disk

    d = default_search()
    assert set(d) == set(SEARCH_KEYS),         f"default_search records {len(d)} of {len(SEARCH_KEYS)}: missing {set(SEARCH_KEYS) - set(d)}"
    assert d == MCTS(None, "cpu").settings(), "default_search has drifted from the constructor"

    rows = versions_on_disk()
    if not rows:
        print("  sealed versions: no versions.json here, manifest part skipped", flush=True)
        return
    for r in rows:
        n = int(r["version"])
        blk = r.get("search")
        assert blk, f"v{n} has NO sealed search: it plays whatever today's defaults are"
        missing = set(SEARCH_KEYS) - set(blk)
        assert not missing, f"v{n} is sealed without {sorted(missing)}: those leak from today"
        assert pinned_search(r["file"]) == blk, f"v{n}'s pinned search is not its sealed block"
    v1 = next((r for r in rows if int(r["version"]) == 1), None)
    if v1 is not None:
        assert v1["search"]["vl"] == 1.0, "v1 predates the virtual-loss change and must keep vl=1.0"
    print(f"  sealed versions: {', '.join('v%d' % int(r['version']) for r in rows)} all frozen, "
          f"all {len(SEARCH_KEYS)} settings pinned", flush=True)


def unit_a_version_is_sealed_the_moment_it_freezes():
    """freeze_version must seal the version it freezes, completely, and touch no other.

    This is the half of frozen-at-release that runs inside the trainer, unattended. So the real
    function runs here, into a scratch directory, against a manifest that already holds a sealed v1
    with vl=1.0, and freezes v2 on a toy model."""
    import os, json, tempfile, torch
    import train_av as T
    from mcts import SEARCH_KEYS, default_search

    d = tempfile.mkdtemp()
    v1_block = dict(default_search(), vl=1.0, tree_policy="puct", fpu_root=-1.0)
    json.dump([{"version": 1, "file": "mayfly_v1.pt", "search": v1_block}],
              open(os.path.join(d, "versions.json"), "w"))
    saved = (T.VERSION_DIR, T.VERSIONS)
    T.VERSION_DIR, T.VERSIONS = d, os.path.join(d, "versions.json")
    try:
        T.freeze_version(2, torch.nn.Linear(2, 2), 100, 48.0,
                         {"q": 0.5, "policy": 3.0, "top1": 0.2}, 12)
    finally:
        T.VERSION_DIR, T.VERSIONS = saved
    rows = {int(r["version"]): r for r in json.load(open(os.path.join(d, "versions.json")))}
    assert 2 in rows, "freeze_version did not record the version it froze"
    blk = rows[2].get("search")
    assert blk, "the version just frozen has no sealed search: it would play whatever comes next"
    assert set(blk) == set(SEARCH_KEYS), f"sealed incompletely, missing {set(SEARCH_KEYS) - set(blk)}"
    assert blk == default_search(), "sealed with something other than the search at the freeze"
    assert rows[1]["search"] == v1_block, "freezing v2 rewrote v1's seal"
    print(f"  freeze_version: v2 sealed at release with all {len(blk)} settings "
          f"({blk['tree_policy']}, vl {blk['vl']}), v1 untouched", flush=True)


def unit_a_mate_in_one_is_played_without_search():
    """From v4 a mate in one is played at once, before the network is asked anything. It is sealed off
    for v1 to v3, where the search still has to find it, so both halves are checked. On, the mate comes
    back with zero evaluations. Off, a real search runs. A network can rank a mate very low, and
    pondering can bury it."""
    import chess
    from mcts import MCTS
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")          # Ra8 is a back-rank mate
    mate = b.parse_san("Ra8#")
    on = MCTS(_det_stub(), "cpu", batch=8).apply({"root_mate": True, "tb_dtz": False})
    mv, info = on.search(b.copy(), visits=64, reuse=False)
    assert mv == mate, f"root_mate on played {b.san(mv)}, not the mate"
    assert on.stats["evals"] == 0, f"{on.stats['evals']} evaluations before a proven mate"
    assert info.get("proof") == "mate in one", f"reported proof {info.get('proof')!r}"
    off = MCTS(_det_stub(), "cpu", batch=8).apply({"root_mate": False, "tb_dtz": False})
    off.search(b.copy(), visits=64, reuse=False)
    assert off.stats["evals"] > 0, "root_mate off still skipped the search"
    print(f"  mate in one: 0 evaluations when on, a real search ({off.stats['evals']} "
          f"evaluations) when off", flush=True)


def unit_pondered_visits_are_discounted():
    """From v4 visits carried over from a ponder are scaled at the new root with every Q kept exactly.
    The move is chosen by visit count, so a ponder tree arrives with its expected reply far ahead and
    a newly found mate can lose the count. At 1.0, which v1 to v3 are sealed with, the carried tree
    must come through untouched."""
    import chess, numpy as np
    from mcts import MCTS
    b = chess.Board()
    b.push_san("e4")
    for d in (1.0, 0.25):
        t = MCTS(_det_stub(), "cpu", batch=8).apply(
            {"ponder_discount": d, "root_mate": False, "tb_dtz": False})
        t.search(b.copy(), visits=300, reuse=False)            # thinking on the person's time
        reply = b.copy()
        reply.push_san("e5")
        kid = t._root_for(reply)
        assert kid is not None and kid.expanded and kid.N.sum() > 0, "the reply was not searched"
        n0, w0 = kid.N.astype(np.float64).copy(), kid.W.astype(np.float64).copy()
        t.search(reply.copy(), visits=1, reuse=True)            # one new visit touches one edge
        assert t.root is kid, "the ponder subtree was not reused"
        moved = np.flatnonzero(np.abs(kid.N - n0 * d) > 1e-3)
        assert len(moved) <= 1, f"{len(moved)} edges changed beyond the discount"
        keep = np.setdiff1d(np.flatnonzero(n0 > 0), moved)
        assert np.allclose(kid.N[keep], n0[keep] * d, rtol=1e-5), f"visits not scaled by {d}"
        assert np.allclose(kid.W[keep] / kid.N[keep], w0[keep] / n0[keep], atol=1e-5),             f"the discount at {d} changed a Q value"
    print("  pondered visits: untouched at 1.0, scaled at 0.25, every Q unchanged", flush=True)


def unit_the_tablebase_root_plays_for_progress():
    """From v4, in a tablebase win the root plays the move that reaches a capture or pawn move soonest.
    Win/draw/loss alone rates every winning move equally, and v3 once shuffled its queen for sixteen
    moves in this position (Black to make its 72nd move, where v3 played Qg8). Needs the tables, and
    says so where they are missing."""
    import chess
    from mcts import MCTS, tablebase
    if tablebase() is None:
        print("  tablebase root: no tables on this machine, skipped", flush=True)
        return
    b = chess.Board("8/7q/8/6P1/p2k2K1/8/8/2B5 b - - 4 72")
    t = MCTS(_det_stub(), "cpu", batch=8).apply({"root_mate": False, "tb_dtz": True})
    got = t._tb_root_move(b.copy(stack=False))
    assert got is not None, "a won six-piece position went unanswered"
    assert b.is_zeroing(got), f"played {b.san(got)}, which does not reset the fifty-move count"
    off = MCTS(_det_stub(), "cpu", batch=8).apply({"root_mate": False, "tb_dtz": False})
    assert off._proven_move(b) is None, "tb_dtz off still answered from the tables"
    print(f"  tablebase root: plays {b.san(got)} where v3 shuffled Qg8; off leaves it to the "
          f"search", flush=True)


# ---------------------------------------------------------------- search features, v4 onwards
def _hand_root(n, w, v2):
    """A root with hand-set statistics and no children, for the move-choice checks."""
    import numpy as np, chess
    from mcts import Node
    r = Node(chess.Board())
    k = len(r.moves)
    z = lambda: np.zeros(k, np.float32)
    r.P, r.N, r.W, r.V2, r.F, r.X = z() + 1.0 / k, z(), z(), z(), z(), z()
    r.kids = [None] * k
    for i, (nn, ww, vv) in enumerate(zip(n, w, v2)):
        r.N[i], r.W[i], r.V2[i] = nn, ww, vv
    r.sum_n = float(r.N.sum())
    return r


def unit_lcb_plays_the_move_it_is_surest_of():
    """With lcb the most visited move is not always played. A has 100 visits averaging 0.10, B has 60
    averaging 0.30, both tight, and C has 3 visits averaging 0.90, below the floor. argmax N plays A,
    the bound plays B, and C's lucky average never plays. Off, it is argmax N exactly."""
    from mcts import MCTS
    # values are A all 0.10, B all 0.30, C all 0.90, so V2 = n * v^2
    root = _hand_root([100, 60, 3], [10.0, 18.0, 2.7], [1.0, 5.4, 2.43])
    off = MCTS(_det_stub(), "cpu").apply({"lcb": False, "proofs": False})
    on = MCTS(_det_stub(), "cpu").apply({"lcb": True, "proofs": False})
    assert off._choose(root) == 0, f"lcb off played edge {off._choose(root)}, not argmax N"
    assert on._choose(root) == 1, f"lcb played edge {on._choose(root)}, not the surer B"
    # spread matters, so B at the same mean but split +-1 is a worse bet than A's steady 0.10
    wild = _hand_root([100, 60, 3], [10.0, 18.0, 2.7], [1.0, 60.0, 2.43])
    assert on._choose(wild) == 0, "a wildly spread B still beat a steady A"
    print("  lcb: argmax N plays A, the bound plays the tighter better B, a wild B loses to A, "
          "a 3-visit fluke never plays", flush=True)


def unit_a_virtual_mean_leaves_q_alone():
    """Under virtual_mean a pending visit narrows U and must not move Q, and taking it back must put W
    and N exactly where they were. A whole search under it keeps the counters balanced and the
    in-flight arrays empty, which the proofs rely on."""
    import numpy as np, chess
    from mcts import MCTS
    # The Q selection reads, which is not always W/N. Under q_init (v4's default) it is
    # (q_n0*Q0 + W)/(q_n0 + N), and the pending visit adds exactly that value, so that is what must
    # not move. Both settings are checked, so the combination v4 ships is tested.
    for qi in (False, True):
        t = MCTS(_det_stub(), "cpu", batch=8, cache_size=0).apply(
            {"virtual_mean": True, "q_init": qi, "root_mate": False, "tb_dtz": False})
        t.search(chess.Board(), visits=120, reuse=False)
        r = t.root
        n0, w0 = r.N.copy(), r.W.copy()
        q_before = np.asarray(t._terms(r, t._fpu_base(r), is_root=True)[0], dtype=np.float64)
        got = t._descend(r)
        assert got is not None, "the descent found nothing to expand"
        path = got[0]
        a = path[0][1]
        q_after = np.asarray(t._terms(r, t._fpu_base(r), is_root=True)[0], dtype=np.float64)
        if qi or n0[a] > 0:
            assert abs(q_after[a] - q_before[a]) < 1e-5,                 f"q_init {qi}: a pending visit moved the Q that selection reads"
        t._unwind(path)
        assert np.array_equal(r.N, n0) and np.allclose(r.W, w0, atol=1e-6), "unwind was not exact"
    for vm in (False, True):
        s = MCTS(_det_stub(), "cpu", batch=16).apply(
            {"virtual_mean": vm, "root_mate": False, "tb_dtz": False})
        s.search(chess.Board(), visits=400, reuse=False)
        rr = s.root
        assert abs(float(rr.N.sum()) - rr.sum_n) < 1e-6, f"vm {vm}: counters drifted"
        assert float(np.abs(rr.F).max()) == 0.0, f"vm {vm}: visits left in flight"
        assert float(np.abs(rr.X).max()) < 1e-4, f"vm {vm}: provisional values left behind"
    print("  virtual mean: Q unmoved by a pending visit, unwind exact, nothing left in flight",
          flush=True)


def unit_a_proof_climbs_the_tree():
    """With proofs on, a mate two moves deep is proven at the root and played, whatever the network
    thinks. In this K+R v K position 1.Kb6 or 1.Kc7 mates next move (checked by brute force), and
    there is no mate in one, so root_mate cannot answer it instead. Off, the root is never marked.
    The terminal conversions are counted, so a proof that never fires fails."""
    import chess
    from mcts import MCTS
    b = chess.Board("k7/8/2K5/8/8/8/8/7R w - - 0 1")
    on = MCTS(_det_stub(), "cpu", batch=16, cache_size=0).apply(
        {"proofs": True, "root_mate": True, "tb_dtz": False})
    # tables off, or every rook move is already a proven tablebase win and the tree rightly plays
    # the most visited of them
    on.use_tb = False
    mv, info = on.search(b.copy(), visits=3000, reuse=False)
    assert on.root.lo >= 1.0, f"the root was not proven won (bounds {on.root.lo}, {on.root.hi})"
    assert mv.uci() in ("c6b6", "c6c7"), f"played {mv.uci()}, not a mating move"
    assert info["solved"] > 0, "no node was converted to a terminal"
    assert info["visits"] < 3000, "the search did not stop once the root was proven"
    edge = [m.uci() for m in on.root.moves].index(mv.uci())
    assert abs(float(on.root.W[edge] / on.root.N[edge]) - 1.0) < 1e-5, \
        "the proven move's average was not corrected to a win"
    off = MCTS(_det_stub(), "cpu", batch=16, cache_size=0).apply(
        {"proofs": False, "root_mate": True, "tb_dtz": False})
    off.use_tb = False
    off.search(b.copy(), visits=600, reuse=False)
    assert off.root.lo == -1.0 and off.stats["solved"] == 0, "proofs off still proved"
    print(f"  proofs: mate in two proven at the root in {info['visits']} visits, "
          f"{info['solved']} nodes solved, plays {b.san(mv)}; off, nothing proven", flush=True)


def unit_the_tree_draws_at_one_hundred_plies():
    """With draw_plies v4 draws where the harness and the arbiter do, at 100 plies without a capture or
    pawn move, while v1 to v3 keep the seventy-five-move 150. A root at the limit is still searched,
    since it has legal moves and a move has to come back."""
    import chess
    from mcts import MCTS, Node
    b = chess.Board("8/8/4k3/8/8/2K5/8/6R1 w - - 100 90")
    assert Node(b, 100).term == 0.0 and Node(b, 150).term is None, "the rule is not the setting"
    assert Node(b, 100).lo == 0.0 == Node(b, 100).hi, "a rules draw is not exact"
    t = MCTS(_det_stub(), "cpu", batch=8).apply({"draw_plies": 100, "tb_dtz": False})
    mv, info = t.search(b.copy(), visits=64, reuse=False)
    assert mv is not None and mv in b.legal_moves, "a root at the limit returned no move"
    print("  draw at 100 plies: a node at the limit is a terminal draw at 100 and not at 150, "
          "and a root there still moves", flush=True)


def unit_a_terminal_root_still_gets_a_move():
    """The game can reach a position the previous tree marked terminal (a repetition or tablebase hit
    met as a grandchild). The search must still return a move from it, not None, which the board
    reads as "no move". Reproduced by marking a grandchild the way `_repeats` would, then searching
    from it."""
    import chess
    from mcts import MCTS
    b = chess.Board()
    t = MCTS(_det_stub(), "cpu", batch=8).apply({"root_mate": False, "tb_dtz": False})
    t.search(b.copy(), visits=300, reuse=False)
    found = None
    for i, kid in enumerate(t.root.kids):
        if kid is not None and kid.expanded:
            for j, g in enumerate(kid.kids):
                if g is not None:
                    found = (t.root.moves[i], kid.moves[j], g)
                    break
        if found:
            break
    assert found, "no grandchild to test with"
    m, h, g = found
    g.term = 0.0                                   # exactly what a repetition mark does
    after = b.copy()
    after.push(m)
    after.push(h)
    mv, info = t.search(after, visits=64, reuse=True)
    assert mv is not None and mv in after.legal_moves, "a terminal grandchild root gave no move"
    print(f"  terminal root: after {m.uci()} {h.uci()}, marked terminal, the search still "
          f"plays {mv.uci()}", flush=True)


def unit_q_init_scores_a_move_by_its_own_value():
    """Under q_init an unvisited edge is worth the q head's value for that move, not its parent's less
    an urgency, and a visited one is (n0 * q0 + W) / (n0 + N)."""
    import numpy as np
    from mcts import MCTS
    root = _hand_root([4, 0, 0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    root.Q0 = np.linspace(-0.5, 0.5, len(root.moves)).astype(np.float32)
    root.value = 0.2
    t = MCTS(_det_stub(), "cpu").apply({"q_init": True, "q_n0": 2.0})
    q, _ = t._terms(root, t._fpu_base(root), is_root=True)
    want0 = (2.0 * float(root.Q0[0]) + 2.0) / (2.0 + 4.0)
    assert abs(float(q[0]) - want0) < 1e-6, f"visited edge {float(q[0])} not {want0}"
    assert np.allclose(q[1:], root.Q0[1:], atol=1e-6), "an unvisited edge is not its own q0"
    off = MCTS(_det_stub(), "cpu").apply({"q_init": False})
    q2, _ = off._terms(root, off._fpu_base(root), is_root=True)
    assert not np.allclose(q2[1:], root.Q0[1:]), "q_init off still read the q head"
    print("  q_init: unvisited edges carry their own q, visited ones start from it", flush=True)


def unit_the_fast_pick_is_the_pick():
    """`_pick_fast` (the selection v4 and v5 play, in fewer numpy calls) must return the edge and the q
    that `_pick` returns, bit for bit, at every node a search touches, since it is a speed change only.
    Each pick is made both ways on the same state, in v5's search and in v4's (c1, c2, q_n0,
    policy_temp, the explore terms and forcing set back), at batch 8 and 48, from a fresh root and from
    a carried one (the ponder discount scales the root's N, W and V2).

    On the Python descent. Under timing_fixes the compiled walk (walk.c) makes the pick itself and never
    calls `_pick_fast`, so the picks are compared with the compiled walk held off. A second check then
    runs the same searches through the walk and expects the same moves and the same root N and W, bit
    for bit, with `_pick_fast` never called."""
    import numpy as np, chess
    import mcts
    from mcts import MCTS

    real, real_walk, seen, bad = MCTS._pick_fast, mcts._native_walk, [0, 0], []

    def both(self, node, is_root, ply1):
        forced = self.stats.get("forced", 0)
        want = MCTS._pick(self, node, self._fpu_base(node), is_root=is_root, ply1=ply1)
        self.stats["forced"] = forced                    # `_pick_fast` counts its own forced pick
        got = real(self, node, is_root, ply1)
        seen[0] += 1
        if got[0] != want[0] or np.float64(got[1]).tobytes() != np.float64(want[1]).tobytes():
            seen[1] += 1
            if len(bad) < 3:
                bad.append(f"{got} against {want}")
        return got

    v4 = {"c1": 2.0 ** 0.5, "c2": 3.0, "q_n0": 2.0, "policy_temp": 1.0, "explore_growth": 0.0,
          "explore_base": 500.0, "forcing": 0, "lcb_alpha": 1e-5, "lcb_floor": 0.10}

    def run(opts, batch):
        t = MCTS(_det_stub(), "cpu", batch=batch, cache_size=100_000).apply(opts)
        b = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9")
        mv, _ = t.search(b, visits=300, reuse=False)
        b.push(mv)
        b.push(sorted(b.legal_moves, key=lambda m: m.uci())[0])
        mv2, _ = t.search(b, visits=300, reuse=True)      # carried, so a discounted root
        return mv.uci(), mv2.uci(), t.root.N.tobytes(), t.root.W.tobytes()

    runs = [(name, opts, batch) for name, opts in (("v5", {}), ("v4", v4)) for batch in (8, 48)]
    python = {}
    MCTS._pick_fast = both
    mcts._native_walk = lambda: None                    # the Python descent, which calls _pick_fast
    try:
        for name, opts, batch in runs:
            python[(name, batch)] = run(opts, batch)
    finally:
        MCTS._pick_fast = real
        mcts._native_walk = real_walk
    check("the fast pick is the pick, bit for bit", seen[0] > 1000 and not seen[1],
          "; ".join(bad) or f"{seen[0]} picks compared, none differ")

    if real_walk() is None:
        print("  compiled walk: not built on this machine, skipped", flush=True)
        return
    calls, differ = [0], []

    def counted(self, node, is_root, ply1):
        calls[0] += 1
        return real(self, node, is_root, ply1)

    MCTS._pick_fast = counted
    try:
        for name, opts, batch in runs:
            if run(opts, batch) != python[(name, batch)]:
                differ.append(f"{name} settings at batch {batch}")
    finally:
        MCTS._pick_fast = real
    check("the compiled walk is the Python descent, bit for bit", not differ and not calls[0],
          "; ".join(differ) or (f"_pick_fast was called {calls[0]} times, so the walk did not run"
                                if calls[0] else f"{len(runs)} searches fresh and carried, same moves, "
                                                 f"same root N and W"))


def unit_the_tablebase_respects_the_fifty_move_counter():
    """For tb_rule50. WDL assumes a fresh fifty-move counter. KQ v K is a win at clock 0, but at clock
    99 no mate or capture is one ply away, so the defender claims a draw, and a tree that reads the
    counter must say 0 where one that does not says 1. Needs the tables."""
    import chess
    from mcts import MCTS, tablebase
    if tablebase() is None:
        print("  tablebase fifty-move rule: no tables on this machine, skipped", flush=True)
        return
    fresh = chess.Board("8/8/8/4k3/8/8/8/3QK3 w - - 0 1")
    stale = chess.Board("8/8/8/4k3/8/8/8/3QK3 w - - 99 60")
    on = MCTS(_det_stub(), "cpu").apply({"tb_rule50": True})
    off = MCTS(_det_stub(), "cpu").apply({"tb_rule50": False})
    assert on._tb_value(fresh) == 1.0 and off._tb_value(fresh) == 1.0, "KQ v K is not a win"
    assert on._tb_value(stale) == 0.0, f"at clock 99 it says {on._tb_value(stale)}, not a draw"
    assert off._tb_value(stale) == 1.0, "tb_rule50 off changed what v1-v3 see"
    print("  tablebase fifty-move rule: KQ v K a win at clock 0, a draw at 99; off, a win at "
          "both", flush=True)


def unit_the_tree_reads_the_head_it_is_sealed_with():
    """v4's network carries a second policy and a second action-value head, on the square tokens.
    Which pair the tree reads changes every prior and every leaf value, so it is sealed like the
    rest, and a checkpoint without those heads has to fall back rather than fail."""
    import numpy as np, torch, chess
    from mcts import MCTS
    from moves import move_to_idx

    b = chess.Board()
    order = [move_to_idx(m, False) for m in b.legal_moves]
    # two moves that are actually legal here, or the spikes sit on indices the softmax over legal
    # moves never sees and the check reads whichever move happens to come first
    i_cls, i_att = order[0], order[7]

    class Two(torch.nn.Module):
        """A summary-token pair that loves one move, an attention pair that loves another."""

        def forward(self, ids, cast, hm, ep):
            n = ids.shape[0]
            cls_p = torch.full((n, 4162), -9.0)
            cls_p[:, i_cls] = 9.0
            att_p = torch.full((n, 4162), -9.0)
            att_p[:, i_att] = 9.0
            return {"h": torch.zeros(n, 4), "policy": cls_p, "policy_att": att_p,
                    "value": torch.zeros(n), "qvalue": torch.full((n, 4162), -4.0),
                    "qvalue_att": torch.full((n, 4162), 4.0)}

    got = {}
    for head in ("cls", "att"):
        t = MCTS(Two(), "cpu", batch=8, cache_size=0).apply(
            {"policy_head": head, "q_head": head, "root_mate": False, "tb_dtz": False})
        t.search(b.copy(), visits=32, reuse=False)
        got[head] = (order[int(np.argmax(t.root.P))], float(t.root.value))
    assert got["cls"][0] == i_cls, f"cls read move index {got['cls'][0]}, not {i_cls}"
    assert got["att"][0] == i_att, f"att read move index {got['att'][0]}, not {i_att}"
    assert got["cls"][1] < 0 < got["att"][1], \
        f"the leaf value did not follow the q head: {got['cls'][1]:.3f}, {got['att'][1]:.3f}"
    # a network from before those heads, where the setting falls back instead of failing
    old = MCTS(_det_stub(), "cpu", batch=8, cache_size=0).apply(
        {"policy_head": "att", "q_head": "att", "root_mate": False, "tb_dtz": False})
    assert old.search(b.copy(), visits=32, reuse=False)[0] is not None, \
        "a checkpoint without attention heads did not fall back"
    print(f"  head switch: cls takes move {got['cls'][0]} at value {got['cls'][1]:+.2f}, att takes "
          f"{got['att'][0]} at {got['att'][1]:+.2f}; an older network falls back", flush=True)


def unit_the_clock_cannot_be_flagged_by_one_move():
    """The allocator's one hard guarantee is that no single move may take enough of the clock to lose
    on time, even after the search extends itself to the maximum. A move that returns nothing is also
    a loss, so it has a floor as well."""
    from timectl import allocate, MAX_SHARE, MOVE_NOW
    bad = []
    for base, inc in ((60, 0), (180, 2), (600, 0), (900, 10)):
        for remaining in (base, base * 0.5, 10.0, 3.0, 0.8, 0.2, 0.0):
            for ply in (0, 20, 60, 140):
                s = allocate(remaining, inc, ply)
                keep = max(0.15, 0.05 * remaining)
                # with the clock already at or under keep there is no room at all, and the only
                # answer that is not a loss is MOVE_NOW, a deadline of now (0.0 means no deadline)
                if s * 1.6 > max(0.0, remaining - keep) + 1e-9 and s > MOVE_NOW + 1e-12:
                    bad.append(f"{base}+{inc} with {remaining}s at ply {ply}: {s:.3f}s, extended "
                               f"{1.6*s:.3f}s, leaves less than {keep:.2f}s")
                if s < 0:
                    bad.append(f"{base}+{inc} with {remaining}s: negative allocation")
                if remaining > 2 and s > remaining * MAX_SHARE + 1e-9:
                    bad.append(f"{base}+{inc} with {remaining}s: {s:.2f}s is over the share cap")
    assert not bad, "; ".join(bad[:4])
    print("  clock: no allocation, extended as far as the search may take it, leaves the clock "
          f"empty; the share cap is {100*MAX_SHARE:.0f}% of what is left", flush=True)


def unit_one_minute_sudden_death_searches_on_its_own_price():
    """With tm_short_sudden (v6 only) a no-increment game that started at a minute or less is planned at
    the tree's network price instead of NET_SUDDEN, so 1+0 searches instead of playing every move on one
    visit. Nothing else moves. 2+0, an increment, a game first seen after its start (a 3+0 whose clocks
    have fallen under a minute) and a version sealed with 0 are all planned exactly as before."""
    import types
    import chess
    import uci

    def allowance(short, start, clocks, ply, inc=0.0, first_ply=2):
        eng = uci.Engine()
        eng.tree = types.SimpleNamespace(tm_extend=True, tm_extend_max=0.6, tm_parity=1.0, tm_deep=0.5,
                                         tm_open_spend=0.4, tm_open_plies=16, prior_fallback=True,
                                         tm_short_sudden=short)
        secs = None
        for p, c in ([(first_ply, start)] if start is not None else []) + [(ply, clocks)]:
            eng.board = chess.Board()
            eng.board.fullmove_number = p // 2 + 1
            eng.board.turn = chess.WHITE if p % 2 == 0 else chess.BLACK
            ms, mi = int(c * 1000), int(inc * 1000)
            secs, _, _ = eng.budget({"wtime": ms, "btime": ms, "winc": mi, "binc": mi})
        return secs

    flat, short = allowance(0.0, 60, 55, 20), allowance(0.8, 60, 55, 20)
    assert flat < uci.INSTANT <= short, f"1+0 at ply 20 with 55 s: {flat:.3f}s flat, {short:.3f}s at 0.8"
    for args in ((120, 100, 20), (60, 55, 20, 1.0), (None, 50, 60)):
        a, b = allowance(0.0, *args), allowance(0.8, *args)
        assert a == b, f"{args}: {a} at 0, {b} at 0.8"
    print(f"  clock: 1+0 at ply 20 gets {short:.2f}s at tm_short_sudden 0.8 against {flat:.2f}s flat; "
          "2+0, 1+1 and a game first seen late are unchanged", flush=True)


def unit_a_close_decision_buys_more_time():
    """Think longer when it matters, meaning the top two moves are too close to separate. Settled means
    ahead on both value and visits, because either alone is the weaker signal, and a settled root
    must not extend."""
    import numpy as np
    from mcts import MCTS
    t = MCTS(_det_stub(), "cpu").apply({"tm_margin": 0.05})
    close = _hand_root([100, 80, 5], [20.0, 15.0, 1.0], [5.0, 4.0, 0.3])    # 0.200 against 0.188
    clear = _hand_root([300, 40, 5], [150.0, 4.0, 1.0], [80.0, 1.0, 0.3])   # 0.500 against 0.100
    thin = _hand_root([120, 100, 5], [60.0, 40.0, 1.0], [32.0, 18.0, 0.3])  # clear value, thin visits
    assert not t._settled(close, 0.05, 1.2), "a 0.012 gap counted as settled"
    assert t._settled(clear, 0.05, 1.2), "a 0.4 gap with 7x the visits did not count as settled"
    assert not t._settled(thin, 0.05, 2.0), "value alone settled it at 1.2x the visits"
    print("  time management: close decisions are unsettled and get the reserve, a clear leader "
          "is settled and hands the time back", flush=True)


def unit_the_book_plays_master_moves_and_nothing_else():
    """The book answers before the tree, only inside its depth, only with a legal move, and only when it
    is switched on. It is sealed off in every released version, since a book is TWIC's strength and
    not the network's, so a ladder row that used one would measure the wrong thing. A book entry for a
    position that cannot occur must never reach the board."""
    import json, tempfile, chess, chess.polyglot
    import book
    from mcts import MCTS

    b = chess.Board()
    after_e4 = chess.Board()
    after_e4.push_san("e4")
    doc = {"plies": 4, "positions": {
        str(chess.polyglot.zobrist_hash(b)): {"e2e4": [900, 0.55], "d2d4": [700, 0.54]},
        str(chess.polyglot.zobrist_hash(after_e4)): {"c7c5": [500, 0.51]}}}
    path = os.path.join(tempfile.mkdtemp(), "book.json")
    json.dump(doc, open(path, "w", encoding="utf-8"))
    book.BOOK_FILE, book._BOOK = path, None

    # sampled, not argmax, so two games are not the same game
    import random
    rng = random.Random(7)
    picks = {book.pick(b.copy(), rng, 4).uci() for _ in range(40)}
    assert picks <= {"e2e4", "d2d4"}, f"the book invented {picks}"
    assert len(picks) == 2, f"the book always played the same move: {picks}"
    assert book.pick(after_e4.copy(), rng, 4).uci() == "c7c5", "the reply was not found"

    # off by default, and off is off
    off = MCTS(_det_stub(), "cpu", batch=8).apply({"book": False, "root_mate": False,
                                                   "tb_dtz": False})
    mv, info = off.search(b.copy(), visits=32, reuse=False)
    assert info["visits"] > 0 and info.get("proof") != "opening book", "the book played with book off"

    on = MCTS(_det_stub(), "cpu", batch=8).apply({"book": True, "book_plies": 4,
                                                  "root_mate": False, "tb_dtz": False})
    mv, info = on.search(b.copy(), visits=32, reuse=False)
    assert info.get("proof") == "opening book" and info["visits"] == 0,         f"the book did not answer: {info.get('proof')}, {info['visits']} visits"
    assert mv in b.legal_moves, "the book played an illegal move"

    # past its depth it is the tree's job again
    deep = chess.Board()
    for u in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"):
        deep.push(chess.Move.from_uci(u))
    mv2, info2 = on.search(deep, visits=32, reuse=False)
    assert info2["visits"] > 0, "the book answered past book_plies"
    print("  book: sampled from master practice inside its depth, silent past it, silent when "
          "off, and every released version is sealed without it", flush=True)


def unit_a_collision_does_not_walk_the_same_path_again():
    """With `collide` off, a walk that reaches a leaf another walk is waiting on unwinds and the next
    walk takes the identical path, because `_pick` is argmax with nothing random in it. With it on the
    virtual loss stays, so the next walk goes elsewhere and the batch fills, and the walks that
    collided are backed up with the leaf's real value when it arrives. It is a search change, so it is
    sealed off through v4.

    It is not a free speedup. A collided walk counts as a visit, so the same budget buys fewer network
    evaluations. Under a clock that is the right trade, at a fixed visit count it may not be.
    """
    import chess
    from mcts import MCTS

    # A narrow tree and a full batch, because that is where collisions happen. At batch 8 this
    # position produces none and the test would pass without testing.
    b = chess.Board("8/8/4k3/8/4K3/8/4P3/8 w - - 0 1")
    seen = {}
    for on in (False, True):
        t = MCTS(_det_stub(), "cpu", batch=48, cache_size=0).apply(
            {"collide": on, "root_mate": False, "tb_dtz": False})
        t.use_tb = False                      # code, not a sealed setting, and this is an ending
        mv, info = t.search(b.copy(), visits=200, reuse=False)
        assert info["visits"] == 200, f"collide={on} spent {info['visits']} of 200"
        assert mv in b.legal_moves, f"collide={on} played an illegal move"
        seen[on] = (t.stats["collisions"], t.stats["evals"])
        # nothing may be left waiting once the search returns
        assert t._collided is None, "a collided path outlived the search"
    assert seen[False][0] > 50, f"the position produced {seen[False][0]} collisions, so this "                                f"test is not testing anything"
    assert seen[True][0] < seen[False][0],         f"keeping the virtual loss did not reduce collisions: {seen[True][0]} against {seen[False][0]}"
    print(f"  collisions: keeping the virtual loss took {seen[True][0]} collisions and "
          f"{seen[True][1]} evaluations against {seen[False][0]} and {seen[False][1]}, "
          f"both spending their full budget", flush=True)


def unit_a_held_collision_is_given_back():
    """collide 2 holds a collided walk's virtual visit while the batch gathers, so the next walk goes
    elsewhere, and gives it back when the batch returns. It is how lc0 treats a collision, and it
    exists for virtual_mean, where a pending visit barely discourages its path, so the first
    collision comes early and every walk after it repeats it until the try cap.

    Three things must hold. Every visit asked for is a real visit and so an evaluation, not a
    duplicate (collide 1 counted collided walks as visits and lost 69 Elo doing it). Nothing is left
    in flight anywhere in the tree. And the batches fill, or the mode does nothing."""
    import chess
    from mcts import MCTS

    b = chess.Board("8/8/4k3/8/4K3/8/4P3/8 w - - 0 1")
    seen = {}
    for mode in (0, 2):
        t = MCTS(_det_stub(), "cpu", batch=48, cache_size=0).apply(
            {"collide": mode, "virtual_mean": True, "root_mate": False, "tb_dtz": False})
        t.use_tb = False                      # code, not a sealed setting, and this is an ending
        mv, info = t.search(b.copy(), visits=200, reuse=False)
        assert info["visits"] == 200, f"collide={mode} spent {info['visits']} of 200"
        assert mv in b.legal_moves, f"collide={mode} played an illegal move"
        assert t._collided is None, "a held path outlived the search"
        f_left, x_left, stack = 0.0, 0.0, [t.root]
        while stack:
            n = stack.pop()
            if n.F is not None:
                f_left += float(abs(n.F).sum())
                x_left = max(x_left, float(abs(n.X).max()))
                stack.extend(k for k in n.kids if k is not None)
        # counts exactly zero, while the pending value is float32 and rounds to about 1e-6
        assert f_left == 0 and x_left < 1e-4,             f"collide={mode} left {f_left:.0f} visits and {x_left:.2g} of value in flight"
        seen[mode] = (t.stats["batches"], t.stats["evals"], t.stats["collisions"])
    assert seen[0][2] > 50, f"the position produced {seen[0][2]} collisions, so this test is "                             f"not testing anything"
    assert seen[2][0] < seen[0][0],         f"holding did not fill the batches: {seen[2][0]} passes against {seen[0][0]}"
    # the budget buys evaluations, not duplicates, within a few of what the storm run got
    assert seen[2][1] >= 0.9 * seen[0][1],         f"held collisions ate the budget: {seen[2][1]} evaluations against {seen[0][1]}"
    print(f"  held collisions: {seen[2][0]} passes and {seen[2][1]} evaluations, against "
          f"{seen[0][0]} and {seen[0][1]} when a collision ends the gather", flush=True)


def unit_a_node_ceiling_holds_the_tree():
    """max_nodes is the memory ceiling pondering runs under (uci.py, 12 GB by default) in place of a
    visit cap. Three things must hold. The bound it reads, tree_nodes(), is never below the nodes a
    walk of the tree counts, so the ceiling can only err low. A search asked for far more visits than
    the ceiling allows stops at it and says so. And a carried tree, whose root the ponder discount
    has scaled down to a quarter, is held to the ceiling too, which the root's own count would not do."""
    import chess
    from mcts import MCTS

    def walk(root):
        seen, stack = set(), [root]
        while stack:
            x = stack.pop()
            if id(x) in seen:
                continue
            seen.add(id(x))
            if x.kids:
                stack.extend(k for k in x.kids if k is not None)
        return len(seen)

    b = chess.Board("r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N1P/PP1P1PP1/RNBQR1K1 b - - 0 9")
    t = MCTS(_det_stub(), "cpu", batch=16, cache_size=0).apply({"ponder_discount": 0.25})
    t.book = False
    t.use_tb = False
    _, info = t.search(b.copy(), visits=100_000, reuse=False, max_nodes=600)
    n = walk(t.root)
    assert info["at_ceiling"], "100,000 visits under a 600 node ceiling did not stop at the ceiling"
    assert n <= 600, f"the ceiling was 600 nodes and the tree holds {n}"
    assert t.tree_nodes() >= n, f"the bound {t.tree_nodes()} is below the {n} nodes walked"
    assert n >= 500, f"the ceiling stopped the tree at {n} of 600 nodes, far too early"
    # the same position again, with the tree carried, its root discounted, and the ceiling raised
    before = t.stats["discounted"]
    _, info2 = t.search(b.copy(), visits=100_000, reuse=True, max_nodes=900)
    n2 = walk(t.root)
    assert t.stats["discounted"] == before + 1 and info2["carried"] >= 500,         "the second search did not carry a discounted tree, so this test is not testing anything"
    assert n2 <= 900, f"a carried, discounted tree grew to {n2} nodes under a 900 node ceiling"
    assert t.tree_nodes() >= n2, f"the bound {t.tree_nodes()} is below the {n2} nodes walked"
    print(f"  node ceiling: {n} nodes under 600, then {n2} under 900 on a carried tree whose root "
          f"reads {int(t.root.sum_n)} visits", flush=True)


def unit_a_cached_pass_is_not_a_saturated_tree():
    """The collection loop stops when 192 descents in a row produce no leaf. That guard is for a tree
    it cannot descend, where every walk collides and unwinds, and must not stop a search whose leaves
    the eval cache has already answered (a position searched twice in one process, as after a
    take-back on the board). It changes which move comes back, so it is sealed (cache_progress), with
    v1 to v3 keeping the old reading and v4 getting the new one.

    The terminal case must still stop. Once the root is proven, every walk ends at a solved node and
    produces no leaf either. The two are told apart by whether the cache answered anything.
    """
    import chess
    from mcts import MCTS

    b = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7")
    for fix, want in ((False, 192), (400, 400)):
        t = MCTS(_det_stub(), "cpu", batch=48, cache_size=100000).apply(
            {"cache_progress": bool(fix), "root_mate": False, "tb_dtz": False, "proofs": False})
        first = t.search(b.copy(), visits=400, reuse=False)[1]["visits"]
        assert first == 400, f"the first search spent {first} of 400"
        again = 400
        for _ in range(4):                       # the cache fills over a repeat or two
            again = t.search(b.copy(), visits=400, reuse=False)[1]["visits"]
        if fix:
            assert again == 400, f"a cached pass ended the search at {again} of 400"
        else:
            assert again <= 192, f"the old reading spent {again}, so this no longer tests it"
    print("  cached pass: the search runs its full budget when the cache answers it, and v1 to "
          "v3 keep the early stop they were released with", flush=True)


def unit_its_own_openings_vary_without_playing_a_worse_move():
    """open_temp samples the opening from the tree's own visit counts, so two games from the start
    position are not the same game, and it still cannot be talked into a move the search rates worse.

    The exponent is 1/temp, so a temperature below one sharpens (0.3 or 0.6 gives near identical
    games). 1.0 is visit-proportional and anything above it flattens. A bare visit floor alone can let
    a barely visited move with a lucky average through, and the value margin is what stops that.
    """
    import types
    import numpy as np
    from mcts import MCTS

    t = MCTS(None, "cpu")
    t.open_temp, t.open_floor, t.open_margin = 2.0, 0.02, 0.05
    t.rng = np.random.default_rng(11)
    # leader, a real alternative, a tail move on noise, and a move it genuinely dislikes
    names = ["leader", "second", "tail", "worse"]
    root = types.SimpleNamespace(
        moves=names,
        N=np.array([600.0, 200.0, 3.0, 150.0]),
        W=np.array([0.0, -4.0, 0.6, -30.0]))   # q = 0.000, -0.020, +0.200, -0.200
    drawn = {names[t._sample_opening(root, 0)] for _ in range(200)}
    assert drawn == {"leader", "second"}, f"sampled {drawn}"

    # the cold limit is the move it would have played anyway, and it gets there without
    # overflowing, since N^(1/temp) at a small temperature is inf and inf/inf is NaN
    t.open_temp = 1e-3
    assert all(t._sample_opening(root, 0) == 0 for _ in range(5)), "a cold draw left the leader"

    # the move it was going to play is always in its own draw, even when the floor excludes it,
    # since with lcb on `best` is not argmax N
    t.open_temp = 2.0
    thin = types.SimpleNamespace(moves=names, N=np.array([600.0, 200.0, 3.0, 150.0]),
                                 W=np.array([0.0, -4.0, 0.6, -30.0]))
    picks = {thin.moves[t._sample_opening(thin, 2)] for _ in range(60)}
    assert "tail" in picks, f"the chosen move was dropped from its own draw: {picks}"
    print("  own openings: drawn from its own visit counts, never the tail it glanced at nor a "
          "move it rates worse, and a cold draw is the move it would have played", flush=True)


def unit_the_root_check_spends_the_same_budget():
    """root_check holds back a share of the visits. Triggered or not, the move is made with the budget
    it was given, and a forced line is released before the next search."""
    import chess
    from mcts import MCTS
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    for margin in (-9.0, 9.0):                   # always armed if unvisited, and never
        t = MCTS(_det_stub(), "cpu", batch=8, cache_size=0).apply(
            {"root_check": True, "root_check_margin": margin, "q_init": True,
             "root_mate": False, "tb_dtz": False})
        mv, info = t.search(b.copy(), visits=300, reuse=False)
        assert mv is not None and info["visits"] == 300, f"spent {info['visits']} of 300"
        assert t._force is None, "the forced line outlived the search"
    print("  root check: the same 300 visits armed or not, nothing forced afterwards", flush=True)


if __name__ == "__main__":
    print("unit checks")
    unit_lcb_plays_the_move_it_is_surest_of()
    unit_a_virtual_mean_leaves_q_alone()
    unit_a_proof_climbs_the_tree()
    unit_the_tree_draws_at_one_hundred_plies()
    unit_a_terminal_root_still_gets_a_move()
    unit_q_init_scores_a_move_by_its_own_value()
    unit_the_fast_pick_is_the_pick()
    unit_the_tablebase_respects_the_fifty_move_counter()
    unit_the_tree_reads_the_head_it_is_sealed_with()
    unit_the_book_plays_master_moves_and_nothing_else()
    unit_a_collision_does_not_walk_the_same_path_again()
    unit_a_held_collision_is_given_back()
    unit_a_cached_pass_is_not_a_saturated_tree()
    unit_a_node_ceiling_holds_the_tree()
    unit_its_own_openings_vary_without_playing_a_worse_move()
    unit_the_clock_cannot_be_flagged_by_one_move()
    unit_one_minute_sudden_death_searches_on_its_own_price()
    unit_a_close_decision_buys_more_time()
    unit_the_root_check_spends_the_same_budget()
    unit_the_visit_counters_balance()
    unit_the_batch_is_filled()
    unit_a_stale_source_is_reported()
    unit_the_power_mean_is_a_mean()
    unit_the_legality_mask_never_hides_the_target()
    unit_the_rate_halves_on_the_version_boundaries()
    unit_the_tree_policies_agree_with_what_the_page_shows()
    unit_the_variance_estimate_is_a_variance()
    unit_gumbel_concentrates_the_budget()
    unit_each_version_keeps_its_own_search()
    unit_every_sealed_version_pins_every_setting()
    unit_a_version_is_sealed_the_moment_it_freezes()
    unit_a_mate_in_one_is_played_without_search()
    unit_pondered_visits_are_discounted()
    unit_the_tablebase_root_plays_for_progress()
    unit_the_graph_merges_transpositions()
    unit_early_stop_never_changes_the_move()
    unit_the_leaf_search_sees_a_hanging_piece()
    unit_the_calibration_stretches_the_middle()
    unit_importing_the_board_does_not_read_our_arguments()
    unit_a_job_can_be_restarted_from_its_marker()
    unit_batched_leaf_values_match_the_loop()
    unit_the_tablebases_are_used()
    unit_a_dead_loader_does_not_hang_the_run()
    unit_the_process_list_is_not_truncated()
    unit_a_stop_does_not_kill_the_next_search()
    unit_move_space()
    unit_clock_and_ep()
    unit_the_batch_encoder_equals_board_to_ids()
    unit_adapter_is_identity_at_birth()
    unit_adapter_only_touches_their_nodes()

    print("")
    print(f"starting a real server on {PORT}", flush=True)
    env = dict(os.environ)
    env.update({"BIND": "127.0.0.1", "OPEN": "0", "PONDER": "0",
                "VISITS": "64" if not FAST else "0", "PASS": "",
                # so this throwaway server does not overwrite the live board's pid file
                "PID_SUFFIX": "_selftest",
                # one minute, so the idle check below is testable in a suite rather than an hour
                "IDLE_MINUTES": "1",
                # the test plays games, and without these it would append its positions to the
                # live labelling queue
                "GAMES": os.path.join(HERE, "selftest_games"),
                "LABEL_QUEUE": os.path.join(HERE, "selftest_games", "positions.txt")})
    # Reap any stray from a previous run first. The teardown below is a `finally`, which does not run
    # when the suite's own process is killed, so an interrupted run can leave a board holding this port
    # and a copy of the model on the GPU, and the next run would fail to bind.
    reap_strays(PORT)
    log = open(os.path.join(HERE, "selftest_server.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "play_web.py"), str(PORT)],
                            cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        if not server_up(proc):
            print("  FAIL  the server never answered")
            FAILED.append("server starts")
        else:
            check("the server answers over http", True)
            print("")
            print("one player")
            http_checks()
            print("")
            print("what the tree says")
            search_report_check()
            print("")
            print("the clock")
            clock_checks()
            print("")
            print("keeping your games")
            game_log_checks()
            print("")
            print("giving the card back when nobody is playing")
            idle_checks()
            print("")
            print("tactics, which is really a check that the search runs")
            tactics_checks()
            print("")
            print("five players at once")
            concurrency_check()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()

    log.close()
    # The server's own output, always, not only on a failure to start. A handler that raises
    # prints there and nowhere else, and the client only sees a missing key.
    said = open(os.path.join(HERE, "selftest_server.log"), encoding="utf-8").read()
    complaints = [l for l in said.splitlines() if "failed:" in l or "Traceback" in l]
    if complaints:
        print("")
        print("what the server said:")
        for l in complaints[:12]:
            print("   " + l.strip())

    print("")
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  failed: {f}")
    raise SystemExit(1 if FAILED else 0)
