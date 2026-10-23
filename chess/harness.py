"""Prove that a change to the search leaves every sealed version's moves bit for bit the same.

Every released version's search is frozen, so a speedup may reach it only if it changes no move, no
visit count and no backed-up value. This checks that mechanically.

    python chess/harness.py record OUT [--real]            golden records from this checkout's code
    python chess/harness.py prove OLD_DIR NEW_DIR [--real]  OLD twice (the control, it must repeat
                                                           itself) and NEW once, each in a fresh process,
                                                           then prints IDENTICAL or the first difference
    python chess/harness.py prove OLD NEW --sealed-only    the same for v1 to v5 only, for a change that
                                                           moves v6's defaults on purpose
    python chess/harness.py invariants                     tree invariants after searches over a grid of
                                                           settings (stub network)
    python chess/harness.py seal                           mcts.py's scrubbed defaults against
                                                           golden/search_v6.json (search_v5.json is v5's)
    python chess/harness.py table                          write golden/search_v6.json from the defaults

HARNESS_CHESS_DIR points record, invariants and seal at another copy of the code, and prove sets it
for each side itself.

A record holds the move, the principal variation, the evaluation count, a hash of the root's float32
N, W and V2, and a hash over N, W, V2 and P of every expanded node. Nothing is rounded, since float32
sums repeat exactly when the operations repeat in the same order. The root's N and Q are included so a
person can read a failure.

The stub network (HashStub) is integer arithmetic until its last step, so its outputs do not depend on
the batch, thread count or machine, and it runs on a CPU with the repo's own versions.json and no
tablebases. --real uses v4's weights on the GPU (from VERSION_DIR) and records the environment. A real
record set is only comparable with one made on the same machine, driver and libraries, which is why
prove records OLD and NEW side by side.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = "--real" in sys.argv
# with --sealed-only, prove compares v1 to v5 alone, for a change that moves v6's defaults on purpose
SEALED_ONLY = "--sealed-only" in sys.argv
CODE = os.environ.get("HARNESS_CHESS_DIR") or HERE     # the code under test
sys.path.insert(0, CODE)
if not REAL:
    # no tablebases, and the repo's own manifest, read only
    os.environ.setdefault("SYZYGY_DIR", os.path.join(tempfile.gettempdir(), "mayfly_harness_no_tables"))
    os.environ.setdefault("VERSION_DIR", HERE)

import chess                    # python-chess (there is no chess.py in this folder)
import torch

POSITIONS = [
    ("start", chess.STARTING_FEN),
    ("italian", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("sicilian", "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 2 6"),
    ("ruy_mid", "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N1P/PP1P1PP1/RNBQR1K1 b - - 0 9"),
    ("hanging_q", "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"),
    ("backrank_m1", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
    ("kpk", "8/8/4k3/8/4K3/8/4P3/8 w - - 0 1"),
    ("rook_r50", "8/5k2/8/3R4/8/2K5/8/6r1 w - - 80 120"),
    ("promo", "8/P6k/8/8/8/8/8/K7 w - - 0 1"),
    ("ep", "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
    ("castles", "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 8"),
    ("qgd", "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4"),
    # a middlegame with captures and checks in reply, so forcing and forcing_apart do something
    ("tactic", "2rqrbk1/3b1pp1/3P2np/1p2p2Q/4P3/1B3N1P/PP3PP1/2RR2K1 b - - 0 26"),
]


def board_with_history():
    """A game with a move stack, so the repetition history is walked (Nf3 Nf6 Ng1 Ng8 Nf3)."""
    b = chess.Board()
    for u in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3"):
        b.push_uci(u)
    return b


class HashStub(torch.nn.Module):
    """A fake network whose every output is a function of the position alone, computed in integers.

    An exact int64 sum over squares, an int64 projection, a prime modulus and a power-of-two divide.
    The clock and the en passant square enter the hash, so cache_clock has something to key on."""

    def __init__(self, seed=20260929):
        super().__init__()
        g = np.random.default_rng(seed)
        self.register_buffer("R", torch.from_numpy(g.integers(-2**15, 2**15, (64, 13, 4), dtype=np.int64)))
        self.register_buffer("Wp", torch.from_numpy(g.integers(-2**15, 2**15, (4, 4162), dtype=np.int64)))
        self.register_buffer("Wq", torch.from_numpy(g.integers(-2**15, 2**15, (4, 4162), dtype=np.int64)))
        self.anchor = torch.nn.Parameter(torch.zeros(1))     # so next(model.parameters()) works

    def forward(self, ids, cast, hm, ep):
        n = ids.shape[0]
        sq = torch.arange(64).unsqueeze(0).expand(n, 64)
        f = self.R[sq, ids.long()].sum(1)
        f = f + torch.round(cast.float() * 4).long() * torch.tensor([3, 5, 7, 11])
        f[:, 0] += hm.long() * 7919
        f[:, 1] += ep.long() * 104729
        lp = torch.remainder(f @ self.Wp, 8191) - 4095
        lq = torch.remainder(f @ self.Wq, 6143) - 3071
        return {"h": torch.zeros(n, 4), "policy": lp.float() / 1024.0,
                "value": torch.zeros(n), "qvalue": lq.float() / 1024.0}


_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        if REAL:
            from conversion_suite import load
            _MODEL = load("mayfly_v4.pt")
        else:
            torch.set_num_threads(1)
            _MODEL = HashStub()
    return _MODEL


def sealed(version):
    from mcts import pinned_search
    blk = pinned_search(f"mayfly_v{version}.pt")
    assert blk, f"no sealed block for v{version} in {os.environ.get('VERSION_DIR')}"
    return blk


def scrubbed_defaults(code=None):
    """mcts.py's defaults with every environment name it reads removed, in a fresh interpreter.

    This is what a freeze (train_av.sealed_search) seals."""
    code = code or CODE
    names = set(re.findall(r'E\("([A-Z0-9_]+)"', open(os.path.join(code, "mcts.py"), encoding="utf-8").read()))
    names.add("MLH")
    env = {k: v for k, v in os.environ.items() if k not in names or k in ("SYZYGY_DIR", "VERSION_DIR")}
    src = ("import json, sys; sys.path.insert(0, %r); from mcts import default_search; "
           "print('SEALED ' + json.dumps(default_search()))" % code)
    r = subprocess.run([sys.executable, "-c", src], cwd=tempfile.gettempdir(), env=env,
                       capture_output=True, text=True, timeout=300)
    for line in r.stdout.splitlines():
        if line.startswith("SEALED "):
            return json.loads(line[7:])
    raise RuntimeError(r.stderr[-800:])


def make_tree(settings):
    from mcts import MCTS
    t = MCTS(model(), "cuda" if REAL else "cpu", batch=48, cache_size=20000).apply(settings)
    t.open_temp = 0.0
    t.rng = np.random.default_rng(0)                       # reset() does not reseed it
    return t


def walk(root):
    """Every node reachable from root once, in edge order."""
    out, seen, stack = [], set(), [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        out.append((node, depth))
        if node.kids:
            for k in reversed(node.kids):
                if k is not None:
                    stack.append((k, depth + 1))
    return out


def tree_digest(root):
    h = hashlib.sha256()
    for node, depth in walk(root):
        h.update(depth.to_bytes(2, "little"))
        if node.N is not None:
            for arr in (node.N, node.W, node.V2, node.P):
                h.update(np.ascontiguousarray(arr, np.float32).tobytes())
        h.update(repr(node.term).encode())
    return h.hexdigest()[:16]


def record(t, board, visits, reuse=False):
    mv, info = t.search(board.copy(), visits=visits, reuse=reuse)
    r = t.root
    rec = {"move": mv.uci() if mv else None, "pv": info.get("pv", ""), "visits": info.get("visits"),
           "evals": info.get("evals"), "forced": int(t.stats.get("forced", 0))}
    if r is not None and r.N is not None and info.get("visits"):
        rec["N"] = [int(x) if float(x).is_integer() else float(x) for x in r.N]
        rec["Q"] = [round(float(w) / float(n), 6) if n > 0 else None for w, n in zip(r.W, r.N)]
        rec["root_bits"] = hashlib.sha256(r.N.tobytes() + r.W.tobytes() + r.V2.tobytes()).hexdigest()[:16]
        rec["tree"] = tree_digest(r)
    return mv, info, rec


def configs():
    """What the golden records cover, every sealed version, the v6 defaults, and v4 in pipelines 1 and 2."""
    out = [(f"v{v}", sealed(v), 0) for v in (1, 2, 3, 4, 5)]
    out.append(("v6defaults", scrubbed_defaults(), 0))
    s4 = sealed(4)
    out += [("v4", s4, 1), ("v4", s4, 2)]
    return out


def golden(visits_list=(200, 400)):
    recs = []
    for tag, s, pipe in configs():
        vl = visits_list if pipe == 0 else (400,)
        for visits in vl:
            for name, fen in POSITIONS + [("history", None)]:
                t = make_tree(dict(s, pipeline=pipe) if pipe else s)
                b = chess.Board(fen) if fen else board_with_history()
                mv, info, rec = record(t, b, visits)
                rec.update(version=tag, pipeline=pipe, visits_asked=visits, pos=name)
                recs.append(rec)
                # the second move of the pair, for tree reuse and, from v4, the ponder discount
                if mv is not None and t.root is not None and t.root.kids:
                    b.push(mv)
                    kid = t.root.kids[t.root.moves.index(mv)]
                    if kid is not None and kid.N is not None and kid.sum_n > 0:
                        b.push(kid.moves[int(np.argmax(kid.N))])
                        if not b.is_game_over():
                            _, _, rec2 = record(t, b, visits, reuse=True)
                            rec2.update(version=tag, pipeline=pipe, visits_asked=visits, pos=name + "+2")
                            recs.append(rec2)
    return recs


def environment():
    env = {"network": "v4 weights" if REAL else "HashStub(20260929)", "torch": torch.__version__,
           "numpy": np.__version__, "python": sys.version.split()[0], "chess": chess.__version__,
           "FAST_EVAL": os.environ.get("FAST_EVAL", ""), "CUDA_GRAPH": os.environ.get("CUDA_GRAPH", "")}
    if REAL and torch.cuda.is_available():
        env.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    return env


KEYS = ("move", "pv", "evals", "root_bits", "tree")


def cmd_record(out):
    t0 = time.time()
    recs = golden()
    doc = {"schema": 1, "code": CODE, "environment": environment(), "records": recs,
           "seconds": round(time.time() - t0, 1)}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, separators=(",", ":"))
    print(f"recorded {len(recs)} searches from {CODE} in {time.time() - t0:.0f} s -> {out}", flush=True)


def compare(a, b):
    keep = (lambda r: r["version"] != "v6defaults") if SEALED_ONLY else (lambda r: True)
    ka = {(r["version"], r["pipeline"], r["visits_asked"], r["pos"]): r for r in a["records"] if keep(r)}
    kb = {(r["version"], r["pipeline"], r["visits_asked"], r["pos"]): r for r in b["records"] if keep(r)}
    if set(ka) != set(kb):
        return f"the record sets differ: {sorted(set(ka) ^ set(kb))[:4]}"
    for k in sorted(ka):
        if any(ka[k].get(f) != kb[k].get(f) for f in KEYS):
            x, y = ka[k], kb[k]
            return (f"FIRST DIFFERENCE {k}: move {x.get('move')} / {y.get('move')}, evals {x.get('evals')} / "
                    f"{y.get('evals')}, tree {x.get('tree')} / {y.get('tree')}\n  N {x.get('N')}\n  N {y.get('N')}")
    return None


def cmd_prove(old, new):
    tmp = tempfile.mkdtemp(prefix="mayfly_harness_")
    runs = [("old", old, "a"), ("old", old, "b"), ("new", new, "a")]
    docs = {}
    for label, code, rep in runs:
        out = os.path.join(tmp, f"{label}_{rep}.json")
        env = dict(os.environ, HARNESS_CHESS_DIR=os.path.abspath(code), PYTHONHASHSEED={"a": "1", "b": "2"}[rep])
        args = [sys.executable, os.path.abspath(__file__), "record", out] + (["--real"] if REAL else [])
        r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0 or not os.path.exists(out):
            print(f"{label} {rep} failed:\n{r.stderr[-1500:]}")
            return 1
        print(r.stdout.strip().splitlines()[-1], flush=True)
        docs[(label, rep)] = json.load(open(out, encoding="utf-8"))
    ctl = compare(docs[("old", "a")], docs[("old", "b")])
    if ctl:
        print("CONTROL FAILED: the old code does not repeat itself, so nothing can be proven here\n" + ctl)
        return 1
    d = compare(docs[("old", "a")], docs[("new", "a")])
    n = sum(1 for r in docs[("old", "a")]["records"] if not SEALED_ONLY or r["version"] != "v6defaults")
    if d:
        print(f"NOT IDENTICAL ({n} records)\n{d}")
        return 1
    print(f"IDENTICAL{' (sealed versions only)' if SEALED_ONLY else ''}: {n} records, old repeated itself and new matched it bit for bit "
          f"({'v4 weights on ' + str(docs[('new', 'a')]['environment'].get('gpu')) if REAL else 'stub network'})")
    return 0


def invariants(t, info, discounted=False):
    """Tree invariants after a search has returned, as a list of violations."""
    bad = []
    root = t.root
    if getattr(t, "_collided", None) is not None:
        bad.append("a collided path outlived the search")
    if getattr(t, "_force", None) is not None:
        bad.append("a forced line outlived the search")
    if root is None or root.N is None:
        return bad
    mcgs, proofs, coll1 = int(t.mcgs), bool(t.proofs), int(t.collide) == 1
    apart = bool(getattr(t, "forcing_apart", False))
    kept_apart = float(t.stats.get("forced_apart", 0)) if apart else 0.0
    for node, depth in walk(root):
        if node.pending:
            bad.append(f"pending node at depth {depth}")
        if node.N is None:
            continue
        k = len(node.moves)
        if not (len(node.N) == len(node.W) == len(node.V2) == len(node.P) == len(node.kids) == k):
            bad.append(f"array lengths disagree at depth {depth}")
            continue
        if float(np.abs(node.F).sum()) != 0.0:
            bad.append(f"F={float(node.F.sum()):.0f} in flight at depth {depth}")
        # X is a float32 sum added and taken back once per descent, so its residue is rounding and the
        # bound scales with the descents

        xtol = 1e-4 + 1e-6 * (float(t.stats.get("collisions", 0)) + 2.0 * float(root.sum_n))
        if float(np.abs(node.X).max(initial=0.0)) > xtol:
            bad.append(f"X={float(np.abs(node.X).max()):.2g} left (bound {xtol:.1g}) at depth {depth}")
        for nm in ("N", "W", "V2", "P"):
            if not np.isfinite(getattr(node, nm)).all():
                bad.append(f"non-finite {nm} at depth {depth}")
        if float(node.N.min()) < 0:
            bad.append(f"negative N at depth {depth}")
        if abs(float(node.P.sum()) - 1.0) > 1e-4 or float(node.P.min()) < 0:
            bad.append(f"P sums to {float(node.P.sum()):.6f} at depth {depth}")
        tol = 1e-3 + 1e-5 * float(node.N.max(initial=0))
        if (np.abs(node.W) > node.N + tol).any():
            bad.append(f"|W| > N at depth {depth}")
        if (node.V2 > node.N + tol).any():
            bad.append(f"V2 > N at depth {depth}")
        vis = node.N > 0
        if (node.V2[vis].astype(np.float64) < (node.W[vis].astype(np.float64) ** 2) / node.N[vis] - tol).any():
            bad.append(f"V2 < W^2/N at depth {depth}")
        s = float(node.N.astype(np.float64).sum())
        if node is root:
            # forcing_apart keeps forced visits in the root's sum_n (the budget) and out of its N
            lo, hi = (-0.5, 0.5) if discounted else (-1e-6, kept_apart + 1e-6)
            if not (lo <= float(node.sum_n) - s <= hi):
                bad.append(f"root sum_n {float(node.sum_n):.1f} against sum N {s:.1f} (kept apart {kept_apart:.0f})")
        elif abs(s - float(node.sum_n)) > 1e-6:
            bad.append(f"sum N {s:.1f} != sum_n {float(node.sum_n):.1f} at depth {depth}")
        for a, kid in enumerate(node.kids):
            if kid is None:
                if node.N[a] > 0 and not mcgs:
                    bad.append(f"edge with N {float(node.N[a]):.0f} and no child at depth {depth}")
                continue
            if kid.N is None or mcgs or (node is root and (discounted or apart)):
                continue
            gap = float(node.N[a]) - float(kid.sum_n)
            if gap < 1.0 - 1e-6:
                bad.append(f"edge N {float(node.N[a]):.0f} < child sum_n + 1 at depth {depth}")
            elif gap > 1.0 + 1e-6 and not coll1 and not proofs and kid.term is None:
                bad.append(f"edge N {float(node.N[a]):.0f} > child sum_n + 1 at depth {depth}")
    mv = info.get("move")
    if mv is not None and mv not in root.board.legal_moves:
        bad.append("illegal move")
    return bad


def cmd_invariants():
    v5 = scrubbed_defaults()
    grid = [("v4", sealed(4)), ("v5", v5), ("v4 pipeline 1", dict(sealed(4), pipeline=1)),
            ("v4 pipeline 2", dict(sealed(4), pipeline=2)), ("v5 collide 1", dict(v5, collide=1)),
            ("v5 proofs", dict(v5, proofs=True)), ("v5 pipeline 1", dict(v5, pipeline=1))]
    from mcts import SEARCH_KEYS
    if "forcing_apart" in SEARCH_KEYS:
        grid += [("v5 forcing_apart", dict(v5, forcing_apart=True)),
                 ("v5 forcing_apart pipeline 1", dict(v5, forcing_apart=True, pipeline=1))]
    total = 0
    for label, s in grid:
        for name, fen in POSITIONS:
            for visits in (240, 800):
                t = make_tree(s)
                _, info = t.search(chess.Board(fen), visits=visits, reuse=False)
                bad = invariants(t, info)
                if bad:
                    print(f"  {label} {name} {visits}: {bad[:3]}")
                    total += 1
    print(f"invariants: {len(grid)} settings x {len(POSITIONS)} positions x 2 budgets, "
          f"{total} searches with a violation")
    return 1 if total else 0


def cmd_seal(write=False):
    path = os.path.join(HERE, "golden", "search_v6.json")
    d = scrubbed_defaults()
    if write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"wrote {len(d)} settings to {path}")
        return 0
    want = json.load(open(path, encoding="utf-8"))
    diff = {k: (want.get(k), d.get(k)) for k in set(want) | set(d) if want.get(k) != d.get(k)}
    if diff:
        print(f"SEAL DRIFT, {len(diff)} of {len(want)} settings (table, defaults): {diff}")
        return 1
    print(f"seal: {CODE}'s scrubbed defaults equal golden/search_v6.json on all {len(d)} settings")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a not in ("--real", "--sealed-only")]
    if not args:
        print(__doc__)
        sys.exit(0)
    cmd = args[0]
    if cmd == "record":
        cmd_record(args[1])
    elif cmd == "prove":
        sys.exit(cmd_prove(args[1], args[2]))
    elif cmd == "invariants":
        sys.exit(cmd_invariants())
    elif cmd == "seal":
        sys.exit(cmd_seal())
    elif cmd == "table":
        sys.exit(cmd_seal(write=True))
    else:
        print(__doc__)
        sys.exit(2)
