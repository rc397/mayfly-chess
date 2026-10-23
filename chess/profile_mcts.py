r"""Where the time in a search goes.

Runs cProfile over real searches on real positions and rolls the per-function numbers up into
buckets.

  selection   walking down the tree, _select and _descend
  expansion   making a node, with legal move generation, the board copy and attaching priors
  network     the forward pass, which synchronises on .cpu() so GPU wait lands here
  backup      pushing the value back up and undoing virtual loss
  adapter     the opponent model's delta, if a game is running
  tablebase   syzygy probes

cProfile adds overhead to every call, so the absolute numbers are inflated and the ratios are what
matter. A wall clock run with the profiler off is printed alongside for scale.

  python chess/profile_mcts.py [POSITIONS] [VISITS]
  env: VERSION=mayfly_v1.pt  ADAPTER=0  PINNED=1 (the version's sealed search)  TOP=12
       PGN=<folder of .pgn files to sample positions from, default <data>/twic>
"""
import os, sys, glob, random, cProfile, pstats, io as _io, time, collections
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths

E = os.environ.get
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
VISITS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PGN_DIR = E("PGN", paths.under("twic"))

# Function name to bucket. Anything unlisted lands in "other", which is printed too, since a big
# "other" means this map is out of date.
BUCKETS = [
    # _pick and _terms are where selection happens
    ("selection", ("_select", "_pick", "_terms", "_descend")),
    ("expansion", ("_expand", "__init__", "generate_legal_moves", "generate_pseudo_legal_moves",
                   "_is_safe", "is_check", "copy", "_transposition_key", "attacks_mask",
                   "is_insufficient_material", "_attackers_mask", "generate_castling_moves",
                   "_slider_blockers", "is_checkmate", "is_seventyfive_moves", "checkers_mask")),
    ("network", ("_evaluate", "board_to_ids", "clock_and_ep", "move_to_idx", "forward", "_call_impl")),
    ("backup", ("_backup", "_unwind")),
    ("adapter", ("delta",)),
    ("tablebase", ("_tb_value", "probe_wdl")),
]


def bucket_of(name, where=""):
    for label, names in BUCKETS:
        if name in names:
            return label
    return "other"


# By where the code lives, since the name map above misses many functions. The split is the tree
# code, python-chess, torch (launching kernels and waiting on the GPU), numpy and batch encoding.
def origin_of(name, where):
    w = where.replace("\\", "/")
    if w.endswith("mcts.py"):
        return {"_evaluate": "torch: launch and wait"}.get(name, "tree: " + bucket_of(name))
    if "/chess/" in w and w.endswith("__init__.py") or name in ("scan_reversed", "scan_forward"):
        return "python-chess"
    if "torch" in w or "torch" in name or "TensorBase" in name or w.endswith(("transformer.py", "module.py", "functional.py")):
        return "torch: launch and wait"
    if "numpy" in w or "numpy" in name or w.endswith(("numeric.py", "fromnumeric.py")) or "ndarray" in name or "ufunc" in name:
        return "numpy (tree arithmetic)"
    if w.endswith(("moves.py", "chess_net.py", "encode.py")):
        return "encoding the batch"
    return "other"


def sample(n, seed=3):
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
    rng.shuffle(files)
    out = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while len(out) < n:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                mv = list(g.mainline_moves())
                if len(mv) < 30:
                    continue
                b = g.board()
                for m in mv[:rng.randrange(14, 40)]:
                    b.push(m)
                if not b.is_game_over():
                    out.append(b.fen())
        if len(out) >= n:
            break
    return out[:n]


if __name__ == "__main__":
    import torch
    from mcts import MCTS, pinned_search
    from play_web import load_version
    from ttt import OpponentModel

    fens = sample(N_POS)
    model, label, _ = load_version(E("VERSION", "mayfly_v1.pt"))
    dev = str(next(model.parameters()).device)
    print(f"{len(fens)} positions, {VISITS} visits each, {label} on {dev}", flush=True)

    def build():
        t = MCTS(model, dev, batch=48, cache_size=20000)
        # The version's own sealed search, since the split of CPU time depends on which settings
        # are on.
        if E("PINNED", "1") == "1":
            t.apply(pinned_search(E("VERSION", "mayfly_v1.pt")))
        if E("ADAPTER", "0") == "1":
            t.opponent = OpponentModel(model, dev)
            t.opp_colour = chess.BLACK
        return t

    def run_all():
        for fen in fens:
            t = build()
            t.search(chess.Board(fen), visits=VISITS, reuse=False)

    # scale first, with the profiler off, because cProfile inflates everything it touches
    t0 = time.time()
    run_all()
    if dev != "cpu":
        torch.cuda.synchronize()
    wall = time.time() - t0
    print(f"  unprofiled: {wall:.2f}s for {len(fens)} moves, {wall/len(fens)*1000:.0f} ms a move, "
          f"{len(fens)*VISITS/wall:,.0f} visits a second", flush=True)

    pr = cProfile.Profile()
    pr.enable()
    run_all()
    pr.disable()

    st = pstats.Stats(pr)
    totals = collections.Counter()
    grand = 0.0
    rows = []
    origins = collections.Counter()
    for (fn_file, fn_line, fn_name), (cc, nc, tt, ct, callers) in st.stats.items():
        totals[bucket_of(fn_name)] += tt
        origins[origin_of(fn_name, fn_file)] += tt
        grand += tt
        rows.append((tt, nc, fn_name, os.path.basename(fn_file)))
    print("")
    print(f"  by where the code lives, {grand:.2f}s of profiled self time")
    for label, t in origins.most_common():
        print(f"    {label:<28} {t:>7.2f}s  {100 * t / grand:>5.1f}%")

    print("")
    print(f"  where the time goes, {grand:.2f}s of profiled self time")
    for label, _names in BUCKETS + [("other", ())]:
        share = 100 * totals[label] / grand if grand else 0
        print(f"    {label:<11} {totals[label]:>7.2f}s  {share:>5.1f}%")

    print("")
    top = int(E("TOP", "12"))
    print(f"  the {top} most expensive functions by self time")
    rows.sort(reverse=True)
    for tt, nc, name, where in rows[:top]:
        print(f"    {tt:>7.2f}s  {nc:>9,} calls  {name:<28} {where}  [{bucket_of(name)}]")
