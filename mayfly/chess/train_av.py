r"""train_av is the main trainer, one network with two kinds of supervision in one forward pass.

  lz and lc   a position with a distribution over its best few moves and a value. Deep (25 million
              nodes a position for Lichess) and comparatively small, about a hundred million.
  av          a position and one move with the win probability of playing it. Shallow (50 ms a move)
              and enormous, about ten billion (ChessBench).

Both go into the same trunk, so a batch is built from both and put through the network once, with each
loss applied to its own slice. A ChessBench record names one move, so its loss touches only that move's
output and says nothing about the moves it was not asked about.

  python chess/train_av.py [ITERS]
  env: AV_BSZ=96 POL_BSZ=32   how the batch is split between the two corpora
       SOURCES=lz,lc  D=1024 LAYERS=16 LR=1.2e-4 WARMUP=12000 W_Q=1.0 TAG=av
       INIT=engine    start from an existing checkpoint that has no action-value head
       VERSION_AT=24,48,96,192,384,768,1536,3072   hours at which to freeze a version
       VERSION_EVERY=3072  and every this many hours after the last of those
       COOL_FROM= COOL_STEPS= COOL_FLOOR=0.01   a cooled copy's rate falls to the floor (cooldown.py)
       CKPT_DIR            working checkpoints, default beside the code (put it on a fast disk)
       VERSION_DIR         frozen versions, default paths.under("versions")
       PRECISION=bf16      or fp16acc, fp16 autocast with fp16 accumulation and a GradScaler (precision.py)
       COMPILE=0           1 runs the training forward under torch.compile (about 2 minutes to compile at start)
                           Together they trained 1.48x as fast as bf16 eager in an A/B.

At each hour in VERSION_AT a copy is frozen as mayfly_v1, v2 and so on, with its metrics and corpus size,
so versions can later be played against each other. A version that was not saved cannot be recovered.
"""
import os, sys, json, time, math, glob
# torch.compile on Windows (1.21x alone, 1.48x with PRECISION=fp16acc). triton-windows ships its own C compiler
# (TinyCC) and a minimal CUDA toolchain, but under some Python installs it finds neither by itself ("Failed to find
# C compiler"), so they are pointed at here, before torch imports triton, for this process only. Each precision
# gets its own inductor cache, since a graph compiled under bf16 could otherwise come back under fp16 autocast.
COMPILE = os.environ.get("COMPILE", "0") == "1"
if COMPILE:
    import importlib.util
    _spec = importlib.util.find_spec("triton")
    if os.name == "nt" and _spec and _spec.submodule_search_locations:
        _tdir = list(_spec.submodule_search_locations)[0]
        if os.path.exists(os.path.join(_tdir, "runtime", "tcc", "tcc.exe")):
            os.environ.setdefault("CC", os.path.join(_tdir, "runtime", "tcc", "tcc.exe"))
        if os.path.exists(os.path.join(_tdir, "backends", "nvidia", "include", "cuda.h")):
            os.environ.setdefault("CUDA_PATH", os.path.join(_tdir, "backends", "nvidia"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(
        os.environ.get("CKPT_DIR") or os.path.dirname(os.path.abspath(__file__)),
        "inductor_" + os.environ.get("PRECISION", "bf16").lower()))
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_net import ChessNet, N_MOVES
from data_packed import PackedLoader
from data_av import AVLoader
from precision import apply_precision, autocast_ctx
from aside import Aside
import paths

E = os.environ.get
SHARDS = E("SHARDS", paths.under("shards"))
SOURCES = None if E("SOURCES", "lz,lc").lower() == "all" else E("SOURCES", "lz,lc").split(",")
D, LAYERS = int(E("D", 1024)), int(E("LAYERS", 16))
NHEAD, FF_MULT = int(E("NHEAD", 16)), float(E("FF_MULT", 2.0))
AV_BSZ, POL_BSZ = int(E("AV_BSZ", 96)), int(E("POL_BSZ", 32))
LR = float(E("LR", 1.2e-4))
# Adam's second moment. At small batches beta2 0.95 averages the squared gradient over only a few hundred
# positions. Holding the half-life constant in samples puts beta2 near 0.999. Measured with arms_beta2.py
# (three arms of 100,000 steps), 0.999 beat 0.95 (legal policy loss 1.8462 against 1.8916, top-1 44.1%
# against 42.7%, q flat), and 0.9995 at twice the LR was worse on everything.
BETA2 = float(E("BETA2", 0.999))
DP_SHARE = float(E("DP_SHARE", 0.0))
W_Q = float(E("W_Q", 1.0))
# MASK_LEGAL changes the objective, so it defaults off. Without it the softmax runs over all 4,162 moves
# and the head spent about a quarter of its mass on moves that cannot be played.
#
# The policy loss is not comparable across this flag. A masked cross-entropy is lower on the same
# weights, and once illegal logits get no gradient the unmasked number may drift upward while the masked
# one improves. Judge the change on `top1_legal`, which is measured the same way on both sides.
MASK_LEGAL = E("MASK_LEGAL", "0") == "1"
# W_V=0 freezes the value head by leaving the term out of the loss rather than multiplying it by zero, so
# the head gets no gradient and the trunk is not pulled toward an objective the search never reads (with
# `value_from_q=True` every leaf is scored from the action-value head). The number is still logged.
W_V = float(E("W_V", 0.5))
# Attention heads on the square tokens, a policy head and an action-value head that score a move from its
# two squares, trained on the same targets as the CLS heads. For ATT_DETACH steps after they first appear
# they learn on a detached trunk, so a random head's gradient cannot shake a trained trunk. After that the
# loss reaches the trunk at weight W_ATT.
ATT = E("ATT", "0") == "1"
W_ATT = float(E("W_ATT", 1.0))
ATT_DETACH = int(E("ATT_DETACH", 20000))
# Distributional heads. DIST is the number of buckets, 0 for off (DeepMind's search model uses 64).
# `qdist` learns one move's win probability as a distribution and reaches the trunk at W_QDIST after its
# own warmup. `vdist` learns the position's value from Leela's search value and is a read-out that never
# pulls the trunk, since the scalar value head is frozen.
DIST = int(E("DIST", 0))
W_QDIST = float(E("W_QDIST", 0.3))
W_VDIST = float(E("W_VDIST", 1.0))
DIST_DETACH = int(E("DIST_DETACH", 20000))
# Moves left. MLH=1 adds the head and puts MLH_SHARE of every policy batch on Leela records that carry
# plies_left, drawn only from shards written since MLH_SINCE (when the converter began keeping it). Those
# rows are ordinary policy rows as well, so the head costs no extra forward pass. For MLH_DETACH steps it
# learns on a detached trunk, then reaches the trunk at W_MLH. The loss is a Huber on plies / 100 with a
# 10-ply knee, Lc0's shape.
MLH = E("MLH", "0") == "1"
W_MLH = float(E("W_MLH", 0.1))
MLH_DETACH = int(E("MLH_DETACH", 30000))
MLH_SHARE = float(E("MLH_SHARE", 0.25))
MLH_SINCE = E("MLH_SINCE", "2026-09-24 19:10")
TAG = E("TAG", "av")
INIT = E("INIT", "")
# Every validation rewrites a 1.72 GB resume checkpoint, so validating often wears out an SSD. 10,000
# steps still leaves thousands of points on the curve.
VAL_EVERY = int(E("VAL_EVERY", 10000))
# The resume checkpoint has its own, shorter clock, so a crash costs about 20 minutes. This is tied to
# supervise.py's STALL_MINUTES, which must stay above roughly two save intervals or the supervisor kills
# the trainer before it can ever save.
SAVE_EVERY = int(E("SAVE_EVERY", 5000))
# Zero, as the architecture sweep chose. At one pass over the corpus there is nothing to regularise.
DROPOUT = float(E("DROPOUT", 0.0))
# the board's "somebody is at the board" flag, and how much of each step to give them
PLAYING_FILE = E("PLAYING_FILE", os.path.join(E("CKPT_DIR", paths.under("ckpt")), "playing.flag"))
YIELD_MS = float(E("YIELD_MS", 40))      # 0 turns the courtesy off
YIELD_FOR = float(E("YIELD_FOR", 90))    # seconds after the last request that still counts
# The release gate's flag (aside.py). While clock_gate.py holds it the trainer stands aside between steps,
# touching train_heartbeat for supervise.py while it waits. "" turns it off. Each run watches its own
# folder's flag, so a cooled copy or an arm is never held by the main run's gate.
ASIDE_FILE = E("ASIDE_FILE", os.path.join(E("CKPT_DIR", paths.under("ckpt")), "gate_aside.flag"))
VAL_BATCHES = int(E("VAL_BATCHES", 40))
# the trainer stops once it passes this many hours
MAX_HOURS = float(E("MAX_HOURS", 5000))
RESUME = E("RESUME", "1") == "1"
SEED = int(E("SEED", 0))
# The training stream must not rewind on a restart. Nothing about the loaders goes into the checkpoint, so
# seeding them from SEED would replay the same rows after every restart. The training loaders get a fresh
# seed every start, printed in the log and kept in the checkpoint, and DATA_SEED=<that number> replays a
# segment exactly. The validation loaders stay on SEED + 1 so the yardstick does not move.
DATA_SEED = int(E("DATA_SEED")) if E("DATA_SEED") else None
# Route 1's dp shards are written a position at a time, about thirty rows each, so the dp loader
# thins its sequential reads by this much to spread a pool over that many times more positions.
# data_av.AVLoader explains the arithmetic. ChessBench is shuffled already and keeps thin=1.
DP_THIN = int(E("DP_THIN", 32))
# Route 1 positions at a lower weight, for older self-play whose search was known to be flawed.
# DP_DOWN_KEYS lists their board hashes (data_av.board_hash_av, sorted .npy), and their rows train
# DP_DOWN_WEIGHT times as often as the rest of the dp slice. Off at weight 1 or with no file.
DP_DOWN_KEYS = E("DP_DOWN_KEYS", "")
DP_DOWN_WEIGHT = float(E("DP_DOWN_WEIGHT", 1.0))
# The Lichess deep evaluations in the action-value slice. lichess_to_av.py writes them as lv_* shards
# (37.7 million positions scored by Stockfish at a median 25 million nodes, 3 to 5 best moves each, on
# ChessBench's win-probability curve). LV_SHARE of AV_BSZ rows come from there, taken out of ChessBench's
# rows, and `lv_q` in the log is the action-value loss on its held-out positions. A position's rows sit
# together, so reads are thinned LV_THIN times. Off at 0.
LV_SHARE = float(E("LV_SHARE", 0.0))
LV_THIN = int(E("LV_THIN", 8))
# Fishtest's Stockfish games in the same slice. fishtest_to_av.py writes them as fs_* shards, one row a
# position, the move Stockfish played valued at its own evaluation (median depth 23, against ChessBench's
# 13). FS_SHARE of AV_BSZ rows come from there, and `fs_q` is the action-value loss on held-out fishtest
# positions. A game's positions sit together, so reads are thinned FS_THIN times. Off at 0.
FS_SHARE = float(E("FS_SHARE", 0.0))
FS_THIN = int(E("FS_THIN", 8))
# The deep second labels in the same slice. dh_* shards are MayFly's own hard positions labelled again much
# deeper. DH_SHARE of AV_BSZ rows come from there, and `dhq` in the log is the action-value loss on held-out
# dh positions. A position's rows sit together, so reads are thinned DH_THIN times. Off at 0.
DH_SHARE = float(E("DH_SHARE", 0.0))
DH_THIN = int(E("DH_THIN", 4))
# MayFly's own search targets in the same slice. st_* shards are pack_search.py's, every move the self-play
# search spent enough visits on, valued at the search's own mean for it. ST_SHARE of AV_BSZ rows come from
# there, and `stq` in the log is the action-value loss on held-out search targets. A position's rows sit
# together, so reads are thinned ST_THIN times. Off at 0 (arms_search.py tests it).
ST_SHARE = float(E("ST_SHARE", 0.0))
ST_THIN = int(E("ST_THIN", 4))
# What the validation numbers are measured against. castle_by_king (pack.py) moved every castling target
# from king-takes-rook onto the king's own square, so `policy_legal` and `top1_legal` read differently
# after it. A best score from before it is reset, or the best-of checkpoint would freeze at old weights.
YARDSTICK = "castle-by-king"
WARMUP = int(E("WARMUP", 12000))
HEARTBEAT = int(E("HEARTBEAT", 2000))
# Versions at doubling hours, not on a calendar. Improvement is roughly logarithmic in time (the policy
# gain per 100,000 steps halved each window), so equal spacing in log time puts roughly equal improvement
# between consecutive versions, which makes the version ladder measurable. 3072 hours is the last
# doubling inside a two hundred day run.
VERSION_AT = [float(x) for x in E("VERSION_AT", "24,48,96,192,384,768,1536,3072").split(",")]
VERSION_EVERY = float(E("VERSION_EVERY", 3072))
# The cooldown, for a copy of the run and never the run itself. A release cools a copy to near zero over
# 20 to 40 hours to become the candidate, and the main run resumes from its own saved state. From iteration
# COOL_FROM the rate is multiplied by a half cosine falling from 1 to COOL_FLOOR over COOL_STEPS steps, and
# stays at the floor after. A function of the iteration, so a cooled copy that dies resumes on the same
# curve. COOL_STEPS 0, the default, changes nothing. chess/cooldown.py runs it.
COOL_FROM = int(E("COOL_FROM", 0))
COOL_STEPS = int(E("COOL_STEPS", 0))
COOL_FLOOR = float(E("COOL_FLOOR", 0.01))


def cool_factor(it):
    """1.0 everywhere unless a cooldown is set, then 1 at COOL_FROM falling to COOL_FLOOR."""
    if COOL_STEPS <= 0 or it < COOL_FROM:
        return 1.0
    x = min(1.0, (it - COOL_FROM) / COOL_STEPS)
    return COOL_FLOOR + (1.0 - COOL_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * x))


def save_atomic(obj, path):
    """Write beside the target, then rename over it.

    torch.save writes straight into the file it names, so a process that dies partway leaves a
    truncated file where the only resume point used to be. A rename on the same volume is atomic,
    so the old checkpoint survives until a complete new one replaces it.
    """
    tmp = path + ".part"
    torch.save(obj, tmp)
    # On Windows the rename is refused while another process has the target open (self-play or a
    # harness in the middle of torch.load), so it waits for the reader instead of dying
    for k in range(60):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == 59:
                raise
            time.sleep(1.0)


def versions_owed(hours):
    """How many versions should exist by this many hours of training."""
    n = sum(1 for h in VERSION_AT if hours >= h)
    if n == len(VERSION_AT) and hours >= VERSION_AT[-1]:
        n += int((hours - VERSION_AT[-1]) // VERSION_EVERY)
    return n
ITERS = int(float(sys.argv[1])) if len(sys.argv) > 1 and sys.argv[1].replace(".", "").isdigit() \
    else int(float(E("ITERS", 4_000_000)))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Where checkpoints go. The resume checkpoint (1.72 GB, rewritten often) belongs on a fast disk, where a
# save takes seconds instead of stalling training. The frozen versions (573 MB each, written rarely) keep
# accumulating and can go on a big slow disk.
CKPT_DIR = E("CKPT_DIR", HERE)
VERSION_DIR = E("VERSION_DIR", paths.under("versions"))
os.makedirs(VERSION_DIR, exist_ok=True)
CKPT = os.path.join(CKPT_DIR, f"ckpt_{TAG}.pt")
LAST = os.path.join(CKPT_DIR, f"ckpt_{TAG}_last.pt")
# Freeze on demand. Create this file and the next validation cuts a version and deletes the file, so a
# release can follow a milestone rather than a timer.
FREEZE_NOW = os.path.join(CKPT_DIR, "freeze_now.flag")
HIST = os.path.join(CKPT_DIR, f"hist_{TAG}.json")
VERSIONS = os.path.join(VERSION_DIR, "versions.json")


@torch.no_grad()
def recentre_policy(model):
    """Subtract the mean row from the policy head's weight and the mean from its bias.

    Every logit of a position moves by the same amount, and softmax cannot see a shift common to all
    moves, so in exact arithmetic this changes nothing. It matters for bfloat16. Uncentred, the logits
    drifted to a common offset near -300, where bf16 resolves only every second integer and many
    positions had a tie for the top move. Centred, they agree with fp32 on the top move 97% of the
    time. AdamW scales each element separately, so the offset drifts back, and this runs after every
    step (one 4162 x 1024 mean). Used from v4 onwards.
    """
    w, b = model.policy.weight, model.policy.bias
    w.sub_(w.mean(0, keepdim=True))
    b.sub_(b.mean())


def lr_at(it, hours=0.0):
    """Warmup, then one halving at each version freeze.

    Every entry of VERSION_AT doubles, so halving the rate at each freeze is a rate proportional to
    1/t, and every change lands on a version boundary where it can be attributed. A constant rate
    converges only to a noise ball around a minimum.

    Driven by `hours` rather than a counter, so a restart recovers the right rate from the checkpoint's
    trained_hours. The offset of two puts the first halving at 96 hours, on v3's freeze.

    A cooled copy (COOL_STEPS above) multiplies all of that by its cosine. The main run never does.
    """
    return (LR * min(1.0, (it + 1) / WARMUP) * 0.5 ** max(0, versions_owed(hours) - 2)
            * cool_factor(it))


def read_versions(tries=3):
    """Every released version, or a loud stop. Never an empty list by accident.

    Reading "no versions" by mistake would make the next validation freeze today's weights over every
    released version. So it reads with utf-8-sig like every other reader, retries a moment in case a
    hand edit is mid-write, and then stops. A missing file with released weights beside it stops too.
    Only a directory with neither is a fresh run.
    """
    if not os.path.exists(VERSIONS):
        released = sorted(glob.glob(os.path.join(VERSION_DIR, "mayfly_v*.pt")))
        if released:
            raise SystemExit(
                f"{VERSIONS} is missing but {len(released)} released version(s) sit beside it "
                f"({', '.join(os.path.basename(p) for p in released[:6])}). Refusing to go on: "
                f"with no manifest the trainer would freeze over them. Restore versions.json.")
        return []
    err = None
    for k in range(tries):
        try:
            with open(VERSIONS, encoding="utf-8-sig") as f:
                rows = json.load(f)
            [int(v["version"]) for v in rows]            # every entry must say which version it is
            return rows
        except Exception as e:
            err = e
            if k + 1 < tries:
                time.sleep(2)
    raise SystemExit(
        f"cannot read {VERSIONS} ({type(err).__name__}: {err}). Refusing to go on: an unreadable "
        f"manifest used to read as 'no versions' and would have frozen today's weights over every "
        f"released one. Fix the file; the supervisor's restart will pick the run up from its last "
        f"save.")


def sealed_search():
    """The search as mcts.py stands on disk now, read by a fresh interpreter.

    Inside this process Python caches mcts as it was at the first import, so a second freeze in a
    long-running trainer would seal an old search. A child process has no stale cache. Reloading in
    place is only a fallback, for a machine that cannot start one.
    """
    import subprocess
    code = ("import json, sys; sys.path.insert(0, %r); from mcts import default_search; "
            "print('SEALED ' + json.dumps(default_search()))" % HERE)
    # The seal is mcts.py's defaults, never this process's environment. mcts takes every default from an
    # environment variable, and some names mean different things here (MLH=1 trains the moves-left head
    # but turns on the tree's moves-left utility in mcts). So every name mcts.py reads is left out.
    import re
    names = set(re.findall(r'E\("([A-Z0-9_]+)"', open(os.path.join(HERE, "mcts.py"), encoding="utf-8").read()))
    names.add("MLH")
    env = {k: v for k, v in os.environ.items() if k not in names}
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True,
                           text=True, timeout=300, env=env)
        for line in reversed(r.stdout.splitlines()):
            if line.startswith("SEALED "):
                return json.loads(line[len("SEALED "):])
        raise RuntimeError(f"exit {r.returncode}: {(r.stderr or '').strip()[-400:]}")
    except Exception as e:
        print(f"  a fresh interpreter could not read the search ({type(e).__name__}: {e}); "
              f"reloading mcts in this process instead", flush=True)
    import importlib
    import mcts
    held = {k: os.environ.pop(k) for k in list(os.environ) if k in names}
    try:
        return importlib.reload(mcts).default_search()
    finally:
        os.environ.update(held)


def freeze_version(n, model, it, hours, val, corpus):
    """Copy the weights aside under a version number, and record what they could do."""
    path = os.path.join(VERSION_DIR, f"mayfly_v{n}.pt")
    # A released version is never written over. If the manifest already names it, or its weights are
    # already on disk, something upstream is wrong.
    all_v = read_versions()
    if any(int(v["version"]) == n for v in all_v) or os.path.exists(path):
        raise SystemExit(
            f"refusing to freeze MayFly v{n}: it is already released "
            f"({'in versions.json' if any(int(v['version']) == n for v in all_v) else path}). "
            f"A frozen version is never overwritten.")
    # Sealed at release. The version being frozen gets the search exactly as it stands now and never
    # changes afterwards. Improvements go into the next version. Earlier versions are not touched.
    # Sealed before the weights are written, since the seal can take minutes and a trainer that died
    # inside it would leave mayfly_vN.pt with no manifest row.
    try:
        pinned = sealed_search()
    except Exception as e:
        print(f"  could not record the search settings: {type(e).__name__}: {e}", flush=True)
        pinned = {}
    save_atomic({"model": model.state_dict(), "iter": it, "val": val,
                 "version": n, "hours": round(hours, 2), "corpus": corpus,
                 "cfg": {"d": D, "layers": LAYERS, "nhead": NHEAD, "ff_mult": FF_MULT,
                         "qhead": True, "att": ATT, "dist": DIST,
                         "sources": SOURCES}}, path)
    entry = {"version": n, "file": os.path.basename(path), "iter": it,
             "hours": round(hours, 2), "corpus": corpus,
             "at": time.strftime("%Y-%m-%d %H:%M:%S"), **val}
    if pinned:
        entry["search"] = pinned
        print(f"  sealed MayFly v{n} at release with all {len(pinned)} search settings: "
              f"{pinned}", flush=True)
    else:
        print(f"  MayFly v{n} FROZE WITHOUT A SEALED SEARCH. It will play today's defaults, "
              f"which is wrong the moment they change. Seal it by hand.", flush=True)
    # Read again before writing, so a hand edit made to versions.json during the seal is not lost.
    if os.path.exists(VERSIONS):
        all_v = read_versions()
        if any(int(v["version"]) == n for v in all_v):
            raise SystemExit(f"refusing to record MayFly v{n}: versions.json gained it during the seal")
    all_v.append(entry)
    all_v.sort(key=lambda v: v["version"])
    tmp = VERSIONS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:           # closed, and so complete, before the swap
        json.dump(all_v, f, indent=1)
    # A reader holding versions.json open (pinned_search, the board's version list, wait_for_version)
    # makes Windows refuse the swap, so it waits instead of dying and leaving mayfly_vN.pt with no row
    for k in range(60):
        try:
            os.replace(tmp, VERSIONS)
            break
        except PermissionError:
            if k == 59:
                raise
            time.sleep(1.0)
    print(f"  FROZE MayFly_v{n} at {hours:.1f}h, iter {it:,}, q {val['q']:.4f}, "
          f"policy {val['policy']:.4f}", flush=True)


def q_loss(out, mv, q):
    """Binary cross entropy on the one move each record names, and nothing else."""
    logit = out["qvalue"].float().gather(1, mv.view(-1, 1)).squeeze(1)
    return F.binary_cross_entropy_with_logits(logit, q), (torch.sigmoid(logit) - q).abs().mean()


_C = None


def centres(device):
    """The bucket centres, made once. The value a distribution means is its mean over these."""
    global _C
    if _C is None or _C.device != torch.device(device):
        from chess_net import bin_centres
        _C = bin_centres(DIST, device)
    return _C


def dist_loss(logits, target, centres):
    """Cross entropy against the HL-Gauss target, and the error of the distribution's mean, which
    is the number a search would read off it."""
    from chess_net import hl_gauss
    logits = logits.float()
    p = hl_gauss(target, logits.shape[-1])
    ce = -(p * F.log_softmax(logits, -1)).sum(-1).mean()
    mean = (F.softmax(logits, -1) * centres).sum(-1)
    return ce, (mean - target.float()).abs().mean()


def policy_ce(logits, pol, legal=None):
    """Cross entropy of the target distribution against the logits, over legal moves if given."""
    logits = logits.float()
    if legal is not None:
        # -1e9 rather than -inf, since a fully masked row would make log_softmax produce nan. The row
        # cannot be fully masked (batch_tensors adds the target's own moves to the mask), but a loss
        # function is the wrong place to rely on that.
        logits = logits.masked_fill(~legal, -1e9)
    return -(pol * F.log_softmax(logits, -1)).sum(-1).mean()


def policy_losses(out, pol, val, mat, king, pawns, mob, bish, aux, legal=None):
    lp = policy_ce(out["policy"], pol, legal)
    lv = F.binary_cross_entropy_with_logits(out["value"].float(), val)
    w = aux.view(-1, 1)
    n = aux.sum().clamp(min=1.0)
    msew = lambda a, b, k: (((a.float() - b) ** 2).mean(-1, keepdim=True) * k).sum() / n
    la = (msew(out["material"], mat, w) + msew(out["king"], king, w)
          + msew(out["pawns"], pawns, w) + msew(out["mob"].unsqueeze(-1), mob.unsqueeze(-1), w)
          + (F.binary_cross_entropy_with_logits(out["bishop"].float(), bish, reduction="none")
             .mean(-1, keepdim=True) * w).sum() / n)
    total = lp + 0.2 * la
    if W_V:
        total = total + W_V * lv
    return total, lp, lv


def mleft_loss(pred, plies, valid):
    """Huber on plies / 100 (knee at 10 plies) over the rows that carry a label, and the MAE in plies."""
    w = valid.float()
    n = w.sum().clamp(min=1.0)
    err = pred.float() - plies / 100.0
    hub = F.huber_loss(pred.float(), plies / 100.0, reduction="none", delta=0.1)
    return (hub * w).sum() / n, (err.abs() * w).sum() / n * 100.0


def step_batch(model, used, av, pol, want_parts=False, dp=None, mlh=None, lv=None, fs=None, dh=None, st=None):
    """One forward pass over both corpora, each loss applied to its own slice.

    `dp` is route 1's own labels (MayFly's positions, every legal move scored by Stockfish), in the
    same record format as the action-value corpus and mixed into the same slice as a fixed share of
    every batch (DP_SHARE). `lv`, `fs`, `dh` and `st` are further fixed shares of the same slice."""
    a = av.next()
    if dp is not None:
        d = dp.next()
        a = tuple(torch.cat([x, y], 0) for x, y in zip(a, d))
    if lv is not None:
        e = lv.next()
        a = tuple(torch.cat([x, y], 0) for x, y in zip(a, e))
    if fs is not None:
        f = fs.next()
        a = tuple(torch.cat([x, y], 0) for x, y in zip(a, f))
    if dh is not None:
        g = dh.next()
        a = tuple(torch.cat([x, y], 0) for x, y in zip(a, g))
    if st is not None:
        h = st.next()
        a = tuple(torch.cat([x, y], 0) for x, y in zip(a, h))
    p = pol.next()
    if mlh is not None:
        p = tuple(torch.cat([x, y], 0) for x, y in zip(p, mlh.next()))
    ml = None
    if MLH:
        p, ml = p[:-2], p[-2:]                      # plies left and whether the row carries it
    ids = torch.cat([a[0], p[0]], 0)
    cast = torch.cat([a[1], p[1]], 0)
    hm = torch.cat([a[2], p[2]], 0)
    ep = torch.cat([a[3], p[3]], 0)
    with autocast_ctx(used):
        out = model(ids, cast, hm, ep)
    n = a[0].shape[0]
    av_out = {"qvalue": out["qvalue"][:n]}
    pol_out = {k: v[n:] for k, v in out.items() if k not in ("qvalue", "qvalue_att")}
    lq, mae = q_loss(av_out, a[4], a[5])
    lpol, lp, lv = policy_losses(pol_out, *p[4:])
    # Every float() here is a GPU sync that stalls the CPU. The training loop never reads `parts` (the
    # log's numbers come from validate), so they are computed only when `want_parts` asks for them.
    d = (lambda t: float(t.detach())) if want_parts else (lambda t: None)
    total = W_Q * lq + lpol
    parts = {"q": d(lq), "mae": d(mae), "policy": d(lp), "value": d(lv)}
    if "policy_att" in out:
        # the same targets as the CLS heads, the same legality mask, the same weights between them
        lq_att, _ = q_loss({"qvalue": out["qvalue_att"][:n]}, a[4], a[5])
        lp_att = policy_ce(out["policy_att"][n:], p[4], p[12] if len(p) > 12 else None)
        total = total + W_ATT * (W_Q * lq_att + lp_att)
        parts.update({"att_q": d(lq_att), "att_policy": d(lp_att)})
    if DIST:
        c = centres(out["h"].device)
        lqd, qd_mae = dist_loss(model.qdist_of(out["h"][:n], a[4]), a[5], c)
        lvd, vd_mae = dist_loss(out["vdist"][n:], p[5], c)
        total = total + W_QDIST * lqd + W_VDIST * lvd
        parts.update({"qdist": d(lqd), "qdist_mae": d(qd_mae),
                      "vdist": d(lvd), "vdist_mae": d(vd_mae)})
    if ml is not None and "mleft" in out:
        lm, m_mae = mleft_loss(out["mleft"][n:], ml[0], ml[1])
        total = total + W_MLH * lm
        parts.update({"mleft": d(lm), "mleft_mae": d(m_mae)})
    return total, parts


@torch.no_grad()
def validate(model, used, av, pol, n, mlv=None, lvv=None, fsv=None, dhv=None, stv=None):
    model.eval()
    tot = {"q": 0.0, "mae": 0.0, "policy": 0.0, "value": 0.0, "top1": 0.0}
    # Both metrics, every time, whether or not the objective is masked. `policy` and `top1` stay as v1
    # to v3 measured them. The `_legal` pair is measured on the distribution `_expand` hands to PUCT and
    # is the only one that compares a masked version with an unmasked one.
    for _ in range(n):
        a, p = av.next(), pol.next()
        with autocast_ctx(used):
            oa = model(a[0], a[1], a[2], a[3])
            op = model(p[0], p[1], p[2], p[3])
        lq, mae = q_loss(oa, a[4], a[5])
        tot["q"] += float(lq)
        tot["mae"] += float(mae)
        tot["policy"] += float(-(p[4] * F.log_softmax(op["policy"].float(), -1)).sum(-1).mean())
        tot["value"] += float(F.binary_cross_entropy_with_logits(op["value"].float(), p[5]))
        tot["top1"] += float((op["policy"].argmax(-1) == p[4].argmax(-1)).float().mean())
        if len(p) > 12:
            ml = op["policy"].float().masked_fill(~p[12], -1e9)
            tot["policy_legal"] = tot.get("policy_legal", 0.0) + float(
                -(p[4] * F.log_softmax(ml, -1)).sum(-1).mean())
            tot["top1_legal"] = tot.get("top1_legal", 0.0) + float(
                (ml.argmax(-1) == p[4].argmax(-1)).float().mean())
            # top-3 beside top-1. Top-1 is peak strength and top-3 measures general play. Earlier
            # versions were scored on the same positions by top3_versions.py.
            tot["top3_legal"] = tot.get("top3_legal", 0.0) + float(
                (ml.topk(3, -1).indices == p[4].argmax(-1, keepdim=True)).any(-1).float().mean())
        if DIST:
            c = centres(oa["h"].device)
            lqd, qd_mae = dist_loss(model.qdist_of(oa["h"], a[4]), a[5], c)
            lvd, vd_mae = dist_loss(op["vdist"], p[5], c)
            for k, v in (("qdist", lqd), ("qdist_mae", qd_mae),
                         ("vdist", lvd), ("vdist_mae", vd_mae)):
                tot[k] = tot.get(k, 0.0) + float(v)
        if "policy_att" in op:
            # measured exactly as the CLS head's `_legal` pair and `q` are, on the same batches,
            # so the two policies can be read against each other point for point
            lq_att, _ = q_loss({"qvalue": oa["qvalue_att"]}, a[4], a[5])
            tot["att_q"] = tot.get("att_q", 0.0) + float(lq_att)
            if len(p) > 12:
                al = op["policy_att"].float().masked_fill(~p[12], -1e9)
                tot["att_policy_legal"] = tot.get("att_policy_legal", 0.0) + float(
                    -(p[4] * F.log_softmax(al, -1)).sum(-1).mean())
                tot["att_top1_legal"] = tot.get("att_top1_legal", 0.0) + float(
                    (al.argmax(-1) == p[4].argmax(-1)).float().mean())
                tot["att_top3_legal"] = tot.get("att_top3_legal", 0.0) + float(
                    (al.topk(3, -1).indices == p[4].argmax(-1, keepdim=True)).any(-1).float().mean())
    if mlv is not None:
        # the moves-left head on its own held-out labelled records, the same count of batches
        for _ in range(n):
            q = mlv.next()
            with autocast_ctx(used):
                oq = model(q[0], q[1], q[2], q[3])
            if "mleft" in oq:
                lm, m_mae = mleft_loss(oq["mleft"], q[-2], q[-1])
                tot["mleft"] = tot.get("mleft", 0.0) + float(lm)
                tot["mleft_mae"] = tot.get("mleft_mae", 0.0) + float(m_mae)
    if lvv is not None:
        # the action values on held-out Lichess positions, the deep labels, scored like `q`
        for _ in range(n):
            a = lvv.next()
            with autocast_ctx(used):
                oa = model(a[0], a[1], a[2], a[3])
            lq, mae = q_loss(oa, a[4], a[5])
            tot["lv_q"] = tot.get("lv_q", 0.0) + float(lq)
            tot["lv_mae"] = tot.get("lv_mae", 0.0) + float(mae)
    if fsv is not None:
        # the same on held-out fishtest positions
        for _ in range(n):
            a = fsv.next()
            with autocast_ctx(used):
                oa = model(a[0], a[1], a[2], a[3])
            lq, mae = q_loss(oa, a[4], a[5])
            tot["fs_q"] = tot.get("fs_q", 0.0) + float(lq)
            tot["fs_mae"] = tot.get("fs_mae", 0.0) + float(mae)
    if dhv is not None:
        # the same on held-out deep second labels (DH_SHARE)
        for _ in range(n):
            a = dhv.next()
            with autocast_ctx(used):
                oa = model(a[0], a[1], a[2], a[3])
            lq, mae = q_loss(oa, a[4], a[5])
            tot["dh_q"] = tot.get("dh_q", 0.0) + float(lq)
            tot["dh_mae"] = tot.get("dh_mae", 0.0) + float(mae)
    if stv is not None:
        # the same on held-out search targets (ST_SHARE)
        for _ in range(n):
            a = stv.next()
            with autocast_ctx(used):
                oa = model(a[0], a[1], a[2], a[3])
            lq, mae = q_loss(oa, a[4], a[5])
            tot["st_q"] = tot.get("st_q", 0.0) + float(lq)
            tot["st_mae"] = tot.get("st_mae", 0.0) + float(mae)
    model.train()
    return {k: v / n for k, v in tot.items()}


if __name__ == "__main__":
    from pidfile import claim
    claim("train_av")

    # Normal priority, whoever starts it. Task Scheduler starts its tasks below normal and children inherit it,
    # so the trainer would compete with CPU-heavy jobs such as a Stockfish labeller and starve its loader and
    # kernel launches (about half the steps an hour, with the GPU mostly idle).
    if os.name == "nt":
        import ctypes
        _k32 = ctypes.WinDLL("kernel32")        # its own handle types, not windll's shared ones (64-bit handles)
        _k32.GetCurrentProcess.restype = ctypes.c_void_p
        _k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
        _k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        _me = _k32.GetCurrentProcess()
        _was = _k32.GetPriorityClass(_me)
        if _was != 0x20 and _k32.SetPriorityClass(_me, 0x20):      # NORMAL_PRIORITY_CLASS
            print(f"[priority] normal (started at class 0x{_was:x})", flush=True)

    torch.manual_seed(SEED)
    # Read before anything slow, so a manifest that cannot be read stops the trainer in seconds
    # rather than at the first validation. See read_versions.
    print(f"versions on record: {len(read_versions())} in {VERSIONS}", flush=True)
    data_seed = DATA_SEED if DATA_SEED is not None else int.from_bytes(os.urandom(4), "little")
    print(f"[data] training loaders seeded {data_seed}, fresh this start so a resume does not "
          f"replay the stream (DATA_SEED={data_seed} reproduces it); validation stays on "
          f"SEED+1={SEED + 1}", flush=True)
    # Route 1's share of every batch. dp_* shards are what deep_label writes from MayFly's own self-play
    # and games, in the same format as av_*. DP_SHARE of AV_BSZ rows come from there and the rest from
    # ChessBench, so the batch size, memory and loss are unchanged. Off at 0 or with no dp shard yet.
    dp_rows = int(round(AV_BSZ * DP_SHARE))
    dp_rows = dp_rows if dp_rows > 0 and glob.glob(os.path.join(SHARDS, "dp_*.bin")) else 0
    lv_rows = int(round(AV_BSZ * LV_SHARE))
    if lv_rows and not glob.glob(os.path.join(SHARDS, "lv_*.bin")):
        print(f"[lichess] LV_SHARE={LV_SHARE:g} but no lv_* shard in {SHARDS}; slice off", flush=True)
        lv_rows = 0
    fs_rows = int(round(AV_BSZ * FS_SHARE))
    if fs_rows and not glob.glob(os.path.join(SHARDS, "fs_*.bin")):
        print(f"[fishtest] FS_SHARE={FS_SHARE:g} but no fs_* shard in {SHARDS}; slice off", flush=True)
        fs_rows = 0
    dh_rows = int(round(AV_BSZ * DH_SHARE))
    if dh_rows and not glob.glob(os.path.join(SHARDS, "dh_*.bin")):
        print(f"[deep] DH_SHARE={DH_SHARE:g} but no dh_* shard in {SHARDS}; slice off", flush=True)
        dh_rows = 0
    st_rows = int(round(AV_BSZ * ST_SHARE))
    if st_rows and not glob.glob(os.path.join(SHARDS, "st_*.bin")):
        print(f"[search] ST_SHARE={ST_SHARE:g} but no st_* shard in {SHARDS}; slice off", flush=True)
        st_rows = 0
    av = AVLoader(SHARDS, batch=AV_BSZ - dp_rows - lv_rows - fs_rows - dh_rows - st_rows, split="train", device=device,
                  seed=data_seed, refresh=600)
    if dp_rows and DP_DOWN_KEYS and DP_DOWN_WEIGHT < 1.0 and not os.path.exists(DP_DOWN_KEYS):
        # a weight asked for and not applied is a silent change of recipe, so stop rather than guess
        raise SystemExit(f"DP_DOWN_KEYS={DP_DOWN_KEYS} does not exist; not training without it")
    dp = (AVLoader(SHARDS, batch=dp_rows, split="train", device=device, seed=data_seed + 2,
                   refresh=600, prefix="dp", thin=DP_THIN, down_keys=DP_DOWN_KEYS or None,
                   down_weight=DP_DOWN_WEIGHT) if dp_rows else None)
    if dp is not None:
        print(f"[route 1] {dp_rows} of every {AV_BSZ} action values come from dp_* shards "
              f"({100 * dp_rows / AV_BSZ:.0f}%), reads thinned {DP_THIN}x", flush=True)
        if dp.down is not None:
            print(f"[route 1] {len(dp.down):,} pre-fix self-play positions drawn at weight "
                  f"{DP_DOWN_WEIGHT:g} ({DP_DOWN_KEYS})", flush=True)
    lv = lvv = None
    if lv_rows:
        lv = AVLoader(SHARDS, batch=lv_rows, split="train", device=device, seed=data_seed + 4,
                      refresh=600, prefix="lv", thin=LV_THIN)
        lvv = AVLoader(SHARDS, batch=AV_BSZ, split="val", device=device, seed=SEED + 1,
                       refresh=600, prefix="lv")
        print(f"[lichess] {lv_rows} of every {AV_BSZ} action values come from lv_* shards "
              f"({100 * lv_rows / AV_BSZ:.0f}%), reads thinned {LV_THIN}x; ChessBench keeps "
              f"{AV_BSZ - dp_rows - lv_rows - fs_rows - dh_rows - st_rows}", flush=True)
    fs = fsv = None
    if fs_rows:
        fs = AVLoader(SHARDS, batch=fs_rows, split="train", device=device, seed=data_seed + 5,
                      refresh=600, prefix="fs", thin=FS_THIN)
        fsv = AVLoader(SHARDS, batch=AV_BSZ, split="val", device=device, seed=SEED + 1,
                       refresh=600, prefix="fs")
        print(f"[fishtest] {fs_rows} of every {AV_BSZ} action values come from fs_* shards "
              f"({100 * fs_rows / AV_BSZ:.0f}%), reads thinned {FS_THIN}x; ChessBench keeps "
              f"{AV_BSZ - dp_rows - lv_rows - fs_rows - dh_rows - st_rows}", flush=True)
    dh = dhv = None
    if dh_rows:
        dh = AVLoader(SHARDS, batch=dh_rows, split="train", device=device, seed=data_seed + 6,
                      refresh=600, prefix="dh", thin=DH_THIN)
        dhv = AVLoader(SHARDS, batch=AV_BSZ, split="val", device=device, seed=SEED + 1,
                       refresh=600, prefix="dh")
        print(f"[deep] {dh_rows} of every {AV_BSZ} action values come from dh_* shards "
              f"({100 * dh_rows / AV_BSZ:.1f}%), reads thinned {DH_THIN}x; ChessBench keeps "
              f"{AV_BSZ - dp_rows - lv_rows - fs_rows - dh_rows - st_rows}", flush=True)
    st = stv = None
    if st_rows:
        st = AVLoader(SHARDS, batch=st_rows, split="train", device=device, seed=data_seed + 7,
                      refresh=600, prefix="st", thin=ST_THIN)
        stv = AVLoader(SHARDS, batch=AV_BSZ, split="val", device=device, seed=SEED + 1,
                       refresh=600, prefix="st")
        print(f"[search] {st_rows} of every {AV_BSZ} action values come from st_* shards "
              f"({100 * st_rows / AV_BSZ:.1f}%), reads thinned {ST_THIN}x; ChessBench keeps "
              f"{AV_BSZ - dp_rows - lv_rows - fs_rows - dh_rows - st_rows}", flush=True)
    avv = AVLoader(SHARDS, batch=AV_BSZ, split="val", device=device, seed=SEED + 1, refresh=600)
    mlh_rows = int(round(POL_BSZ * MLH_SHARE)) if MLH else 0
    pol = PackedLoader(SHARDS, batch=POL_BSZ - mlh_rows, split="train", device=device,
                       seed=data_seed + 3, sources=SOURCES, refresh=600, legal=MASK_LEGAL,
                       mleft=MLH)
    mlh = mlv = None
    if MLH:
        since = time.mktime(time.strptime(MLH_SINCE, "%Y-%m-%d %H:%M"))
        mlh = PackedLoader(SHARDS, batch=mlh_rows, split="train", device=device,
                           seed=data_seed + 5, sources=("lz",), refresh=600, legal=MASK_LEGAL,
                           mleft=True, need_flag=8, min_mtime=since)
        mlv = PackedLoader(SHARDS, batch=POL_BSZ, split="val", device=device, seed=SEED + 2,
                           sources=("lz",), refresh=0, legal=True, mleft=True, need_flag=8,
                           min_mtime=since, val_size=8192,
                           val_cache=os.path.join(CKPT_DIR, f"val_mleft_{TAG}.npy"))
        # The cache is read back on every start, so a set drawn while few labelled records existed would
        # be the yardstick for good. Until it holds nearly 8,192 distinct positions it is drawn again.
        uniq = len({r.tobytes() for r in mlv.fixed})
        if uniq < 0.95 * 8192:
            try:
                os.remove(os.path.join(CKPT_DIR, f"val_mleft_{TAG}.npy"))
            except OSError:
                pass
        print(f"[moves left] {mlh_rows} of every {POL_BSZ} policy rows carry plies left, from "
              f"{len(mlh.paths)} shard(s) written since {MLH_SINCE}; validation on "
              f"{len(mlv.fixed):,} labelled positions ({uniq:,} distinct"
              f"{', not kept, drawn again next start' if uniq < 0.95 * 8192 else ''})", flush=True)
    # The validation loader always masks, even when the objective does not, so a version trained without
    # the mask still records what it would score with one.
    polv = PackedLoader(SHARDS, batch=POL_BSZ, split="val", device=device, seed=SEED + 1,
                        sources=SOURCES, refresh=600, legal=True,
                        val_cache=os.path.join(CKPT_DIR, f"val_policy_{TAG}.npy"))
    print(av.describe(), flush=True)
    print(pol.describe(), flush=True)

    model = ChessNet(d=D, nhead=NHEAD, layers=LAYERS, dim_ff=int(FF_MULT * D), qhead=True,
                     dropout=DROPOUT,
                     att=ATT, dist=DIST, mlh=MLH).to(device)
    model, used = apply_precision(model)
    # fp16 scales its loss. Five exponent bits flush the smallest gradients to zero unless the loss is multiplied
    # up first and the gradients divided back down before the clip (bf16 has fp32's range and does not need it).
    # The scaler lowers the scale when a step overflows, which the skip below counts. It starts at 1,024, since
    # the default 65,536 overflowed the very first step.
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 10) if used == "fp16acc" else None
    scaler_state = lambda: scaler.state_dict() if scaler is not None else None      # rides in every resume save
    # Validation stays in bf16 whatever the training precision, so the history stays comparable and the network is
    # measured the way the search runs it (bf16 autocast).
    used_val = "bf16" if used == "fp16acc" else used
    # The training forward goes through torch.compile with COMPILE=1. Validation, the versions and every save use the
    # plain module, which shares the same weights.
    fwd = torch.compile(model) if COMPILE else model
    if COMPILE:
        print(f"torch.compile on: the first step compiles (about 2 minutes); inductor cache "
              f"{os.environ.get('TORCHINDUCTOR_CACHE_DIR')}", flush=True)
    # The optional heads get optimizer groups of their own, after the original two, so a checkpoint written
    # before they existed still loads its AdamW moments. In the same groups they would be a size mismatch,
    # and the fallback would be fresh moments for the whole model.
    is_att = lambda name: name.split(".")[0] in ("ln_sq", "att", "att_q")
    is_dist = lambda name: name.split(".")[0] in ("qdist", "vdist")
    is_mlh = lambda name: name.split(".")[0] == "mleft"
    named = [(n, p) for n, p in model.named_parameters()
             if not is_att(n) and not is_dist(n) and not is_mlh(n)]
    # Fused AdamW is one kernel for the whole update instead of a launch per tensor. Guarded because it
    # needs every parameter on CUDA and a build that has it.
    def _adamw(**kw):
        return torch.optim.AdamW(
            [{"params": [p for _, p in named if p.ndim > 1], "weight_decay": 0.05},
             {"params": [p for _, p in named if p.ndim <= 1], "weight_decay": 0.0}],
            lr=LR, betas=(0.9, BETA2), **kw)
    try:
        opt = _adamw(fused=True)
    except (RuntimeError, ValueError, TypeError) as e:
        print(f"  fused AdamW unavailable ({e}), using the unfused one", flush=True)
        opt = _adamw()
    # Each family of optional heads gets its own group, appended in a fixed order, so the new ones simply
    # start fresh.
    extra = []
    if ATT:
        extra.append({"params": [p for n, p in model.named_parameters() if is_att(n)],
                      "weight_decay": 0.05})
    if DIST:
        extra.append({"params": [p for n, p in model.named_parameters() if is_dist(n)],
                      "weight_decay": 0.05})
    if MLH:
        extra.append({"params": [p for n, p in model.named_parameters() if is_mlh(n)],
                      "weight_decay": 0.05})
    print(f"{model.num_params()/1e6:.1f}M params | d{D} x {LAYERS} | "
          f"batch {AV_BSZ} action values + {POL_BSZ} positions | {used} | {device}", flush=True)

    start, hist, best, trained, att_since, dist_since = 0, [], 1e9, 0.0, None, None
    mlh_since = None
    # RESUME=1 means continue the run. If its checkpoint is not there, something is misconfigured, and a
    # fresh model must not be trained over the top of the real run.
    if RESUME and not os.path.exists(LAST) and E("FRESH", "0") != "1":
        raise SystemExit(
            f"RESUME is on but {LAST} does not exist. Refusing to start from scratch: check "
            f"CKPT_DIR (currently {CKPT_DIR}). Set FRESH=1 if a new run really is what you want.")
    if RESUME and os.path.exists(LAST):
        ck = torch.load(LAST, map_location=device)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        # strict except for the new heads, which a checkpoint from before them lacks
        bad = [k for k in missing if not is_att(k) and not is_dist(k) and not is_mlh(k)]
        bad += list(unexpected)
        if bad:
            raise SystemExit(f"checkpoint does not match the model: {bad[:6]}")
        if missing:
            print(f"{len(missing)} head tensor(s) NEW in this run, at their init: "
                  f"{', '.join(sorted({k.split('.')[0] for k in missing}))}", flush=True)
        # A family the checkpoint had is one none of whose weights were missing from it. One switched on
        # later can sit between those in the fixed order (DIST on after MLH), so the groups are matched by
        # family rather than counted, and put back in the fixed order after the load.
        name_of = {id(p): k for k, p in model.named_parameters()}
        knew = [g for g in extra if not any(name_of[id(p)] in missing for p in g["params"])]
        first = {id(g["params"][0]): i for i, g in enumerate(extra)}
        try:
            for g in knew:                           # groups the checkpoint already knew about
                opt.add_param_group(g)
            opt.load_state_dict(ck["optim"])         # the moments every old parameter had
        except (ValueError, KeyError) as e:
            print(f"optimiser state not loadable ({type(e).__name__}); fresh moments", flush=True)
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])     # the loss scale it had reached, not the default again
        added = {id(g["params"][0]) for g in opt.param_groups}
        for g in extra:                              # and fresh ones for the new heads only
            if id(g["params"][0]) not in added:
                opt.add_param_group(g)
        opt.param_groups[2:] = sorted(opt.param_groups[2:], key=lambda g: first[id(g["params"][0])])
        start, hist, best = ck["iter"] + 1, ck.get("hist", []), ck.get("best", 1e9)
        # the saved param groups carry the betas they were saved with and override the constructor,
        # so BETA2 is written back here or a resumed run would ignore it
        for g in opt.param_groups:
            g["betas"] = (g["betas"][0], BETA2)
        if ck.get("yardstick") != YARDSTICK:
            print(f"validation yardstick is now '{YARDSTICK}' (the checkpoint's was "
                  f"{ck.get('yardstick')!r}): best {best:.4f} was measured the old way and is "
                  f"reset, so the best-of checkpoint follows the new numbers", flush=True)
            best = 1e9
        trained = ck.get("trained_hours", 0.0)
        att_since = ck.get("att_since")
        if ATT and att_since is None:
            att_since = start
        dist_since = ck.get("dist_since")
        if DIST and dist_since is None:
            dist_since = start
        mlh_since = ck.get("mlh_since")
        if MLH and mlh_since is None:
            mlh_since = start
        print(f"RESUMED at iter {start:,}, best {best:.4f}, "
              f"{trained:.1f}h of training behind it", flush=True)
        # centred before the first forward, so the first validation after a resume is already
        # read at full resolution rather than one step late
        off = float(model.policy.weight.mean(0) @ model.policy.weight.mean(0)) ** 0.5
        recentre_policy(model)
        print(f"policy head centred (mean-row norm was {off:.3f})", flush=True)
    elif INIT:
        p = os.path.join(HERE, f"ckpt_{INIT}.pt")
        ck = torch.load(p, map_location=device)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"started from {os.path.basename(p)} at iter {ck.get('iter', '?')}; "
              f"{len(missing)} new tensor(s) left at init: {', '.join(missing) or 'none'}",
              flush=True)

    # the loaded checkpoint sits on the card (map_location=device) and its weights were copied into
    # the parameters, so keeping the dict would hold a second copy of them for the whole run
    ck = None
    for g in extra[len(opt.param_groups) - 2:]:
        opt.add_param_group(g)                       # a fresh run, or one started from INIT
    if ATT and att_since is None:
        att_since = start
    if DIST and dist_since is None:
        dist_since = start
    if MLH and mlh_since is None:
        mlh_since = start
    n_of = lambda pred: sum(p.numel() for n, p in model.named_parameters() if pred(n))
    if MLH:
        print(f"moves-left head: {n_of(is_mlh)/1e3:.0f}K params, own optimiser group, on a "
              f"detached trunk until iter {mlh_since + MLH_DETACH:,}, then end to end at "
              f"weight {W_MLH}", flush=True)
    if ATT:
        print(f"attention heads: {n_of(is_att)/1e6:.2f}M params, own optimiser group, on a "
              f"detached trunk until iter {att_since + ATT_DETACH:,}, then end to end at "
              f"weight {W_ATT}", flush=True)
    if DIST:
        print(f"distributional heads: {n_of(is_dist)/1e6:.2f}M params, {DIST} buckets; qdist "
              f"detached until iter {dist_since + DIST_DETACH:,} then at weight {W_QDIST}; "
              f"vdist is a read-out and never reaches the trunk", flush=True)

    if COOL_STEPS > 0:
        print(f"COOLDOWN: the rate falls by a half cosine to {COOL_FLOOR:g} of itself over iterations "
              f"{COOL_FROM:,} to {COOL_FROM + COOL_STEPS:,}: {lr_at(max(start, COOL_FROM), trained):.3g} now, "
              f"{lr_at(COOL_FROM + COOL_STEPS, trained):.3g} at the end", flush=True)
    t0 = time.time()
    gnorm, skipped = 1.0, 0
    aside = Aside(ASIDE_FILE, os.path.join(CKPT_DIR, "train_heartbeat")) if ASIDE_FILE else None
    model.train()
    for it in range(start, ITERS + 1):
        if it % VAL_EVERY == 0:
            v = validate(model, used_val, avv, polv, VAL_BATCHES, mlv=mlv, lvv=lvv, fsv=fsv, dhv=dhv, stv=stv)
            el = time.time() - t0
            hours = trained + el / 3600
            rate = (it - start) * (AV_BSZ + POL_BSZ) / max(el, 1)
            hist.append({"it": it, **v, "t": round(el, 1), "pos_s": round(rate),
                         "av_corpus": len(av), "pol_corpus": len(pol),
                         "grad": round(gnorm, 3), "skipped": skipped})
            json.dump(hist, open(HIST, "w"), indent=1)
            # Select on the objective actually being trained. With MASK_LEGAL on, the unmasked `policy`
            # may drift upward, so scoring on it would freeze the best-of checkpoint at old weights.
            score = v["q"] + v.get("policy_legal", v["policy"])
            if score < best:
                best = score
                save_atomic({"model": model.state_dict(), "iter": it, "val": v,
                             "yardstick": YARDSTICK,
                             "cfg": {"d": D, "layers": LAYERS, "nhead": NHEAD,
                                     "ff_mult": FF_MULT, "qhead": True, "att": ATT,
                                     "dist": DIST, "mlh": MLH, "sources": SOURCES,
                                     "dropout": DROPOUT}}, CKPT)
            save_atomic({"model": model.state_dict(), "optim": opt.state_dict(), "iter": it,
                         "hist": hist, "best": best, "trained_hours": hours,
                         "att_since": att_since, "dist_since": dist_since, "mlh_since": mlh_since,
                         "yardstick": YARDSTICK, "data_seed": data_seed, "scaler": scaler_state()}, LAST)
            # A version is owed whenever the clock has passed another multiple, so a restart picks
            # up the count from the hours rather than from a counter that could be lost.
            want = versions_owed(hours)
            # read_versions stops the run rather than guess, after the save above, so the stop costs
            # no training
            have = max([int(v["version"]) for v in read_versions()] or [0])
            if os.path.exists(FREEZE_NOW):
                want = max(want, have + 1)
                print(f"  freeze_now: cutting version {want} on request", flush=True)
                try:
                    os.remove(FREEZE_NOW)
                except OSError:
                    pass
            for n in range(have + 1, want + 1):
                freeze_version(n, model, it, hours, v, len(av))
            print(f"  it {it:8,}  lr {lr_at(it, hours):.3g}  "
                  f"q {v['q']:.4f}  mae {v['mae']:.4f}  "
                  f"policy {v['policy']:.4f}  top1 {v['top1']:.4f}"
                  + (f"  top1L {v['top1_legal']:.4f}" if "top1_legal" in v else "")
                  + (f"  attL {v['att_policy_legal']:.4f}  att1L {v['att_top1_legal']:.4f}"
                     f"  attq {v['att_q']:.4f}" if "att_policy_legal" in v else "")
                  + (f"  qdist {v['qdist']:.4f}/{v['qdist_mae']:.4f}"
                     f"  vdist {v['vdist']:.4f}/{v['vdist_mae']:.4f}" if "qdist" in v else "")
                  + (f"  lvq {v['lv_q']:.4f}/{v['lv_mae']:.4f}" if "lv_q" in v else "")
                  + (f"  fsq {v['fs_q']:.4f}/{v['fs_mae']:.4f}" if "fs_q" in v else "")
                  + (f"  dhq {v['dh_q']:.4f}/{v['dh_mae']:.4f}" if "dh_q" in v else "")
                  + (f"  stq {v['st_q']:.4f}/{v['st_mae']:.4f}" if "st_q" in v else "")
                  + f"  value {v['value']:.4f}  "
                  f"{rate:.0f} pos/s  grad {gnorm:.2f}  "
                  f"av {len(av)/1e9:.2f}B  {hours:.2f}h", flush=True)
            if el / 3600 > MAX_HOURS:
                print("time cap reached", flush=True)
                break
        if it == ITERS:
            if it % VAL_EVERY != 0 and it != start:
                # the run ends between validations, so keep what was trained since the last save
                save_atomic({"model": model.state_dict(), "optim": opt.state_dict(), "iter": it,
                             "hist": hist, "best": best,
                             "trained_hours": trained + (time.time() - t0) / 3600,
                             "att_since": att_since, "dist_since": dist_since,
                             "mlh_since": mlh_since, "yardstick": YARDSTICK,
                             "data_seed": data_seed, "scaler": scaler_state()}, LAST)
            break
        # The resume point between validations. It only copies the weights off the GPU, so it costs
        # seconds. The heartbeat line below is cheap proof of life between validations.
        if SAVE_EVERY and it % SAVE_EVERY == 0 and it != start and it % VAL_EVERY != 0:
            save_atomic({"model": model.state_dict(), "optim": opt.state_dict(), "iter": it,
                         "hist": hist, "best": best,
                         "trained_hours": trained + (time.time() - t0) / 3600,
                         "att_since": att_since, "dist_since": dist_since, "mlh_since": mlh_since,
                         "yardstick": YARDSTICK, "data_seed": data_seed, "scaler": scaler_state()}, LAST)
        if HEARTBEAT and it % HEARTBEAT == 0 and it != start:
            print(f"    {it:,}  {(it-start)*(AV_BSZ+POL_BSZ)/max(time.time()-t0,1):.0f} pos/s  "
                  f"grad {gnorm:.1f}  {trained + (time.time()-t0)/3600:.2f}h", flush=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(it, trained + (time.time() - t0) / 3600)
        if ATT:
            model.att_detach = (it - att_since) < ATT_DETACH
        if DIST:
            model.dist_detach = (it - dist_since) < DIST_DETACH
        if MLH:
            model.mlh_detach = (it - mlh_since) < MLH_DETACH
        # Standing aside for the release gate (aside.py) returns at once unless clock_gate.py holds the flag.
        # t0 moves on by the wait, so the trained hours, the pace and the history leave it out.
        if aside is not None:
            t0 += aside.wait(it, lambda m: print(m, flush=True))
        # Give the GPU back while somebody is playing. A live game's few milliseconds of work would queue
        # behind a training step. The board touches PLAYING_FILE on every real request, and while that is
        # fresh the trainer pauses a few tens of milliseconds between steps. The os.stat itself is free.
        if YIELD_MS:
            try:
                playing = time.time() - os.stat(PLAYING_FILE).st_mtime < YIELD_FOR
            except OSError:
                playing = False
            if playing:
                time.sleep(YIELD_MS / 1000.0)
        loss, _ = step_batch(fwd, used, av, pol, dp=dp, mlh=mlh, lv=lv, fs=fs, dh=dh, st=st)
        opt.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                     # true gradients before the clip, so 1.0 still means 1.0
        else:
            loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(gn) or gn > 50.0:
            skipped += 1
            opt.zero_grad()
            if scaler is not None:
                scaler.update()                      # an overflow lowers the scale, without this it skips forever
            if skipped % 20 == 1:
                print(f"  skipped a batch at it {it:,}, grad norm {gn:.1f} ({skipped} so far"
                      f"{f', loss scale {scaler.get_scale():g}' if scaler is not None else ''})", flush=True)
            continue
        gnorm = 0.98 * gnorm + 0.02 * gn
        # The optimiser steps itself, never scaler.step(opt). The gradients are already unscaled and proved finite, and
        # update() needs only the inf check unscale_ recorded. scaler.step would hand a fused optimiser found_inf, but a
        # resumed run's older param groups come back as fused=None and step on the multi-tensor path, which asserts
        # found_inf is None.
        opt.step()
        if scaler is not None:
            scaler.update()
        recentre_policy(model)

    print(f"\nDONE {TAG}  best {best:.4f}  {(time.time()-t0)/3600:.2f}h", flush=True)
    av.close(); avv.close(); pol.close(); polv.close()
