"""
train_ttt is the outer training loop for the game memory, differentiating through the inner updates.

The claim under test is that predicting an opponent's move gets easier as a game goes on because of
the memory, not because late positions are easier. Two controls check that.

  memory off      same net and games, no fast weight. A late-game gain here belongs to chess.
  scrambled       the memory is fed the plies of a different game. A gain that survives this is
                  generic structure, not the opponent.

The inner loss needs no labels. It reconstructs the position that actually followed, which at
inference is just the move the opponent made. A self-chosen objective tends to collapse to one that
is trivially easy to satisfy, and this one cannot.

  python chess/train_ttt.py [STEPS] [GAMES_PER_BATCH]
  env: D=256 NHEAD=16 LAYERS=4 FF=512 LR=1e-4 BPTT=24 MAXP=96 WARM=ckpt_long_ctrl.pt
       NOMEM=1 to train the memory-off baseline   TAG=ttt   VAL_EVERY=200
"""
import os, sys, glob, json, time, random
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from chess_ttt import ChessTTT

E = os.environ.get
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
BSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 8
D, NHEAD = int(E("D", 256)), int(E("NHEAD", 16))
LAYERS, FF = int(E("LAYERS", 4)), int(E("FF", 512))
LR = float(E("LR", 1e-4))
BPTT = int(E("BPTT", 24))
MAXP = int(E("MAXP", 96))
NOMEM = E("NOMEM", "0") == "1"
TAG = E("TAG", "ttt_nomem" if NOMEM else "ttt")
VAL_EVERY = int(E("VAL_EVERY", 200))
WARM = E("WARM", "ckpt_long_ctrl.pt")
GAMES = os.path.join(HERE, "ttt_games")
HOLDOUT = 10                      # every 10th game per player is validation
BUCKETS = [(0, 20), (20, 40), (40, 70), (70, 999)]
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_players():
    """One entry per player file. Games stay grouped, because the grouping is the experiment."""
    out = []
    for fp in sorted(glob.glob(os.path.join(GAMES, "*.npz"))):
        z = np.load(fp)
        out.append({"who": os.path.splitext(os.path.basename(fp))[0], "ids": z["ids"],
                    "cast": z["cast"], "move": z["move"], "stm": z["stm"], "plies": z["plies"]})
    if not out:
        raise SystemExit(f"no game files in {GAMES}, run gen_ttt_games.py first")
    return out


def sample(players, rng, want_val, bsz):
    """A batch of whole games, cropped to MAXP. Train and validation never share a game."""
    ids_b, cast_b, mv_b, stm_b = [], [], [], []
    while len(ids_b) < bsz:
        p = players[rng.randrange(len(players))]
        g = rng.randrange(len(p["plies"]))
        if (g % HOLDOUT == 0) != want_val:
            continue
        n = min(int(p["plies"][g]), MAXP)
        ids_b.append(p["ids"][g, :n]); cast_b.append(p["cast"][g, :n])
        mv_b.append(p["move"][g, :n]); stm_b.append(p["stm"][g, :n])
    P = max(len(x) for x in ids_b)
    B = len(ids_b)
    ids = np.zeros((B, P, 64), np.uint8); cast = np.zeros((B, P, 4), np.float32)
    mv = np.full((B, P), -1, np.int64); stm = np.zeros((B, P), np.uint8)
    for i in range(B):
        n = len(ids_b[i])
        ids[i, :n] = ids_b[i]; cast[i, :n] = cast_b[i]
        mv[i, :n] = mv_b[i]; stm[i, :n] = stm_b[i]
    t = lambda a, d: torch.as_tensor(a, dtype=d, device=device)
    return t(ids, torch.long), t(cast, torch.float), t(mv, torch.long), t(stm, torch.bool)


def walk(model, ids, cast, use_mem, scramble=False):
    """Per-ply policy logits. use_mem False runs the plain net, which is the ablation.

    scramble predicts these positions using a memory built from a different game in the batch.
    The positions scored are the same and only the history is wrong, so a late-game advantage that
    survives it was never about the opponent.
    """
    if not use_mem:
        return [model(ids[:, t], cast[:, t]) for t in range(ids.shape[1])]
    if not scramble:
        outs, _ = model.play_game(ids, cast, bptt=BPTT)
        return outs

    B, P = ids.shape[0], ids.shape[1]
    perm = torch.roll(torch.arange(B, device=ids.device), 1)   # every game gets its neighbour's
    hs = [model.encode(ids[:, t], cast[:, t]) for t in range(P)]
    W = model.mem.start(B, ids.device)
    outs = []
    for t in range(P):
        outs.append(model.heads(model.mem.read(hs[t], W)))
        if t + 1 < P:
            W, _ = model.mem.update(hs[t][perm], hs[t + 1][perm], W)
    return outs


def ply_loss(outs, mv, stm, only_tracked=True):
    """Cross-entropy on the move actually played, and per-ply correctness for the buckets."""
    losses, correct, count = [], [], []
    for t, o in enumerate(outs):
        m = (mv[:, t] >= 0)
        if only_tracked:
            m = m & stm[:, t]
        if not m.any():
            losses.append(None); correct.append(None); count.append(0)
            continue
        lg, tg = o["policy"][m], mv[:, t][m]
        losses.append(F.cross_entropy(lg, tg))
        correct.append((lg.argmax(-1) == tg).float().sum())
        count.append(int(m.sum()))
    n = sum(count)
    tot = sum(l * c for l, c in zip(losses, count) if l is not None) / max(n, 1)
    return tot, correct, count


def bucket_acc(correct, count):
    out = {}
    for lo, hi in BUCKETS:
        c = sum(float(correct[t]) for t in range(len(count)) if lo <= t < hi and count[t])
        n = sum(count[t] for t in range(len(count)) if lo <= t < hi)
        out[f"{lo}-{hi if hi < 999 else ''}"] = (c / n) if n else float("nan")
    return out


@torch.no_grad()
def validate(model, players, rng, n_batches):
    model.eval()
    acc = {k: {b: [0.0, 0] for b in ("0-20", "20-40", "40-70", "70-")} for k in
           ("mem", "nomem", "scram")}
    loss = {k: [] for k in acc}
    for _ in range(n_batches):
        ids, cast, mv, stm = sample(players, rng, True, BSZ)
        for key, use, scr in (("mem", not NOMEM, False), ("nomem", False, False),
                              ("scram", not NOMEM, True)):
            if key != "nomem" and NOMEM:
                continue
            outs = walk(model, ids, cast, use, scr)
            l, correct, count = ply_loss(outs, mv, stm)
            loss[key].append(float(l))
            for t in range(len(count)):
                if not count[t]:
                    continue
                for lo, hi in BUCKETS:
                    if lo <= t < hi:
                        b = f"{lo}-{hi if hi < 999 else ''}"
                        acc[key][b][0] += float(correct[t]); acc[key][b][1] += count[t]
    model.train()
    res = {}
    for k in acc:
        if not loss[k]:
            continue
        res[k] = {"loss": sum(loss[k]) / len(loss[k]),
                  "acc": {b: (v[0] / v[1] if v[1] else float("nan")) for b, v in acc[k].items()}}
    return res


if __name__ == "__main__":
    torch.manual_seed(0)
    players = load_players()
    ngames = sum(len(p["plies"]) for p in players)
    print(f"{len(players)} players, {ngames} games, {sum(int(p['plies'].sum()) for p in players)} plies",
          flush=True)

    model = ChessTTT(d=D, nhead=NHEAD, layers=LAYERS, dim_ff=FF).to(device)
    wp = os.path.join(HERE, WARM)
    if os.path.exists(wp):
        v = model.load_base(wp, map_location=device)
        print(f"warm started from {WARM} (policy {v.get('policy', float('nan')):.4f})", flush=True)
    else:
        print(f"no warm start, {WARM} not found", flush=True)
    print(f"{'MEMORY OFF (baseline)' if NOMEM else 'MEMORY ON'} | "
          f"{model.num_params()/1e6:.3f}M params | bsz {BSZ} games | bptt {BPTT} | {device}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    rng, vrng = random.Random(1234), random.Random(99)
    hist, t0 = [], time.time()

    for it in range(STEPS + 1):
        if it % VAL_EVERY == 0:
            res = validate(model, players, random.Random(99), 6)
            rec = {"it": it, "t": time.time() - t0, **res}
            hist.append(rec)
            json.dump(hist, open(os.path.join(HERE, f"hist_{TAG}.json"), "w"), indent=1)
            torch.save({"model": model.state_dict(), "it": it, "nomem": NOMEM,
                        "cfg": {"d": D, "nhead": NHEAD, "layers": LAYERS, "ff": FF}},
                       os.path.join(HERE, f"ckpt_{TAG}.pt"))
            parts = []
            for k in ("mem", "nomem", "scram"):
                if k in res:
                    a = res[k]["acc"]
                    parts.append(f"{k} {res[k]['loss']:.4f} [" +
                                 " ".join(f"{a[b]*100:.1f}" for b in ("0-20", "20-40", "40-70", "70-")) + "]")
            print(f"  it {it:6d}  " + "  ".join(parts) + f"   {time.time()-t0:6.0f}s", flush=True)
        if it == STEPS:
            break
        ids, cast, mv, stm = sample(players, rng, False, BSZ)
        outs = walk(model, ids, cast, use_mem=not NOMEM)
        loss, _, _ = ply_loss(outs, mv, stm)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"\nDONE {TAG}  {time.time()-t0:.0f}s  history hist_{TAG}.json", flush=True)
