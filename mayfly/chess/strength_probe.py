r"""strength_probe measures the network's strength by what its moves do, not by agreement with the teacher.

Everything here plays the bare network with no search, the policy head's top legal move with the head and
castling rule the version is sealed with (as mcts.policy_move and the ladder do). Three measures.

  regret    3,000 held-out route 1 positions (MayFly's own self-play) where Stockfish scored every legal move.
            Reports the mean win probability the network's move gives up against the best move, how often it
            gives up 15 points or more (a blunder by Lichess's definition), and how often it plays a best move.
            The action-value head's own pick is reported too.
  puzzles   Lichess puzzle rating on 2,503 settled puzzles (100 per 100-point band), fitted as in
            lichess_puzzles.py.
  games     Score against full-strength Stockfish at a fixed node count. Skill Level and UCI_Elo make Stockfish
            blunder at random, and fixed nodes keep the opponent the same however busy the machine is. 50
            openings from both sides, at most 400 plies, then Stockfish adjudicates at 300 cp. The Elo gap is
            only against that opponent.

About five minutes per checkpoint on a 16 GB GPU. The live checkpoint is skipped if it has not moved since its
last row.

  python chess/strength_probe.py build_regret          build the held-out positions (once)
  python chess/strength_probe.py score [CKPT ...]      append a row per checkpoint to SP_OUT (default live)
  env: SP_NODES=1000,4000  SP_GAMES=100  SP_WORKERS=16  SP_SKIP=regret,puzzles,games
       SP_MIN_FREE_GB=2.5  SP_PUZZLES  SP_REGRET  SP_OUT  SP_LIVE  SP_DEVICE  SHARDS
"""
import glob
import json
import math
import os
import queue
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402
E = os.environ.get
SHARDS = E("SHARDS", paths.under("shards"))
REGRET = E("SP_REGRET", paths.under("strength", "regret_positions.npz"))
PUZZLES = E("SP_PUZZLES", os.path.join(HERE, "strength_puzzles.json"))
OUT = E("SP_OUT", paths.under("ckpt", "strength_mayfly.jsonl"))
LIVE = E("SP_LIVE", paths.under("ckpt", "ckpt_mayfly_last.pt"))
VDIR = E("VERSION_DIR", paths.under("versions"))
NODES = [int(x) for x in E("SP_NODES", "1000,4000").split(",") if x]   # v5 scored 46.5% and 17% against these
GAMES = int(E("SP_GAMES", 100))
WORKERS = int(E("SP_WORKERS", 16))
SKIP = set(filter(None, E("SP_SKIP", "").split(",")))
MAX_PLIES, ADJ_CP, ADJ_NODES = 400, 300, 1_000_000
BLUNDER = 0.15            # win probability, which is Lichess's 0.3 of win chance
POSITIONS = int(E("SP_POSITIONS", 3000))
MIN_FREE_GB = float(E("SP_MIN_FREE_GB", 2.5))


# ---------------------------------------------------------------------------------------------- checkpoints
def load_any(path, dev):
    """Load any MayFly checkpoint (version file, trainer checkpoint, cooled export) as an eval-mode ChessNet.
    Loads on the CPU first since the trainer's file also holds the optimizer state."""
    import torch
    from chess_net import ChessNet
    from mcts import versions_on_disk
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd, c = dict(ck["model"]), ck.get("cfg", {})
    m = ChessNet(d=c.get("d", 1024), nhead=c.get("nhead", 16), layers=c.get("layers", 16),
                 dim_ff=int(c.get("ff_mult", 2.0) * c.get("d", 1024)),
                 qhead=bool(c.get("qhead", "qvalue.weight" in sd)), att=bool(c.get("att", "att.q.weight" in sd)),
                 dist=int(c.get("dist", 64 if "vdist.weight" in sd else 0)),
                 mlh=bool(c.get("mlh", "mleft.0.weight" in sd)))
    # older checkpoints have a smaller move space, so copy what overlaps (as mcts.load does)
    cur = m.state_dict()
    for k, v in list(sd.items()):
        if k in cur and cur[k].shape != v.shape:
            t = cur[k].clone()
            cut = tuple(slice(0, min(a, b)) for a, b in zip(t.shape, v.shape))
            t[cut] = v[cut]
            sd[k] = t
    m.load_state_dict(sd, strict=False)
    m = m.to(dev).eval()
    meta = {"iter": int(ck.get("iter", 0)), "hours": ck.get("hours", ck.get("trained_hours")),
            "version": ck.get("version"), "cooled": bool(ck.get("cooled")) or "cooled" in os.path.basename(path)}
    # version files don't always say which version they are, so check versions.json
    rec = next((r for r in versions_on_disk(VDIR) if r.get("file") == os.path.basename(path)), {})
    if rec and os.path.normcase(os.path.dirname(os.path.abspath(path))) == os.path.normcase(os.path.abspath(VDIR)):
        meta["version"] = meta["version"] or rec.get("version")
        meta["cooled"] = meta["cooled"] or bool(rec.get("cooled"))
        meta["hours"] = meta["hours"] or rec.get("hours")
        meta["iter"] = meta["iter"] or int(rec.get("iter", 0))
    del ck, sd
    return m, meta


def seal_for(path, meta):
    """Policy head and castling rule from the version's seal. Unreleased checkpoints use v5's, cls and king."""
    from mcts import pinned_search
    s = pinned_search(os.path.basename(path), VDIR) if meta.get("version") else {}
    return {"policy_head": s.get("policy_head", "cls"), "castle_prior": s.get("castle_prior", "king")}


def label_for(path, meta):
    if meta.get("version"):
        return f"v{meta['version']}"
    b = os.path.basename(path).replace(".pt", "")
    known = {"ckpt_mayfly_last": "live", "ckpt_mayfly": "best-of"}
    return known.get(b) or f"{os.path.basename(os.path.dirname(path))}/{b}"


# --------------------------------------------------------------------------------------------------- regret
def legal_count(ids, cast, ep):
    """Number of legal moves for a position in the network's encoding (side to move shown as White).
    Mirroring a position doesn't change its legal moves, so the board is rebuilt in that frame."""
    import chess
    b = chess.Board(None)
    b.turn = chess.WHITE
    for s in range(64):
        t = int(ids[s])
        if t:
            b.set_piece_at(s, chess.Piece((t - 1) % 6 + 1, chess.WHITE if t <= 6 else chess.BLACK))
    b.castling_rights = ((chess.BB_H1 if cast[0] else 0) | (chess.BB_A1 if cast[1] else 0)
                         | (chess.BB_H8 if cast[2] else 0) | (chess.BB_A8 if cast[3] else 0))
    b.castling_rights = b.clean_castling_rights()
    b.ep_square = int(ep) - 1 if ep else None
    return b.legal_moves.count()


def build_regret():
    """Collect held-out route 1 positions where Stockfish scored every legal move.

    Uses the trainer's own split (hash of the position), so none of these were trained on. About half of route 1
    comes from the Kaggle labeller's top lines (5 or 10 moves plus MayFly's own), where the network's pick among
    the scored moves is always decent, so only positions with every legal move scored are kept. ChessBench's shards
    were shuffled when packed and only have about two moves per position, so they're not used."""
    from data_av import board_hash_av
    from pack_av import DT_AV, batch_tensors_av
    keep, hashes = [], []
    for f in sorted(glob.glob(os.path.join(SHARDS, "dp_*.bin"))):
        mm = np.memmap(f, DT_AV, mode="r")
        for a in range(0, len(mm), 1_000_000):
            rows = np.array(mm[a:a + 1_000_000])
            h = board_hash_av(rows)
            v = (h % np.uint64(64)) == 0
            keep.append(rows[v])
            hashes.append(h[v])
    rows, h = np.concatenate(keep), np.concatenate(hashes)
    order = np.argsort(h, kind="stable")
    rows, h = rows[order], h[order]
    starts = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
    ends = np.r_[starts[1:], len(h)]
    complete = []
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        i_, c_, mv, q, hm_, ep_ = batch_tensors_av(rows[s:e])
        merged = {}
        for m_, q_ in zip(mv.tolist(), q.tolist()):
            merged.setdefault(m_, []).append(q_)
        if len(merged) >= 2 and len(merged) == legal_count(i_[0], c_[0], ep_[0]):
            complete.append((i_[0], c_[0], hm_[0], ep_[0], merged))
    # an even spread in hash order, the same every build
    pick = np.linspace(0, len(complete) - 1, min(POSITIONS, len(complete))).astype(int)
    ids, cast, hm, ep, off, mvs, qs = [], [], [], [], [0], [], []
    for gi in pick:
        i_, c_, h_, e_, merged = complete[gi]
        ids.append(i_), cast.append(c_), hm.append(h_), ep.append(e_)
        for m_, ql in merged.items():
            mvs.append(m_), qs.append(sum(ql) / len(ql))
        off.append(len(mvs))
    os.makedirs(os.path.dirname(REGRET), exist_ok=True)
    np.savez_compressed(REGRET, dp_ids=np.array(ids, np.uint8), dp_cast=np.array(cast, np.float32),
                        dp_hm=np.array(hm, np.int64), dp_ep=np.array(ep, np.int64), dp_off=np.array(off, np.int64),
                        dp_mv=np.array(mvs, np.int64), dp_q=np.array(qs, np.float32))
    print(f"{len(rows):,} held-out rows, {len(complete):,} positions with every legal move scored; kept {len(ids):,} "
          f"positions and {len(mvs):,} moves -> {REGRET}", flush=True)


def regret(model, seal, dev):
    """Mean win probability given up by the policy's move, blunder rate and best-move rate, and the same for the q head."""
    import torch
    d = np.load(REGRET)
    pk = "policy_att" if seal["policy_head"] == "att" else "policy"
    qk = "qvalue_att" if seal["policy_head"] == "att" else "qvalue"
    ids, cast, hm, ep = d["dp_ids"], d["dp_cast"], d["dp_hm"], d["dp_ep"]
    off, mv, q = d["dp_off"], d["dp_mv"], d["dp_q"]
    rp, rq = [], []
    for a in range(0, len(ids), 256):
        b = min(a + 256, len(ids))
        with torch.no_grad():
            out = model(torch.from_numpy(ids[a:b].astype(np.int64)).to(dev), torch.from_numpy(cast[a:b]).to(dev),
                        torch.from_numpy(hm[a:b]).to(dev), torch.from_numpy(ep[a:b]).to(dev))
        pol = out[pk if out.get(pk) is not None else "policy"].float().cpu().numpy()
        qv = out[qk].float().cpu().numpy() if out.get(qk) is not None else None
        for i in range(a, b):
            m_, q_ = mv[off[i]:off[i + 1]], q[off[i]:off[i + 1]]
            best = q_.max()
            rp.append(best - q_[int(np.argmax(pol[i - a, m_]))])
            if qv is not None:
                rq.append(best - q_[int(np.argmax(qv[i - a, m_]))])
    rp, rq = np.array(rp), np.array(rq)
    res = {"n": len(rp), "regret": round(float(rp.mean()), 5), "blunders": round(float((rp >= BLUNDER).mean()), 4),
           "best": round(float((rp <= 0.005).mean()), 4)}
    if len(rq):
        res.update({"q_regret": round(float(rq.mean()), 5), "q_blunders": round(float((rq >= BLUNDER).mean()), 4),
                    "q_best": round(float((rq <= 0.005).mean()), 4)})
    return {"dp": res}


# -------------------------------------------------------------------------------------------------- puzzles
def puzzles(model, seal, dev):
    """Puzzle rating, solving with lichess_puzzles.py's rule (each solution move in turn, or any mate in one).
    64 puzzles run at once through NetServer."""
    import chess
    from lichess_puzzles import fit, LO
    suite = json.load(open(PUZZLES, encoding="utf-8"))["puzzles"]
    net, solved_each, todo, t0 = NetServer(model, dev, seal), [None] * len(suite), queue.Queue(), time.time()
    for i in range(len(suite)):
        todo.put(i)

    def solver():
        while True:
            try:
                i = todo.get_nowait()
            except queue.Empty:
                return
            p = suite[i]
            board = chess.Board(p["fen"])
            board.push_uci(p["moves"][0])
            solved = True
            for j in range(1, len(p["moves"]), 2):
                mv = net.move(board)
                if mv == chess.Move.from_uci(p["moves"][j]):
                    board.push(mv)
                    if j + 1 < len(p["moves"]):
                        board.push_uci(p["moves"][j + 1])
                    continue
                if mv is not None and mv in board.legal_moves:
                    board.push(mv)
                    solved = board.is_checkmate()
                else:
                    solved = False
                break
            solved_each[i] = solved

    ts = [threading.Thread(target=solver, daemon=True) for _ in range(64)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    results = [(p["rating"], bool(s)) for p, s in zip(suite, solved_each)]
    rating, se, bound = fit(results)
    if not bound and rating < LO:
        rating, se, bound = 0.0, None, True
    k = sum(s for _, s in results)
    return {"n": len(results), "solved": k, "rating": round(rating), "se": None if se is None else round(se, 1),
            "bound": bound, "seconds": round(time.time() - t0)}


# ---------------------------------------------------------------------------------------------------- games
class NetServer:
    """Batches the bare network's moves for many threads. Each thread asks for a move and waits, and one thread runs
    every waiting position through the network together. Same moves as policy_move (checked on 418 positions)."""

    def __init__(self, model, dev, seal):
        self.model, self.dev, self.seal, self.q = model, dev, seal, queue.Queue()
        threading.Thread(target=self._serve, daemon=True).start()

    def move(self, board):
        slot = [board, None, threading.Event()]
        self.q.put(slot)
        slot[2].wait()
        if isinstance(slot[1], Exception):
            raise slot[1]
        return slot[1]

    def _serve(self):
        import torch
        from encode import board_to_ids, clock_and_ep
        from mcts import policy_index
        key = "policy_att" if self.seal["policy_head"] == "att" else "policy"
        while True:
            batch = [self.q.get()]
            try:
                while len(batch) < 64:
                    batch.append(self.q.get(timeout=0.002))
            except queue.Empty:
                pass
            try:
                enc = []
                for b, _, _ in batch:
                    ids, cast, flip = board_to_ids(b)
                    hm, ep = clock_and_ep(b, flip)
                    enc.append((ids, cast, hm, ep, flip))
                wd = next(self.model.parameters()).dtype
                with torch.no_grad():
                    out = self.model(torch.tensor([e[0] for e in enc], device=self.dev),
                                     torch.tensor([e[1] for e in enc], device=self.dev, dtype=wd),
                                     torch.tensor([e[2] for e in enc], device=self.dev),
                                     torch.tensor([e[3] for e in enc], device=self.dev))
                pol = out[key if out.get(key) is not None else "policy"].float()
                for i, (slot, e) in enumerate(zip(batch, enc)):
                    legal = list(slot[0].legal_moves)
                    idx = torch.from_numpy(policy_index(slot[0], legal, e[4], self.seal["castle_prior"])).to(self.dev)
                    slot[1] = legal[int(pol[i][idx].argmax())]
            except Exception as ex:
                for slot in batch:
                    slot[1] = ex
            for slot in batch:
                slot[2].set()


def elo(p, n):
    """Elo difference for score p over n games, clamped so a clean sweep stays finite."""
    p = min(max(p, 0.5 / n), 1 - 0.5 / n)
    return 400 * math.log10(p / (1 - p))


def games(model, seal, dev):
    """Games against Stockfish at each node count in NODES. One Stockfish (1 thread, hash cleared each game) per
    worker thread, and the network is shared through NetServer."""
    import chess
    import chess.engine
    book = [(o["name"], o["fen"]) for o in json.load(open(os.path.join(HERE, "round_robin_book.json"),
                                                          encoding="utf-8"))["openings"]]
    sf = open(os.path.join(HERE, "stockfish_path.txt")).read().strip()
    tasks = queue.Queue()
    for nodes in NODES:
        for g in range(GAMES):
            tasks.put((nodes, g))
    net, rows, errors = NetServer(model, dev, seal), {n: [] for n in NODES}, []

    def worker():
        eng = chess.engine.SimpleEngine.popen_uci(sf)
        try:
            eng.configure({"Threads": 1, "Hash": 16})
            while True:
                try:
                    nodes, g = tasks.get_nowait()
                except queue.Empty:
                    return
                name, fen = book[(g // 2) % len(book)]
                mf_white, board, game, plies = g % 2 == 0, chess.Board(fen), object(), 0
                while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
                    if (board.turn == chess.WHITE) == mf_white:
                        mv = net.move(board)
                    else:
                        mv = eng.play(board, chess.engine.Limit(nodes=nodes), game=game).move
                    board.push(mv)
                    plies += 1
                if board.is_game_over(claim_draw=True):
                    ws, how = {"1-0": 1.0, "0-1": 0.0}.get(board.result(claim_draw=True), 0.5), "played"
                else:
                    sc = eng.analyse(board, chess.engine.Limit(nodes=ADJ_NODES), game=game)["score"].white()
                    cp = sc.score(mate_score=100000)
                    ws, how = (1.0 if cp >= ADJ_CP else 0.0 if cp <= -ADJ_CP else 0.5), "adjudicated"
                rows[nodes].append((g, ws if mf_white else 1.0 - ws, plies, how))
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
        finally:
            eng.quit()

    t0 = time.time()
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errors:
        raise RuntimeError(f"a game worker failed: {errors[0]}")
    res = {}
    for nodes, rs in rows.items():
        s = [x[1] for x in rs]
        n, mean = len(s), sum(s) / max(1, len(s))
        half = 1.96 * (sum((x - mean) ** 2 for x in s) / max(1, n - 1)) ** 0.5 / max(1, n) ** 0.5
        res[str(nodes)] = {"n": n, "score": round(mean, 4), "w": s.count(1.0), "d": s.count(0.5), "l": s.count(0.0),
                           "elo": round(elo(mean, n)), "elo_lo": round(elo(mean - half, n)),
                           "elo_hi": round(elo(mean + half, n)),
                           "adjudicated": sum(x[3] == "adjudicated" for x in rs),
                           "plies": round(sum(x[2] for x in rs) / max(1, n))}
    res["seconds"] = round(time.time() - t0)
    return res


# ---------------------------------------------------------------------------------------------------- score
def measured(label):
    """Iterations already measured under this label."""
    try:
        return {json.loads(l).get("iter") for l in open(OUT, encoding="utf-8")
                if l.strip() and json.loads(l).get("label") == label}
    except OSError:
        return set()


def score(paths):
    import torch
    dev = E("SP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda":
        # Don't squeeze onto a full card. Past its memory Windows spills into system RAM and slows everything
        # else on it. nvidia-smi's dedicated memory is the number that matters, since torch's mem_get_info can
        # report far more free memory than there is.
        try:
            import subprocess
            used, total = (float(x) for x in subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30).stdout.split("\n")[0].split(","))
            free = (total - used) / 1024
        except Exception:
            free = torch.cuda.mem_get_info()[0] / 2 ** 30
        if free < MIN_FREE_GB:
            print(f"only {free:.1f} GB free on the card (need {MIN_FREE_GB}); skipping this run", flush=True)
            return
    for path in paths or [LIVE]:
        t0 = time.time()
        model, meta = load_any(path, dev)
        seal, label = seal_for(path, meta), label_for(path, meta)
        if label == "live" and meta["iter"] in measured("live"):
            print(f"live: iteration {meta['iter']:,} already measured, skipping", flush=True)
            del model
            continue
        row = {"when": time.strftime("%Y-%m-%d %H:%M"), "label": label, "path": path, **meta, "seal": seal,
               "nodes": NODES, "games_a_cell": GAMES}
        print(f"{label}: iteration {meta['iter']:,}, {meta['hours']} h, seal {seal}", flush=True)
        if "regret" not in SKIP:
            row["regret"] = regret(model, seal, dev)
            print(f"  regret {row['regret']}", flush=True)
        if "puzzles" not in SKIP:
            row["puzzles"] = puzzles(model, seal, dev)
            print(f"  puzzles {row['puzzles']}", flush=True)
        if "games" not in SKIP:
            row["games"] = games(model, seal, dev)
            print(f"  games {row['games']}", flush=True)
        row["seconds"] = round(time.time() - t0)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "score") == "build_regret":
        build_regret()
    else:
        score(sys.argv[2:])
