r"""selfplay_remote runs self-play with the network served from one accelerator to many searches at once.

This puts a TPU (or an otherwise idle GPU, such as Kaggle's T4s) to work running MayFly's own network for
self-play. selfplay.py runs one game at a time with the network in its own process. Here the two are split.

  servers   one process owns the accelerator and runs the network on big padded batches made of many searches'
            requests. "jax" drives every TPU chip from one process with the batch sharded across them
            (chessnet_jax.py), "torch" is one process per CUDA card (float16 on a card without bfloat16),
            "onnx" runs volunteer/export_net.py's file on DirectML (the volunteer app) and "cpu" is for tests.
  workers   CPU processes, one game each by default. SP_THREADS above 1 runs that many games a process on
            threads and switches the compiled tree walk off (MCTS_NATIVE=0), since two trees searching at once
            in one process stall it. A worker's tree is selfplay.py's, with the version's network and sealed
            search and selfplay.play's temperature, root noise and mixed starts. Only the forward pass is
            elsewhere. RemoteMCTS sends _prepare's arrays to the server and reads back what
            MCTS._dispatch_one_copy computes on the card, so the tree code is the same.

The pass runs at the server's precision on batches of other games' positions, so its numbers can differ from a
local search in the last bits. For self-play that is fine (selfplay_remote_check compares the moves).

  python chess/selfplay_remote.py
  env: SP_BACKEND=auto (jax with a TPU, torch with a CUDA card, else cpu)  SP_SECONDS=3600  SP_WORKERS=cores
       SP_THREADS=1  SP_OUT=./sp_out  SP_CKPT (default mayfly_v5.pt in paths.VERSIONS, or SP_NPZ, chessnet_jax's
       weights)  VERSION=mayfly_v5.pt  VERSION_DIR (default paths.VERSIONS)  SP_STARTS= (a file of FENs for the
       mixed starts, else selfplay's corpus shards)  SP_BUCKETS=  SP_SEED=0  and selfplay's VISITS (1600 here),
       TEMP_PLIES, OPENING_FRAC, MAX_PLIES, BATCH, ROOT_NOISE, NOISE_ALPHA
"""
import os
import sys
import time
import json
import pickle
import random
import threading
import multiprocessing as mp
from multiprocessing.connection import wait

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
E = os.environ.get
W_FIX = 224                    # every legal-move row padded to this (218 is the most any position has)


# ---------------------------------------------------------------- the wire

def pack_request(prep):
    """_prepare's arrays, kept small, with the board tokens and clocks as bytes and the move indices as uint16."""
    I = prep["I"]
    Ip = None if prep["Ip"] is I else prep["Ip"].astype(np.uint16)
    return pickle.dumps((np.asarray(prep["ids"], np.uint8), np.asarray(prep["cast"], np.float32),
                         np.asarray(prep["hm"], np.uint8), np.asarray(prep["ep"], np.uint8),
                         I.astype(np.uint16), Ip, np.asarray(prep["M"], bool)), 5)


# ---------------------------------------------------------------- the servers

def load_weights(ckpt=None, npz=None):
    """(cfg, numpy params) from a checkpoint (needs torch) or chessnet_jax's npz (cfg inside as JSON)."""
    if npz:
        z = np.load(npz, allow_pickle=False)
        cfg = json.loads(str(z["__cfg__"]))
        flat = {k: z[k] for k in z.files if k != "__cfg__"}
        p = {k: v for k, v in flat.items() if not k.startswith("layers.")}
        n = int(cfg["layers"])
        p["layers"] = [{k.split(".", 2)[2]: flat[k] for k in flat if k.startswith(f"layers.{i}.")} for i in range(n)]
        return cfg, p
    import torch
    from chessnet_jax import params_from_torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    return ck["cfg"], params_from_torch(ck["model"], ck["cfg"])


def save_npz(ckpt, out):
    """chessnet_jax's weights (float32) and the cfg in one npz, so a TPU host needs no torch to load them."""
    cfg, p = load_weights(ckpt)
    flat = {k: v for k, v in p.items() if k != "layers"}
    for i, q in enumerate(p["layers"]):
        for k, v in q.items():
            flat[f"layers.{i}.{k}"] = v
    flat["__cfg__"] = np.array(json.dumps(cfg))
    np.savez(out, **flat)


class Merger:
    """Requests from many connections into one padded batch, and the answers back to each."""

    def __init__(self, buckets):
        self.buckets = sorted(buckets)

    def take(self, conns, timeout, cap):
        """Every request ready within `timeout`, up to `cap` rows, as [(conn, arrays)]. A connection that closed is
        dropped from `conns`."""
        got, rows = [], 0
        for c in wait(conns, timeout):
            if rows >= cap:
                break
            try:
                a = pickle.loads(c.recv_bytes())
            except (EOFError, OSError):
                conns.remove(c)
                continue
            got.append((c, a))
            rows += len(a[0])
        return got

    def stack(self, got):
        """The batch padded to a bucket (ids, cast, hm, ep, I, Ip, M as numpy) and each request's row range."""
        n = sum(len(a[0]) for _, a in got)
        size = next((b for b in self.buckets if b >= n), None)
        if size is None:
            raise ValueError(f"{n} rows is more than the largest bucket {self.buckets[-1]}")
        ids = np.zeros((size, 64), np.int32)
        cast = np.zeros((size, 4), np.float32)
        hm = np.zeros(size, np.int32)
        ep = np.zeros(size, np.int32)
        I = np.zeros((size, W_FIX), np.int32)
        Ip = np.zeros((size, W_FIX), np.int32)
        M = np.zeros((size, W_FIX), bool)
        M[:, 0] = True                       # padding rows get one "legal" move, so their softmax is finite
        spans, r = [], 0
        for c, (a_ids, a_cast, a_hm, a_ep, a_I, a_Ip, a_M) in got:
            k, w = a_I.shape
            ids[r:r + k] = a_ids
            cast[r:r + k] = a_cast
            hm[r:r + k] = a_hm
            ep[r:r + k] = a_ep
            I[r:r + k, :w] = a_I
            Ip[r:r + k, :w] = a_I if a_Ip is None else a_Ip
            M[r:r + k] = False
            M[r:r + k, :w] = a_M
            spans.append((c, r, k, w))
            r += k
        return (ids, cast, hm, ep, I, Ip, M), spans, n

    @staticmethod
    def reply(spans, pg, qg, val):
        for c, r, k, w in spans:
            out = np.empty((k, 2 * w + 1), np.float32)
            out[:, :w] = pg[r:r + k, :w]
            out[:, w:2 * w] = qg[r:r + k, :w]
            out[:, 2 * w] = val[r:r + k]
            try:
                c.send_bytes(pickle.dumps(out, 5))
            except (BrokenPipeError, OSError):
                pass


def serve(backend, device, conns, weights, ready, stop, evals, buckets, log):
    """One server, the network on `device` answering `conns` until `stop` and every connection is gone."""
    try:
        _serve(backend, device, conns, weights, ready, stop, evals, buckets, log)
    except Exception as e:
        import traceback
        log.put(f"server {device}: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
        raise


def _serve(backend, device, conns, weights, ready, stop, evals, buckets, log):
    merger = Merger(buckets)
    cap = merger.buckets[-1] - 64             # leave room for one more request of up to 64 rows
    t0 = time.time()
    if backend == "onnx":
        _serve_onnx(device, conns, weights["onnx"], ready, evals, merger, cap, log, t0)
        return
    cfg, params = load_weights(**weights)
    if backend == "jax":
        import jax
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from chessnet_jax import make_forward
        devs = jax.devices()
        mesh = Mesh(np.array(devs), ("d",))
        rep, dat = NamedSharding(mesh, P()), NamedSharding(mesh, P("d"))
        params = jax.device_put(params, rep)
        fwd = jax.jit(make_forward(int(cfg["nhead"]), E("SP_DTYPE", "bfloat16")),
                      in_shardings=(rep,) + (dat,) * 7, out_shardings=(dat, dat, dat))
        for b in merger.buckets:                 # compile every bucket before the workers start
            if b % len(devs):
                raise ValueError(f"bucket {b} does not split over {len(devs)} devices")
            arrs =(np.zeros((b, 64), np.int32), np.zeros((b, 4), np.float32), np.zeros(b, np.int32),
                    np.zeros(b, np.int32), np.zeros((b, W_FIX), np.int32), np.zeros((b, W_FIX), np.int32),
                    np.ones((b, W_FIX), bool))
            jax.block_until_ready(fwd(params, *arrs))
        log.put(f"server jax: {len(devs)} device(s) ({devs[0].platform}, {devs[0].device_kind}), buckets "
                f"{merger.buckets} compiled in {time.time() - t0:.0f}s")
        ready.set()
        inflight = None
        while conns or inflight is not None:
            got = merger.take(conns, 0.0 if inflight is not None else 0.05, cap) if conns else []
            nxt = None
            if got:
                arrs, spans, n = merger.stack(got)
                nxt = (fwd(params, *arrs), spans, n)    # dispatched, runs while the last batch is answered
            if inflight is not None:
                (pg, qg, val), spans, n = inflight
                merger.reply(spans, np.asarray(pg), np.asarray(qg), np.asarray(val))
                with evals.get_lock():
                    evals.value += n
            inflight = nxt
            if not conns and inflight is None:
                break
        return
    import torch
    from chess_net import ChessNet
    dev = torch.device(device)
    if dev.type == "cuda":
        major = torch.cuda.get_device_capability(dev)[0]
        adt = torch.bfloat16 if major >= 8 else torch.float16
    else:
        adt = None
    net = _torch_net(cfg, params).to(dev).eval()
    log.put(f"server torch {device}: {'autocast ' + str(adt).split('.')[-1] if adt else 'float32'}, "
            f"buckets {merger.buckets}, loaded in {time.time() - t0:.0f}s")
    ready.set()
    with torch.inference_mode():
        while conns:
            got = merger.take(conns, 0.05, cap)
            if not got:
                continue
            (ids, cast, hm, ep, I, Ip, M), spans, n = merger.stack(got)
            T = lambda a, dt=torch.long: torch.from_numpy(a).to(dev, dt, non_blocking=True)  # noqa: E731
            It, Ipt, Mt = T(I), T(Ip), torch.from_numpy(M).to(dev)
            with torch.autocast(dev.type, adt, enabled=adt is not None):
                pol, qv = net(T(ids), T(cast, torch.float32), T(hm), T(ep))
            pg = torch.gather(pol.float(), 1, Ipt)
            qg = torch.gather(torch.sigmoid(qv.float()), 1, It)
            w = torch.softmax(pg.masked_fill(~Mt, float("-inf")), -1)
            val = (w * qg).sum(1) * 2.0 - 1.0
            merger.reply(spans, pg.cpu().numpy(), (qg * 2.0 - 1.0).cpu().numpy(), val.cpu().numpy())
            with evals.get_lock():
                evals.value += n


def _serve_onnx(device, conns, path, ready, evals, merger, cap, log, t0):
    """The volunteer app's server. volunteer/export_net.py's graph runs in ONNX Runtime, on DirectML (any Windows
    graphics card, device being its DXGI number) or on the CPU. The graph ends where the torch server's code does,
    so its three outputs are the reply as they come."""
    import onnxruntime as ort
    prov = [("DmlExecutionProvider", {"device_id": int(device)}), "CPUExecutionProvider"] \
        if device != "cpu" and "DmlExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(path, opts, providers=prov)
    kinds = {i.name: i.type for i in sess.get_inputs()}
    cast_dt = np.float16 if kinds.get("cast") == "tensor(float16)" else np.float32
    for b in merger.buckets:                 # one pass at every bucket before the workers start
        sess.run(None, {"ids": np.zeros((b, 64), np.int64), "cast": np.zeros((b, 4), cast_dt),
                        "hm": np.zeros(b, np.int64), "ep": np.zeros(b, np.int64),
                        "I": np.zeros((b, W_FIX), np.int64), "Ip": np.zeros((b, W_FIX), np.int64),
                        "M": np.ones((b, W_FIX), bool)})
    log.put(f"server onnx {sess.get_providers()[0]} device {device}: buckets {merger.buckets}, "
            f"ready in {time.time() - t0:.0f}s")
    ready.set()
    while conns:
        got = merger.take(conns, 0.05, cap)
        if not got:
            continue
        (ids, cast, hm, ep, I, Ip, M), spans, n = merger.stack(got)
        pg, qg, val = sess.run(None, {"ids": ids.astype(np.int64), "cast": cast.astype(cast_dt),
                                      "hm": hm.astype(np.int64), "ep": ep.astype(np.int64),
                                      "I": I.astype(np.int64), "Ip": Ip.astype(np.int64), "M": M})
        merger.reply(spans, pg, qg, val)
        with evals.get_lock():
            evals.value += n


def _torch_net(cfg, params):
    """ChessNet's trunk with the two heads a v5 search reads, from chessnet_jax's numpy params."""
    import torch
    import torch.nn as nn

    d, H = int(cfg["d"]), int(cfg["nhead"])
    ff = int(float(cfg.get("ff_mult", 2.0)) * d)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731

    class Lean(nn.Module):
        def __init__(self):
            super().__init__()
            self.piece = nn.Embedding(13, d)
            self.pos = nn.Embedding(64, d)
            self.hm = nn.Embedding(102, d)
            self.cast = nn.Linear(4, d)
            self.cls = nn.Parameter(torch.zeros(d))
            self.ep_mark = nn.Parameter(torch.zeros(d))
            layer = nn.TransformerEncoderLayer(d, H, ff, dropout=0.0, batch_first=True, activation="gelu",
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, int(cfg["layers"]), enable_nested_tensor=False)
            self.ln = nn.LayerNorm(d)
            self.policy = nn.Linear(d, params["pol_w"].shape[1])
            self.qvalue = nn.Linear(d, params["q_w"].shape[1])

        def forward(self, ids, cast, hm, ep):
            B = ids.shape[0]
            x = self.piece(ids) + self.pos.weight[None]
            has = (ep > 0).to(x.dtype)[:, None]
            onehot = torch.nn.functional.one_hot((ep - 1).clamp(min=0), 64).to(x.dtype) * has
            x = x + onehot[:, :, None] * self.ep_mark.to(x.dtype)
            cls = self.cls[None] + self.cast(cast) + self.hm(hm)
            z = self.enc(torch.cat([cls[:, None, :], x], 1))
            h = self.ln(z[:, 0])
            return self.policy(h), self.qvalue(h)

    m = Lean()
    with torch.no_grad():
        m.piece.weight.copy_(t(params["piece"])); m.pos.weight.copy_(t(params["pos"]))
        m.hm.weight.copy_(t(params["hm"])); m.cls.copy_(t(params["cls"])); m.ep_mark.copy_(t(params["ep_mark"]))
        m.cast.weight.copy_(t(params["cast_w"]).T); m.cast.bias.copy_(t(params["cast_b"]))
        m.ln.weight.copy_(t(params["ln_g"])); m.ln.bias.copy_(t(params["ln_b"]))
        m.policy.weight.copy_(t(params["pol_w"]).T); m.policy.bias.copy_(t(params["pol_b"]))
        m.qvalue.weight.copy_(t(params["q_w"]).T); m.qvalue.bias.copy_(t(params["q_b"]))
        for L, q in zip(m.enc.layers, params["layers"]):
            L.self_attn.in_proj_weight.copy_(t(q["in_w"]).T); L.self_attn.in_proj_bias.copy_(t(q["in_b"]))
            L.self_attn.out_proj.weight.copy_(t(q["out_w"]).T); L.self_attn.out_proj.bias.copy_(t(q["out_b"]))
            L.linear1.weight.copy_(t(q["l1_w"]).T); L.linear1.bias.copy_(t(q["l1_b"]))
            L.linear2.weight.copy_(t(q["l2_w"]).T); L.linear2.bias.copy_(t(q["l2_b"]))
            L.norm1.weight.copy_(t(q["n1_g"])); L.norm1.bias.copy_(t(q["n1_b"]))
            L.norm2.weight.copy_(t(q["n2_g"])); L.norm2.bias.copy_(t(q["n2_b"]))
    return m


# ---------------------------------------------------------------- the workers

def remote_tree_class():
    """MCTS with the forward pass on the server. _dispatch sends the batch and _collect reads the answer the way it
    reads sw25_one_copy's packed rows. Built in the worker, after selfplay's settings are in its environment."""
    from mcts import MCTS, _Cut

    class RemoteMCTS(MCTS):
        def __init__(self, conn, batch):
            self._conn = conn
            super().__init__(None, device="cpu", batch=batch)

        def _dispatch(self, prep):
            if "I" not in prep or "theirs" in prep:
                raise RuntimeError("RemoteMCTS serves value_from_q searches without an opponent adapter")
            self._conn.send_bytes(pack_request(prep))
            return {"nodes": prep["nodes"], "remote": True, "widths": prep["widths"]}

        def _collect(self, h):
            if "remote" not in h:
                return super()._collect(h)
            nodes = h["nodes"]
            t_wait = time.perf_counter() if self.timing_fixes else None
            arr = pickle.loads(self._conn.recv_bytes())
            if t_wait is not None:
                self.stats["wait_s"] = self.stats.get("wait_s", 0.0) + (time.perf_counter() - t_wait)
            widths = h["widths"]
            W = (arr.shape[1] - 1) // 2
            pol = [_Cut(arr[i, :widths[i]].copy()) for i in range(len(nodes))]
            val = arr[:, 2 * W].copy()
            qv = [arr[i, W:W + widths[i]].astype(np.float32) for i in range(len(nodes))]
            self._ml = None
            self.stats["evals"] += len(nodes)
            self.stats["batches"] += 1
            return pol, val, qv

    return RemoteMCTS


def work(widx, conns, cfg, soft_stop, made, log):
    """One worker process, len(conns) games at a time, each on its own thread and connection."""
    for k, v in cfg["env"].items():
        os.environ[k] = str(v)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # the volunteer app has no PyTorch, so its workers import the search code over volunteer/torch_stub.py
    if cfg.get("torch_stub"):
        from volunteer.torch_stub import install
        install(force=cfg["torch_stub"] == "force")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    import chess
    import selfplay as sp
    from mcts import pinned_search
    RemoteMCTS = remote_tree_class()
    starts = cfg.get("starts") or []
    if starts:
        def corpus_board(rng):
            for _ in range(10):
                b = chess.Board(rng.choice(starts))
                if b.is_valid() and not b.is_game_over() and b.legal_moves.count() > 2:
                    return b
            return chess.Board()
        sp.random_corpus_board = corpus_board
    sealed = pinned_search(cfg["version"], cfg["vdir"]) if cfg["version"] else None
    os.makedirs(cfg["out"], exist_ok=True)
    pos_path = os.path.join(cfg["out"], f"positions_w{widx:03d}.txt")
    tgt_path = os.path.join(cfg["out"], f"targets_w{widx:03d}.jsonl")
    fast_path = os.path.join(cfg["out"], f"fast_w{widx:03d}.jsonl")      # rows from cheap moves
    lock = threading.Lock()
    # Every tree is built and warmed here, on the main thread, after its sealed search is applied. A search thread
    # must never import (mcts._warm). v5's sealed search turns on lcb, whose scipy import would otherwise happen
    # inside a game thread and stall it.
    trees = []
    for c in conns:
        tree = RemoteMCTS(c, sp.BATCH)
        if sealed:
            tree.apply(sealed)
        tree.engine = "python"       # the network is on the server, so there's no C++ tree here
        tree._warm()
        trees.append(tree)

    def games(t, conn):
        try:
            tree = trees[t]
            rng = random.Random(f"{cfg['seed']}/{widx}/{t}/{time.time_ns()}")
            sp.with_noise(tree, rng)
            while time.time() < soft_stop.value:
                rows = [] if sp.TARGETS else None
                fast = [] if sp.TARGETS and sp.FULL_FRAC < 1.0 else None
                fens = sp.play(None, "cpu", rng, tree, rows, fast)
                with lock:
                    with open(pos_path, "a", encoding="utf-8") as f:
                        f.write("".join(x + "\n" for x in fens))
                    if fast:
                        with open(fast_path, "a", encoding="utf-8") as f:
                            f.write("".join(json.dumps(r) + "\n" for r in fast))
                    if rows:
                        with open(tgt_path, "a", encoding="utf-8") as f:
                            f.write("".join(json.dumps(r) + "\n" for r in rows))
                with made.get_lock():
                    made[0] += len(fens)
                    made[1] += 1
        except Exception as e:
            import traceback
            log.put(f"worker {widx} thread {t}: {type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}")
        finally:
            try:
                conn.close()
            except OSError:
                pass

    ts = [threading.Thread(target=games, args=(t, c), daemon=True) for t, c in enumerate(conns)]
    for x in ts:
        x.start()
    for x in ts:
        x.join()


# ---------------------------------------------------------------- the run

def detect_backend():
    want = E("SP_BACKEND", "auto")
    if want != "auto":
        return want
    try:
        import jax
        if jax.devices()[0].platform == "tpu":
            return "jax"
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "torch"
    except ImportError:
        pass
    return "cpu"


def run(seconds=None, out=None):
    """Servers, then workers, until the budget is spent. Returns the totals."""
    if mp.current_process().name != "MainProcess":
        # spawn runs the calling script again in every child, so a script without a __main__ guard would start a
        # whole run inside each server and worker
        raise SystemExit("selfplay_remote.run() called in a child process: guard the calling script's main")
    import paths
    backend = detect_backend()
    seconds = float(seconds if seconds is not None else E("SP_SECONDS", 3600))
    out = out or E("SP_OUT", os.path.join(os.getcwd(), "sp_out"))
    P = int(E("SP_WORKERS", max(1, (os.cpu_count() or 2) - 2)))
    T = int(E("SP_THREADS", 1))
    weights = ({"onnx": E("SP_ONNX")} if backend == "onnx" else {"npz": E("SP_NPZ")} if E("SP_NPZ")
               else {"ckpt": E("SP_CKPT", paths.under("versions", "mayfly_v5.pt"))})
    default_buckets = {"jax": "512,1024,2048,4096,8192", "torch": "64,128,256,512,1024",
                       "onnx": "32,64,128,256,512", "cpu": "16,32,64,128,256"}[backend]
    buckets = [int(x) for x in E("SP_BUCKETS", default_buckets).split(",")]
    starts = []
    if E("SP_STARTS") and os.path.exists(E("SP_STARTS")):
        starts = [l.strip() for l in open(E("SP_STARTS"), encoding="utf-8") if l.strip()]
    env = {k: E(k) for k in ("VISITS", "TEMP_PLIES", "OPENING_FRAC", "MAX_PLIES", "BATCH", "ROOT_NOISE",
                             "NOISE_ALPHA", "TARGETS", "MCTS_NATIVE", "TB", "FULL_FRAC", "FAST_VISITS")
           if E(k) is not None}
    env.setdefault("VISITS", "1600")
    if T > 1:
        env["MCTS_NATIVE"] = "0"          # the compiled walk is not safe for two trees at once in one process
    cfg = {"env": env, "version": E("VERSION", "mayfly_v5.pt"), "vdir": E("VERSION_DIR", paths.under("versions")),
           "out": out, "seed": int(E("SP_SEED", 0)), "starts": starts, "torch_stub": E("SP_TORCH_STUB")}
    ctx = mp.get_context("spawn")
    log = ctx.Queue()
    ready_ev, stop_ev = [], ctx.Event()
    evals = ctx.Value("q", 0)
    made = ctx.Array("q", 2)
    soft_stop = ctx.Value("d", time.time() + 10 * 365 * 86400)
    worker_ends = [[] for _ in range(P)]
    server_ends = []
    for w in range(P):
        for _ in range(T):
            a, b = ctx.Pipe()
            worker_ends[w].append(a)
            server_ends.append(b)
    if backend == "torch":
        import torch
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif backend == "onnx":
        devices = [E("SP_DEVICE", "0")]       # the graphics card's DXGI number, or "cpu"
    else:
        devices = [backend]
    split = [server_ends[i::len(devices)] for i in range(len(devices))]
    servers = []
    for dvc, ends in zip(devices, split):
        r = ctx.Event()
        ready_ev.append(r)
        p = ctx.Process(target=serve, args=(backend if backend != "cpu" else "torch",
                                            "cpu" if backend == "cpu" else dvc, ends, weights, r, stop_ev,
                                            evals, buckets, log), daemon=True)
        p.start()
        servers.append(p)
    t_ready = time.time()
    while not all(r.is_set() for r in ready_ev):
        while not log.empty():
            print(log.get(), flush=True)
        if any(not p.is_alive() for p in servers):
            while not log.empty():
                print(log.get(), flush=True)
            raise RuntimeError("a server died before it was ready")
        time.sleep(1)
    while not log.empty():
        print(log.get(), flush=True)
    t0 = time.time()
    end = t0 + seconds
    # no new game starts in the last stretch, so most games end inside the budget
    soft_stop.value = end - min(600.0, 0.1 * seconds)
    workers = []
    for w in range(P):
        p = ctx.Process(target=work, args=(w, worker_ends[w], cfg, soft_stop, made, log), daemon=True)
        p.start()
        workers.append(p)
    for ends in worker_ends:              # the parent's copies, each worker holds its own
        for c in ends:
            c.close()
    print(f"{backend}: {len(servers)} server(s), {P} workers x {T} games, {env['VISITS']} visits a move, "
          f"{seconds / 3600:.2f}h (ready in {t0 - t_ready:.0f}s after loading)", flush=True)
    last = (time.time(), 0, 0)
    while time.time() < end and any(p.is_alive() for p in workers):
        time.sleep(5)
        while not log.empty():
            print(log.get(), flush=True)
        now = time.time()
        if now - last[0] >= float(E("SP_REPORT", 300)):
            ev, pos = evals.value, made[0]
            print(f"  {(now - t0) / 3600:.2f}h: {made[1]:,} games, {pos:,} positions "
                  f"({(pos - last[2]) / (now - last[0]) * 3600:,.0f} an hour), "
                  f"{(ev - last[1]) / (now - last[0]):,.0f} evals a second", flush=True)
            last = (now, ev, pos)
    soft_stop.value = 0.0
    for p in workers:                     # a game still running at the end is given up
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    stop_ev.set()
    for p in servers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    while not log.empty():
        print(log.get(), flush=True)
    took = time.time() - t0
    res = {"backend": backend, "workers": P, "threads": T, "seconds": round(took), "games": made[1],
           "positions": made[0], "evals": evals.value, "evals_per_s": round(evals.value / max(took, 1)),
           "positions_per_hour": round(made[0] / max(took, 1) * 3600)}
    print("done:", json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "npz":
        save_npz(sys.argv[2], sys.argv[3])
    else:
        run()
