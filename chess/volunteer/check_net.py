"""check_net -- is the app's ONNX network the same player as the real one? The same real positions go through the
torch server's own code (selfplay_remote._torch_net, float32 on the CPU, the reference) and through the ONNX file on
each place the app can run it (DirectML on every graphics card, and the CPU). Reported: how often the favourite move
agrees, and the largest differences in the policy, the move values (q) and the position value. Then the speed of
each card at the app's batch sizes.

  python chess/volunteer/check_net.py MODEL.onnx CKPT.pt POSITIONS.txt [N]
"""
import os
import sys
import time

import chess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from encode import encode_keys  # noqa: E402
from moves import move_to_idx  # noqa: E402
from selfplay_remote import W_FIX  # noqa: E402


def batch(fens):
    boards = [chess.Board(f) for f in fens]
    ids, cast, hm, ep = encode_keys([b._transposition_key() for b in boards], [b.halfmove_clock for b in boards])
    n = len(boards)
    I = np.zeros((n, W_FIX), np.int64)
    M = np.zeros((n, W_FIX), bool)
    for i, b in enumerate(boards):
        mv = list(b.legal_moves)
        I[i, :len(mv)] = [move_to_idx(m, b.turn == chess.BLACK) for m in mv]
        M[i, :len(mv)] = True
    return {"ids": ids.astype(np.int64), "cast": cast.astype(np.float32), "hm": hm.astype(np.int64),
            "ep": ep.astype(np.int64), "I": I, "Ip": I.copy(), "M": M}


def reference(ckpt, inp):
    import torch
    sys.path.insert(0, HERE)
    from export_net import Served
    from selfplay_remote import _torch_net, load_weights
    cfg, params = load_weights(ckpt=ckpt)
    model = Served(_torch_net(cfg, params)).eval()
    with torch.no_grad():
        out = model(*[torch.from_numpy(inp[k]) for k in ("ids", "cast", "hm", "ep", "I", "Ip", "M")])
    return [o.numpy() for o in out]


def compare(name, ref, got, M):
    pg_r, q_r, v_r = ref
    pg_g, q_g, v_g = got
    neg = np.where(M, 0, -1e9)
    top_r, top_g = (pg_r + neg).argmax(1), (pg_g + neg).argmax(1)
    dp = np.abs(pg_r - pg_g)[M].max()
    dq = np.abs(q_r - q_g)[M].max()
    dv = np.abs(v_r - v_g).max()
    print(f"  {name:34s} favourite move agrees {100 * (top_r == top_g).mean():5.1f}%   "
          f"policy {dp:.3f}  q {dq:.4f}  value {dv:.4f}  (largest differences)", flush=True)


def main():
    import onnxruntime as ort
    sys.path.insert(0, HERE)
    from gpus import list_gpus
    model, ckpt, pos = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    fens = [l.split("|")[0].strip() for l in open(pos, encoding="utf-8") if l.strip()][:n]
    inp = batch(fens)
    print(f"{len(fens)} positions from {os.path.basename(pos)}", flush=True)
    ref = reference(ckpt, inp)
    places = [(f"DirectML {g['index']} {g['name']}", [("DmlExecutionProvider", {"device_id": g["index"]})])
              for g in list_gpus()] + [("CPU", ["CPUExecutionProvider"])]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    for name, prov in places:
        try:
            sess = ort.InferenceSession(model, opts, providers=prov)
        except Exception as e:
            print(f"  {name}: cannot load ({type(e).__name__}: {str(e)[:120]})")
            continue
        feed = dict(inp, cast=inp["cast"].astype(np.float16))
        got = sess.run(None, feed)
        compare(name, ref, got, inp["M"])
        if name == "CPU":
            continue
        for b in (64, 256):
            f = {k: np.resize(v, (b,) + v.shape[1:]) for k, v in feed.items()}
            sess.run(None, f)
            t0, k = time.time(), 0
            while time.time() - t0 < 8:
                sess.run(None, f)
                k += 1
            print(f"      batch {b}: {k * b / (time.time() - t0):,.0f} positions a second", flush=True)


if __name__ == "__main__":
    main()
