r"""
fp32, bfloat16, FP8 and fp16-accumulation training, selected by one environment variable.

FP8 needs three things right, or it crashes or silently trains badly.

  shapes      an FP8 matmul needs both dimensions divisible by 16. ChessNet has a Linear(4, d)
              carrying castling rights, so small layers stay in bf16, a negligible share of compute.
  master      FP8 is for the matmul only. Weights, optimiser moments and accumulation stay in
              higher precision (torchao handles this), so FP8 saves time, not much memory.
  numerics    four exponent bits and three mantissa bits is a narrow range, so check_numerics
              trains the same batches in both modes and compares the loss curves.

  from precision import apply_precision, autocast_ctx
  model = apply_precision(model)                 honours PRECISION
  with autocast_ctx():
      out = model(ids, cast)

  env: PRECISION=fp32|bf16|fp8|fp16acc   (default bf16)

  python chess/precision.py            speed and numerics for fp32, bf16 and FP8

fp16acc is for GPUs that run a bf16 matmul with fp32 accumulation at half the tensor rate, as GeForce
Blackwell cards do, while fp16 with fp16 accumulation runs at the full rate. On MayFly's model it trained
about 1.15x faster alone and 1.48x under torch.compile (train_av's COMPILE=1), with a loss closer to fp32's
than bf16's, since fp16 keeps three more mantissa bits than bf16. Its narrow range is why the trainer scales
the loss (a GradScaler) in this mode. Weights, optimiser moments and checkpoints stay fp32, so switching
back is only the environment variable.
"""
import os
import contextlib
import torch
import torch.nn as nn

E = os.environ.get
PRECISION = E("PRECISION", "bf16").lower()
FP8_MIN_DIM = 16


def fp8_eligible(mod, name=""):
    """An FP8 matmul needs both dimensions divisible by 16, so Linear(4, d) cannot take part."""
    return (isinstance(mod, nn.Linear)
            and mod.in_features % FP8_MIN_DIM == 0
            and mod.out_features % FP8_MIN_DIM == 0)


def fp8_available():
    if not torch.cuda.is_available():
        return False, "no cuda"
    major = torch.cuda.get_device_capability(0)[0]
    if major < 9:
        return False, f"compute capability {major}.x, FP8 needs 9.0 or newer"
    try:
        from torchao.float8 import convert_to_float8_training  # noqa: F401
    except Exception as e:
        return False, f"torchao unusable: {type(e).__name__}"
    return True, "ok"


def apply_precision(model, precision=None, verbose=True):
    """Convert the model for the requested precision. Returns the model and the mode actually used.

    Falls back to bf16 with a printed reason rather than failing, so a long run does not die at the
    start over a precision flag.
    """
    p = (precision or PRECISION).lower()
    if p == "fp16acc":
        if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
            if verbose:
                print(f"fp16acc requested but this torch ({torch.__version__}) has no fp16 accumulation "
                      f"(it needs 2.7 or newer); using bf16", flush=True)
            p = "bf16"
        else:
            # process-wide, every fp16 matmul from here on accumulates in fp16, which is where the speed is
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
    if p == "fp8":
        ok, why = fp8_available()
        if not ok:
            if verbose:
                print(f"FP8 requested but unavailable ({why}); using bf16", flush=True)
            p = "bf16"
        else:
            from torchao.float8 import convert_to_float8_training
            n_before = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
            convert_to_float8_training(model, module_filter_fn=fp8_eligible)
            n_fp8 = sum(1 for m in model.modules()
                        if type(m).__name__.lower().startswith("float8"))
            if verbose:
                print(f"FP8: {n_fp8} of {n_before} linear layers converted, "
                      f"the rest stay bf16 for shape reasons", flush=True)
    if verbose and p != "fp8":
        print(f"precision: {p}", flush=True)
    return model, p


def autocast_ctx(mode=None):
    """The autocast context for a precision mode.

    fp32 runs bare. bf16 and FP8 both run under bf16 autocast, since FP8 replaces the matmul inside
    layers autocast already manages. fp16acc runs under fp16 autocast."""
    p = (mode or PRECISION).lower()
    if p == "fp32":
        return contextlib.nullcontext()
    if p == "fp16acc":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cuda", dtype=torch.bfloat16)


def check_numerics(build, steps=120, seed=0, device="cuda"):
    """Train the same data in fp32 and in the requested mode, and compare the loss curves.

    A faster precision that diverges is worthless, and the failure is quiet, the loss just stops
    falling where it should not. This makes it visible.
    """
    import torch.nn.functional as F
    out = {}
    for mode in ("fp32", PRECISION):
        torch.manual_seed(seed)
        m = build().to(device)
        m, used = apply_precision(m, mode, verbose=False)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        g = torch.Generator(device="cpu").manual_seed(seed)
        losses = []
        for i in range(steps):
            ids = torch.randint(0, 13, (64, 64), generator=g).to(device)
            cast = torch.rand(64, 4, generator=g).to(device)
            tgt = torch.rand(64, m.policy.out_features, generator=g).to(device)
            tgt = tgt / tgt.sum(-1, keepdim=True)
            with autocast_ctx(used):
                o = m(ids, cast)
                l = -(tgt * F.log_softmax(o["policy"].float(), -1)).sum(-1).mean()
            opt.zero_grad(); l.backward(); opt.step()
            losses.append(float(l))
        out[mode] = losses
        del m, opt
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    import sys, time, gc
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chess_net import ChessNet
    import torch.nn.functional as F

    ok, why = fp8_available()
    print(f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"FP8 available: {ok} ({why})\n")

    D, L = int(E("D", 1024)), int(E("LAYERS", 16))
    build = lambda: ChessNet(d=D, nhead=16, layers=L, dim_ff=2 * D)

    print(f"speed, d{D} x {L} layers, batch 128, minimum of five bursts")
    print(f"{'mode':8s} {'ms/step':>9s} {'pos/s':>8s} {'GB':>6s}")
    res = {}
    for mode in ("fp32", "bf16", "fp8"):
        try:
            torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
            m, used = apply_precision(build().to("cuda"), mode, verbose=False)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            ids = torch.randint(0, 13, (128, 64), device="cuda")
            cast = torch.rand(128, 4, device="cuda")
            # the policy's own width, 4,162 since underpromotions
            tgt = torch.rand(128, m.policy.out_features, device="cuda"); tgt = tgt / tgt.sum(-1, keepdim=True)

            def step():
                with autocast_ctx(used):
                    o = m(ids, cast)
                    l = -(tgt * F.log_softmax(o["policy"].float(), -1)).sum(-1).mean()
                opt.zero_grad(); l.backward(); opt.step()

            for _ in range(6):
                step()
            best = 1e9
            for _ in range(5):
                torch.cuda.synchronize(); t0 = time.time()
                for _ in range(6):
                    step()
                torch.cuda.synchronize(); best = min(best, (time.time() - t0) / 6)
            res[mode] = 128 / best
            print(f"{mode:8s} {best*1000:9.1f} {128/best:8.0f} "
                  f"{torch.cuda.max_memory_allocated()/1e9:6.1f}")
            del m, opt; torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            print(f"{mode:8s} failed: {str(e)[:70]}")
    if "fp32" in res:
        for k in ("bf16", "fp8"):
            if k in res:
                print(f"  {k} is {res[k]/res['fp32']:.2f}x fp32")
