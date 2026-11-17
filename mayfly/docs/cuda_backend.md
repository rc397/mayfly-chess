# The CUDA backend

`chess/cudanet/` is a hand-written CUDA implementation of the search's forward pass. It computes only what the search
reads, which is the trunk on 65 tokens, the final layer norm of the summary token, the policy and action-value heads
and the reduction over the legal moves (the masked softmax, the sigmoid on each move's action value and the leaf
value). The contract every file follows is in `cudanet_spec.h`.

## Design

- **Kernels compiled at run time.** The kernels are CUDA C source (`kernels_trunk.cu`, `kernels_heads.cu`,
  `kernels_fuse.cu`, `kernels_gemm.cu`). NVRTC compiles them on first use to a cubin for the GPU's own
  architecture and caches it on disk. No nvcc is needed, and since no PTX is shipped a driver older than the
  toolkit still works.
- **Matrix products in cuBLASLt**, with fp16 inputs and fp32 accumulation and the bias and residual fused into the
  epilogue. GELU is its own kernel, because cuBLASLt's GELU epilogue is the tanh approximation and the network uses
  the exact one.
- **Libraries loaded at run time.** The driver API, the CUDA runtime, cuBLASLt and NVRTC are loaded dynamically
  (`cuda_dyn.cpp`), so no CUDA import library is linked. If a process has already loaded them, for example through
  PyTorch, the same copies are used.
- **One CUDA graph per batch size**, captured on first use and replayed for every later batch of that size.
- **Attention on tensor cores.** Both attention products use `mma.sync` (m16n8k16, fp16 inputs, fp32
  accumulation). The 65 rows and keys are padded to 80 and the scores, softmax and probabilities stay in registers.
- **fp32 where it matters.** The residual stream and the layer norms are fp32. The big matrices are stored in fp16.
- **Two batches in flight.** `begin` queues a batch and `end` collects the oldest one, so the C++ search can gather
  the next batch while the GPU runs the last.

The code is Windows only as written (it loads libraries through the Windows API). It has been run on sm_86 (RTX 3070)
and sm_120 (RTX 5070 Ti).

## The weight file

`export_cudanet.py` turns a checkpoint into a `.cnw` file. It starts with the magic `MFCNW001` and a JSON header with
the architecture and a fingerprint of the network, then holds every tensor at a 256-byte aligned offset, the big
matrices in fp16 and the rest in fp32. For v5 it is 287 MB. The search uses a `.cnw` only when its fingerprint
matches the network the tree holds.

```
python chess/cudanet/export_cudanet.py data/versions/mayfly_v6.pt data/versions/mayfly_v6.cnw --check
```

`--check` reads the file back and compares every tensor with the checkpoint.

## Building and using it

Both builds need Visual Studio 2022 (MSVC) and the CUDA 12 headers, named by `CUDA_KIT` (a folder with `include\`).
At run time the CUDA runtime, cuBLASLt and NVRTC DLLs are found through `MAYFLY_CUDA_BIN` or the `lib` folder of a
loaded PyTorch. `cuda_shim/` holds host-only stand-ins for two headers that NVIDIA's CUDA pip wheels leave out, for
builds against those wheels.

- **From Python.** `build_cudanet.bat` builds `build\cudanet.dll`, and `cudanet_py.py` wraps it through ctypes
  (`MAYFLY_CUDANET_DLL` names another build).
- **Inside the C++ search.** `chess\cpp\build_rig.bat` builds `mayfly_cppsearch` with LibTorch and the CUDA
  backend together. A tree whose `engine` setting is `cpp` and whose `cuda_net` setting is on (`CUDA_NET=1` for an
  unsealed tree) looks for `<version>.cnw` beside the version, or the file in `CPP_CNW`, and falls back to the
  LibTorch trace with a warning when none matches.

### Switches

| variable | effect |
|---|---|
| `MAYFLY_CUDANET_ACC16` | 1 accumulates the qkv and first feed-forward products in fp16, 2 all four products, unset keeps fp32 |
| `MAYFLY_CUDANET_ATT` | 0 runs attention on the scalar kernel instead of tensor cores |
| `MAYFLY_CUDANET_FUSE` | 1 fuses the embedding into the first norm and folds the residual adds, with identical results |
| `MAYFLY_CUDANET_TUNE` | 1 times cuBLASLt's candidate algorithms for each shape on first use and caches the fastest per GPU, driver and library, 2 also sweeps the configurable variants |
| `MAYFLY_CUDANET_GEMM` | hand-written tensor-core products instead of cuBLASLt, 1 with fp16 accumulation, 2 with fp16 inside each tile of 64 and fp32 across tiles |
| `MAYFLY_CUDANET_GEMM_MINB` | the batch size below which cuBLASLt is used anyway (24 for mode 1, 4 for mode 2) |
| `MAYFLY_CUDANET_MAX_BATCH` | the largest batch the buffers hold, 48 by default and at most 1024 |
| `MAYFLY_CUDANET_GRAPH` | 0 turns the CUDA graphs off |
| `MAYFLY_CUDANET_SRC` | the folder with the `.cu` sources, by default the folder of the module |

### Tools

- `test_cudanet.py numerics|speed|profile` checks the backend against `ChessNet` in float32 on the CPU, times it
  against the PyTorch paths, and profiles a pass step by step.
- `tune_cudanet.py fill` fills the tuning cache for every batch size, and `tune_cudanet.py ab` compares tuned with
  untuned.
- `bench_search.py` runs the same C++ search on the LibTorch trace or on a `.cnw` and reports time and evaluations
  a search.
- `bench/gemm_bench.cpp` times the hand-written products against cuBLASLt.

## Accuracy

Differences from `ChessNet` in float32 on a fixed set of test positions (RTX 3070), with the tensor-core attention.

| setting | largest policy logit error | top move agrees |
|---|---|---|
| default, fp32 accumulation | 0.0052 | 100% |
| `ACC16=1` | 0.019 | 100% |
| `ACC16=2` | 0.026 | 99.6%, one near tie |
| `GEMM=2` with `FUSE` and `TUNE` | 0.0069 | 100% |
| `ACC16=2` with `GEMM=1`, `FUSE` and `TUNE` | 0.0255 | 100% |
| TensorRT 10.16, strongly typed fp16 engine | 0.027 | 99.2% |
| the bf16 autocast pass the Python search plays with | 0.088 | 98.4% |

Every setting stays inside the error of the bf16 pass. On sm_120 the default setting measured 0.0070, with the top
move agreeing on every position checked.

## Speed

### Against TensorRT

Milliseconds a batch for a full evaluate call on an RTX 3070, each setting in its own process, the order rotated each
round, medians of three rounds. The fast setting is `ACC16=2` with `GEMM=1`, `FUSE` and `TUNE`, and the accurate
setting is `GEMM=2` with `FUSE` and `TUNE`.

| batch | 1 | 8 | 16 | 32 | 40 | 48 |
|---|---|---|---|---|---|---|
| TensorRT 10.16 | 2.92 | 4.96 | 9.38 | 18.25 | 21.86 | 23.94 |
| fast setting | 1.56 | 4.78 | 9.45 | 17.81 | 21.52 | 23.73 |
| accurate setting | 2.37 | 5.69 | 11.04 | 20.65 | 24.77 | 27.91 |

The fast setting is ahead of TensorRT at every size but 16, where they are level, by 1.88 times at batch 1 and 1
to 4% from batch 8 up. The accurate setting is 12 to 15% behind TensorRT at the large batches and 23% ahead of
plain fp32 accumulation, which took 34.31 ms at batch 48. TensorRT chose fp16-accumulating kernels and a fused
attention kernel on its own.

### Against PyTorch

The first version of the backend (fp32 accumulation, scalar attention) against the paths the search used before it.

- RTX 3070, against the Python search's bf16 pass with a CUDA graph, in evaluations a second. Batch 1 397 against 120
  (3.30 times), batch 48 1,264 against 888 (1.42 times).
- RTX 5070 Ti with other GPU work running, against the LibTorch trace driven from Python. 5.29 times at batch 1,
  falling to 1.72 times at batch 48 (1,706 against 989).

Tensor-core attention and fp16 accumulation in all four products later took the batch 48 pass on the RTX 3070 from
37.7 ms to 25.6 ms, 1,875 evaluations a second. Before the tensor-core attention, the four products took 29.5 of the
39.2 ms of a batch 48 pass, and attention took 17% of the time for under 2% of the arithmetic.

### In the search

The same C++ search with the same settings on each backend, an RTX 5070 Ti with other GPU work running, 24 positions
at 1,600 visits and a batch of 48.

| network | ms a search | evaluations a second |
|---|---|---|
| LibTorch trace | 823 | 1,211 |
| CUDA backend, fp32 accumulation | 596 to 613 | 1,625 to 1,671 |
| CUDA backend, `ACC16=1` | 517 | 1,927 |

### On the clock

A match at 60 seconds plus 1 a move with pondering on an RTX 5070 Ti, the same C++ search and network on both sides,
the CUDA backend (`ACC16=1`, scalar attention) against the LibTorch trace. Over 199 games the CUDA backend scored
+39 =157 -3, about +64 Elo (95% interval +43 to +85). It searched 6,294 visits a move against 3,994 in the same 1.51
seconds a move.

## Tried and left out

- **FP8.** Simulated on the float32 model, no scaling scheme kept the policy inside the accuracy bar. With all four
  products in FP8 the largest policy error was 0.34 to 0.36 and the top move agreed 95 to 96% of the time.
- **INT8** weights and activations with a randomized Hadamard rotation. It reached the bf16 pass's accuracy but was
  only 1.12 times as fast at batch 48.
- **GELU in the cuBLASLt epilogue.** The tanh approximation raised the largest policy error from 0.026 to 0.043, so
  it stays off.

## The TensorRT path

`chess/trtnet/` builds the same forward pass as a TensorRT engine, as a yardstick for the backend.

- `setup_venv.bat` makes a separate virtual environment with TensorRT 10 for CUDA 12, onnx and a CPU build of
  PyTorch (`TRT_VENV` names the folder).
- `export_onnx.py` writes the pass as ONNX with the backend's inputs and outputs, either typed the way the backend
  computes or all fp32.
- `build_engine.py` builds an engine for batches 1 to 48 and writes a layer dump beside it, which names the kernel
  TensorRT chose for each layer.
- `trtnet_py.py` runs an engine behind the same interface as `cudanet_py.py`, so `test_cudanet.py` can measure it
  unchanged, and `test_trtnet.py` tests it.
