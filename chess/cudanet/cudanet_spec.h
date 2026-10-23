// The shared contract of MayFly's hand-tuned CUDA backend. Every file in chess/cudanet/ is written against this
// header, so change it only together with everything that uses it.
//
// It computes the search's forward pass and nothing else (cpp/netbackend.h, cpp/search2.h mayfly2::Net). That is
// ChessNet's trunk on 65 tokens (the CLS token, then the 64 squares in the side to move's frame), the final layer norm
// on the CLS row, the policy and action-value heads, and _collect's post-processing (sigmoid on q, the masked softmax of
// the policy at the legal moves, val = sum(w * qg) * 2 - 1, qv = qg * 2 - 1). The attention, distributional and
// moves-left heads are not read by the search and are not computed.
//
// Kernels are CUDA C source compiled at runtime by NVRTC to a cubin for the device's own architecture (for example
// sm_86 or sm_120). PTX is never shipped, because an older driver cannot load PTX from a newer toolkit. The matrix
// products are cuBLASLt (FP16 inputs, FP32 accumulation), every CUDA library is loaded at runtime (no import
// libraries), and the whole forward pass for a batch size is captured once as a CUDA graph and replayed.
// It builds against the CUDA 12.8 headers and loads cudart64_12.dll, cublasLt64_12.dll, nvrtc64_120_0.dll and
// nvrtc-builtins64_128.dll (see cuda_dyn.h for where from). The driver API comes from nvcuda.dll.
#pragma once
#include <cstdint>

namespace cudanet {

// ---------------------------------------------------------------------------------------------------- the network
constexpr int kD = 1024;          // width
constexpr int kLayers = 16;
constexpr int kHeads = 16;
constexpr int kDh = 64;           // kD / kHeads
constexpr int kFF = 2048;         // feed-forward width
constexpr int kTokens = 65;       // CLS + 64 squares, token 0 is CLS
constexpr int kMoves = 4162;      // policy rows (chess/moves.py)
constexpr int kMaxBatch = 48;     // the search never asks for more (netbackend.h), the default buffer size
constexpr int kMaxBatchLimit = 1024;   // env MAYFLY_CUDANET_MAX_BATCH may size the buffers for up to this many
constexpr float kLnEps = 1e-5f;
// The architecture numbers above are v5's and v6's. The weight file carries its own and the backend refuses a file
// whose numbers differ from these.

// ---------------------------------------------------------------------------------------------------- the weight file
// export_cudanet.py writes it from a train_av or versions checkpoint, and weights.cpp reads it.
//   bytes 0..7    magic "MFCNW001"
//   bytes 8..11   uint32 little endian: length of the JSON header that follows
//   JSON header   {"version": 1, "d": 1024, "layers": 16, "heads": 16, "ff": 2048, "tokens": 65, "moves": 4162,
//                  "fingerprint": "<mcts_cpp.fingerprint of the model>", "source": "<checkpoint path>",
//                  "tensors": [{"name": ..., "dtype": "f16" | "f32", "shape": [...], "offset": <bytes from the data
//                  start>, "nbytes": ...}, ...]}
//   then zero padding to the next multiple of 256 (the data start), then the tensors, each at a 256-aligned offset.
// Tensor names, dtypes and shapes (row-major, a linear weight keeps torch's layout, [out, in])
//   "piece"   f32 [13, kD]          "pos" f32 [64, kD]     "cls" f32 [kD]
//   "cast_w"  f32 [kD, 4]           "cast_b" f32 [kD]      "hm" f32 [102, kD]     "ep_mark" f32 [kD]
//   per layer i (0..kLayers-1):
//   "L{i}.n1_g", "L{i}.n1_b"  f32 [kD]
//   "L{i}.qkv_w" f16 [3*kD, kD]     "L{i}.qkv_b" f32 [3*kD]     (rows: q, then k, then v; torch's in_proj)
//   "L{i}.out_w" f16 [kD, kD]       "L{i}.out_b" f32 [kD]
//   "L{i}.n2_g", "L{i}.n2_b"  f32 [kD]
//   "L{i}.ff1_w" f16 [kFF, kD]      "L{i}.ff1_b" f32 [kFF]
//   "L{i}.ff2_w" f16 [kD, kFF]      "L{i}.ff2_b" f32 [kD]
//   "ln_g", "ln_b"  f32 [kD]        (the final layer norm)
//   "heads_w" f16 [2*kMoves, kD]    "heads_b" f32 [2*kMoves]    (rows 0..kMoves-1 policy, then kMoves rows qvalue)

// ---------------------------------------------------------------------------------------------------- the buffers
// rows = B * kTokens, row r = b * kTokens + t. Every buffer is sized for the largest batch, kMaxBatch, or env
// MAYFLY_CUDANET_MAX_BATCH (1..kMaxBatchLimit) read once when a CudaNet is made (about 1.4 MB of buffers a position).
//   x      f32 [rows, kD]       the residual stream
//   h      f16 [rows, kD]       a layer norm's output (the next product's input)
//   qkv    f16 [rows, 3*kD]     columns q (kD), k (kD), v (kD), and within each head j owns columns j*kDh..j*kDh+63
//   att    f16 [rows, kD]       attention output, heads concatenated (the out projection's input)
//   ff     f16 [rows, kFF]      the feed-forward hidden layer, after GELU
//   cls    f16 [B, kD]          the final layer norm of each CLS row
//   logits f32 [B, 2*kMoves]    the two heads: policy raw logits, then raw qvalue logits
// Inputs on the device (copied from pinned host memory each batch): ids int64 [B*64], cast f32 [B*4], hm int64 [B],
// ep int64 [B], I int64 [B*wide], M uint8 [B*wide]. Outputs: pol f32 [B*wide] (the policy logits at each row's legal
// moves, 0 on padding), val f32 [B], qv f32 [B*wide].

// ---------------------------------------------------------------------------------------------------- the products
// Every product is Y[rows, out] = X[rows, in] * W^T (+ epilogue), with W as stored in the file ([out, in] row-major).
//   qkv:  X = h,   W = qkv_w, Y = qkv f16, epilogue bias (qkv_b)
//   out:  X = att, W = out_w, Y = x   f32, epilogue bias (out_b) plus the residual: x = x + att * W^T + b (in place)
//   ff1:  X = h,   W = ff1_w, Y = ff  f16, epilogue bias (ff1_b), then gelu_f16 (exact, erf) in place
//   ff2:  X = ff,  W = ff2_w, Y = x   f32, epilogue bias (ff2_b) plus the residual, in place
//   heads: X = cls [B, kD], W = heads_w, Y = logits f32, epilogue bias (heads_b)
// cuBLASLt's own GELU epilogue is the tanh approximation, which ChessNet does not use, so GELU is a kernel.

// ---------------------------------------------------------------------------------------------------- the kernels
// kernels_trunk.cu and kernels_heads.cu hold CUDA C for NVRTC: extern "C" __global__, no host code, no #include beyond
// <cuda_fp16.h>, sizes passed as arguments. One forward pass for B rows:
//   embed(ids, cast, hm, ep, piece, pos, cls, cast_w, cast_b, hm_emb, ep_mark, x, B)
//       x[b*65+0] = cls + cast_w * cast[b] + cast_b + hm_emb[hm[b]]
//       x[b*65+1+s] = piece[ids[b*64+s]] + pos[s] + (ep[b] > 0 && s == ep[b]-1 ? ep_mark : 0)
//   for each layer:
//     layernorm_f32_f16(x, n1_g, n1_b, h, rows, kD, eps)
//     [qkv product]
//     attention_mma(qkv, att, B, scale = 1/sqrt(kDh))          softmax over all 65 keys, no mask, on tensor cores
//                                                              (attention_f16, the scalar one, when MAYFLY_CUDANET_ATT=0)
//     [out product, residual]
//     layernorm_f32_f16(x, n2_g, n2_b, h, rows, kD, eps)
//     [ff1 product]; gelu_f16(ff, rows * kFF)
//     [ff2 product, residual]
//   cls_layernorm(x, ln_g, ln_b, cls, B, eps)                    rows b*65 only
//   [heads product]
//   heads_post(logits, I, M, B, wide, pol, val, qv)              _collect's arithmetic, see the top of this file

// ---------------------------------------------------------------------------------------------------- the classes
// cuda_dyn.h      struct Cuda with function pointers for the driver API (nvcuda.dll), the runtime (cudart64_12.dll),
//                 cuBLASLt (cublasLt64_12.dll) and NVRTC (nvrtc64_120_0.dll), loaded from MAYFLY_CUDA_BIN, then
//                 the CUDA 12.8 install folder, then the directory of the running torch's lib. Cuda& cuda() loads
//                 once and throws std::runtime_error with the missing DLL or symbol.
// nvrtc_loader.h  class KernelModule(source, name, device): compile to a cubin for the device's compute capability
//                 (cached as %LOCALAPPDATA%\mayfly_cudanet\<sha1 of source and options>_sm<cc>.cubin), load it,
//                 CUfunction get(name); launch_kernel(fn, LaunchCfg, stream, args).
// gemm.h          class Gemm(stream): linear(X f16, W f16, bias f32, Y, rows, out, in, mode) with mode F16_BIAS or
//                 F32_BIAS_RESIDUAL (Y += X W^T + bias, f32), plans and algorithms cached per (rows, out, in, mode),
//                 one workspace.
// weights.h       class Weights(path, device): parses the file above, uploads every tensor once, ptr(name), the
//                 header's numbers and fingerprint.
// cudanet.h       class CudaNet(weights_path, device): all buffers for max_batch(), one stream, a CUDA graph per batch
//                 size captured on its first use, evaluate(B, wide, ids, cast, hm, ep, I, M, pol, val, qv) on host
//                 pointers (pinned staging inside), synchronous.
// cudanet_backend.cpp  mayfly2::Net and mayfly::NetBackend adapters over CudaNet (cpp/search2.h, cpp/netbackend.h).
// cudanet_capi.cpp     the C ABI of cudanet.dll for Python (ctypes): see cudanet_capi.h.
}  // namespace cudanet
