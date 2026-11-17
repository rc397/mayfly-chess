// See gemm.h. Every CUDA call goes through cudanet::cuda() (cuda_dyn.h). Nothing links cuBLASLt directly.
//
// Row-major on a column-major library. cuBLASLt computes D = alpha * op(A) op(B) + beta * C + bias with column-major
// matrices. A row-major [r, c] buffer with row stride c is the same memory as a column-major [c, r] matrix with leading
// dimension c, i.e. its transpose. So, with our row-major buffers:
//   X [rows, in]  is column-major X^T [in, rows],  ld = in
//   W [out, in]   is column-major W^T [in, out],   ld = in
//   Y [rows, out] is column-major Y^T [out, rows], ld = out
// and Y = X W^T is the same as Y^T = W X^T. In cuBLASLt's terms, m = out, n = rows, k = in:
//   A = W's buffer, layout [in, out] ld in, TRANSA = T   -> op(A) = W   [out, in]  (m x k)
//   B = X's buffer, layout [in, rows] ld in, TRANSB = N  -> op(B) = X^T [in, rows] (k x n)
//   C = D = Y's buffer, layout [out, rows] ld out        -> D = Y^T     [out, rows] (m x n)
// This is the "TN" case (both operands contiguous along k), the fast one for tensor cores. D has out rows, so
// CUBLASLT_EPILOGUE_BIAS adds bias[i] to D[i, j] for every j, one bias per output feature, as torch does. D stays in
// column order (cuBLASLt's bias epilogue does not support CUBLASLT_ORDER_ROW for D).
//
// Bias type. In cublasLt.h (12.8, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE) bias elements are D's type, the only exceptions
// being IMMA (int8) and fp8 kernels. So for an f16 D the bias must be f16, and F16_BIAS keeps a half copy of each f32 bias
// (made once per bias pointer, rounded to nearest). The epilogue still adds it in fp32 before rounding Y to f16, so
// this matches torch's fp16 Linear, which casts the bias to f16 as well. F32 modes use the f32 bias as it is.
//
// Scalars and graph capture. Pointer mode is host, so cuBLASLt reads alpha and beta on the host when cublasLtMatmul is
// called and passes their values to the kernels as launch arguments, which graph capture copies into the kernel
// nodes. So the values are frozen at capture and nothing reads the host scalars at replay. They are members of the
// Gemm anyway (one_, zero_), constant and alive as long as the Gemm, so even a captured pointer would stay correct.
//
// In place. F32_BIAS_RESIDUAL passes Y as both C and D with the same layout object (cuBLASLt requires ldc == ldd and
// the same order when C == D) and beta = 1.
#include "gemm.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_dyn.h"

namespace cudanet {

namespace {

constexpr int kRequestAlgos = 8;   // heuristic results to ask for, the first that fits the workspace wins

// Largest power of two (capped at 256) that divides the address. nullptr counts as 256.
unsigned align_of(const void* p) {
  uintptr_t v = reinterpret_cast<uintptr_t>(p);
  if (v == 0 || (v & 255) == 0) return 256;
  return static_cast<unsigned>(v & (~v + 1));
}

template <typename T>
void set_desc(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr, const T& value, const char* what) {
  check_lt(cuda().cublasLtMatmulDescSetAttribute(desc, attr, &value, sizeof(value)), what);
}

template <typename T>
void set_pref(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, const T& value,
              const char* what) {
  check_lt(cuda().cublasLtMatmulPreferenceSetAttribute(pref, attr, &value, sizeof(value)), what);
}

}  // namespace

Gemm::Gemm(cudaStream_t stream, size_t workspace_bytes) : stream_(stream), ws_bytes_(workspace_bytes) {
  tune_ = tune_level();
  check_lt(cuda().cublasLtCreate(&lt_), "gemm: cublasLtCreate");
  if (ws_bytes_ > 0) {
    cudaError_t e = cuda().cudaMalloc(&ws_, ws_bytes_);
    if (e != cudaSuccess) {
      cuda().cublasLtDestroy(lt_);
      lt_ = nullptr;
      check_rt(e, "gemm: cudaMalloc workspace");
    }
  }
}

Gemm::~Gemm() {
  // no throwing here, errors at teardown are ignored
  for (auto& kv : plans_) destroy(kv.second);
  plans_.clear();
  for (auto& kv : half_bias_) cuda().cudaFree(kv.second.first);
  half_bias_.clear();
  if (ws_) cuda().cudaFree(ws_);
  if (lt_) cuda().cublasLtDestroy(lt_);
}

void Gemm::destroy(Plan& p) {
  if (p.d) cuda().cublasLtMatrixLayoutDestroy(p.d);
  if (p.b) cuda().cublasLtMatrixLayoutDestroy(p.b);
  if (p.a) cuda().cublasLtMatrixLayoutDestroy(p.a);
  if (p.desc) cuda().cublasLtMatmulDescDestroy(p.desc);
  p = Plan{};
}

void Gemm::plan(int rows, int out, int in, Mode mode, bool with_bias) {
  get_plan(rows, out, in, mode, with_bias, 256, nullptr);
}

Gemm::Plan& Gemm::get_plan(int rows, int out, int in, Mode mode, bool has_bias, unsigned align,
                           const void* bias_dev) {
  if (rows <= 0 || out <= 0 || in <= 0) throw std::invalid_argument("gemm: rows, out and in must be positive");
  Key key{rows, out, in, static_cast<int>(mode), has_bias ? 1 : 0, align};
  auto it = plans_.find(key);
  if (it != plans_.end()) return it->second;

  const bool acc16 = mode == Mode::F16_BIAS_ACC16 || mode == Mode::F16_GELU_BIAS_ACC16;
  const bool gelu = mode == Mode::F16_GELU_BIAS || mode == Mode::F16_GELU_BIAS_ACC16;
  const bool f16_out = mode == Mode::F16_BIAS || acc16 || gelu;
  const cudaDataType_t dtype = f16_out ? CUDA_R_16F : CUDA_R_32F;
  Plan p;
  cublasLtMatmulPreference_t pref = nullptr;
  try {
    check_lt(cuda().cublasLtMatmulDescCreate(&p.desc, acc16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
                                             acc16 ? CUDA_R_16F : CUDA_R_32F), "gemm: MatmulDescCreate");
    const int32_t pmode = CUBLASLT_POINTER_MODE_HOST;
    set_desc(p.desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, pmode, "gemm: pointer mode");
    const int32_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;   // see the top of this file
    set_desc(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, ta, "gemm: transa");
    set_desc(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, tb, "gemm: transb");
    const uint32_t epi = has_bias ? (gelu ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS)
                                  : (gelu ? CUBLASLT_EPILOGUE_GELU : CUBLASLT_EPILOGUE_DEFAULT);
    set_desc(p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, epi, "gemm: epilogue");
    if (has_bias) {
      const int32_t btype = dtype;   // bias in D's type (see the top of this file)
      set_desc(p.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, btype, "gemm: bias type");
      // the heuristic reads the bias pointer only for its alignment, so a 256-aligned dummy when none is given yet
      const void* bp = bias_dev ? bias_dev : reinterpret_cast<const void*>(uintptr_t(256));
      set_desc(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, bp, "gemm: bias pointer");
    }

    // A = W [in, out] ld in, B = X [in, rows] ld in, C = D = Y [out, rows] ld out (column-major, the default order)
    check_lt(cuda().cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16F, uint64_t(in), uint64_t(out), int64_t(in)),
             "gemm: layout A");
    check_lt(cuda().cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16F, uint64_t(in), uint64_t(rows), int64_t(in)),
             "gemm: layout B");
    check_lt(cuda().cublasLtMatrixLayoutCreate(&p.d, dtype, uint64_t(out), uint64_t(rows), int64_t(out)),
             "gemm: layout D");

    check_lt(cuda().cublasLtMatmulPreferenceCreate(&pref), "gemm: PreferenceCreate");
    const uint64_t ws = ws_bytes_;
    set_pref(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ws, "gemm: pref workspace");
    const uint32_t al = align;
    set_pref(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, al, "gemm: pref align A");
    set_pref(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, al, "gemm: pref align B");
    set_pref(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, al, "gemm: pref align C");
    set_pref(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, al, "gemm: pref align D");
    if (f16_out) {
      // split-k partial sums stay in fp32, with no reduction in the f16 output itself
      const uint32_t mask = CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
      set_pref(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, mask, "gemm: pref reduction");
    }

    cublasLtMatmulHeuristicResult_t res[kRequestAlgos] = {};
    int n = 0;
    cublasStatus_t hs =
        cuda().cublasLtMatmulAlgoGetHeuristic(lt_, p.desc, p.a, p.b, p.d, p.d, pref, kRequestAlgos, res, &n);
    if (hs == CUBLAS_STATUS_NOT_SUPPORTED) n = 0;   // no algorithm at all, and the message below says which shape
    else check_lt(hs, "gemm: AlgoGetHeuristic");
    int pick = -1;
    for (int i = 0; i < n; ++i) {
      if (res[i].state == CUBLAS_STATUS_SUCCESS && res[i].workspaceSize <= ws_bytes_) {
        pick = i;
        break;
      }
    }
    if (pick < 0) {
      throw std::runtime_error("gemm: no cuBLASLt algorithm for rows " + std::to_string(rows) + ", out " +
                               std::to_string(out) + ", in " + std::to_string(in) + ", mode " +
                               std::to_string(static_cast<int>(mode)) + ", align " + std::to_string(align));
    }
    p.algo = res[pick].algo;
    p.ws = res[pick].workspaceSize;
    cuda().cublasLtMatmulPreferenceDestroy(pref);
    pref = nullptr;
  } catch (...) {
    if (pref) cuda().cublasLtMatmulPreferenceDestroy(pref);
    destroy(p);
    throw;
  }
  return plans_.emplace(key, p).first->second;
}

const __half* Gemm::half_bias(const float* bias, int out) {
  auto it = half_bias_.find(bias);
  if (it != half_bias_.end()) {
    if (it->second.second < out) throw std::runtime_error("gemm: bias reused with a larger out");
    return it->second.first;
  }
  // Needs a sync, so it cannot run inside graph capture. There the sync fails and this throws.
  check_rt(cuda().cudaStreamSynchronize(stream_),
           "gemm: sync for a half bias copy (run one forward pass eagerly before graph capture)");
  std::vector<float> f(out);
  check_rt(cuda().cudaMemcpyAsync(f.data(), bias, sizeof(float) * out, cudaMemcpyDeviceToHost, stream_),
           "gemm: bias to host");
  check_rt(cuda().cudaStreamSynchronize(stream_), "gemm: sync bias to host");
  std::vector<__half> h(out);
  for (int i = 0; i < out; ++i) h[i] = __float2half_rn(f[i]);
  void* dev = nullptr;
  check_rt(cuda().cudaMalloc(&dev, sizeof(__half) * out), "gemm: cudaMalloc half bias");
  cudaError_t e = cuda().cudaMemcpyAsync(dev, h.data(), sizeof(__half) * out, cudaMemcpyHostToDevice, stream_);
  if (e == cudaSuccess) e = cuda().cudaStreamSynchronize(stream_);   // h must outlive the copy
  if (e != cudaSuccess) {
    cuda().cudaFree(dev);
    check_rt(e, "gemm: half bias to device");
  }
  half_bias_.emplace(bias, std::make_pair(static_cast<__half*>(dev), out));
  return static_cast<const __half*>(dev);
}

void Gemm::linear(const __half* X, const __half* W, const float* bias, void* Y, int rows, int out, int in,
                  Mode mode) {
  const bool has_bias = bias != nullptr;
  const void* bias_dev = bias;
  const bool acc16 = mode == Mode::F16_BIAS_ACC16 || mode == Mode::F16_GELU_BIAS_ACC16;
  const bool f16_out = acc16 || mode == Mode::F16_BIAS || mode == Mode::F16_GELU_BIAS;
  if (has_bias && f16_out) bias_dev = half_bias(bias, out);

  unsigned align = align_of(X);
  if (align_of(W) < align) align = align_of(W);
  if (align_of(Y) < align) align = align_of(Y);
  if (has_bias && align_of(bias_dev) < align) align = align_of(bias_dev);

  Plan& p = get_plan(rows, out, in, mode, has_bias, align, bias_dev);
  // one plan serves every layer of a shape, so the bias pointer is set per call (host only, fine during capture)
  if (has_bias) set_desc(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias_dev, "gemm: bias pointer");
  const void* alpha = acc16 ? static_cast<const void*>(&one_h_) : static_cast<const void*>(&one_);
  const void* beta = acc16 ? static_cast<const void*>(&zero_h_)
                           : static_cast<const void*>(mode == Mode::F32_BIAS_RESIDUAL ? &one_ : &zero_);
  if (tune_ && !p.tuned)
    tune_plan(p, Key{rows, out, in, static_cast<int>(mode), has_bias ? 1 : 0, align}, X, W, alpha, beta);
  check_lt(cuda().cublasLtMatmul(lt_, p.desc, alpha, W, p.a, X, p.b, beta, Y, p.d, Y, p.d, &p.algo, ws_, p.ws,
                                 stream_),
           "gemm: cublasLtMatmul");
}

}  // namespace cudanet
