// The matrix products of MayFly's CUDA backend on cuBLASLt, loaded at runtime through cuda_dyn.h.
// See cudanet_spec.h, "the products".
//
// Every product is Y[rows, out] = X[rows, in] * W^T + bias, all row-major, W as torch's Linear stores it ([out, in]).
// FP16 inputs, FP32 accumulation and FP32 scalars (CUBLAS_COMPUTE_32F).
//   Mode::F16_BIAS           Y is __half, Y = X W^T + bias
//   Mode::F32_BIAS_RESIDUAL  Y is float, read and written in place, Y = X W^T + bias + Y
//   Mode::F32_BIAS           Y is float, Y = X W^T + bias (the heads, no residual, Y need not be cleared)
// bias is f32 on the device, length out. nullptr means no bias.
//
// For graph capture, the workspace is allocated in the constructor, and plans (descriptors, layouts, algorithm) are cached
// per (rows, out, in, mode, alignment), so a captured forward pass replays exactly the calls of the eager one. Building
// a plan is host work only and is safe during capture. One thing is not: the first F16_BIAS call with a given bias
// pointer makes a half copy of that bias (cuBLASLt wants the bias in D's type), which needs a stream sync. So run one
// forward pass (any batch size) eagerly before the first capture. A copy attempted during capture throws.
// Running each new batch size once eagerly before capturing it is also wise (cuBLASLt loads its kernels lazily).
//
// With tuning on (MAYFLY_CUDANET_TUNE=1 or 2, gemm_tune.cpp), the first eager call of each plan times cuBLASLt's candidate
// algorithms on its own X and W and keeps the fastest, cached on disk per GPU, driver and cuBLASLt version. During a
// capture only that cache is read, so the eager pass before each capture is where the tuning happens.
//
// One Gemm per stream, not thread safe. All device pointers belong to the current device.
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <tuple>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

namespace cudanet {

// F16_BIAS_ACC16: as F16_BIAS but accumulated in fp16, which GeForce cards run at twice the fp32-accumulate rate
// F16_GELU_BIAS(_ACC16): as F16_BIAS(_ACC16) with cuBLASLt's GELU epilogue (the tanh approximation, not erf)
enum class Mode { F16_BIAS, F32_BIAS_RESIDUAL, F32_BIAS, F16_BIAS_ACC16, F16_GELU_BIAS, F16_GELU_BIAS_ACC16 };

// env MAYFLY_CUDANET_TUNE: 0 (unset) the heuristic's first algorithm, 1 the fastest heuristic candidate, 2 also the
// configurable variants (gemm_tune.cpp)
int tune_level();

class Gemm {
 public:
  Gemm(cudaStream_t stream, size_t workspace_bytes = 32 << 20);
  ~Gemm();
  Gemm(const Gemm&) = delete;
  Gemm& operator=(const Gemm&) = delete;

  // Y[rows, out] = X[rows, in] * W^T + bias (+ Y in F32_BIAS_RESIDUAL). Y is __half* for F16_BIAS, float* otherwise.
  void linear(const __half* X, const __half* W, const float* bias, void* Y, int rows, int out, int in, Mode mode);

  // Builds the plan of a shape ahead of time (optional, linear builds it on first use anyway).
  void plan(int rows, int out, int in, Mode mode, bool with_bias = true);

  size_t workspace_bytes() const { return ws_bytes_; }
  size_t plan_count() const { return plans_.size(); }

 private:
  struct Plan {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr;   // W
    cublasLtMatrixLayout_t b = nullptr;   // X
    cublasLtMatrixLayout_t d = nullptr;   // Y, also C
    cublasLtMatmulAlgo_t algo{};
    size_t ws = 0;
    bool tuned = false;   // tune_plan has run (or tried) for this plan
  };
  // rows, out, in, mode, has bias, pointer alignment in bytes
  using Key = std::tuple<int, int, int, int, int, unsigned>;

  // gemm_tune.cpp: picks p's algorithm from the tuning cache, or times the candidates when not capturing
  void tune_plan(Plan& p, const Key& key, const __half* X, const __half* W, const void* alpha, const void* beta);
  int tune_ = 0;   // tune_level() at construction

  Plan& get_plan(int rows, int out, int in, Mode mode, bool has_bias, unsigned align, const void* bias_dev);
  const __half* half_bias(const float* bias, int out);
  static void destroy(Plan& p);

  cudaStream_t stream_ = nullptr;
  cublasLtHandle_t lt_ = nullptr;
  void* ws_ = nullptr;
  size_t ws_bytes_ = 0;
  std::map<Key, Plan> plans_;
  std::map<const float*, std::pair<__half*, int>> half_bias_;   // f32 bias pointer -> its half copy and length
  // alpha and beta live as long as the Gemm, so even a captured pointer to them stays valid and constant
  const float one_ = 1.0f;
  const float zero_ = 0.0f;
  const __half one_h_ = __float2half(1.0f);    // the scalars of the fp16-accumulate mode
  const __half zero_h_ = __float2half(0.0f);
};

}  // namespace cudanet
