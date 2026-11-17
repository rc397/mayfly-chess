// MayFly's own tensor core matmul (kernels_gemm.cu) for the trunk's products.
// Y[M, N] = X[M, K] W^T (+ epilogue), the same contract as gemm.h's Gemm::linear, plus the epilogues cuBLASLt
// cannot do, exact GELU (ff1) and the residual straight into the f32 stream with an fp16 accumulated product.
// cudanet.cpp uses it under MAYFLY_CUDANET_GEMM (default off). gemm_bench (bench/) measures it against cuBLASLt.
//
// One MmGemm per stream. It compiles kernels_gemm.cu once (NVRTC, cached like the other modules). Launches are plain
// kernel launches, so they capture into a CUDA graph like any other.
#pragma once
#include <memory>
#include <string>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "nvrtc_loader.h"

namespace cudanet {

// kernels_gemm.cu's epilogues
enum class MmEpi : int { Bias = 0, BiasGelu = 1, BiasResidual = 2, BiasF32 = 3 };

// one compiled configuration (kernels_gemm.cu MM_SET). acc is 0 fp32, 1 fp16, 2 fp16 within a k tile then fp32,
// and minb the blocks per SM its registers are capped for
struct MmCfg {
    int acc, bm, bn, bk, wm, wn, st, minb;
    std::string name() const;
    int threads() const { return wm * wn * 32; }
    int shmem() const { return st * (bm + bn) * bk * 2; }
    bool fits(int N, int K) const { return N % bn == 0 && K % bk == 0; }
};

// every configuration kernels_gemm.cu instantiates, acc 1 first
const MmCfg* mm_configs(int* count);

class MmGemm {
public:
    // kernel_dir holds kernels_gemm.cu. defines go in front of the source (gemm_bench passes "#define MM_BENCH_ACC 1")
    MmGemm(const std::string& kernel_dir, int device, cudaStream_t stream, const std::string& defines = "");
    ~MmGemm();
    MmGemm(const MmGemm&) = delete;
    MmGemm& operator=(const MmGemm&) = delete;

    // Yh for Bias and BiasGelu, Yf (read and written in place for BiasResidual) otherwise. bias may be null.
    void run(const MmCfg& cfg, const __half* X, const __half* W, const float* bias, __half* Yh, float* Yf, int M,
             int N, int K, MmEpi epi);

    // the configuration cudanet.cpp uses for a shape and accumulation (acc 1 or 2), picked from gemm_bench's
    // numbers on an RTX 3070 at batch 48
    static MmCfg pick(int N, int K, int acc);

private:
    CUfunction fn(const MmCfg& cfg);

    int device_ = 0;
    cudaStream_t stream_ = nullptr;
    std::unique_ptr<KernelModule> mod_;
    struct Fn;
    std::unique_ptr<Fn> fns_;
};

}  // namespace cudanet
