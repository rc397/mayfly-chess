// See mmgemm.h.
#include "mmgemm.h"

#include <map>
#include <stdexcept>

#include "cuda_dyn.h"

namespace cudanet {

namespace {

// kernels_gemm.cu MM_SET, in the same order, once per accumulation
const MmCfg kShapes[] = {
    {0, 128, 128, 32, 2, 2, 3, 2}, {0, 128, 128, 32, 2, 2, 4, 1}, {0, 128, 128, 32, 2, 4, 3, 2},
    {0, 128, 128, 32, 4, 2, 3, 2}, {0, 128, 128, 32, 2, 4, 4, 1}, {0, 128, 128, 32, 4, 2, 4, 1},
    {0, 128, 128, 64, 2, 2, 3, 1}, {0, 128, 128, 64, 4, 2, 3, 1}, {0, 128, 256, 32, 2, 4, 3, 1},
    {0, 128, 256, 32, 2, 4, 4, 1}, {0, 256, 128, 32, 4, 2, 3, 1}, {0, 256, 128, 32, 4, 2, 4, 1},
    {0, 128, 64, 32, 2, 2, 4, 2},  {0, 64, 128, 32, 2, 2, 4, 2},  {0, 128, 64, 64, 2, 2, 3, 1},
    {0, 64, 128, 64, 2, 2, 3, 1},
};
constexpr int kNShapes = sizeof(kShapes) / sizeof(kShapes[0]);
constexpr int kAccOrder[3] = {1, 2, 0};

struct Table {
    MmCfg all[3 * kNShapes];
    Table()
    {
        int n = 0;
        for (int a : kAccOrder)
            for (const MmCfg& s : kShapes) {
                all[n] = s;
                all[n].acc = a;
                ++n;
            }
    }
};

const Table& table()
{
    static const Table t;
    return t;
}

}  // namespace

std::string MmCfg::name() const
{
    return "mm" + std::to_string(acc) + "_" + std::to_string(bm) + "x" + std::to_string(bn) + "x" +
           std::to_string(bk) + "_" + std::to_string(wm) + "x" + std::to_string(wn) + "_s" + std::to_string(st) +
           "_b" + std::to_string(minb);
}

const MmCfg* mm_configs(int* count)
{
    if (count) *count = 3 * kNShapes;
    return table().all;
}

struct MmGemm::Fn {
    std::map<std::string, CUfunction> by_name;
};

MmGemm::MmGemm(const std::string& kernel_dir, int device, cudaStream_t stream, const std::string& defines)
    : device_(device), stream_(stream), fns_(std::make_unique<Fn>())
{
    std::string path = kernel_dir;
    if (!path.empty() && path.back() != '\\' && path.back() != '/') path += "\\";
    path += "kernels_gemm.cu";
    mod_ = std::make_unique<KernelModule>(defines + read_text_file(path), "kernels_gemm", device);
}

MmGemm::~MmGemm() = default;

CUfunction MmGemm::fn(const MmCfg& cfg)
{
    const std::string n = cfg.name();
    auto it = fns_->by_name.find(n);
    if (it != fns_->by_name.end()) return it->second;
    CUfunction f = mod_->get(n.c_str());
    if (cfg.shmem() > 48 * 1024) set_max_dynamic_shmem(f, cfg.shmem());
    fns_->by_name.emplace(n, f);
    return f;
}

void MmGemm::run(const MmCfg& cfg, const __half* X, const __half* W, const float* bias, __half* Yh, float* Yf, int M,
                 int N, int K, MmEpi epi)
{
    if (M <= 0 || !cfg.fits(N, K))
        throw std::runtime_error("mmgemm: " + cfg.name() + " does not fit M " + std::to_string(M) + ", N " +
                                 std::to_string(N) + ", K " + std::to_string(K));
    CUfunction f = fn(cfg);
    LaunchCfg c;
    c.gx = static_cast<unsigned>(((M + cfg.bm - 1) / cfg.bm) * (N / cfg.bn));
    c.bx = static_cast<unsigned>(cfg.threads());
    c.shmem = static_cast<unsigned>(cfg.shmem());
    int e = static_cast<int>(epi);
    void* args[] = {&X, &W, &bias, &Yh, &Yf, &M, &N, &K, &e};
    launch_kernel(f, c, stream_, args);
}

MmCfg MmGemm::pick(int N, int K, int acc)
{
    // From gemm_bench at batch 48 on an RTX 3070 (best of interleaved rounds). fp16 accumulation uses 128 x 128 x 32
    // tiles, 3 stages, 2 blocks an SM, warps 4 x 2 for out (N 1024, K 1024), 2 x 4 for ff2 (N 1024, K 2048), 2 x 2 for
    // qkv and ff1. acc 2 keeps fp32 accumulators too, so it needs the registers of one block an SM, with 128 x 128 x 64,
    // warps 4 x 2, 3 stages for every shape. Every pick must be instantiated in kernels_gemm.cu's backend list.
    if (acc == 2) return MmCfg{acc, 128, 128, 64, 4, 2, 3, 1};
    if (N == 1024 && K == 1024) return MmCfg{acc, 128, 128, 32, 4, 2, 3, 2};
    if (N == 1024) return MmCfg{acc, 128, 128, 32, 2, 4, 3, 2};
    return MmCfg{acc, 128, 128, 32, 2, 2, 3, 2};
}

}  // namespace cudanet
