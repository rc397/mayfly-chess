// kernels_gemm.cu (mmgemm.h) against cuBLASLt (gemm.h) on the trunk's four products.
//
//   gemm_bench.exe [--src DIR] [--acc 1] [--B 48[,32,...]] [--reps 30] [--shapes qkv,out,ff1,ff2] [--cfg SUBSTR]
//                  [--compile SM]
//
// For each batch size B (rows M = 65 B) and shape it makes random X (N(0,1)), four copies of W (N(0, 1/K), rotated while
// timing so W comes from DRAM as it does layer after layer), bias N(0, 0.02), and for out and ff2 a residual stream
// N(0,1). The reference is cuBLASLt with fp32 accumulation (F16_BIAS, then GELU on the host for ff1,
// F32_BIAS_RESIDUAL for out and ff2). For cuBLASLt's fp16 accumulated product (F16_BIAS_ACC16, the
// MAYFLY_CUDANET_ACC16=2 path), its fp32 one, and each kernels_gemm configuration it prints ms, TFLOPS, and the largest and
// mean absolute difference from the reference. Also the trunk kernels the fused epilogues remove (gelu_f16, and the
// residual add inside add_layernorm_f32_f16 over plain layernorm_f32_f16).
// --acc picks which accumulation's candidates to compile and run (kernels_gemm.cu MM_BENCH_ACC: 0, 1 or 2).
// --compile SM only compiles kernels_gemm.cu for sm_SM with NVRTC (no device needed, --acc -1 for the backend's own
// list) and reports the cubin size. For SM 86 it also writes the cubin into KernelModule's cache, so a timed run
// on the card does not spend card time compiling.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../cuda_dyn.h"
#include "../gemm.h"
#include "../mmgemm.h"
#include "../nvrtc_loader.h"

using namespace cudanet;

#define RT(call) check_rt(cuda().call, #call)

namespace {

struct Shape {
    const char* name;
    int N, K;
    MmEpi epi;
};
const Shape kShapes[] = {{"qkv", 3072, 1024, MmEpi::Bias},
                         {"out", 1024, 1024, MmEpi::BiasResidual},
                         {"ff1", 2048, 1024, MmEpi::BiasGelu},
                         {"ff2", 1024, 2048, MmEpi::BiasResidual}};

std::vector<int> ints(const std::string& s)
{
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        v.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return v;
}

// a copy ordered after everything already queued on any stream (the bench's stream is non-blocking, so a plain
// cudaMemcpy on the legacy stream would not wait for its kernels)
void copy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind)
{
    RT(cudaDeviceSynchronize());
    RT(cudaMemcpy(dst, src, bytes, kind));
    RT(cudaDeviceSynchronize());
}

// name contains one of the comma separated parts of filter
bool matches(const std::string& name, const std::string& filter)
{
    size_t i = 0;
    while (i <= filter.size()) {
        size_t j = filter.find(',', i);
        if (j == std::string::npos) j = filter.size();
        if (j > i && name.find(filter.substr(i, j - i)) != std::string::npos) return true;
        i = j + 1;
    }
    return false;
}

template <class T>
T* dev(size_t n)
{
    void* p = nullptr;
    RT(cudaMalloc(&p, n * sizeof(T)));
    return static_cast<T*>(p);
}

void up_half(__half* d, const std::vector<float>& f)
{
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half_rn(f[i]);
    copy(d, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice);
}

std::vector<float> down_half(const __half* d, size_t n)
{
    std::vector<__half> h(n);
    copy(h.data(), d, n * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = __half2float(h[i]);
    return f;
}

std::vector<float> down_f32(const float* d, size_t n)
{
    std::vector<float> f(n);
    copy(f.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return f;
}

std::vector<float> randn(size_t n, float sd, std::mt19937& rng)
{
    std::normal_distribution<float> nd(0.0f, sd);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng);
    return v;
}

float gelu(float v)
{
    return v * 0.5f * (1.0f + std::erf(v * 0.70710678118654752440f));
}

struct Err {
    double max = 0, mean = 0;
};

Err diff(const std::vector<float>& a, const std::vector<float>& b)
{
    Err e;
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        if (!(d <= e.max)) e.max = d;   // NaN counts as the largest
        s += d;
    }
    e.mean = s / (double)a.size();
    return e;
}

// mean ms of fn(i) over reps calls after 3 warm-up calls
template <class F>
double time_ms(cudaStream_t s, int reps, F fn)
{
    for (int i = 0; i < 3; ++i) fn(i);
    cudaEvent_t a, b;
    RT(cudaEventCreateWithFlags(&a, cudaEventDefault));
    RT(cudaEventCreateWithFlags(&b, cudaEventDefault));
    RT(cudaEventRecord(a, s));
    for (int i = 0; i < reps; ++i) fn(i);
    RT(cudaEventRecord(b, s));
    RT(cudaEventSynchronize(b));
    float t = 0;
    RT(cudaEventElapsedTime(&t, a, b));
    cuda().cudaEventDestroy(a);
    cuda().cudaEventDestroy(b);
    return t / reps;
}

// NVRTC only, no device. Compiles kernels_gemm.cu (acc < 0 for the backend's own list, else MM_BENCH_ACC acc) for
// sm_<sm>. With cache, the cubin goes where KernelModule looks for it (the same options and key as nvrtc_loader.cpp),
// so the timed run on the card loads it instead of compiling.
int compile_only(const std::string& src_dir, int acc, int sm, bool cache)
{
    std::string src = read_text_file(src_dir + "\\kernels_gemm.cu");
    if (acc >= 0) src = "#define MM_BENCH_ACC " + std::to_string(acc) + "\n" + src;
    const char* inc_env = std::getenv("MAYFLY_CUDA_INCLUDE");
    std::string inc = inc_env ? inc_env : "";
    while (!inc.empty() && (inc.back() == '\\' || inc.back() == '/' || inc.back() == ' ')) inc.pop_back();
    if (inc.empty()) inc = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include";
    const std::vector<std::string> opts = {"--gpu-architecture=sm_" + std::to_string(sm), "-std=c++17",
                                           "-default-device", "--include-path=" + inc};
    Cuda& c = cuda();
    nvrtcProgram prog = nullptr;
    check_nvrtc(c.nvrtcCreateProgram(&prog, src.c_str(), "kernels_gemm", 0, nullptr, nullptr), "create");
    std::vector<const char*> argv;
    for (const std::string& o : opts) argv.push_back(o.c_str());
    const char* verbose = std::getenv("MM_PTXAS_V");   // set: ptxas prints registers and spills (no cache then)
    if (verbose) {
        argv.push_back("--ptxas-options=-v");
        cache = false;
    }
    nvrtcResult r = c.nvrtcCompileProgram(prog, (int)argv.size(), argv.data());
    size_t ln = 0;
    c.nvrtcGetProgramLogSize(prog, &ln);
    std::string log(ln, '\0');
    if (ln) c.nvrtcGetProgramLog(prog, &log[0]);
    std::string cubin;
    if (r == NVRTC_SUCCESS) {
        size_t n = 0;
        c.nvrtcGetCUBINSize(prog, &n);
        cubin.assign(n, '\0');
        c.nvrtcGetCUBIN(prog, &cubin[0]);
    }
    c.nvrtcDestroyProgram(&prog);
    std::printf("sm_%d compile (acc %d): %s, cubin %zu bytes\n%s\n", sm, acc, r == NVRTC_SUCCESS ? "ok" : "FAILED",
                cubin.size(), log.c_str());
    if (r != NVRTC_SUCCESS) return 1;
    if (cache) {
        int vmaj = 0, vmin = 0;
        check_nvrtc(c.nvrtcVersion(&vmaj, &vmin), "nvrtcVersion");
        std::string key = src;
        key.push_back('\0');
        for (const std::string& o : opts) {
            key += o;
            key.push_back('\0');
        }
        key += "nvrtc " + std::to_string(vmaj) + "." + std::to_string(vmin);
        const char* lad = std::getenv("LOCALAPPDATA");
        if (!lad) throw std::runtime_error("no LOCALAPPDATA");
        const std::string dir = std::string(lad) + "\\mayfly_cudanet";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        const std::string path = dir + "\\" + sha1_hex(key) + "_sm" + std::to_string(sm) + ".cubin";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + path);
        std::fwrite(cubin.data(), 1, cubin.size(), f);
        std::fclose(f);
        std::printf("cached %s\n", path.c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    // --src defaults to chess\cudanet relative to the working directory (the repository root)
    std::string src = "chess\\cudanet", shapes = "qkv,out,ff1,ff2", filter;
    std::vector<int> Bs = {48};
    int reps = 30, acc = 1, rounds = 5;
    int compile_sm = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
            return argv[++i];
        };
        if (a == "--src") src = next();
        else if (a == "--B") Bs = ints(next());
        else if (a == "--reps") reps = std::atoi(next().c_str());
        else if (a == "--acc") acc = std::atoi(next().c_str());
        else if (a == "--rounds") rounds = std::atoi(next().c_str());
        else if (a == "--shapes") shapes = next();
        else if (a == "--cfg") filter = next();
        else if (a == "--compile") compile_sm = std::atoi(next().c_str());
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    try {
        if (compile_sm) return compile_only(src, acc, compile_sm, compile_sm == 86);
        RT(cudaSetDevice(0));
        cudaStream_t s = nullptr;
        RT(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        MmGemm mm(src, 0, s, "#define MM_BENCH_ACC " + std::to_string(acc) + "\n");
        KernelModule trunk(read_text_file(src + "\\kernels_trunk.cu"), "kernels_trunk", 0);
        CUfunction k_gelu = trunk.get("gelu_f16"), k_ln = trunk.get("layernorm_f32_f16"),
                   k_add_ln = trunk.get("add_layernorm_f32_f16");
        int ncfg = 0;
        const MmCfg* cfgs = mm_configs(&ncfg);
        std::mt19937 rng(1234);

        for (int B : Bs) {
            const int M = 65 * B;
            for (const Shape& sh : kShapes) {
                if (shapes.find(sh.name) == std::string::npos) continue;
                const int N = sh.N, K = sh.K;
                const size_t mk = (size_t)M * K, nk = (size_t)N * K, mn = (size_t)M * N;
                const double gflop = 2.0 * M * N * K * 1e-9;
                __half* X = dev<__half>(mk);
                __half* Wc[4];
                for (auto& w : Wc) w = dev<__half>(nk);
                float* bias = dev<float>(N);
                float* x0 = dev<float>(mn);
                float* Yf = dev<float>(mn);
                __half* Yh = dev<__half>(mn);
                up_half(X, randn(mk, 1.0f, rng));
                for (auto& w : Wc) up_half(w, randn(nk, 1.0f / std::sqrt((float)K), rng));
                const std::vector<float> hb = randn(N, 0.02f, rng), hx0 = randn(mn, 1.0f, rng);
                copy(bias, hb.data(), N * sizeof(float), cudaMemcpyHostToDevice);
                copy(x0, hx0.data(), mn * sizeof(float), cudaMemcpyHostToDevice);
                const bool res = sh.epi == MmEpi::BiasResidual;
                // a fresh cuBLASLt wrapper per shape, because Gemm caches the half copy of a bias by its pointer and a new
                // bias buffer can reuse a freed one's address
                Gemm lt(s);

                // the reference: fp32 accumulation on cuBLASLt, with W copy 0
                std::vector<float> ref, prod, raw;   // prod is the product alone, raw is before GELU
                if (res) {
                    copy(Yf, x0, mn * sizeof(float), cudaMemcpyDeviceToDevice);
                    lt.linear(X, Wc[0], bias, Yf, M, N, K, Mode::F32_BIAS_RESIDUAL);
                    ref = down_f32(Yf, mn);
                    prod.resize(mn);
                    for (size_t i = 0; i < mn; ++i) prod[i] = ref[i] - hx0[i];
                } else {
                    lt.linear(X, Wc[0], bias, Yh, M, N, K, Mode::F16_BIAS);
                    ref = down_half(Yh, mn);
                    raw = ref;
                    if (sh.epi == MmEpi::BiasGelu)
                        for (float& v : ref) v = gelu(v);
                    prod = ref;
                }
                std::printf("\nB=%d %s: M=%d N=%d K=%d (%.2f GFLOP), best of %d rounds of %d calls\n", B, sh.name, M,
                            N, K, gflop, rounds, reps);
                std::printf("  %-26s %8s %8s %10s %10s\n", "kernel", "ms", "TFLOPS", "max err", "mean err");

                // the contenders, cuBLASLt fp16 accumulation (the product alone into half, as ACC16=2 runs it),
                // cuBLASLt fp32 accumulation, then each kernels_gemm candidate. Errors first, then timing rounds.
                struct Row {
                    std::string name;
                    std::function<void(int)> fn;
                    Err e;
                    bool has_err = false;
                    double best = 1e30;
                };
                std::vector<Row> rows_;
                {
                    Row r{"cublasLt acc16", [&](int i) {
                              lt.linear(X, Wc[i & 3], bias, Yh, M, N, K, Mode::F16_BIAS_ACC16);
                          }};
                    r.fn(0);
                    std::vector<float> got = down_half(Yh, mn);
                    if (sh.epi == MmEpi::BiasGelu)
                        for (float& v : got) v = gelu(v);
                    r.e = diff(got, prod);
                    r.has_err = true;
                    rows_.push_back(r);
                }
                rows_.push_back(Row{res ? "cublasLt acc32 residual" : "cublasLt acc32", [&](int i) {
                                        if (res) lt.linear(X, Wc[i & 3], bias, Yf, M, N, K, Mode::F32_BIAS_RESIDUAL);
                                        else lt.linear(X, Wc[i & 3], bias, Yh, M, N, K, Mode::F16_BIAS);
                                    }});
                bool raw_done = false;
                for (int ci = 0; ci < ncfg; ++ci) {
                    const MmCfg cfg = cfgs[ci];
                    const std::string name = cfg.name();
                    if (cfg.acc != acc) continue;
                    if (!filter.empty() && !matches(name, filter)) continue;
                    if (!cfg.fits(N, K)) continue;
                    if (sh.epi == MmEpi::BiasGelu && !raw_done) {
                        // the product without GELU once, against the reference before GELU
                        mm.run(cfg, X, Wc[0], bias, Yh, Yf, M, N, K, MmEpi::Bias);
                        const Err e = diff(down_half(Yh, mn), raw);
                        std::printf("  %-26s %8s %8s %10.3g %10.3g\n", (name + " no gelu").c_str(), "", "", e.max,
                                    e.mean);
                        raw_done = true;
                    }
                    if (res) copy(Yf, x0, mn * sizeof(float), cudaMemcpyDeviceToDevice);
                    mm.run(cfg, X, Wc[0], bias, Yh, Yf, M, N, K, sh.epi);
                    const std::vector<float> got = res ? down_f32(Yf, mn) : down_half(Yh, mn);
                    Row r{name, [&mm, cfg, X, &Wc, bias, Yh, Yf, M, N, K, &sh](int i) {
                              mm.run(cfg, X, Wc[i & 3], bias, Yh, Yf, M, N, K, sh.epi);
                          }};
                    r.e = diff(got, ref);
                    r.has_err = true;
                    rows_.push_back(r);
                }
                for (int round = 0; round < rounds; ++round)
                    for (Row& r : rows_) r.best = std::min(r.best, time_ms(s, reps, r.fn));
                for (const Row& r : rows_) {
                    if (r.has_err)
                        std::printf("  %-26s %8.4f %8.2f %10.3g %10.3g\n", r.name.c_str(), r.best, gflop / r.best,
                                    r.e.max, r.e.mean);
                    else
                        std::printf("  %-26s %8.4f %8.2f\n", r.name.c_str(), r.best, gflop / r.best);
                }
                // what the fused epilogues remove
                if (sh.epi == MmEpi::BiasGelu) {
                    const int n = M * N;
                    LaunchCfg c;
                    c.gx = std::min(8192u, (unsigned)((n / 8 + 255) / 256));
                    c.bx = 256;
                    const double t = time_ms(s, reps, [&](int) {
                        __half* p = Yh;
                        int nn = n;
                        void* a[] = {&p, &nn};
                        launch_kernel(k_gelu, c, s, a);
                    });
                    std::printf("  %-26s %8.4f\n", "gelu_f16 (removed)", t);
                }
                if (res) {
                    __half* h = Yh;
                    LaunchCfg c;
                    c.gx = M;
                    c.bx = 256;
                    int rows = M, d = N;
                    float eps = 1e-5f;
                    float* xx = Yf;
                    const float* g = bias;   // any finite vectors serve as gamma and beta
                    const double t_ln = time_ms(s, reps, [&](int) {
                        void* a[] = {&xx, &g, &g, &h, &rows, &d, &eps};
                        launch_kernel(k_ln, c, s, a);
                    });
                    const double t_add = time_ms(s, reps, [&](int) {
                        void* a[] = {&xx, &h, &g, &g, &h, &rows, &d, &eps};
                        launch_kernel(k_add_ln, c, s, a);
                    });
                    std::printf("  %-26s %8.4f  (add_layernorm %.4f - layernorm %.4f)\n", "residual add (removed)",
                                t_add - t_ln, t_add, t_ln);
                }
                std::fflush(stdout);
                void* bufs[] = {X, Wc[0], Wc[1], Wc[2], Wc[3], bias, x0, Yf, Yh};
                for (void* p : bufs) cuda().cudaFree(p);
            }
        }
        RT(cudaStreamSynchronize(s));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gemm_bench: %s\n", e.what());
        return 1;
    }
    return 0;
}
