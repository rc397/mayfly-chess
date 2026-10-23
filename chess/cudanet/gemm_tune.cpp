// cuBLASLt algorithm autotuning for the products (MAYFLY_CUDANET_TUNE). See gemm.h.
//
// gemm.cpp takes cuBLASLt's first heuristic algorithm for each plan. With MAYFLY_CUDANET_TUNE=1 the first eager call
// of each plan (rows, out, in, mode, bias, alignment) gathers many candidates instead (the best-fit heuristic with 64
// results, and the heuristic limited to each algorithm id, 16 results each), times every one on the call's own X and
// W into a scratch output, as a CUDA graph of products (the way the search replays them) each after a write of twice
// the L2 (so W comes from memory, as every layer's weights do in a real pass, with the flush's cost taken off),
// and keeps the fastest. MAYFLY_CUDANET_TUNE=2 adds a sweep of the configurable variants that cublasLtMatmulAlgoCheck
// accepts (tile, stages, split-K and its reduction scheme, CTA swizzle, custom option), at most MAYFLY_CUDANET_TUNE_MAX
// of them (400). MAYFLY_CUDANET_TUNE_FLUSH=0 times without the flush (a warm L2), and MAYFLY_CUDANET_TUNE_LOG=0 keeps the
// one line a tuned plan prints to stderr quiet. f16 outputs keep the reduction rule of gemm.cpp (split-K partial sums
// in the compute type, never in the f16 output).
//
// The choice. A short screen of every candidate, then the heuristic's pick and the six best in 8 interleaved rounds.
// The heuristic's pick stays unless the best median is at least 2% faster and it won at least 6 of the 8 rounds pair
// by pair. Another process on the card makes the timings nonsense, so a choice counts only when the card looked quiet
// (the median round within 20% of the fastest, for both). Otherwise this process keeps the heuristic's pick, nothing
// is saved and the next process tunes again.
//
// The cache. The choices persist in one text file per GPU name, compute capability, SM count, driver (the CUDA driver
// version and, when NVML loads, the display driver's), cuBLASLt version, cudart version and workspace size, kept in
// %LOCALAPPDATA%\mayfly_cudanet\gemm_tune_<sha1 of that identity>.txt, or MAYFLY_CUDANET_TUNE_FILE. So the tuning is
// paid once per machine and library. cuBLASLt documents that an algorithm descriptor may be stored as bytes and reused
// with the same library version. Each one read back is checked again with cublasLtMatmulAlgoCheck. An entry tuned at
// level 2 serves level 1 too. Several processes may share the file, since a save merges with what is there and
// replaces the file in one rename.
//
// Graph capture. Timing syncs the stream, so it runs only while the stream is not capturing. CudaNet runs every batch
// size eagerly before it captures it, so every plan is tuned in that eager pass. During a capture only the cache is
// read (host work). Any failure keeps the heuristic's algorithm and never fails the product.
#include "gemm.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_dyn.h"
#include "nvrtc_loader.h"   // sha1_hex

namespace fs = std::filesystem;

namespace cudanet {

namespace {

constexpr int kHeurAlgos = 64;     // best-fit heuristic results asked for
constexpr int kPerIdAlgos = 16;    // results asked for per algorithm id
constexpr int kFinalists = 6;      // re-timed after the screen, beside the heuristic's pick
constexpr int kRounds = 8;         // final rounds, and the winner must beat the heuristic's pick in 3 of 4
constexpr double kMargin = 0.98;   // another algorithm must beat the heuristic's pick by 2%
constexpr double kQuiet = 1.20;    // a quiet card: each finalist's median round within 20% of its fastest

// the cuBLASLt calls only the tuner needs, from the cublasLt64_12.dll that cuda_dyn loaded
struct LtTune {
    decltype(&::cublasLtMatmulAlgoGetIds) getIds = nullptr;
    decltype(&::cublasLtMatmulAlgoInit) init = nullptr;
    decltype(&::cublasLtMatmulAlgoCheck) check = nullptr;
    decltype(&::cublasLtMatmulAlgoCapGetAttribute) capGet = nullptr;
    decltype(&::cublasLtMatmulAlgoConfigSetAttribute) cfgSet = nullptr;
    decltype(&::cublasLtMatmulAlgoConfigGetAttribute) cfgGet = nullptr;
    bool ok = false;
};

std::wstring widen(const std::string& s)
{
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n > 0 ? n : 0, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
    return w;
}

const LtTune& lt_tune()
{
    static LtTune t = [] {
        LtTune x;
        HMODULE h = GetModuleHandleW(widen(cuda().cublaslt_path).c_str());
        if (!h) h = GetModuleHandleW(L"cublasLt64_12.dll");
        if (!h) return x;
        x.getIds = reinterpret_cast<decltype(x.getIds)>(GetProcAddress(h, "cublasLtMatmulAlgoGetIds"));
        x.init = reinterpret_cast<decltype(x.init)>(GetProcAddress(h, "cublasLtMatmulAlgoInit"));
        x.check = reinterpret_cast<decltype(x.check)>(GetProcAddress(h, "cublasLtMatmulAlgoCheck"));
        x.capGet = reinterpret_cast<decltype(x.capGet)>(GetProcAddress(h, "cublasLtMatmulAlgoCapGetAttribute"));
        x.cfgSet = reinterpret_cast<decltype(x.cfgSet)>(GetProcAddress(h, "cublasLtMatmulAlgoConfigSetAttribute"));
        x.cfgGet = reinterpret_cast<decltype(x.cfgGet)>(GetProcAddress(h, "cublasLtMatmulAlgoConfigGetAttribute"));
        x.ok = x.getIds && x.init && x.check && x.capGet && x.cfgSet && x.cfgGet;
        return x;
    }();
    return t;
}

int env_int(const char* name, int fallback)
{
    const char* e = std::getenv(name);
    return (e && e[0]) ? std::atoi(e) : fallback;
}

// ---------------------------------------------------------------------------------------------------- the cache
// the display driver's version through NVML (System32, or NVSMI on old drivers), empty when NVML is not there
std::string display_driver()
{
    HMODULE h = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!h) h = LoadLibraryW(L"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    if (!h) return std::string();
    using Init = int (*)();
    using Ver = int (*)(char*, unsigned);
    using Shut = int (*)();
    auto init = reinterpret_cast<Init>(GetProcAddress(h, "nvmlInit_v2"));
    auto ver = reinterpret_cast<Ver>(GetProcAddress(h, "nvmlSystemGetDriverVersion"));
    auto shut = reinterpret_cast<Shut>(GetProcAddress(h, "nvmlShutdown"));
    std::string out;
    if (init && ver && init() == 0) {
        char buf[96] = {};
        if (ver(buf, sizeof(buf)) == 0) out = buf;
        if (shut) shut();
    }
    FreeLibrary(h);
    return out;
}

// what a tuned choice is valid for (the current device), built once per device and workspace size
std::string identity(size_t ws_bytes)
{
    Cuda& c = cuda();
    int dev = 0;
    c.cudaGetDevice(&dev);
    static std::map<std::pair<int, size_t>, std::string> known;
    auto it = known.find({dev, ws_bytes});
    if (it != known.end()) return it->second;
    cudaDeviceProp prop{};
    c.cudaGetDeviceProperties(&prop, dev);
    int drv = 0, rt = 0;
    c.cuDriverGetVersion(&drv);
    c.cudaRuntimeGetVersion(&rt);
    char buf[640];
    std::snprintf(buf, sizeof(buf), "gpu=%s|sm=%d%d|sms=%d|cuda_driver=%d|driver=%s|cublaslt=%zu|cudart=%d|ws=%zu|v=1",
                  prop.name, prop.major, prop.minor, prop.multiProcessorCount, drv, display_driver().c_str(),
                  c.cublasLtGetVersion(), rt, ws_bytes);
    known[{dev, ws_bytes}] = buf;
    return buf;
}

fs::path cache_file(const std::string& ident)
{
    const char* f = std::getenv("MAYFLY_CUDANET_TUNE_FILE");
    if (f && f[0]) return fs::path(widen(f));
    wchar_t buf[MAX_PATH * 2];
    DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", buf, (DWORD)(sizeof(buf) / sizeof(buf[0])));
    fs::path dir = (n > 0 && n < sizeof(buf) / sizeof(buf[0])) ? fs::path(std::wstring(buf, n)) : fs::temp_directory_path();
    dir /= L"mayfly_cudanet";
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir / widen("gemm_tune_" + sha1_hex(ident).substr(0, 16) + ".txt");
}

struct Entry {
    cublasLtMatmulAlgo_t algo{};
    size_t ws = 0;
    int level = 0;
    double ms = 0.0, heur_ms = 0.0;
    std::string what;   // the configuration in words, for people reading the file
};

// one file per identity, keyed by "rows out in mode bias align"
struct Store {
    std::string ident;
    fs::path path;
    std::map<std::string, Entry> entries;
};

std::mutex g_mu;

std::string hex_algo(const cublasLtMatmulAlgo_t& a)
{
    std::string s;
    char b[17];
    for (uint64_t v : a.data) {
        std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v));
        s += b;
    }
    return s;
}

bool parse_algo(const std::string& s, cublasLtMatmulAlgo_t& a)
{
    if (s.size() != 16 * 8) return false;
    for (int i = 0; i < 8; ++i) a.data[i] = std::strtoull(s.substr(16 * i, 16).c_str(), nullptr, 16);
    return true;
}

// the entries of the file at path when its identity line is ident
std::map<std::string, Entry> read_file(const fs::path& path, const std::string& ident)
{
    std::map<std::string, Entry> out;
    std::ifstream in(path);
    if (!in) return out;
    std::string line;
    if (!std::getline(in, line) || !std::getline(in, line) || line != "# " + ident) return out;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int r, o, i, m, b;
        unsigned al;
        Entry e;
        std::string hx;
        if (!(ss >> r >> o >> i >> m >> b >> al >> e.level >> e.ws >> e.ms >> e.heur_ms >> hx)) continue;
        if (!parse_algo(hx, e.algo)) continue;
        std::getline(ss, e.what);
        out[std::to_string(r) + " " + std::to_string(o) + " " + std::to_string(i) + " " + std::to_string(m) + " " +
            std::to_string(b) + " " + std::to_string(al)] = e;
    }
    return out;
}

// merged with what other processes wrote, then replaced in one rename. Failures only lose the cache.
void save(Store& s)
{
    std::map<std::string, Entry> merged = read_file(s.path, s.ident);
    for (auto& kv : s.entries) merged[kv.first] = kv.second;
    s.entries = merged;
    fs::path tmp = s.path;
    tmp += widen(".tmp" + std::to_string(GetCurrentProcessId()));
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) return;
        f << "# mayfly cudanet gemm tune (gemm_tune.cpp): rows out in mode bias align level ws ms heur_ms algo what\n";
        f << "# " << s.ident << "\n";
        char num[64];
        for (auto& kv : s.entries) {
            const Entry& e = kv.second;
            f << kv.first << " " << e.level << " " << e.ws;
            std::snprintf(num, sizeof(num), " %.5f %.5f ", e.ms, e.heur_ms);
            f << num << hex_algo(e.algo) << e.what << "\n";
        }
        if (!f) return;
    }
    std::error_code ec;
    fs::rename(tmp, s.path, ec);
    if (ec) fs::remove(tmp, ec);
}

// the store of the current device and workspace size, read from disk on first use (call with g_mu held)
Store& store_for(size_t ws_bytes)
{
    static std::map<std::string, Store> stores;
    const std::string ident = identity(ws_bytes);
    auto it = stores.find(ident);
    if (it != stores.end()) return it->second;
    Store s;
    s.ident = ident;
    s.path = cache_file(ident);
    s.entries = read_file(s.path, ident);
    return stores.emplace(ident, std::move(s)).first->second;
}

// ---------------------------------------------------------------------------------------------------- timing
// one product on the call's X and W into the scratch output S (also C, so the residual mode adds into S)
struct Bench {
    cublasLtHandle_t lt;
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t a, b, d;
    const void *alpha, *beta, *W, *X;
    void* S;
    void* ws;
    cudaStream_t stream;
    // written before each timed product so W and X come from memory, as each layer's weights do in a real pass
    // (back to back on the same W would time a warm L2, 4 MB on an RTX 3070 and 48 MB on an RTX 5070 Ti)
    void* flush = nullptr;
    size_t flush_bytes = 0;

    cublasStatus_t run(const cublasLtMatmulAlgo_t& algo, size_t wsz) const
    {
        return cuda().cublasLtMatmul(lt, desc, alpha, W, a, X, b, beta, S, d, S, d, &algo, ws, wsz, stream);
    }
};

// n products back to back as one graph, each after the L2 flush, null when the algorithm fails. One eager call comes
// first, which loads the kernel and refuses a bad configuration outside any capture. A null algo times the flushes alone.
cudaGraphExec_t make_graph(const Bench& c, const cublasLtMatmulAlgo_t* algo, size_t wsz, int n)
{
    Cuda& cu = cuda();
    if (algo && (c.run(*algo, wsz) != CUBLAS_STATUS_SUCCESS || cu.cudaStreamSynchronize(c.stream) != cudaSuccess)) {
        cu.cudaGetLastError();
        return nullptr;
    }
    if (cu.cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        cu.cudaGetLastError();
        return nullptr;
    }
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) {
        if (c.flush) ok = cu.cudaMemsetAsync(c.flush, i & 0xff, c.flush_bytes, c.stream) == cudaSuccess;
        if (ok && algo) ok = c.run(*algo, wsz) == CUBLAS_STATUS_SUCCESS;
    }
    cudaGraph_t g = nullptr;
    const cudaError_t e = cu.cudaStreamEndCapture(c.stream, &g);
    cudaGraphExec_t x = nullptr;
    if (ok && e == cudaSuccess && g && cu.cudaGraphInstantiate(&x, g, 0) != cudaSuccess) x = nullptr;
    if (g) cu.cudaGraphDestroy(g);
    if (!x) cu.cudaGetLastError();
    return x;
}

// ms a product, the fastest of `launches` timed launches of a graph of n products, less `less` ms (the flush),
// or a huge number on failure
double time_graph(cudaGraphExec_t x, int n, int launches, cudaStream_t s, cudaEvent_t a, cudaEvent_t b,
                  double less = 0.0)
{
    Cuda& cu = cuda();
    double best = 1e30;
    for (int l = 0; l < launches; ++l) {
        if (cu.cudaEventRecord(a, s) != cudaSuccess || cu.cudaGraphLaunch(x, s) != cudaSuccess ||
            cu.cudaEventRecord(b, s) != cudaSuccess || cu.cudaEventSynchronize(b) != cudaSuccess) {
            cu.cudaGetLastError();
            return 1e30;
        }
        float t = 0.0f;
        if (cu.cudaEventElapsedTime(&t, a, b) == cudaSuccess) best = std::min(best, double(t) / n - less);
    }
    return best;
}

double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n == 0 ? 1e30 : (n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

struct Cand {
    cublasLtMatmulAlgo_t algo{};
    size_t ws = 0;
    double ms = 1e30;
};

void add_cand(std::vector<Cand>& v, const cublasLtMatmulAlgo_t& a, size_t ws)
{
    for (const Cand& c : v)
        if (std::memcmp(&c.algo, &a, sizeof(a)) == 0) return;
    Cand c;
    c.algo = a;
    c.ws = ws;
    v.push_back(c);
}

template <typename T>
bool pref_set(cublasLtMatmulPreference_t p, cublasLtMatmulPreferenceAttributes_t attr, const T& v)
{
    return cuda().cublasLtMatmulPreferenceSetAttribute(p, attr, &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
}

template <typename T>
T cfg_get(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoConfigAttributes_t attr)
{
    T v{};
    size_t n = 0;
    lt_tune().cfgGet(&a, attr, &v, sizeof(v), &n);
    return v;
}

template <typename T>
bool cfg_set(cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoConfigAttributes_t attr, const T& v)
{
    return lt_tune().cfgSet(&a, attr, &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
}

std::string describe(const cublasLtMatmulAlgo_t& a)
{
    char b[160];
    std::snprintf(b, sizeof(b), " algo %d tile %u stages %u splitk %d reduction %u swizzle %u custom %u",
                  cfg_get<int32_t>(a, CUBLASLT_ALGO_CONFIG_ID), cfg_get<uint32_t>(a, CUBLASLT_ALGO_CONFIG_TILE_ID),
                  cfg_get<uint32_t>(a, CUBLASLT_ALGO_CONFIG_STAGES_ID),
                  cfg_get<int32_t>(a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
                  cfg_get<uint32_t>(a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),
                  cfg_get<uint32_t>(a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
                  cfg_get<uint32_t>(a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION));
    return b;
}

std::vector<uint32_t> cap_list(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr)
{
    size_t bytes = 0;
    if (lt_tune().capGet(&a, attr, nullptr, 0, &bytes) != CUBLAS_STATUS_SUCCESS || bytes == 0) return {0u};
    std::vector<uint32_t> v(bytes / sizeof(uint32_t));
    size_t got = 0;
    if (lt_tune().capGet(&a, attr, v.data(), v.size() * sizeof(uint32_t), &got) != CUBLAS_STATUS_SUCCESS) return {0u};
    v.resize(got / sizeof(uint32_t));
    if (v.empty()) v.push_back(0u);
    return v;
}

template <typename T>
T cap_get(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr)
{
    T v{};
    size_t n = 0;
    lt_tune().capGet(&a, attr, &v, sizeof(v), &n);
    return v;
}

}  // namespace

int tune_level()
{
    const char* e = std::getenv("MAYFLY_CUDANET_TUNE");
    if (!e || !e[0]) return 0;
    const int v = std::atoi(e);
    return v < 0 ? 0 : (v > 2 ? 2 : v);
}

void Gemm::tune_plan(Plan& p, const Key& key, const __half* X, const __half* W, const void* alpha, const void* beta)
{
    p.tuned = true;   // one attempt per plan, whatever happens
    const LtTune& lt = lt_tune();
    if (!lt.ok) return;
    const int rows = std::get<0>(key), out = std::get<1>(key), in = std::get<2>(key);
    const Mode mode = static_cast<Mode>(std::get<3>(key));
    const unsigned align = std::get<5>(key);
    const bool acc16 = mode == Mode::F16_BIAS_ACC16;
    const bool f16_out = mode == Mode::F16_BIAS || acc16;
    const cudaDataType_t dtype = f16_out ? CUDA_R_16F : CUDA_R_32F;
    const cublasComputeType_t ctype = acc16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;
    const cudaDataType_t stype = acc16 ? CUDA_R_16F : CUDA_R_32F;
    // the reduction schemes gemm.cpp allows, since f16 outputs never sum split-K parts in the output itself
    const uint32_t red_allowed = f16_out ? (CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)
                                         : uint32_t(CUBLASLT_REDUCTION_SCHEME_MASK);
    const std::string k = std::to_string(rows) + " " + std::to_string(out) + " " + std::to_string(in) + " " +
                          std::to_string(std::get<3>(key)) + " " + std::to_string(std::get<4>(key)) + " " +
                          std::to_string(align);
    Cuda& cu = cuda();

    // the cache first, host work only, so this is fine even during a capture
    std::lock_guard<std::mutex> lock(g_mu);
    Store* store = nullptr;
    try {
        store = &store_for(ws_bytes_);
    } catch (...) {
        return;
    }
    auto hit = store->entries.find(k);
    if (hit != store->entries.end() && hit->second.level >= tune_) {
        cublasLtMatmulHeuristicResult_t r{};
        if (lt.check(lt_, p.desc, p.a, p.b, p.d, p.d, &hit->second.algo, &r) == CUBLAS_STATUS_SUCCESS &&
            r.workspaceSize <= ws_bytes_) {
            p.algo = hit->second.algo;
            p.ws = r.workspaceSize;
            return;
        }
    }

    // timing syncs the stream, so never during a capture
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cu.cudaStreamIsCapturing(stream_, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) {
        cu.cudaGetLastError();
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();

    // the candidates, the heuristic's pick first, then best fit, then per algorithm id
    std::vector<Cand> cands;
    add_cand(cands, p.algo, p.ws);
    size_t n_best = 1;
    cublasLtMatmulPreference_t pref = nullptr;
    if (cu.cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return;
    const uint64_t wsb = ws_bytes_;
    const uint32_t al = align;
    bool pok = pref_set(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, wsb) &&
               pref_set(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, al) &&
               pref_set(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, al) &&
               pref_set(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, al) &&
               pref_set(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, al);
    if (pok && f16_out) pok = pref_set(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, red_allowed);
    std::vector<int> ids(256);
    int n_ids = 0;
    if (lt.getIds(lt_, ctype, stype, CUDA_R_16F, CUDA_R_16F, dtype, dtype, (int)ids.size(), ids.data(), &n_ids) !=
        CUBLAS_STATUS_SUCCESS)
        n_ids = 0;
    ids.resize(n_ids);
    if (pok) {
        std::vector<cublasLtMatmulHeuristicResult_t> res(kHeurAlgos);
        int n = 0;
        if (cu.cublasLtMatmulAlgoGetHeuristic(lt_, p.desc, p.a, p.b, p.d, p.d, pref, kHeurAlgos, res.data(), &n) ==
            CUBLAS_STATUS_SUCCESS)
            for (int i = 0; i < n; ++i)
                if (res[i].state == CUBLAS_STATUS_SUCCESS && res[i].workspaceSize <= ws_bytes_)
                    add_cand(cands, res[i].algo, res[i].workspaceSize);
        n_best = cands.size();
        const uint32_t by_id = CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID;
        if (pref_set(pref, CUBLASLT_MATMUL_PREF_SEARCH_MODE, by_id)) {
            for (int id : ids) {
                cublasLtMatmulAlgo_t a{};
                if (lt.init(lt_, ctype, stype, CUDA_R_16F, CUDA_R_16F, dtype, dtype, id, &a) != CUBLAS_STATUS_SUCCESS)
                    continue;
                std::vector<cublasLtMatmulHeuristicResult_t> r(kPerIdAlgos);
                for (auto& x : r) x.algo = a;   // the id to search, given as an initialised algorithm
                int m = 0;
                if (cu.cublasLtMatmulAlgoGetHeuristic(lt_, p.desc, p.a, p.b, p.d, p.d, pref, kPerIdAlgos, r.data(),
                                                      &m) != CUBLAS_STATUS_SUCCESS)
                    continue;
                for (int i = 0; i < m; ++i)
                    if (r[i].state == CUBLAS_STATUS_SUCCESS && r[i].workspaceSize <= ws_bytes_)
                        add_cand(cands, r[i].algo, r[i].workspaceSize);
            }
        }
    }
    cu.cublasLtMatmulPreferenceDestroy(pref);
    const size_t n_heur = cands.size();

    // level 2: every configuration AlgoCheck accepts, a fixed random sample when there are too many
    if (tune_ >= 2) {
        const int kSplits[] = {1, 2, 3, 4, 6, 8, 12, 16};
        std::vector<Cand> sweep;
        for (int id : ids) {
            cublasLtMatmulAlgo_t a{};
            if (lt.init(lt_, ctype, stype, CUDA_R_16F, CUDA_R_16F, dtype, dtype, id, &a) != CUBLAS_STATUS_SUCCESS)
                continue;
            const std::vector<uint32_t> tiles = cap_list(a, CUBLASLT_ALGO_CAP_TILE_IDS);
            const std::vector<uint32_t> stages = cap_list(a, CUBLASLT_ALGO_CAP_STAGES_IDS);
            const int32_t splitk = cap_get<int32_t>(a, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT);
            const uint32_t red_mask = cap_get<uint32_t>(a, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK) & red_allowed;
            const uint32_t swz = cap_get<uint32_t>(a, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
            const int32_t custom_max = std::min(cap_get<int32_t>(a, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX), 3);
            for (uint32_t tile : tiles)
                for (uint32_t st : stages)
                    for (int32_t cus = 0; cus <= custom_max; ++cus)
                        for (uint32_t sw = 0; sw <= (swz ? 1u : 0u); ++sw)
                            for (int s : kSplits) {
                                if (s > 1 && (!splitk || in / s < 64)) continue;
                                for (uint32_t red = 0; red <= 4; red = red ? red << 1 : 1) {
                                    // one split needs no reduction, more need an allowed scheme
                                    if ((s == 1) != (red == 0)) continue;
                                    if (red && !(red_mask & red)) continue;
                                    cublasLtMatmulAlgo_t v = a;
                                    if (!cfg_set(v, CUBLASLT_ALGO_CONFIG_TILE_ID, tile) ||
                                        !cfg_set(v, CUBLASLT_ALGO_CONFIG_STAGES_ID, st) ||
                                        !cfg_set(v, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, uint32_t(cus)) ||
                                        !cfg_set(v, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, sw) ||
                                        !cfg_set(v, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, int32_t(s)) ||
                                        !cfg_set(v, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, red))
                                        continue;
                                    cublasLtMatmulHeuristicResult_t r{};
                                    if (lt.check(lt_, p.desc, p.a, p.b, p.d, p.d, &v, &r) == CUBLAS_STATUS_SUCCESS &&
                                        r.workspaceSize <= ws_bytes_)
                                        add_cand(sweep, v, r.workspaceSize);
                                }
                            }
        }
        const size_t cap = static_cast<size_t>(std::max(1, env_int("MAYFLY_CUDANET_TUNE_MAX", 400)));
        if (sweep.size() > cap) {
            std::mt19937 rng(12345);
            std::shuffle(sweep.begin(), sweep.end(), rng);
            sweep.resize(cap);
        }
        for (const Cand& c : sweep) add_cand(cands, c.algo, c.ws);
    }

    // a scratch output, so the real Y (and the residual stream) stay as they are
    const size_t d_bytes = static_cast<size_t>(rows) * out * (f16_out ? 2 : 4);
    void* S = nullptr;
    if (cu.cudaMalloc(&S, d_bytes) != cudaSuccess) {
        cu.cudaGetLastError();
        return;
    }
    cudaEvent_t ea = nullptr, eb = nullptr;
    cu.cudaEventCreateWithFlags(&ea, cudaEventDefault);
    cu.cudaEventCreateWithFlags(&eb, cudaEventDefault);
    Bench b{lt_, p.desc, p.a, p.b, p.d, alpha, beta, W, X, S, ws_, stream_};
    // the L2 flush: twice the L2, at least 8 MB (MAYFLY_CUDANET_TUNE_FLUSH=0 times a warm L2 instead)
    if (env_int("MAYFLY_CUDANET_TUNE_FLUSH", 1)) {
        int dev = 0, l2 = 0;
        cu.cudaGetDevice(&dev);
        cu.cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
        b.flush_bytes = std::max<size_t>(size_t(2) * size_t(l2 > 0 ? l2 : 0), size_t(8) << 20);
        if (cu.cudaMalloc(&b.flush, b.flush_bytes) != cudaSuccess) {
            cu.cudaGetLastError();
            b.flush = nullptr;
        }
    }
    double base_ms = 1e30, flush_ms = 0.0;
    size_t timed = 0, pick = 0;   // pick: the chosen candidate (0, the heuristic's, unless another clearly wins)
    int wins = 0;
    bool quiet = false;
    if (ea && eb && cu.cudaMemsetAsync(S, 0, d_bytes, stream_) == cudaSuccess) {
        // the heuristic's pick sets the graph lengths, about 0.3 ms a screen launch and 1 ms a final one
        if (cudaGraphExec_t g = make_graph(b, &cands[0].algo, cands[0].ws, 8)) {
            time_graph(g, 8, 3, stream_, ea, eb);   // clocks up
            base_ms = time_graph(g, 8, 3, stream_, ea, eb);
            cu.cudaGraphExecDestroy(g);
        }
        const double est = base_ms < 1e29 ? std::max(base_ms, 1e-3) : 0.1;
        const int n_screen = std::max(2, std::min(64, static_cast<int>(0.3 / est)));
        const int n_final = std::max(4, std::min(128, static_cast<int>(1.0 / est)));
        // what the flushes alone cost a product, taken off every time
        if (b.flush) {
            if (cudaGraphExec_t g = make_graph(b, nullptr, 0, n_final)) {
                time_graph(g, n_final, 1, stream_, ea, eb);
                flush_ms = time_graph(g, n_final, 4, stream_, ea, eb);
                if (flush_ms >= 1e29) flush_ms = 0.0;
                cu.cudaGraphExecDestroy(g);
            }
        }
        // the screen: every candidate briefly
        for (Cand& c : cands) {
            if (cudaGraphExec_t g = make_graph(b, &c.algo, c.ws, n_screen)) {
                time_graph(g, n_screen, 1, stream_, ea, eb);
                c.ms = time_graph(g, n_screen, 2, stream_, ea, eb, flush_ms);
                cu.cudaGraphExecDestroy(g);
                ++timed;
            }
        }
        // the finals, the heuristic's pick and the best of the screen in interleaved rounds, each scored by its
        // median round. Only finalists can be picked.
        std::vector<size_t> order(cands.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin() + 1, order.end(), [&](size_t x, size_t y) { return cands[x].ms < cands[y].ms; });
        order.resize(std::min(order.size(), static_cast<size_t>(1 + kFinalists)));
        std::vector<cudaGraphExec_t> gs(order.size(), nullptr);
        std::vector<std::vector<double>> t(order.size());
        for (size_t i = 0; i < order.size(); ++i)
            if (cands[order[i]].ms < 1e29) gs[i] = make_graph(b, &cands[order[i]].algo, cands[order[i]].ws, n_final);
        for (int round = 0; round < kRounds; ++round)
            for (size_t j = 0; j < order.size(); ++j) {
                const size_t i = (round & 1) ? order.size() - 1 - j : j;
                if (gs[i]) t[i].push_back(time_graph(gs[i], n_final, 1, stream_, ea, eb, flush_ms));
            }
        for (Cand& c : cands) c.ms = 1e30;
        for (size_t i = 0; i < order.size(); ++i) {
            if (gs[i]) cu.cudaGraphExecDestroy(gs[i]);
            if (t[i].size() == static_cast<size_t>(kRounds)) cands[order[i]].ms = median(t[i]);
        }
        base_ms = cands[0].ms;
        // the winner must also beat the heuristic's pick in most rounds, pair by pair
        size_t fi = 0;
        for (size_t i = 1; i < order.size(); ++i)
            if (cands[order[i]].ms < cands[order[fi]].ms) fi = i;
        if (fi > 0 && base_ms < 1e29) {
            for (int r = 0; r < kRounds; ++r) wins += t[fi][r] < t[0][r] ? 1 : 0;
            if (cands[order[fi]].ms <= kMargin * base_ms && wins * 4 >= kRounds * 3) pick = order[fi];
        }
        // a quiet card has the median round within 20% of the fastest, for both (clocks alone move a few percent,
        // a second process on the card moved them 2 to 10 times). Otherwise the heuristic's pick, not saved.
        if (base_ms < 1e29) {
            const double mw = cands[pick].ms;
            const auto& tw = t[pick == 0 ? 0 : fi];
            quiet = base_ms < kQuiet * *std::min_element(t[0].begin(), t[0].end()) &&
                    mw < kQuiet * *std::min_element(tw.begin(), tw.end());
            if (!quiet) pick = 0;
        }
        cu.cudaStreamSynchronize(stream_);
    }
    if (ea) cu.cudaEventDestroy(ea);
    if (eb) cu.cudaEventDestroy(eb);
    if (b.flush) cu.cudaFree(b.flush);
    cu.cudaFree(S);
    cu.cudaGetLastError();
    if (timed == 0 || base_ms >= 1e29) return;   // nothing measured, so the heuristic's pick, not saved

    p.algo = cands[pick].algo;
    p.ws = cands[pick].ws;

    Entry e;
    e.algo = p.algo;
    e.ws = p.ws;
    e.level = tune_;
    e.ms = cands[pick].ms;
    e.heur_ms = base_ms;
    e.what = describe(p.algo);
    // a busy card (another process on it) times nonsense, so nothing is saved
    if (quiet) {
        store->entries[k] = e;
        try {
            save(*store);
        } catch (...) {
        }
    }
    if (env_int("MAYFLY_CUDANET_TUNE_LOG", 1)) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr,
                     "cudanet tune: rows %d out %d in %d mode %d: %zu candidates (best fit %zu, by id %zu, sweep %zu), "
                     "%zu timed, %.1f s; flush %.4f ms; heuristic %.4f ms, picked %.4f ms (x%.3f, won %d of %d)%s%s\n",
                     rows, out, in, std::get<3>(key), cands.size(), n_best, n_heur - n_best, cands.size() - n_heur,
                     timed, s, flush_ms, base_ms, e.ms, base_ms / std::max(e.ms, 1e-9), wins, kRounds,
                     e.what.c_str(), quiet ? "" : " (busy card: not saved)");
    }
}

}  // namespace cudanet
