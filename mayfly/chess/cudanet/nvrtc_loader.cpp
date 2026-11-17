// See nvrtc_loader.h.
#include "nvrtc_loader.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace cudanet {

namespace {

// ---------------------------------------------------------------------------------------------------- SHA-1
// FIPS 180-4, only the cache key needs it
struct Sha1 {
    uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
    unsigned char buf[64] = {};
    size_t n = 0;        // bytes waiting in buf
    uint64_t total = 0;  // message length in bytes

    static uint32_t rol(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

    void block(const unsigned char* p)
    {
        uint32_t w[80];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 |
                   (uint32_t)p[4 * i + 3];
        for (int i = 16; i < 80; i++) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20) {
                f = (b & c) | (~b & d);
                k = 0x5A827999u;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1u;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDCu;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6u;
            }
            uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rol(b, 30);
            b = a;
            a = t;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
    }

    void bytes(const unsigned char* p, size_t len)
    {
        while (len) {
            size_t take = 64 - n < len ? 64 - n : len;
            std::memcpy(buf + n, p, take);
            n += take;
            p += take;
            len -= take;
            if (n == 64) {
                block(buf);
                n = 0;
            }
        }
    }

    void update(const void* data, size_t len)
    {
        total += len;
        bytes(static_cast<const unsigned char*>(data), len);
    }

    // pads with 0x80, zeros to 56 mod 64, then the bit length big endian. Call once.
    std::string finish()
    {
        uint64_t bits = total * 8;
        unsigned char one = 0x80, zero = 0;
        bytes(&one, 1);
        while (n != 56) bytes(&zero, 1);
        unsigned char len[8];
        for (int i = 0; i < 8; i++) len[i] = (unsigned char)(bits >> (56 - 8 * i));
        bytes(len, 8);
        static const char* hexd = "0123456789abcdef";
        std::string out;
        out.reserve(40);
        for (int i = 0; i < 5; i++)
            for (int s = 28; s >= 0; s -= 4) out.push_back(hexd[(h[i] >> s) & 0xF]);
        return out;
    }
};

// ---------------------------------------------------------------------------------------------------- files
std::wstring wenv(const wchar_t* name)
{
    DWORD n = GetEnvironmentVariableW(name, nullptr, 0);
    if (n == 0) return std::wstring();
    std::wstring v(n, L'\0');
    n = GetEnvironmentVariableW(name, &v[0], n);
    v.resize(n);
    return v;
}

std::string env(const char* name)
{
    DWORD n = GetEnvironmentVariableA(name, nullptr, 0);
    if (n == 0) return std::string();
    std::string v(n, '\0');
    n = GetEnvironmentVariableA(name, &v[0], n);
    v.resize(n);
    return v;
}

// MAYFLY_CUDA_INCLUDE names the directory holding cuda_fp16.h, else the CUDA 12.8 install's include folder
std::string include_dir()
{
    std::string d = env("MAYFLY_CUDA_INCLUDE");
    while (!d.empty() && (d.back() == '\\' || d.back() == '/' || d.back() == ' ')) d.pop_back();
    return d.empty() ? std::string("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include") : d;
}

// %LOCALAPPDATA%\mayfly_cudanet, created if missing, empty when there is nowhere to cache
fs::path cache_dir()
{
    std::wstring base = wenv(L"LOCALAPPDATA");
    if (base.empty()) return fs::path();
    fs::path d = fs::path(base) / L"mayfly_cudanet";
    std::error_code ec;
    fs::create_directories(d, ec);
    if (ec && !fs::is_directory(d, ec)) return fs::path();
    return d;
}

std::string read_bytes(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) return std::string();
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// written beside the target then renamed over it, so a reader never sees half a file. Failures only lose the cache.
void write_bytes_atomic(const fs::path& p, const std::string& data)
{
    fs::path tmp = p;
    tmp += L".tmp" + std::to_wstring(GetCurrentProcessId()) + L"_" + std::to_wstring(GetCurrentThreadId());
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return;
        f.write(data.data(), (std::streamsize)data.size());
        if (!f) {
            f.close();
            std::error_code ec;
            fs::remove(tmp, ec);
            return;
        }
    }
    std::error_code ec;
    fs::rename(tmp, p, ec);
    if (ec) fs::remove(tmp, ec);
}

std::string narrow_path(const fs::path& p)
{
    try {
        return p.u8string();
    } catch (...) {
        return std::string("(path)");
    }
}

// ---------------------------------------------------------------------------------------------------- NVRTC
struct ProgramGuard {
    nvrtcProgram prog = nullptr;
    ~ProgramGuard()
    {
        if (prog) cuda().nvrtcDestroyProgram(&prog);
    }
};

std::string compile_cubin(const std::string& source, const std::string& name, const std::vector<std::string>& opts,
                          int sm)
{
    Cuda& c = cuda();
    ProgramGuard g;
    check_nvrtc(c.nvrtcCreateProgram(&g.prog, source.c_str(), name.c_str(), 0, nullptr, nullptr),
                "nvrtcCreateProgram");
    std::vector<const char*> argv;
    for (const std::string& o : opts) argv.push_back(o.c_str());
    nvrtcResult r = c.nvrtcCompileProgram(g.prog, (int)argv.size(), argv.data());

    std::string log;
    size_t log_n = 0;
    if (c.nvrtcGetProgramLogSize(g.prog, &log_n) == NVRTC_SUCCESS && log_n > 1) {
        log.assign(log_n, '\0');
        if (c.nvrtcGetProgramLog(g.prog, &log[0]) != NVRTC_SUCCESS) log.clear();
        while (!log.empty() && log.back() == '\0') log.pop_back();
    }
    if (r != NVRTC_SUCCESS) {
        const char* es = c.nvrtcGetErrorString(r);
        std::string msg = "cudanet: NVRTC could not compile " + name + " for sm_" + std::to_string(sm) + " (" +
                          (es ? es : "error") + "); options:";
        for (const std::string& o : opts) msg += " " + o;
        msg += "\n" + log;
        throw std::runtime_error(msg);
    }

    size_t n = 0;
    check_nvrtc(c.nvrtcGetCUBINSize(g.prog, &n), "nvrtcGetCUBINSize");
    if (n == 0) throw std::runtime_error("cudanet: NVRTC produced no cubin for " + name);
    std::string cubin(n, '\0');
    check_nvrtc(c.nvrtcGetCUBIN(g.prog, &cubin[0]), "nvrtcGetCUBIN");
    return cubin;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------- KernelModule
KernelModule::KernelModule(const std::string& source, const std::string& name, int device) : device_(device)
{
    Cuda& c = cuda();

    // the runtime's primary context, which cudaFree(0) makes if needed and leaves current on this thread
    check_rt(c.cudaSetDevice(device), "cudaSetDevice");
    check_rt(c.cudaFree(nullptr), "cudaFree(0)");
    check_cu(c.cuInit(0), "cuInit");
    check_cu(c.cuCtxGetCurrent(&ctx_), "cuCtxGetCurrent");
    if (!ctx_) throw std::runtime_error("cudanet: no current CUDA context after cudaSetDevice");

    int major = 0, minor = 0;
    check_rt(c.cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
             "cudaDeviceGetAttribute(major)");
    check_rt(c.cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device),
             "cudaDeviceGetAttribute(minor)");
    sm_ = major * 10 + minor;

    int vmaj = 0, vmin = 0;
    check_nvrtc(c.nvrtcVersion(&vmaj, &vmin), "nvrtcVersion");

    // no --use_fast_math, so erf, exp and the divisions stay exact
    const std::vector<std::string> opts = {
        "--gpu-architecture=sm_" + std::to_string(sm_),
        "-std=c++17",
        "-default-device",
        "--include-path=" + include_dir(),
    };

    // the key is source, options and NVRTC version, each ended by a zero byte
    Sha1 sh;
    sh.update(source.data(), source.size());
    sh.update("\0", 1);
    for (const std::string& o : opts) {
        sh.update(o.data(), o.size());
        sh.update("\0", 1);
    }
    std::string ver = "nvrtc " + std::to_string(vmaj) + "." + std::to_string(vmin);
    sh.update(ver.data(), ver.size());
    std::string file = sh.finish() + "_sm" + std::to_string(sm_) + ".cubin";

    fs::path dir = cache_dir();
    fs::path cached = dir.empty() ? fs::path() : dir / file;
    if (!cached.empty()) cache_path_ = narrow_path(cached);

    // a cached cubin that the driver refuses (truncated, or a driver change) is compiled again
    if (!cached.empty()) {
        std::string cubin = read_bytes(cached);
        if (!cubin.empty()) {
            CUmodule m = nullptr;
            if (c.cuModuleLoadData(&m, cubin.data()) == CUDA_SUCCESS && m) {
                mod_ = m;
                from_cache_ = true;
                return;
            }
        }
    }

    std::string cubin = compile_cubin(source, name, opts, sm_);
    std::string what = "cuModuleLoadData(" + name + ", sm_" + std::to_string(sm_) + ")";
    check_cu(c.cuModuleLoadData(&mod_, cubin.data()), what.c_str());
    if (!cached.empty()) write_bytes_atomic(cached, cubin);
}

KernelModule::~KernelModule()
{
    if (!mod_) return;
    try {
        Cuda& c = cuda();
        // unload in the module's own context, then give the thread back whatever it had
        CUcontext prev = nullptr;
        c.cuCtxGetCurrent(&prev);
        if (prev != ctx_) c.cuCtxSetCurrent(ctx_);
        c.cuModuleUnload(mod_);
        if (prev != ctx_) c.cuCtxSetCurrent(prev);
    } catch (...) {
    }
    mod_ = nullptr;
}

CUfunction KernelModule::get(const char* kernel) const
{
    CUfunction f = nullptr;
    CUresult r = cuda().cuModuleGetFunction(&f, mod_, kernel);
    if (r != CUDA_SUCCESS || !f) {
        std::string what = std::string("cuModuleGetFunction(") + kernel + ")";
        check_cu(r != CUDA_SUCCESS ? r : CUDA_ERROR_NOT_FOUND, what.c_str());
    }
    return f;
}

// ---------------------------------------------------------------------------------------------------- launches
void launch_kernel(CUfunction f, const LaunchCfg& g, CUstream s, void** args)
{
    check_cu(cuda().cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz, g.shmem, s, args, nullptr),
             "cuLaunchKernel");
}

void set_max_dynamic_shmem(CUfunction f, int bytes)
{
    check_cu(cuda().cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes),
             "cuFuncSetAttribute(max dynamic shared memory)");
}

// ---------------------------------------------------------------------------------------------------- helpers
std::string read_text_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cudanet: cannot read " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    if (f.bad()) throw std::runtime_error("cudanet: error reading " + path);
    return ss.str();
}

std::string sha1_hex(const std::string& bytes)
{
    Sha1 sh;
    sh.update(bytes.data(), bytes.size());
    return sh.finish();
}

}  // namespace cudanet
