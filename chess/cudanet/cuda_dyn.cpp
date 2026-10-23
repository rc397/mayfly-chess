// See cuda_dyn.h. LoadLibraryExW and GetProcAddress for the four CUDA libraries, once per process.
#include "cuda_dyn.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <stdexcept>
#include <string>
#include <vector>

// two levels so the argument is macro expanded first and GetProcAddress gets the exported (versioned) name
#define CUDANET_STR2(x) #x
#define CUDANET_STR(x) CUDANET_STR2(x)

namespace cudanet {

namespace {

std::string narrow(const std::wstring& w)
{
    if (w.empty()) return std::string();
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n : 0, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}

std::wstring env(const wchar_t* name)
{
    DWORD n = GetEnvironmentVariableW(name, nullptr, 0);
    if (n == 0) return std::wstring();
    std::wstring v(n, L'\0');
    n = GetEnvironmentVariableW(name, &v[0], n);
    v.resize(n);
    return v;
}

// MAYFLY_CUDA_BIN (may be a ';' list), then the CUDA 12.8 install, then TORCH_LIB and the running torch's lib
std::vector<std::wstring> search_dirs()
{
    std::vector<std::wstring> dirs;
    auto add_list = [&](const std::wstring& list) {
        size_t start = 0;
        while (start <= list.size()) {
            size_t end = list.find(L';', start);
            if (end == std::wstring::npos) end = list.size();
            std::wstring d = list.substr(start, end - start);
            while (!d.empty() && (d.back() == L'\\' || d.back() == L'/' || d.back() == L' ')) d.pop_back();
            while (!d.empty() && d.front() == L' ') d.erase(d.begin());
            if (!d.empty()) dirs.push_back(d);
            start = end + 1;
        }
    };
    add_list(env(L"MAYFLY_CUDA_BIN"));
    dirs.push_back(L"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin");
    add_list(env(L"TORCH_LIB"));
    // a torch already in the process, whose lib folder carries the same CUDA 12 libraries
    if (HMODULE t = GetModuleHandleW(L"c10.dll")) {
        wchar_t buf[MAX_PATH];
        DWORD n = GetModuleFileNameW(t, buf, MAX_PATH);
        std::wstring p(buf, n);
        size_t cut = p.find_last_of(L"\\/");
        if (n && cut != std::wstring::npos) dirs.push_back(p.substr(0, cut));
    }
    return dirs;
}

bool file_exists(const std::wstring& path)
{
    DWORD a = GetFileAttributesW(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

struct Loaded {
    HMODULE h = nullptr;
    std::wstring dir;
    std::wstring path;
};

// the first search directory holding the file, with its own dependencies looked for beside it
Loaded load_from_dirs(const wchar_t* file)
{
    // already loaded (by a torch beside us), so share that copy rather than load a second runtime
    if (HMODULE have = GetModuleHandleW(file)) {
        wchar_t buf[MAX_PATH];
        DWORD n = GetModuleFileNameW(have, buf, MAX_PATH);
        std::wstring path(buf, n);
        size_t cut = path.find_last_of(L"\\/");
        HMODULE h = LoadLibraryExW(path.c_str(), nullptr,
                                   LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (h && n && cut != std::wstring::npos) return Loaded{h, path.substr(0, cut), path};
    }
    std::string tried;
    for (const std::wstring& d : search_dirs()) {
        std::wstring path = d + L"\\" + file;
        if (!tried.empty()) tried += ", ";
        tried += narrow(d);
        if (!file_exists(path)) continue;
        HMODULE h = LoadLibraryExW(path.c_str(), nullptr,
                                   LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (h) return Loaded{h, d, path};
        tried += " (present but failed to load, error " + std::to_string(GetLastError()) + ")";
    }
    throw std::runtime_error("cudanet: cannot load " + narrow(file) + "; looked in: " + tried +
                             " (set MAYFLY_CUDA_BIN to the directory that holds it)");
}

// the driver is the installed one, never a copy beside the toolkit
Loaded load_driver()
{
    HMODULE h = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!h) h = LoadLibraryW(L"nvcuda.dll");
    if (!h)
        throw std::runtime_error("cudanet: cannot load nvcuda.dll (error " + std::to_string(GetLastError()) +
                                 "); is the NVIDIA driver installed?");
    wchar_t buf[MAX_PATH * 4];
    DWORD n = GetModuleFileNameW(h, buf, (DWORD)(sizeof(buf) / sizeof(buf[0])));
    return Loaded{h, std::wstring(), std::wstring(buf, n)};
}

// NVRTC opens nvrtc-builtins64_XY.dll only when it first compiles. Load it now from nvrtc's own directory so the
// later load by name finds the module already in the process.
void preload_nvrtc_builtins(const std::wstring& dir)
{
    WIN32_FIND_DATAW fd;
    std::wstring pattern = dir + L"\\nvrtc-builtins64_*.dll";
    HANDLE f = FindFirstFileW(pattern.c_str(), &fd);
    if (f == INVALID_HANDLE_VALUE) return;
    do {
        std::wstring path = dir + L"\\" + fd.cFileName;
        LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    } while (FindNextFileW(f, &fd));
    FindClose(f);
}

template <class F>
void resolve(HMODULE h, const std::wstring& dll, const char* sym, F& out)
{
    FARPROC p = GetProcAddress(h, sym);
    if (!p) throw std::runtime_error("cudanet: " + narrow(dll) + " has no symbol " + sym);
    out = reinterpret_cast<F>(p);
}

Cuda load_all()
{
    Cuda c;

    Loaded drv = load_driver();
    c.driver_path = narrow(drv.path);
#define CUDANET_RESOLVE(name) resolve(drv.h, drv.path, CUDANET_STR(name), c.name);
    CUDANET_DRIVER_FNS(CUDANET_RESOLVE)
#undef CUDANET_RESOLVE

    Loaded rt = load_from_dirs(L"cudart64_12.dll");
    c.runtime_path = narrow(rt.path);
#define CUDANET_RESOLVE(name) resolve(rt.h, rt.path, CUDANET_STR(name), c.name);
#define CUDANET_RESOLVE_T(name, type) resolve(rt.h, rt.path, CUDANET_STR(name), c.name);
    CUDANET_RUNTIME_FNS(CUDANET_RESOLVE)
    CUDANET_RUNTIME_FNS_T(CUDANET_RESOLVE_T)
#undef CUDANET_RESOLVE
#undef CUDANET_RESOLVE_T

    Loaded lt = load_from_dirs(L"cublasLt64_12.dll");
    c.cublaslt_path = narrow(lt.path);
#define CUDANET_RESOLVE(name) resolve(lt.h, lt.path, CUDANET_STR(name), c.name);
    CUDANET_CUBLASLT_FNS(CUDANET_RESOLVE)
#undef CUDANET_RESOLVE

    Loaded rtc = load_from_dirs(L"nvrtc64_120_0.dll");
    c.nvrtc_path = narrow(rtc.path);
    preload_nvrtc_builtins(rtc.dir);
#define CUDANET_RESOLVE(name) resolve(rtc.h, rtc.path, CUDANET_STR(name), c.name);
    CUDANET_NVRTC_FNS(CUDANET_RESOLVE)
#undef CUDANET_RESOLVE

    // the handles stay loaded for the life of the process
    return c;
}

}  // namespace

Cuda& cuda()
{
    // a function static, built once, thread safe, and built again on the next call if the first attempt threw
    static Cuda c = load_all();
    return c;
}

void fail_cu(CUresult r, const char* what)
{
    const char* s = nullptr;
    const char* n = nullptr;
    Cuda& c = cuda();
    if (c.cuGetErrorString) c.cuGetErrorString(r, &s);
    if (c.cuGetErrorName) c.cuGetErrorName(r, &n);
    throw std::runtime_error(std::string("cudanet: ") + what + ": " + (n ? n : "CUresult") + " (" +
                             std::to_string((int)r) + ") " + (s ? s : "unknown error"));
}

void fail_rt(cudaError_t r, const char* what)
{
    Cuda& c = cuda();
    const char* s = c.cudaGetErrorString ? c.cudaGetErrorString(r) : nullptr;
    const char* n = c.cudaGetErrorName ? c.cudaGetErrorName(r) : nullptr;
    throw std::runtime_error(std::string("cudanet: ") + what + ": " + (n ? n : "cudaError") + " (" +
                             std::to_string((int)r) + ") " + (s ? s : "unknown error"));
}

void fail_lt(cublasStatus_t r, const char* what)
{
    Cuda& c = cuda();
    const char* s = c.cublasLtGetStatusString ? c.cublasLtGetStatusString(r) : nullptr;
    const char* n = c.cublasLtGetStatusName ? c.cublasLtGetStatusName(r) : nullptr;
    throw std::runtime_error(std::string("cudanet: ") + what + ": " + (n ? n : "cublasStatus") + " (" +
                             std::to_string((int)r) + ") " + (s ? s : "unknown error"));
}

void fail_nvrtc(nvrtcResult r, const char* what)
{
    Cuda& c = cuda();
    const char* s = c.nvrtcGetErrorString ? c.nvrtcGetErrorString(r) : nullptr;
    throw std::runtime_error(std::string("cudanet: ") + what + ": " + (s ? s : "nvrtc error") + " (" +
                             std::to_string((int)r) + ")");
}

}  // namespace cudanet
