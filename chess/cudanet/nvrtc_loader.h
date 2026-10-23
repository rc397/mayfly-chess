// CUDA C source to a loaded module. NVRTC compiles it to a cubin for the device's own compute capability (never
// PTX, which a driver older than the toolkit cannot load), the cubin is cached on disk as
// %LOCALAPPDATA%\mayfly_cudanet\<sha1 of source, options and NVRTC version>_sm<XY>.cubin, and the module is loaded
// into the runtime's primary context of the device, the same context cudart, cuBLASLt and the graphs use.
#pragma once

#include <string>

#include "cuda_dyn.h"

#ifndef CUDANET_LAUNCHCFG
#define CUDANET_LAUNCHCFG
namespace cudanet {
struct LaunchCfg {
    unsigned gx = 1, gy = 1, gz = 1, bx = 1, by = 1, bz = 1, shmem = 0;
};
}  // namespace cudanet
#endif

namespace cudanet {

class KernelModule {
public:
    // compiles (or reads the cached cubin) and loads it, and throws std::runtime_error with the NVRTC log on a
    // compile failure. name is the program's name in NVRTC's messages (e.g. "kernels_trunk.cu").
    KernelModule(const std::string& source, const std::string& name, int device);
    ~KernelModule();
    KernelModule(const KernelModule&) = delete;
    KernelModule& operator=(const KernelModule&) = delete;

    // the extern "C" __global__ function of that name, and throws if the module has none
    CUfunction get(const char* kernel) const;

    int device() const { return device_; }
    int sm() const { return sm_; }  // major * 10 + minor, e.g. 86 or 120
    CUcontext context() const { return ctx_; }
    CUmodule module() const { return mod_; }
    bool from_cache() const { return from_cache_; }
    const std::string& cache_path() const { return cache_path_; }

private:
    int device_ = 0;
    int sm_ = 0;
    CUcontext ctx_ = nullptr;
    CUmodule mod_ = nullptr;
    bool from_cache_ = false;
    std::string cache_path_;
};

// cuLaunchKernel on the stream (cudaStream_t and CUstream are the same type). args holds one pointer per kernel
// argument, in order. Works inside a stream capture like any other launch. The calling thread must have the
// device's primary context current, which any runtime call on it (cudaSetDevice(device) is enough) arranges.
void launch_kernel(CUfunction f, const LaunchCfg& c, CUstream s, void** args);

// lets f use more than 48 KB of dynamic shared memory (needed once per function before such a launch)
void set_max_dynamic_shmem(CUfunction f, int bytes);

// the whole file as bytes, throws if it cannot be read
std::string read_text_file(const std::string& path);

// lowercase hex SHA-1 of the bytes (the cache key)
std::string sha1_hex(const std::string& bytes);

}  // namespace cudanet
