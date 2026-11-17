// Every CUDA entry point the backend uses, loaded at run time (no import libraries).
// The driver API comes from nvcuda.dll (System32). The runtime, cuBLASLt and NVRTC come from the first directory that
// holds them, out of MAYFLY_CUDA_BIN (one directory or a ';' list), the CUDA 12.8 install's bin folder
// (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin), TORCH_LIB, and the lib folder of a torch already
// loaded in the process.
//
// Each member of Cuda is named exactly as the API function and typed from the official header, so a call reads
// cudanet::cuda().cudaMemcpyAsync(...) or cudanet::cuda().cuLaunchKernel(...). The headers' versioning macros
// (cudaGetDeviceProperties is cudaGetDeviceProperties_v2 in 12.x, cuda.h maps some names to _v2) expand the same way
// in the declaration, in every use and in the symbol name handed to GetProcAddress.
//
// A full CUDA toolkit's include folder is enough for host code. Headers taken from pip wheels lack nvcc's
// crt\ folder and CCCL's nv\target, which cuda_runtime_api.h and cublasLt.h (through cuda_fp16.h) include, so then
// put chess\cudanet\cuda_shim (host-only stand-ins) before them. Never put the shim on an NVRTC include path.
//
// A function pointer carries no default arguments. cudaGraphInstantiate takes its flags argument (pass 0) and
// cudaStreamCreateWithFlags, cudaEventCreate and the rest take every argument the header lists.
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <nvrtc.h>

#include <string>

// The plain members. The type is the header's own declaration, and the argument is macro expanded before it is
// pasted, so the member's name is the versioned name when the header maps one.
#define CUDANET_FN(name) decltype(&::name) name = nullptr;
// The members cuda_runtime.h overloads in C++ (cudaMalloc, cudaMallocHost, cudaHostAlloc, cudaEventCreate,
// cudaGraphInstantiate). The C entry point's type is spelled out so this works whether or not cuda_runtime.h is
// included.
#define CUDANET_FN_T(name, type) type name = nullptr;

#define CUDANET_DRIVER_FNS(X) \
    X(cuInit) \
    X(cuDriverGetVersion) \
    X(cuDeviceGet) \
    X(cuDevicePrimaryCtxRetain) \
    X(cuDevicePrimaryCtxRelease) \
    X(cuCtxSetCurrent) \
    X(cuCtxGetCurrent) \
    X(cuCtxSynchronize) \
    X(cuModuleLoadData) \
    X(cuModuleUnload) \
    X(cuModuleGetFunction) \
    X(cuLaunchKernel) \
    X(cuFuncSetAttribute) \
    X(cuFuncGetAttribute) \
    X(cuGetErrorString) \
    X(cuGetErrorName)

#define CUDANET_RUNTIME_FNS(X) \
    X(cudaRuntimeGetVersion) \
    X(cudaSetDevice) \
    X(cudaGetDevice) \
    X(cudaGetDeviceCount) \
    X(cudaGetDeviceProperties) \
    X(cudaDeviceGetAttribute) \
    X(cudaFree) \
    X(cudaFreeHost) \
    X(cudaMemcpy) \
    X(cudaMemcpyAsync) \
    X(cudaMemset) \
    X(cudaMemsetAsync) \
    X(cudaMemGetInfo) \
    X(cudaStreamCreateWithFlags) \
    X(cudaStreamDestroy) \
    X(cudaStreamSynchronize) \
    X(cudaStreamBeginCapture) \
    X(cudaStreamEndCapture) \
    X(cudaStreamIsCapturing) \
    X(cudaGraphLaunch) \
    X(cudaGraphUpload) \
    X(cudaGraphExecDestroy) \
    X(cudaGraphDestroy) \
    X(cudaGetLastError) \
    X(cudaPeekAtLastError) \
    X(cudaGetErrorString) \
    X(cudaGetErrorName) \
    X(cudaEventCreateWithFlags) \
    X(cudaEventRecord) \
    X(cudaEventSynchronize) \
    X(cudaEventElapsedTime) \
    X(cudaEventDestroy) \
    X(cudaDeviceSynchronize)

// the C entry points' types (aliases, so no comma travels through a macro argument)
namespace cudanet {
using fn_cudaMalloc = cudaError_t(CUDARTAPI*)(void**, size_t);
using fn_cudaMallocHost = cudaError_t(CUDARTAPI*)(void**, size_t);
using fn_cudaHostAlloc = cudaError_t(CUDARTAPI*)(void**, size_t, unsigned int);
using fn_cudaEventCreate = cudaError_t(CUDARTAPI*)(cudaEvent_t*);
using fn_cudaGraphInstantiate = cudaError_t(CUDARTAPI*)(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
}  // namespace cudanet

#define CUDANET_RUNTIME_FNS_T(X) \
    X(cudaMalloc, fn_cudaMalloc) \
    X(cudaMallocHost, fn_cudaMallocHost) \
    X(cudaHostAlloc, fn_cudaHostAlloc) \
    X(cudaEventCreate, fn_cudaEventCreate) \
    X(cudaGraphInstantiate, fn_cudaGraphInstantiate)

#define CUDANET_CUBLASLT_FNS(X) \
    X(cublasLtGetVersion) \
    X(cublasLtCreate) \
    X(cublasLtDestroy) \
    X(cublasLtMatmulDescCreate) \
    X(cublasLtMatmulDescDestroy) \
    X(cublasLtMatmulDescSetAttribute) \
    X(cublasLtMatrixLayoutCreate) \
    X(cublasLtMatrixLayoutDestroy) \
    X(cublasLtMatrixLayoutSetAttribute) \
    X(cublasLtMatmulPreferenceCreate) \
    X(cublasLtMatmulPreferenceDestroy) \
    X(cublasLtMatmulPreferenceSetAttribute) \
    X(cublasLtMatmulAlgoGetHeuristic) \
    X(cublasLtMatmul) \
    X(cublasLtGetStatusString) \
    X(cublasLtGetStatusName)

#define CUDANET_NVRTC_FNS(X) \
    X(nvrtcVersion) \
    X(nvrtcCreateProgram) \
    X(nvrtcDestroyProgram) \
    X(nvrtcCompileProgram) \
    X(nvrtcGetProgramLogSize) \
    X(nvrtcGetProgramLog) \
    X(nvrtcGetCUBINSize) \
    X(nvrtcGetCUBIN) \
    X(nvrtcGetErrorString)

namespace cudanet {

struct Cuda {
    CUDANET_DRIVER_FNS(CUDANET_FN)
    CUDANET_RUNTIME_FNS(CUDANET_FN)
    CUDANET_RUNTIME_FNS_T(CUDANET_FN_T)
    CUDANET_CUBLASLT_FNS(CUDANET_FN)
    CUDANET_NVRTC_FNS(CUDANET_FN)

    // where each library came from (for logs and error messages)
    std::string driver_path, runtime_path, cublaslt_path, nvrtc_path;
};

// Loads every library once, thread safe. A failed load throws std::runtime_error naming the missing DLL or symbol,
// and the next call tries again.
Cuda& cuda();

// throws std::runtime_error("cudanet: <what>: <the API's error string>") when the call failed
[[noreturn]] void fail_cu(CUresult r, const char* what);
[[noreturn]] void fail_rt(cudaError_t r, const char* what);
[[noreturn]] void fail_lt(cublasStatus_t r, const char* what);
[[noreturn]] void fail_nvrtc(nvrtcResult r, const char* what);

inline void check_cu(CUresult r, const char* what)
{
    if (r != CUDA_SUCCESS) fail_cu(r, what);
}
inline void check_rt(cudaError_t r, const char* what)
{
    if (r != cudaSuccess) fail_rt(r, what);
}
inline void check_lt(cublasStatus_t r, const char* what)
{
    if (r != CUBLAS_STATUS_SUCCESS) fail_lt(r, what);
}
inline void check_nvrtc(nvrtcResult r, const char* what)
{
    if (r != NVRTC_SUCCESS) fail_nvrtc(r, what);
}

}  // namespace cudanet
