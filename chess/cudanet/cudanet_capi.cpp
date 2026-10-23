// cudanet_capi.h over cudanet::CudaNet, the C ABI of cudanet.dll (Python reaches it through ctypes).
// A handle is a new'd CudaNet. Every exception stops here and comes back as text in err.
#ifndef CUDANET_BUILD
#define CUDANET_BUILD
#endif
#include "cudanet_capi.h"

#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "cudanet.h"

namespace {

// copy msg into err (err_len bytes, always NUL terminated, cut if it does not fit)
void put(char* err, int err_len, const char* msg)
{
    if (!err || err_len <= 0) return;
    size_t n = std::strlen(msg);
    if (n > (size_t)(err_len - 1)) n = (size_t)(err_len - 1);
    std::memcpy(err, msg, n);
    err[n] = 0;
}

}  // namespace

extern "C" {

CUDANET_API void* cudanet_create(const char* weights_path, int device, char* err, int err_len)
{
    put(err, err_len, "");
    if (!weights_path) {
        put(err, err_len, "cudanet_create: no weights path");
        return nullptr;
    }
    try {
        return new cudanet::CudaNet(std::string(weights_path), device);
    } catch (const std::exception& e) {
        put(err, err_len, e.what());
    } catch (...) {
        put(err, err_len, "cudanet_create: unknown exception");
    }
    return nullptr;
}

CUDANET_API int cudanet_evaluate(void* handle, int B, int wide, const int64_t* ids, const float* cast,
                                 const int64_t* hm, const int64_t* ep, const int64_t* I, const uint8_t* M,
                                 float* pol, float* val, float* qv, char* err, int err_len)
{
    put(err, err_len, "");
    if (!handle) {
        put(err, err_len, "cudanet_evaluate: null handle");
        return -1;
    }
    try {
        static_cast<cudanet::CudaNet*>(handle)->evaluate(B, wide, ids, cast, hm, ep, I, M, pol, val, qv);
        return 0;
    } catch (const std::exception& e) {
        put(err, err_len, e.what());
    } catch (...) {
        put(err, err_len, "cudanet_evaluate: unknown exception");
    }
    return -1;
}

// returns the fingerprint's length (it may have been cut to buf_len - 1), or -1 on a null handle
CUDANET_API int cudanet_fingerprint(void* handle, char* buf, int buf_len)
{
    if (!handle) {
        put(buf, buf_len, "");
        return -1;
    }
    try {
        const std::string& fp = static_cast<cudanet::CudaNet*>(handle)->fingerprint();
        put(buf, buf_len, fp.c_str());
        return (int)fp.size();
    } catch (...) {
        put(buf, buf_len, "");
        return -1;
    }
}

CUDANET_API int cudanet_max_batch(void* handle)
{
    return handle ? static_cast<cudanet::CudaNet*>(handle)->max_batch() : -1;
}

CUDANET_API int cudanet_profile(void* handle, int B, int reps, double* ms, int n, char* err, int err_len)
{
    put(err, err_len, "");
    if (!handle || !ms) {
        put(err, err_len, "cudanet_profile: null handle or output");
        return -1;
    }
    try {
        const std::vector<double> got = static_cast<cudanet::CudaNet*>(handle)->profile(B, reps);
        const int k = (int)got.size() < n ? (int)got.size() : n;
        for (int i = 0; i < k; ++i) ms[i] = got[(size_t)i];
        return k;
    } catch (const std::exception& e) {
        put(err, err_len, e.what());
    } catch (...) {
        put(err, err_len, "cudanet_profile: unknown exception");
    }
    return -1;
}

CUDANET_API int cudanet_profile_names(char* buf, int buf_len)
{
    const char* names = cudanet::CudaNet::profile_names();
    put(buf, buf_len, names);
    return (int)std::strlen(names);
}

CUDANET_API void cudanet_destroy(void* handle)
{
    try {
        delete static_cast<cudanet::CudaNet*>(handle);
    } catch (...) {
        // a destructor that throws on the way out, and nothing the caller can do
    }
}

}  // extern "C"
