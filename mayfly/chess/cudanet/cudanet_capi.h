// The C ABI of cudanet.dll, for Python through ctypes (cudanet_py.py) and anything else without C++.
#pragma once
#include <stdint.h>

#ifdef CUDANET_BUILD
#define CUDANET_API __declspec(dllexport)
#else
#define CUDANET_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Load the weight file onto `device`. Returns a handle, or NULL with the reason in err (err_len bytes).
CUDANET_API void* cudanet_create(const char* weights_path, int device, char* err, int err_len);

// One batch, synchronous. Host pointers ids [B*64], cast [B*4], hm [B], ep [B], I [B*wide], M [B*wide] in, and
// pol [B*wide], val [B], qv [B*wide] out. Returns 0, or -1 with the reason in err.
CUDANET_API int cudanet_evaluate(void* handle, int B, int wide, const int64_t* ids, const float* cast,
                                 const int64_t* hm, const int64_t* ep, const int64_t* I, const uint8_t* M,
                                 float* pol, float* val, float* qv, char* err, int err_len);

// The weight file's fingerprint (as mcts_cpp.fingerprint computes it), copied into buf.
CUDANET_API int cudanet_fingerprint(void* handle, char* buf, int buf_len);

// The largest batch the handle takes (kMaxBatch, or env MAYFLY_CUDANET_MAX_BATCH when it was made), -1 on a null handle.
CUDANET_API int cudanet_max_batch(void* handle);

// Milliseconds a pass by step at batch B, the mean of reps passes (cudanet.h CudaNet::profile), into ms (n slots).
// Returns how many were written, or -1 with the reason in err.
CUDANET_API int cudanet_profile(void* handle, int B, int reps, double* ms, int n, char* err, int err_len);

// The step names for cudanet_profile, comma separated, copied into buf. Returns their length.
CUDANET_API int cudanet_profile_names(char* buf, int buf_len);

CUDANET_API void cudanet_destroy(void* handle);

#ifdef __cplusplus
}
#endif
