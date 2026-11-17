// Launch configs for kernels_heads.cu. Host only, no CUDA headers.
// Neither kernel uses dynamic shared memory (each has a static 33 float scratch).
#pragma once

#ifndef CUDANET_LAUNCHCFG
#define CUDANET_LAUNCHCFG
namespace cudanet { struct LaunchCfg { unsigned gx = 1, gy = 1, gz = 1, bx = 1, by = 1, bz = 1, shmem = 0; }; }
#endif

namespace cudanet { namespace launch {

// One block per batch row, 256 threads (4 of the 1024 columns each).
inline LaunchCfg cls_layernorm(int B) {
    LaunchCfg c;
    c.gx = B > 0 ? (unsigned)B : 1u;  // B = 0 still launches one block, which returns at once
    c.bx = 256;
    c.shmem = 0;
    return c;
}

// One block per batch row, threads = wide rounded up to a multiple of 32, between 32 and 256 (wider rows loop).
inline LaunchCfg heads_post(int B, int wide) {
    LaunchCfg c;
    c.gx = B > 0 ? (unsigned)B : 1u;
    int t = (wide + 31) / 32 * 32;
    if (t < 32) t = 32;
    if (t > 256) t = 256;
    c.bx = (unsigned)t;
    c.shmem = 0;
    return c;
}

} }  // namespace cudanet::launch
