// Launch configurations for the kernels in kernels_trunk.cu.
// Host side, no CUDA headers. Keep in step with the comments at the top of each kernel.
// No trunk kernel uses dynamic shared memory (shmem = 0 everywhere, all shared memory is static,
// at most about 19 KB per block in attention_f16).
#pragma once

#ifndef CUDANET_LAUNCHCFG
#define CUDANET_LAUNCHCFG
namespace cudanet { struct LaunchCfg { unsigned gx = 1, gy = 1, gz = 1, bx = 1, by = 1, bz = 1, shmem = 0; }; }
#endif

namespace cudanet {
namespace launch {

// embed: one block per (token, position), 256 threads each own one float4 of the 1024 wide row.
inline LaunchCfg embed(int B)
{
    LaunchCfg c;
    c.gx = 65;
    c.gy = B > 0 ? (unsigned)B : 1u;
    c.bx = 256;
    return c;
}

// layernorm_f32_f16: one block per row, 256 threads (one float4 each at d = 1024, d up to 4096).
inline LaunchCfg layernorm(int rows)
{
    LaunchCfg c;
    c.gx = rows > 0 ? (unsigned)rows : 1u;
    c.bx = 256;
    return c;
}

// attention_f16: grid (16 heads, B, nz), 128 threads (the kernel's __launch_bounds__ and its
// per warp shared arrays assume at most 4 warps). nz splits the 65 query rows so small batches
// still give enough blocks: about 160 blocks or more, nz at most 5 (13 rows per slice).
inline LaunchCfg attention(int B)
{
    LaunchCfg c;
    const unsigned b = B > 0 ? (unsigned)B : 1u;
    unsigned nz = (160u + 16u * b - 1u) / (16u * b);
    if (nz < 1u) nz = 1u;
    if (nz > 5u) nz = 5u;
    c.gx = 16;
    c.gy = b;
    c.gz = nz;
    c.bx = 128;
    return c;
}

// attention_mma: grid (16 heads, B), 160 threads (five warps, one per 16 query rows, the kernel's
// __launch_bounds__). About 23 KB of static shared memory a block.
inline LaunchCfg attention_mma(int B)
{
    LaunchCfg c;
    c.gx = 16;
    c.gy = B > 0 ? (unsigned)B : 1u;
    c.bx = 160;
    return c;
}

// add_f16_f32: 256 threads, grid stride, one thread step covers 4 floats (n % 4 == 0). Grid capped.
inline LaunchCfg add_f16_f32(int n)
{
    LaunchCfg c;
    unsigned long long blocks = ((n > 0 ? (unsigned long long)n : 4ull) / 4ull + 255ull) / 256ull;
    if (blocks < 1ull) blocks = 1ull;
    if (blocks > 8192ull) blocks = 8192ull;
    c.gx = (unsigned)blocks;
    c.bx = 256;
    return c;
}

// gelu_f16: 256 threads, grid stride loop, one thread step covers 8 halves when n % 8 == 0
// (always true for rows * 2048), else 2 or 1. Grid capped, the loop covers the rest.
inline LaunchCfg gelu(int n)
{
    LaunchCfg c;
    const unsigned long long nn = n > 0 ? (unsigned long long)n : 1ull;
    const unsigned long long items = (nn % 8ull == 0ull) ? nn / 8ull : ((nn % 2ull == 0ull) ? nn / 2ull : nn);
    unsigned long long blocks = (items + 255ull) / 256ull;
    if (blocks < 1ull) blocks = 1ull;
    if (blocks > 8192ull) blocks = 8192ull;
    c.gx = (unsigned)blocks;
    c.bx = 256;
    return c;
}

}  // namespace launch
}  // namespace cudanet
