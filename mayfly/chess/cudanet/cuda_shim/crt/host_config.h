/* A stand-in, not NVIDIA's file (see host_defines.h beside it). NVIDIA's crt/host_config.h checks the host
 * compiler for nvcc and is wrapped in __CUDACC__. For a host-only MSVC translation unit it defines nothing, so this
 * one is empty. */
#pragma once
#if defined(__CUDACC__)
#error "cuda_shim is for host-only MSVC compilation"
#endif
