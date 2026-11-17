/* cuda_shim_gcc/crt/host_config.h. A stand-in, not NVIDIA's file: see host_defines.h beside it.
 * NVIDIA's crt/host_config.h checks the host compiler for nvcc and is wrapped in __CUDACC__; for a host-only gcc
 * translation unit it defines nothing, so this one is empty. */
#pragma once
#if defined(__CUDACC__)
#error "cuda_shim_gcc is for host-only gcc compilation"
#endif
