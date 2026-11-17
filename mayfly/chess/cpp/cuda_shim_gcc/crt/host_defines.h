/* cuda_shim_gcc/crt/host_defines.h. A stand-in, not NVIDIA's file: the gcc twin of the MSVC shim
 * build_rig.bat uses (chess/cudanet/cuda_shim has a copy).
 *
 * The CUDA runtime headers torch's pip install brings (nvidia-cuda-runtime-cu12, the CUDA torch was built with)
 * include "crt/host_defines.h", and the crt folder ships with nvcc, which a pip install does not have. The real
 * file defines the function and type annotations of the CUDA headers. For a HOST-ONLY translation unit compiled by
 * gcc (no __CUDACC__) they reduce to the definitions below: the device annotations expand to nothing, alignment to
 * __attribute__((aligned)), the API calling convention to nothing. Only the NET_GRAPH build (build_linux.sh) reads
 * it, and its CUDA calls all go through torch's own functions and the CUDA runtime torch loads.
 */
#pragma once
#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#error "cuda_shim_gcc is for host-only gcc compilation"
#endif
#if !defined(__GNUC__)
#error "cuda_shim_gcc is written for gcc"
#endif
#define __HOST_DEFINES_H__

#define __no_return__ __attribute__((noreturn))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __inline_hint__
#define __align__(n) __attribute__((aligned(n)))
#define __maxnreg__(n)
#define __local_maxnreg__(n)
#define __thread__ __thread
#define __import__
#define __export__
#define __cdecl
#define __annotate__(a)
#define __location__(a)
#define CUDARTAPI
#define CUDARTAPI_CDECL
#define __specialization_static
#define __builtin_align__(a) __align__(a)

#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __managed__
#define __grid_constant__
#define __nv_pure__
#define __launch_bounds__(...)
#define __cluster_dims__(...)
#define __block_size__(...)
#define __cudart_builtin__
#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __CUDA_ALIGN__(n) __attribute__((aligned(n)))
