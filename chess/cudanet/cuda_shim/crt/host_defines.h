/* A stand-in, not NVIDIA's file.
 *
 * The headers in pip's nvidia-cuda-runtime-cu12 (nvidia\cuda_runtime\include) have a host_defines.h that only
 * includes "crt/host_defines.h", and the crt\ folder ships with nvcc. The real file defines the function and type
 * annotations of the CUDA headers. For a host-only translation unit compiled by MSVC (no __CUDACC__) they reduce
 * to the definitions below. The device annotations expand to nothing, alignment to __declspec(align) and the API
 * calling convention to __stdcall. Used by the NET_GRAPH build, whose CUDA calls all go through torch's own
 * exported functions.
 */
#pragma once
#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#error "cuda_shim is for host-only MSVC compilation"
#endif
#if !defined(_MSC_VER)
#error "cuda_shim is written for MSVC"
#endif
#define __HOST_DEFINES_H__

#define __inline__ __inline
#define __no_return__ __declspec(noreturn)
#define __noinline__ __declspec(noinline)
#define __forceinline__ __forceinline
#define __inline_hint__
#define __align__(n) __declspec(align(n))
#define __thread__ __declspec(thread)
#define __import__ __declspec(dllimport)
#define __export__ __declspec(dllexport)
#define __annotate__(a) __declspec(a)
#define __location__(a) __annotate__(__##a##__)
#define CUDARTAPI __stdcall
#define CUDARTAPI_CDECL __cdecl
#define __specialization_static static
#define __builtin_align__(a) __align__(a)
#define __restrict__ __restrict

#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __managed__
#define __grid_constant__
#define __launch_bounds__(...)
#define __cluster_dims__(...)
#define __cudart_builtin__
#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __nv_pure__
#define __CUDA_ALIGN__(n) __declspec(align(n))
#define CUDA_ALIGN(n) __declspec(align(n))
