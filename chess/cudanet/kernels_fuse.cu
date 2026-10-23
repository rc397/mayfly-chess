// Fused trunk kernels of MayFly's CUDA backend, CUDA C for NVRTC, used only when MAYFLY_CUDANET_FUSE=1.
// Each one gives the same bits as the kernels it replaces (kernels_trunk.cu, kernels_heads.cu), with the same
// arithmetic in the same order and only fewer passes over the residual stream.
//
// Kernels:
//   embed_ln             grid (65, B), block 256     embed, then layer 0's first norm of the same row (no reread of x)
//   add_layernorm_keep   grid (rows), block 256      the norm of x + r without writing x back (the ACC16=2 second norm)
//   add2_layernorm       grid (rows), block 256      x = (x + ra) + rb written back, then its norm (the first norm)
//   cls_layernorm_add2   grid (B), block 256         the final norm of the CLS rows of (x + ra) + rb, CLS rows only
// With ACC16=2 the residual stream is written once a layer instead of twice. The out product's half result ra is
// added in registers by the second norm and again, together with the ff2 product rb, by the next layer's first norm.
// fp32 addition of the same values in the same order gives the same x, so h and the logits are unchanged.
// Block size must be 256 and d = 1024 (one float4 a thread, as layernorm_f32_f16 runs at d = 1024).

#include <cuda_fp16.h>

#define FU_D 1024
#define FU_TOK 65
#define FU_PIECES 13
#define FU_HM 102
#define FU_MAXV 4

__device__ __forceinline__ float fu_warp_sum(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// tk_block_sum of kernels_trunk.cu
__device__ __forceinline__ float fu_block_sum(float v, float* red)
{
    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    const int nw = (blockDim.x + 31) >> 5;
    v = fu_warp_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = lane < nw ? red[lane] : 0.0f;
        t = fu_warp_sum(t);
        if (lane == 0) red[32] = t;
    }
    __syncthreads();
    return red[32];
}

__device__ __forceinline__ float4 fu_add4(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// v += four halves
__device__ __forceinline__ void fu_addh4(float4& v, uint2 u)
{
    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    v.x += lo.x; v.y += lo.y; v.z += hi.x; v.w += hi.y;
}

// tk_ln_row of kernels_trunk.cu, unchanged
__device__ __forceinline__ void fu_ln_row(const float4* v, float s, const float* __restrict__ g,
                                          const float* __restrict__ b, __half* __restrict__ y, int row, int d,
                                          float eps, float* red)
{
    const int d4 = d >> 2;
    const float inv_d = 1.0f / (float)d;
    const float mean = fu_block_sum(s, red) * inv_d;

    float q = 0.0f;
    #pragma unroll
    for (int k = 0; k < FU_MAXV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < d4) {
            const float a = v[k].x - mean, bq = v[k].y - mean, c = v[k].z - mean, e = v[k].w - mean;
            q += (a * a + bq * bq) + (c * c + e * e);
        }
    }
    const float var = fu_block_sum(q, red) * inv_d;
    const float rstd = rsqrtf(var + eps);

    const float4* g4 = reinterpret_cast<const float4*>(g);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    uint2* yr = reinterpret_cast<uint2*>(y + (size_t)row * d);   // 4 halves per store
    #pragma unroll
    for (int k = 0; k < FU_MAXV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < d4) {
            const float4 gg = g4[i], bb = b4[i];
            const __half2 lo = __floats2half2_rn((v[k].x - mean) * rstd * gg.x + bb.x,
                                                 (v[k].y - mean) * rstd * gg.y + bb.y);
            const __half2 hi = __floats2half2_rn((v[k].z - mean) * rstd * gg.z + bb.z,
                                                 (v[k].w - mean) * rstd * gg.w + bb.w);
            uint2 o;
            o.x = *reinterpret_cast<const unsigned int*>(&lo);
            o.y = *reinterpret_cast<const unsigned int*>(&hi);
            yr[i] = o;
        }
    }
}

// the norm of one row whose float4 for this thread is v0 (the others zero), as layernorm_f32_f16 at block 256
__device__ __forceinline__ void fu_ln_one(float4 v0, const float* __restrict__ g, const float* __restrict__ b,
                                          __half* __restrict__ y, int row, float eps, float* red)
{
    float4 v[FU_MAXV];
    v[0] = v0;
    #pragma unroll
    for (int k = 1; k < FU_MAXV; ++k) v[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float s = 0.0f;
    s += (v0.x + v0.y) + (v0.z + v0.w);
    fu_ln_row(v, s, g, b, y, row, FU_D, eps, red);
}

// ------------------------------------------------------------------------------------------------ embed + norm
// embed of kernels_trunk.cu for one row (thread c owns float4 c), then layernorm_f32_f16 of that row into h.
extern "C" __global__ void __launch_bounds__(256)
embed_ln(const long long* __restrict__ ids, const float* __restrict__ cast, const long long* __restrict__ hm,
         const long long* __restrict__ ep, const float* __restrict__ piece, const float* __restrict__ pos,
         const float* __restrict__ cls, const float* __restrict__ cast_w, const float* __restrict__ cast_b,
         const float* __restrict__ hm_emb, const float* __restrict__ ep_mark, float* __restrict__ x,
         const float* __restrict__ g, const float* __restrict__ bn, __half* __restrict__ h, int B, float eps)
{
    __shared__ float red[33];
    const int t = blockIdx.x;
    const int b = blockIdx.y;
    if (t >= FU_TOK || b >= B) return;   // uniform over the block
    const int row = b * FU_TOK + t;
    float4* xr = reinterpret_cast<float4*>(x + (size_t)row * FU_D);
    const int c = threadIdx.x;   // < 256 = FU_D / 4
    float4 val;

    if (t == 0) {
        const float c0 = cast[b * 4 + 0], c1 = cast[b * 4 + 1];
        const float c2 = cast[b * 4 + 2], c3 = cast[b * 4 + 3];
        long long hi = hm[b];
        hi = hi < 0 ? 0 : (hi >= FU_HM ? FU_HM - 1 : hi);
        const float4* cl4 = reinterpret_cast<const float4*>(cls);
        const float4* cb4 = reinterpret_cast<const float4*>(cast_b);
        const float4* he4 = reinterpret_cast<const float4*>(hm_emb + (size_t)hi * FU_D);
        const float4* cw4 = reinterpret_cast<const float4*>(cast_w);
        const float4 w0 = cw4[4 * c + 0], w1 = cw4[4 * c + 1];
        const float4 w2 = cw4[4 * c + 2], w3 = cw4[4 * c + 3];
        const float4 bb = cb4[c];
        float4 lin;
        lin.x = bb.x + (w0.x * c0 + w0.y * c1 + w0.z * c2 + w0.w * c3);
        lin.y = bb.y + (w1.x * c0 + w1.y * c1 + w1.z * c2 + w1.w * c3);
        lin.z = bb.z + (w2.x * c0 + w2.y * c1 + w2.z * c2 + w2.w * c3);
        lin.w = bb.w + (w3.x * c0 + w3.y * c1 + w3.z * c2 + w3.w * c3);
        val = fu_add4(fu_add4(cl4[c], lin), he4[c]);
    } else {
        const int s = t - 1;
        long long p = ids[(size_t)b * 64 + s];
        p = p < 0 ? 0 : (p >= FU_PIECES ? FU_PIECES - 1 : p);
        const long long e = ep[b];
        const bool mark = e > 0 && (long long)s == e - 1;
        const float4* pc4 = reinterpret_cast<const float4*>(piece + (size_t)p * FU_D);
        const float4* ps4 = reinterpret_cast<const float4*>(pos + (size_t)s * FU_D);
        const float4* em4 = reinterpret_cast<const float4*>(ep_mark);
        val = fu_add4(pc4[c], ps4[c]);
        if (mark) val = fu_add4(val, em4[c]);
    }
    xr[c] = val;
    fu_ln_one(val, g, bn, h, row, eps, red);
}

// ------------------------------------------------------------------------------------------------ residual norms
// h = norm(x + r), x left as it is (the next first norm adds r again, with the next product).
extern "C" __global__ void __launch_bounds__(256)
add_layernorm_keep(const float* __restrict__ x, const __half* __restrict__ r, const float* __restrict__ g,
                   const float* __restrict__ b, __half* __restrict__ y, int rows, float eps)
{
    __shared__ float red[33];
    const int row = blockIdx.x;
    if (row >= rows) return;   // uniform over the block
    const size_t o = (size_t)row * (FU_D / 4) + threadIdx.x;
    float4 v = reinterpret_cast<const float4*>(x)[o];
    fu_addh4(v, reinterpret_cast<const uint2*>(r)[o]);
    fu_ln_one(v, g, b, y, row, eps, red);
}

// x = (x + ra) + rb, written back, then h = norm(x).
extern "C" __global__ void __launch_bounds__(256)
add2_layernorm(float* __restrict__ x, const __half* __restrict__ ra, const __half* __restrict__ rb,
               const float* __restrict__ g, const float* __restrict__ b, __half* __restrict__ y, int rows, float eps)
{
    __shared__ float red[33];
    const int row = blockIdx.x;
    if (row >= rows) return;   // uniform over the block
    const size_t o = (size_t)row * (FU_D / 4) + threadIdx.x;
    float4 v = reinterpret_cast<float4*>(x)[o];
    const uint2 ua = reinterpret_cast<const uint2*>(ra)[o];
    const uint2 ub = reinterpret_cast<const uint2*>(rb)[o];
    fu_addh4(v, ua);
    fu_addh4(v, ub);
    reinterpret_cast<float4*>(x)[o] = v;
    fu_ln_one(v, g, b, y, row, eps, red);
}

// ------------------------------------------------------------------------------------------------ the final norm
// cls_layernorm of kernels_heads.cu on (x + ra) + rb, the last layer's two products added for the CLS rows only.
// hd_block_sum of kernels_heads.cu (one barrier more than fu_block_sum)
__device__ float fu_block_sum3(float v, float* sh)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = lane < nw ? sh[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (lane == 0) sh[32] = t;
    }
    __syncthreads();
    return sh[32];
}

__device__ __forceinline__ float fu_res(const float* row, const __half* a, const __half* b, int i)
{
    return (row[i] + __half2float(a[i])) + __half2float(b[i]);
}

extern "C" __global__ void cls_layernorm_add2(const float* __restrict__ x, const __half* __restrict__ ra,
                                              const __half* __restrict__ rb, const float* __restrict__ g,
                                              const float* __restrict__ bt, __half* __restrict__ cls, int B,
                                              float eps)
{
    __shared__ float sh[33];
    const int b = blockIdx.x;
    if (b >= B) return;   // the whole block leaves together
    const long long off = (long long)b * FU_TOK * FU_D;
    const float* row = x + off;
    const __half* a = ra + off;
    const __half* c = rb + off;

    float s = 0.0f;
    for (int i = threadIdx.x; i < FU_D; i += blockDim.x) s += fu_res(row, a, c, i);
    const float mean = fu_block_sum3(s, sh) / (float)FU_D;

    float v = 0.0f;
    for (int i = threadIdx.x; i < FU_D; i += blockDim.x) {
        const float d = fu_res(row, a, c, i) - mean;
        v += d * d;
    }
    const float var = fu_block_sum3(v, sh) / (float)FU_D;
    const float rstd = rsqrtf(var + eps);

    __half* out = cls + (long long)b * FU_D;
    for (int i = threadIdx.x; i < FU_D; i += blockDim.x)
        out[i] = __float2half_rn((fu_res(row, a, c, i) - mean) * rstd * g[i] + bt[i]);
}
