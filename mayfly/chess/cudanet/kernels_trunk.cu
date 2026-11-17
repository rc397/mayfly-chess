// The trunk kernels of MayFly's CUDA backend, CUDA C for NVRTC.
// Compiled at runtime to a cubin (for example sm_86 or sm_120). No host code, no header beyond cuda_fp16.h.
// The contract is cudanet_spec.h and the launch configs are in kernels_trunk_launch.h (keep the two in step).
//
// Kernels:
//   embed              grid (65, B), block 256      the 65 input tokens of each position into x (f32)
//   layernorm_f32_f16  grid (rows), block 256       x f32 -> h f16, one block per row
//   add_layernorm_f32_f16 grid (rows), block 256    x += r (a half product), then the same norm
//   add_f16_f32        grid stride, block 256       x += r
//   attention_f16      grid (16, B, nz), block 128  one block per (head, position, query slice)
//   attention_mma      grid (16, B), block 160      the same on tensor cores, one block per (head, position)
//   gelu_f16           grid stride, block 256       exact erf GELU in place
// All shared memory is static, and no kernel needs dynamic shared memory.
// Every block size must be a multiple of 32 (full warp shuffles).
// Assumes: every pointer comes from cudaMalloc or the 256 aligned weight file (16 byte vector loads),
// cuda_fp16.h is found by NVRTC (the loader passes the CUDA include dir), no fast math needed.

#include <cuda_fp16.h>

#define TK_D 1024      // width
#define TK_TOK 65      // CLS + 64 squares
#define TK_DH 64       // head dim
#define TK_HEADS 16
#define TK_QKV 3072    // qkv row length (q, k, v)
#define TK_PIECES 13
#define TK_HM 102

// ------------------------------------------------------------------------------------------------ helpers

__device__ __forceinline__ float tk_warp_sum(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float tk_warp_max(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Sum over the block. red holds 33 floats. Every thread of the block must call it.
// Two barriers are enough to reuse red in the next call. The next call writes red[w] only
// after warp 0 has read it (second barrier) and red[32] only after everyone read it (first barrier).
__device__ __forceinline__ float tk_block_sum(float v, float* red)
{
    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    const int nw = (blockDim.x + 31) >> 5;
    v = tk_warp_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = lane < nw ? red[lane] : 0.0f;
        t = tk_warp_sum(t);
        if (lane == 0) red[32] = t;
    }
    __syncthreads();
    return red[32];
}

__device__ __forceinline__ float4 tk_add4(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// ------------------------------------------------------------------------------------------------ embed
// x f32 [B*65, 1024]. Row b*65 is CLS: cls + cast_w [1024, 4] * cast[b] + cast_b + hm_emb[hm[b]].
// Row b*65+1+s: piece[ids[b*64+s]] + pos[s] (+ ep_mark when ep[b] > 0 and s == ep[b]-1).
// Launch: grid (65, B), block 256 (each thread one float4 of the row, any multiple of 32 works).
// Indices are clamped to their tables so a bad input cannot read out of bounds.
extern "C" __global__ void embed(const long long* __restrict__ ids, const float* __restrict__ cast,
                                 const long long* __restrict__ hm, const long long* __restrict__ ep,
                                 const float* __restrict__ piece, const float* __restrict__ pos,
                                 const float* __restrict__ cls, const float* __restrict__ cast_w,
                                 const float* __restrict__ cast_b, const float* __restrict__ hm_emb,
                                 const float* __restrict__ ep_mark, float* __restrict__ x, int B)
{
    const int t = blockIdx.x;
    const int b = blockIdx.y;
    if (t >= TK_TOK || b >= B) return;
    float4* xr = reinterpret_cast<float4*>(x + ((size_t)b * TK_TOK + t) * TK_D);

    if (t == 0) {
        const float c0 = cast[b * 4 + 0], c1 = cast[b * 4 + 1];
        const float c2 = cast[b * 4 + 2], c3 = cast[b * 4 + 3];
        long long hi = hm[b];
        hi = hi < 0 ? 0 : (hi >= TK_HM ? TK_HM - 1 : hi);
        const float4* cl4 = reinterpret_cast<const float4*>(cls);
        const float4* cb4 = reinterpret_cast<const float4*>(cast_b);
        const float4* he4 = reinterpret_cast<const float4*>(hm_emb + (size_t)hi * TK_D);
        const float4* cw4 = reinterpret_cast<const float4*>(cast_w);   // row j of cast_w is cw4[j]
        for (int c = threadIdx.x; c < TK_D / 4; c += blockDim.x) {
            const float4 w0 = cw4[4 * c + 0], w1 = cw4[4 * c + 1];
            const float4 w2 = cw4[4 * c + 2], w3 = cw4[4 * c + 3];
            const float4 bb = cb4[c];
            // the cast linear first, as torch does, then cls + linear + hm
            float4 lin;
            lin.x = bb.x + (w0.x * c0 + w0.y * c1 + w0.z * c2 + w0.w * c3);
            lin.y = bb.y + (w1.x * c0 + w1.y * c1 + w1.z * c2 + w1.w * c3);
            lin.z = bb.z + (w2.x * c0 + w2.y * c1 + w2.z * c2 + w2.w * c3);
            lin.w = bb.w + (w3.x * c0 + w3.y * c1 + w3.z * c2 + w3.w * c3);
            xr[c] = tk_add4(tk_add4(cl4[c], lin), he4[c]);
        }
    } else {
        const int s = t - 1;
        long long p = ids[(size_t)b * 64 + s];
        p = p < 0 ? 0 : (p >= TK_PIECES ? TK_PIECES - 1 : p);
        const long long e = ep[b];
        const bool mark = e > 0 && (long long)s == e - 1;
        const float4* pc4 = reinterpret_cast<const float4*>(piece + (size_t)p * TK_D);
        const float4* ps4 = reinterpret_cast<const float4*>(pos + (size_t)s * TK_D);
        const float4* em4 = reinterpret_cast<const float4*>(ep_mark);
        for (int c = threadIdx.x; c < TK_D / 4; c += blockDim.x) {
            float4 v = tk_add4(pc4[c], ps4[c]);
            if (mark) v = tk_add4(v, em4[c]);
            xr[c] = v;
        }
    }
}

// ------------------------------------------------------------------------------------------------ layer norm
// y = (x - mean) * rsqrt(var + eps) * g + b, biased variance, all in fp32, rounded to half once.
// Two passes over registers (mean, then sum of squared deviations), like torch's accuracy or better.
// Launch: grid (rows), block 256. Needs d % 4 == 0 and d <= 4 * 4 * blockDim (4096 at 256 threads).
#define LN_MAXV 4

// the norm of one row already in registers (this thread's float4s v, their sum s) into y's row
__device__ __forceinline__ void tk_ln_row(const float4* v, float s, const float* __restrict__ g,
                                          const float* __restrict__ b, __half* __restrict__ y, int row, int d,
                                          float eps, float* red);

extern "C" __global__ void __launch_bounds__(256)
layernorm_f32_f16(const float* __restrict__ x, const float* __restrict__ g, const float* __restrict__ b,
                  __half* __restrict__ y, int rows, int d, float eps)
{
    __shared__ float red[33];
    const int row = blockIdx.x;
    if (row >= rows) return;   // uniform over the block
    const int d4 = d >> 2;
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * d);

    float4 v[LN_MAXV];
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < LN_MAXV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < d4) {
            v[k] = xr[i];
            s += (v[k].x + v[k].y) + (v[k].z + v[k].w);
        } else {
            v[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    tk_ln_row(v, s, g, b, y, row, d, eps, red);
}

// The same after adding a half product into the residual stream: x += r (written back), then the norm of x.
// The fp16-accumulated out and ff2 products (MAYFLY_CUDANET_ACC16=2) land in r instead of adding into x.
extern "C" __global__ void __launch_bounds__(256)
add_layernorm_f32_f16(float* __restrict__ x, const __half* __restrict__ r, const float* __restrict__ g,
                      const float* __restrict__ b, __half* __restrict__ y, int rows, int d, float eps)
{
    __shared__ float red[33];
    const int row = blockIdx.x;
    if (row >= rows) return;   // uniform over the block
    const int d4 = d >> 2;
    float4* xr = reinterpret_cast<float4*>(x + (size_t)row * d);
    const uint2* rr = reinterpret_cast<const uint2*>(r + (size_t)row * d);   // 4 halves per load

    float4 v[LN_MAXV];
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < LN_MAXV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < d4) {
            const uint2 u = rr[i];
            const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
            const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
            v[k] = xr[i];
            v[k].x += lo.x; v[k].y += lo.y; v[k].z += hi.x; v[k].w += hi.y;
            xr[i] = v[k];
            s += (v[k].x + v[k].y) + (v[k].z + v[k].w);
        } else {
            v[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    tk_ln_row(v, s, g, b, y, row, d, eps, red);
}

// x += r over n floats (n % 4 == 0), grid stride: the last layer's ff2 product before the CLS norm
extern "C" __global__ void __launch_bounds__(256)
add_f16_f32(float* __restrict__ x, const __half* __restrict__ r, int n)
{
    const int n4 = n >> 2;
    float4* x4 = reinterpret_cast<float4*>(x);
    const uint2* r4 = reinterpret_cast<const uint2*>(r);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        const uint2 u = r4[i];
        const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
        float4 v = x4[i];
        v.x += lo.x; v.y += lo.y; v.z += hi.x; v.w += hi.y;
        x4[i] = v;
    }
}

__device__ __forceinline__ void tk_ln_row(const float4* v, float s, const float* __restrict__ g,
                                          const float* __restrict__ b, __half* __restrict__ y, int row, int d,
                                          float eps, float* red)
{
    const int d4 = d >> 2;
    const float inv_d = 1.0f / (float)d;
    const float mean = tk_block_sum(s, red) * inv_d;

    float q = 0.0f;
    #pragma unroll
    for (int k = 0; k < LN_MAXV; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < d4) {
            const float a = v[k].x - mean, bq = v[k].y - mean, c = v[k].z - mean, e = v[k].w - mean;
            q += (a * a + bq * bq) + (c * c + e * e);
        }
    }
    const float var = tk_block_sum(q, red) * inv_d;
    const float rstd = rsqrtf(var + eps);

    const float4* g4 = reinterpret_cast<const float4*>(g);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    uint2* yr = reinterpret_cast<uint2*>(y + (size_t)row * d);   // 4 halves per store
    #pragma unroll
    for (int k = 0; k < LN_MAXV; ++k) {
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

// ------------------------------------------------------------------------------------------------ attention
// qkv f16 [B*65, 3072]: q cols 0..1023, k 1024..2047, v 2048..3071, head j owns cols j*64..j*64+63
// of each. att f16 [B*65, 1024], heads concatenated. 16 heads, 65 tokens, no mask.
// Scores, softmax and the weighted sum of V are fp32, and the output is rounded to half once.
//
// Launch: grid (16 heads, B, nz), block 128 (4 warps), no dynamic shared memory.
//   blockIdx.z splits the 65 query rows into nz slices of ceil(65/nz) rows, so a small batch still
//   fills the GPU (B = 1 gives 16 * nz blocks). Each slice reloads K and V of its head from L2.
//   Any nz >= 1 is correct, since the kernel reads gridDim.z.
// Per block: K and V of one head in shared memory (65 x 64 halves each). K rows are padded to 66
// halves (33 words) so that 32 lanes reading 32 different key rows hit 32 different banks.
// Per query row (one warp), lane l scores keys l and l+32 against q (q in shared, broadcast reads),
// and key 64 is a warp reduction. Softmax with warp shuffles. Then lane l owns output dims 2l, 2l+1 and
// sums p_j * V[j] over the 65 keys (V reads are 32 consecutive words, conflict free).
#define ATT_WARPS 4
#define ATT_KSTR 33   // K row stride in half2 words
extern "C" __global__ void __launch_bounds__(128)
attention_f16(const __half* __restrict__ qkv, __half* __restrict__ att, int B, float scale)
{
    __shared__ __align__(16) __half2 ks[TK_TOK * ATT_KSTR];
    __shared__ __align__(16) __half2 vs[TK_TOK * 32];
    __shared__ __align__(16) float qs[ATT_WARPS][TK_DH];
    __shared__ __align__(16) float ps[ATT_WARPS][TK_TOK + 3];

    const int h = blockIdx.x;
    const int b = blockIdx.y;
    if (h >= TK_HEADS || b >= B) return;   // uniform over the block
    const __half* base = qkv + (size_t)b * TK_TOK * TK_QKV + h * TK_DH;

    // K and V of this head: 65 rows x 8 chunks of 16 bytes each, coalesced 128 byte rows
    for (int i = threadIdx.x; i < TK_TOK * 8; i += blockDim.x) {
        const int t = i >> 3, c = i & 7;
        const __half* row = base + (size_t)t * TK_QKV + c * 8;
        const uint4 kk = *reinterpret_cast<const uint4*>(row + TK_D);
        const uint4 vv = *reinterpret_cast<const uint4*>(row + 2 * TK_D);
        unsigned int* kd = reinterpret_cast<unsigned int*>(ks + t * ATT_KSTR + c * 4);   // 4 byte aligned only
        kd[0] = kk.x; kd[1] = kk.y; kd[2] = kk.z; kd[3] = kk.w;
        *reinterpret_cast<uint4*>(vs + t * 32 + c * 4) = vv;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    if (w >= ATT_WARPS) return;   // no barrier after this point
    const int nz = gridDim.z;
    const int per = (TK_TOK + nz - 1) / nz;
    const int r0 = blockIdx.z * per;
    const int r1 = (r0 + per < TK_TOK) ? r0 + per : TK_TOK;

    float* q = qs[w];
    float* p = ps[w];
    const __half2* k0 = ks + lane * ATT_KSTR;            // key lane
    const __half2* k1 = ks + (lane + 32) * ATT_KSTR;     // key lane + 32
    const float2 k64 = __half22float2(ks[64 * ATT_KSTR + lane]);   // key 64, dims 2l and 2l+1
    const __half2* vcol = vs + lane;                     // V dims 2l, 2l+1

    for (int i = r0 + w; i < r1; i += ATT_WARPS) {
        const size_t r = (size_t)b * TK_TOK + i;
        const float2 qf = __half22float2(reinterpret_cast<const __half2*>(qkv + r * TK_QKV + h * TK_DH)[lane]);
        reinterpret_cast<float2*>(q)[lane] = qf;
        __syncwarp();

        // scores for keys lane and lane+32
        float a0 = 0.0f, a1 = 0.0f;
        #pragma unroll 8
        for (int d2 = 0; d2 < 32; ++d2) {
            const float2 qq = reinterpret_cast<const float2*>(q)[d2];   // broadcast
            const float2 x0 = __half22float2(k0[d2]);
            const float2 x1 = __half22float2(k1[d2]);
            a0 = fmaf(qq.x, x0.x, a0); a0 = fmaf(qq.y, x0.y, a0);
            a1 = fmaf(qq.x, x1.x, a1); a1 = fmaf(qq.y, x1.y, a1);
        }
        // key 64: each lane its two dims, then a warp sum (every lane gets it)
        const float a2 = tk_warp_sum(fmaf(qf.x, k64.x, qf.y * k64.y));

        const float s0 = a0 * scale, s1 = a1 * scale, s2 = a2 * scale;
        const float m = tk_warp_max(fmaxf(fmaxf(s0, s1), s2));
        const float e0 = expf(s0 - m), e1 = expf(s1 - m), e2 = expf(s2 - m);
        const float sum = tk_warp_sum(e0 + e1) + e2;   // e2 is the same on every lane, count it once
        p[lane] = e0;
        p[lane + 32] = e1;
        if (lane == 0) p[64] = e2;
        __syncwarp();

        float ox = 0.0f, oy = 0.0f;
        #pragma unroll 5
        for (int j = 0; j < TK_TOK; ++j) {
            const float pj = p[j];   // broadcast
            const float2 vv = __half22float2(vcol[j * 32]);
            ox = fmaf(pj, vv.x, ox);
            oy = fmaf(pj, vv.y, oy);
        }
        const float inv = 1.0f / sum;
        reinterpret_cast<__half2*>(att + r * TK_D + h * TK_DH)[lane] = __floats2half2_rn(ox * inv, oy * inv);
        __syncwarp();   // q and p are rewritten by the next row
    }
}

// ------------------------------------------------------------------------------------------------ attention on tensor cores
// The same attention as attention_f16 with both products on tensor cores (mma.sync m16n8k16, fp16 inputs, fp32
// accumulation), the scores, softmax and probabilities never leaving registers. The scores and softmax are fp32 as
// before. The one new rounding is each probability to half for the second product (about 5e-4 relative, the size
// of the output's own rounding to half).
//
// Launch: grid (16 heads, B), block 160 (5 warps), no dynamic shared memory.
// Rows and keys are padded from 65 to 80 (five 16 row tiles, ten 8 key tiles). Warp w owns query rows 16w..16w+15.
// Its Q fragments come straight from global memory (rows past 64 are zero), K comes from shared memory row major,
// V from shared memory transposed (dims x keys) so both B fragments are single 32 bit loads. Keys past 64 are
// zero in shared memory and masked to -inf before the softmax. Rows past 64 are computed and never stored.
// Fragment layouts (PTX ISA, mma.m16n8k16 .f16): g = lane / 4, c = lane % 4.
//   A 16x16: a0 (g, 2c..2c+1), a1 (g+8, 2c..), a2 (g, 2c+8..), a3 (g+8, 2c+8..)
//   B 16x8:  b0 (k 2c..2c+1, n g), b1 (k 2c+8..2c+9, n g)
//   C 16x8:  c0 c1 (g, 2c..2c+1), c2 c3 (g+8, 2c..2c+1)
// A C fragment pair of key tiles 2j, 2j+1 is exactly the A fragment of key step j, so P goes from the scores to the
// second product without a shuffle. Shared row strides 72 and 88 halves keep the 32 lanes on 32 different banks.
#define AM_ROWS 80
#define AM_KSTR 72    // K row stride in halves (64 dims + 8)
#define AM_VSTR 88    // V transposed row stride in halves (80 keys + 8)
#define AM_NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ void am_mma(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned am_pack(float lo, float hi)
{
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&v);
}

extern "C" __global__ void __launch_bounds__(160)
attention_mma(const __half* __restrict__ qkv, __half* __restrict__ att, int B, float scale)
{
    __shared__ __align__(16) __half ks[AM_ROWS * AM_KSTR];
    __shared__ __align__(16) __half vt[TK_DH * AM_VSTR];

    const int h = blockIdx.x;
    const int b = blockIdx.y;
    if (h >= TK_HEADS || b >= B) return;   // uniform over the block
    const __half* base = qkv + (size_t)b * TK_TOK * TK_QKV + h * TK_DH;

    // K rows: 65 rows x 8 chunks of 16 bytes, coalesced
    for (int i = threadIdx.x; i < TK_TOK * 8; i += blockDim.x) {
        const int t = i >> 3, c = i & 7;
        *reinterpret_cast<uint4*>(ks + t * AM_KSTR + c * 8) =
            *reinterpret_cast<const uint4*>(base + (size_t)t * TK_QKV + TK_D + c * 8);
    }
    // V transposed, rows fastest so a warp's 2 byte stores land on consecutive words
    for (int i = threadIdx.x; i < TK_TOK * 8; i += blockDim.x) {
        const int t = i % TK_TOK, c = i / TK_TOK;
        const uint4 vv = *reinterpret_cast<const uint4*>(base + (size_t)t * TK_QKV + 2 * TK_D + c * 8);
        const __half* v8 = reinterpret_cast<const __half*>(&vv);
        #pragma unroll
        for (int e = 0; e < 8; ++e) vt[(c * 8 + e) * AM_VSTR + t] = v8[e];
    }
    // the padding: K rows 65..79 and V columns 65..79 are zero
    const __half zero = __float2half(0.0f);
    for (int i = threadIdx.x; i < (AM_ROWS - TK_TOK) * TK_DH; i += blockDim.x) {
        const int t = TK_TOK + i / TK_DH, d = i % TK_DH;
        ks[t * AM_KSTR + d] = zero;
        vt[d * AM_VSTR + t] = zero;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;
    const int g = lane >> 2, c = lane & 3;
    const int r0 = w * 16 + g, r1 = r0 + 8;   // this thread's two query rows

    // Q fragments for the four 16 dim steps
    unsigned qa[4][4];
    {
        const unsigned* q0 = reinterpret_cast<const unsigned*>(base + (size_t)(r0 < TK_TOK ? r0 : 0) * TK_QKV);
        const unsigned* q1 = reinterpret_cast<const unsigned*>(base + (size_t)(r1 < TK_TOK ? r1 : 0) * TK_QKV);
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            qa[s][0] = r0 < TK_TOK ? q0[s * 8 + c] : 0u;
            qa[s][1] = r1 < TK_TOK ? q1[s * 8 + c] : 0u;
            qa[s][2] = r0 < TK_TOK ? q0[s * 8 + 4 + c] : 0u;
            qa[s][3] = r1 < TK_TOK ? q1[s * 8 + 4 + c] : 0u;
        }
    }

    // scores: ten key tiles of 8
    float sc[10][4];
    #pragma unroll
    for (int n = 0; n < 10; ++n) {
        sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.0f;
        const unsigned* kr = reinterpret_cast<const unsigned*>(ks + (n * 8 + g) * AM_KSTR);
        #pragma unroll
        for (int s = 0; s < 4; ++s)
            am_mma(sc[n], qa[s][0], qa[s][1], qa[s][2], qa[s][3], kr[s * 8 + c], kr[s * 8 + 4 + c]);
    }

    // softmax of rows r0 (elements 0, 1) and r1 (2, 3) in fp32, with a row's 80 keys on the 4 lanes of a quad
    float m0 = AM_NEG_INF, m1 = AM_NEG_INF;
    #pragma unroll
    for (int n = 0; n < 10; ++n) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
            const bool live = n * 8 + 2 * c + e < TK_TOK;
            sc[n][e] = live ? sc[n][e] * scale : AM_NEG_INF;
            sc[n][2 + e] = live ? sc[n][2 + e] * scale : AM_NEG_INF;
            m0 = fmaxf(m0, sc[n][e]);
            m1 = fmaxf(m1, sc[n][2 + e]);
        }
    }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    float s0 = 0.0f, s1 = 0.0f;
    #pragma unroll
    for (int n = 0; n < 10; ++n) {
        sc[n][0] = expf(sc[n][0] - m0);
        sc[n][1] = expf(sc[n][1] - m0);
        sc[n][2] = expf(sc[n][2] - m1);
        sc[n][3] = expf(sc[n][3] - m1);
        s0 += sc[n][0] + sc[n][1];
        s1 += sc[n][2] + sc[n][3];
    }
    s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 2);

    // out = P V: five 16 key steps, eight 8 dim tiles
    float o[8][4];
    #pragma unroll
    for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    #pragma unroll
    for (int j = 0; j < 5; ++j) {
        const unsigned p0 = am_pack(sc[2 * j][0], sc[2 * j][1]);
        const unsigned p1 = am_pack(sc[2 * j][2], sc[2 * j][3]);
        const unsigned p2 = am_pack(sc[2 * j + 1][0], sc[2 * j + 1][1]);
        const unsigned p3 = am_pack(sc[2 * j + 1][2], sc[2 * j + 1][3]);
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            const unsigned* vr = reinterpret_cast<const unsigned*>(vt + (n * 8 + g) * AM_VSTR + j * 16);
            am_mma(o[n], p0, p1, p2, p3, vr[c], vr[4 + c]);
        }
    }

    const float i0 = 1.0f / s0, i1 = 1.0f / s1;
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
        const int d = h * TK_DH + n * 8 + 2 * c;
        if (r0 < TK_TOK)
            *reinterpret_cast<__half2*>(att + ((size_t)b * TK_TOK + r0) * TK_D + d) =
                __floats2half2_rn(o[n][0] * i0, o[n][1] * i0);
        if (r1 < TK_TOK)
            *reinterpret_cast<__half2*>(att + ((size_t)b * TK_TOK + r1) * TK_D + d) =
                __floats2half2_rn(o[n][2] * i1, o[n][3] * i1);
    }
}

// ------------------------------------------------------------------------------------------------ GELU
// Exact GELU in place, 0.5 * x * (1 + erf(x / sqrt(2))) in fp32, as torch's half kernel does.
// Launch: block 256, any grid (grid stride loop). Paths: 8 halves per thread step when n % 8 == 0
// and h is 16 byte aligned, half2 when n is even and h is 4 byte aligned, else one half at a time.
__device__ __forceinline__ float tk_gelu(float v)
{
    return v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f));
}

__device__ __forceinline__ unsigned int tk_gelu2(unsigned int u)
{
    __half2 hv = *reinterpret_cast<const __half2*>(&u);
    const float2 f = __half22float2(hv);
    hv = __floats2half2_rn(tk_gelu(f.x), tk_gelu(f.y));
    return *reinterpret_cast<const unsigned int*>(&hv);
}

extern "C" __global__ void gelu_f16(__half* __restrict__ h, int n)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t addr = (size_t)h;
    if ((n & 7) == 0 && (addr & 15) == 0) {
        uint4* p = reinterpret_cast<uint4*>(h);
        const size_t n8 = (size_t)(n >> 3);
        for (size_t i = tid; i < n8; i += stride) {
            uint4 u = p[i];
            u.x = tk_gelu2(u.x); u.y = tk_gelu2(u.y);
            u.z = tk_gelu2(u.z); u.w = tk_gelu2(u.w);
            p[i] = u;
        }
    } else if ((n & 1) == 0 && (addr & 3) == 0) {
        unsigned int* p = reinterpret_cast<unsigned int*>(h);
        const size_t n2 = (size_t)(n >> 1);
        for (size_t i = tid; i < n2; i += stride) p[i] = tk_gelu2(p[i]);
    } else {
        for (size_t i = tid; i < (size_t)n; i += stride) h[i] = __float2half_rn(tk_gelu(__half2float(h[i])));
    }
}
