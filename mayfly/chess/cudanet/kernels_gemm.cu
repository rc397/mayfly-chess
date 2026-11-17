// MayFly's own tensor core matmul for the trunk's four products (MAYFLY_CUDANET_GEMM).
// CUDA C for NVRTC, compiled at runtime to a cubin (sm_80 and up, for example sm_86 or sm_120).
// No host code, no header beyond cuda_fp16.h. The host side is mmgemm.h (the config table there must match the
// instantiations at the bottom of this file).
//
// Y[M, N] = X[M, K] W^T (+ epilogue), X row-major f16, W row-major [N, K] f16 (torch's Linear layout), so both
// operands run along k. A fragments come from X rows and B fragments from W rows with plain (non transposed)
// ldmatrix. mma.sync m16n8k16, a cp.async pipeline of ST stages, fragments double buffered in registers.
// Shared tiles are unpadded rows of BK halves with the 16 byte chunks XOR swizzled by row, so the 8 rows of each
// ldmatrix land on 8 different 16 byte bank groups (conflict free) and no shared memory is lost to padding.
//
// Accumulation (template ACC):
//   0  fp32 accumulators (the mma's f32 form, which GeForce runs at half the fp16 rate)
//   1  fp16 accumulators over the whole k (as cuBLASLt's CUBLAS_COMPUTE_16F, MAYFLY_CUDANET_ACC16=2)
//   2  fp16 accumulators within one k tile of BK, then added into fp32 ones, for the fp16 rate and nearly fp32 accuracy
// Epilogues (argument epi), all in fp32 from the accumulators:
//   0  Yh = acc + bias                 (qkv)
//   1  Yh = gelu(acc + bias)           (ff1, exact erf GELU, so no gelu kernel after it)
//   2  Yf = Yf + acc + bias            (out and ff2, the residual straight into the f32 stream, in place)
//   3  Yf = acc + bias
// bias may be null (no bias).
//
// Needs N % BN == 0, K % BK == 0, X, W, Y 16 byte aligned, bias 8 byte aligned. M is any positive count. Rows of the
// last tile past M are zero filled on load (cp.async with a source size of 0) and never stored.
// Launch: grid (ceil(M / BM) * (N / BN)), block WM * WN * 32, dynamic shared memory ST * (BM + BN) * BK * 2 bytes
// (more than 48 KB needs the function attribute set once). Tiles go out in groups of MM_GROUP row tiles per column
// tile, so the blocks running at once share their W and X tiles in L2.

#include <cuda_fp16.h>

#define MM_GROUP 8    // row tiles per group in the tile order

__device__ __forceinline__ unsigned mm_smem(const void* p)
{
    return (unsigned)__cvta_generic_to_shared(p);
}

// 16 bytes global to shared, bypassing L1. full == false writes 16 zero bytes and reads nothing.
__device__ __forceinline__ void mm_cp16(unsigned dst, const void* src, bool full)
{
    const int n = full ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}

__device__ __forceinline__ void mm_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void mm_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void mm_ldsm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// D += A B with fp16 accumulators: d0 = (g, 2c..2c+1), d1 = (g+8, 2c..2c+1) as half2
__device__ __forceinline__ void mm_mma_h(unsigned* d, const unsigned* a, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// D += A B with fp32 accumulators: d0 d1 = (g, 2c..2c+1), d2 d3 = (g+8, 2c..2c+1)
__device__ __forceinline__ void mm_mma_f(float* d, const unsigned* a, unsigned b0, unsigned b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float2 mm_h2f(unsigned u)
{
    return __half22float2(*reinterpret_cast<const __half2*>(&u));
}

__device__ __forceinline__ float mm_gelu(float v)
{
    return v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f));
}

// the swizzled 16 byte chunk of (row, chunk) in a row of CPR chunks, so 8 consecutive rows at one chunk hit 8
// different 16 byte bank groups. CPR 4 (BK 32, 64 byte rows) uses chunk ^ ((row >> 1) & 3), CPR 8 (BK 64) chunk ^ (row & 7)
template <int CPR>
__device__ __forceinline__ int mm_swz(int row, int ch)
{
    if constexpr (CPR == 8) return ch ^ (row & 7);
    else return ch ^ ((row >> 1) & 3);
}

template <int BM, int BN, int BK, int WM, int WN, int ST, int ACC>
__device__ __forceinline__ void mm_body(const __half* __restrict__ X, const __half* __restrict__ W,
                                        const float* __restrict__ bias, __half* __restrict__ Yh,
                                        float* __restrict__ Yf, int M, int N, int K, int epi)
{
    constexpr int NTH = WM * WN * 32;
    constexpr int WTM = BM / WM, WTN = BN / WN;
    constexpr int MT = WTM / 16, NT = WTN / 8;
    constexpr int CPR = BK / 8;               // 16 byte chunks per row of a k tile
    constexpr int KK = BK / 16;               // mma k steps per k tile
    constexpr int A_IT = BM * CPR / NTH, B_IT = BN * CPR / NTH;
    constexpr int ROWB = BK * 2;              // bytes per shared row
    constexpr int A_STAGE = BM * ROWB, B_STAGE = BN * ROWB;
    static_assert(WTM % 16 == 0 && WTN % 16 == 0, "warp tile must be a multiple of 16 x 16");
    static_assert((BM * CPR) % NTH == 0 && (BN * CPR) % NTH == 0, "load chunks must divide among the threads");
    static_assert((BK == 32 || BK == 64) && ST >= 2, "BK 32 or 64, at least 2 stages");

    extern __shared__ __align__(128) unsigned char mm_sh[];
    const unsigned sA = mm_smem(mm_sh);
    const unsigned sB = sA + ST * A_STAGE;

    // grouped tile order
    const int tiles_m = (M + BM - 1) / BM, tiles_n = N / BN;
    const int pid = blockIdx.x;
    const int per_group = MM_GROUP * tiles_n;
    const int first_m = (pid / per_group) * MM_GROUP;
    const int gsize = min(tiles_m - first_m, MM_GROUP);
    const int tm = first_m + (pid % per_group) % gsize;
    const int tn = (pid % per_group) / gsize;
    if (tm >= tiles_m || tn >= tiles_n) return;   // uniform over the block
    const int m0 = tm * BM, n0 = tn * BN;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WM, wn = warp / WM;

    // this thread's load chunks, fixed rows and columns, with k moving by BK a tile
    const __half* a_src[A_IT];
    unsigned a_dst[A_IT];
    bool a_ok[A_IT];
    #pragma unroll
    for (int j = 0; j < A_IT; ++j) {
        const int i = tid + j * NTH, r = i / CPR, ch = i % CPR;
        const int gr = m0 + r;
        a_ok[j] = gr < M;
        a_src[j] = X + (size_t)(gr < M ? gr : M - 1) * K + ch * 8;
        a_dst[j] = (unsigned)(r * ROWB + mm_swz<CPR>(r, ch) * 16);
    }
    const __half* b_src[B_IT];
    unsigned b_dst[B_IT];
    #pragma unroll
    for (int j = 0; j < B_IT; ++j) {
        const int i = tid + j * NTH, r = i / CPR, ch = i % CPR;
        b_src[j] = W + (size_t)(n0 + r) * K + ch * 8;
        b_dst[j] = (unsigned)(r * ROWB + mm_swz<CPR>(r, ch) * 16);
    }
    const int KT = K / BK;
    auto load = [&](int stage, int kt) {
        const int k0 = kt * BK;
        #pragma unroll
        for (int j = 0; j < A_IT; ++j) mm_cp16(sA + stage * A_STAGE + a_dst[j], a_src[j] + k0, a_ok[j]);
        #pragma unroll
        for (int j = 0; j < B_IT; ++j) mm_cp16(sB + stage * B_STAGE + b_dst[j], b_src[j] + k0, true);
    };

    // ldmatrix addresses. A x4: matrices (rows 0-7, k 0-7), (8-15, 0-7), (0-7, 8-15), (8-15, 8-15) = a0..a3, so
    // lane l gives row l & 15, chunk l >> 4. B x4 over two n tiles: (n 0-7, k 0-7), (0-7, 8-15), (8-15, 0-7),
    // (8-15, 8-15) = b0 b1 of the first n tile, then of the second: row (l & 7) + (l >> 4) * 8, chunk (l >> 3) & 1.
    // Every warp tile row starts at a multiple of 16, so the swizzle depends on lane & 7 alone (and on the chunk).
    unsigned a_row[MT], b_row[NT / 2];
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi) a_row[mi] = (unsigned)((wm * WTM + mi * 16 + (lane & 15)) * ROWB);
    #pragma unroll
    for (int p = 0; p < NT / 2; ++p) b_row[p] = (unsigned)((wn * WTN + p * 16 + (lane & 7) + (lane >> 4) * 8) * ROWB);
    unsigned a_col[KK], b_col[KK];
    #pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
        a_col[kk] = (unsigned)(mm_swz<CPR>(lane & 7, 2 * kk + (lane >> 4)) * 16);
        b_col[kk] = (unsigned)(mm_swz<CPR>(lane & 7, 2 * kk + ((lane >> 3) & 1)) * 16);
    }

    float accf[ACC == 1 ? 1 : MT][ACC == 1 ? 1 : NT][4];
    unsigned acch[ACC == 0 ? 1 : MT][ACC == 0 ? 1 : NT][2];
    if constexpr (ACC != 1) {
        #pragma unroll
        for (int mi = 0; mi < MT; ++mi)
            #pragma unroll
            for (int ni = 0; ni < NT; ++ni) accf[mi][ni][0] = accf[mi][ni][1] = accf[mi][ni][2] = accf[mi][ni][3] = 0.0f;
    }
    if constexpr (ACC != 0) {
        #pragma unroll
        for (int mi = 0; mi < MT; ++mi)
            #pragma unroll
            for (int ni = 0; ni < NT; ++ni) acch[mi][ni][0] = acch[mi][ni][1] = 0u;
    }

    unsigned af[2][MT][4], bf[2][NT][2];
    auto frags = [&](int buf, unsigned abase, unsigned bbase, int kk) {
        #pragma unroll
        for (int mi = 0; mi < MT; ++mi)
            mm_ldsm4(af[buf][mi][0], af[buf][mi][1], af[buf][mi][2], af[buf][mi][3], abase + a_row[mi] + a_col[kk]);
        #pragma unroll
        for (int p = 0; p < NT / 2; ++p)
            mm_ldsm4(bf[buf][2 * p][0], bf[buf][2 * p][1], bf[buf][2 * p + 1][0], bf[buf][2 * p + 1][1],
                     bbase + b_row[p] + b_col[kk]);
    };

    // prologue, with ST - 1 tiles in flight, the first one landed and its first fragments in registers
    #pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < KT) load(s, s);
        mm_commit();
    }
    mm_wait<ST - 2>();
    __syncthreads();
    int rd = 0;            // the stage being read
    int wr = ST - 1;       // the stage the next load fills
    frags(0, sA, sB, 0);

    for (int kt = 0; kt < KT; ++kt) {
        #pragma unroll
        for (int kk = 0; kk < KK; ++kk) {
            const int cur = kk & 1;
            if (kk == KK - 1) {
                // the next tile has landed for everyone, and everyone is past the stage the load below refills
                mm_wait<ST - 2>();
                __syncthreads();
                rd = rd + 1 == ST ? 0 : rd + 1;
                if (kt + 1 < KT) frags(cur ^ 1, sA + rd * A_STAGE, sB + rd * B_STAGE, 0);
            } else {
                frags(cur ^ 1, sA + rd * A_STAGE, sB + rd * B_STAGE, kk + 1);
            }
            if (kk == 0) {
                // refill the stage read in the previous tile (free since the barrier that ended it)
                const int nk = kt + ST - 1;
                if (nk < KT) load(wr, nk);
                mm_commit();   // an empty group when nothing was loaded keeps the count right
                wr = wr + 1 == ST ? 0 : wr + 1;
            }
            #pragma unroll
            for (int mi = 0; mi < MT; ++mi)
                #pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    if constexpr (ACC == 0) mm_mma_f(accf[mi][ni], af[cur][mi], bf[cur][ni][0], bf[cur][ni][1]);
                    else mm_mma_h(acch[mi][ni], af[cur][mi], bf[cur][ni][0], bf[cur][ni][1]);
                }
        }
        if constexpr (ACC == 2) {
            // this k tile's fp16 sums into the fp32 ones
            #pragma unroll
            for (int mi = 0; mi < MT; ++mi)
                #pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const float2 lo = mm_h2f(acch[mi][ni][0]), hi = mm_h2f(acch[mi][ni][1]);
                    accf[mi][ni][0] += lo.x; accf[mi][ni][1] += lo.y;
                    accf[mi][ni][2] += hi.x; accf[mi][ni][3] += hi.y;
                    acch[mi][ni][0] = acch[mi][ni][1] = 0u;
                }
        }
    }
    mm_wait<0>();

    // epilogue, where a thread holds rows g and g+8 of each 16 row tile and columns 2c and 2c+1 of each 8 column tile
    const int g = lane >> 2, c = lane & 3;
    float2 bv[NT];
    #pragma unroll
    for (int ni = 0; ni < NT; ++ni) {
        const int col = n0 + wn * WTN + ni * 8 + 2 * c;
        bv[ni] = bias ? *reinterpret_cast<const float2*>(bias + col) : make_float2(0.0f, 0.0f);
    }
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi)
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int row = m0 + wm * WTM + mi * 16 + g + 8 * hh;
            if (row >= M) continue;
            #pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int col = n0 + wn * WTN + ni * 8 + 2 * c;
                float v0, v1;
                if constexpr (ACC == 1) {
                    const float2 f = mm_h2f(acch[mi][ni][hh]);
                    v0 = f.x; v1 = f.y;
                } else {
                    v0 = accf[mi][ni][2 * hh]; v1 = accf[mi][ni][2 * hh + 1];
                }
                v0 += bv[ni].x;
                v1 += bv[ni].y;
                const size_t o = (size_t)row * N + col;
                if (epi == 0) {
                    *reinterpret_cast<__half2*>(Yh + o) = __floats2half2_rn(v0, v1);
                } else if (epi == 1) {
                    *reinterpret_cast<__half2*>(Yh + o) = __floats2half2_rn(mm_gelu(v0), mm_gelu(v1));
                } else if (epi == 2) {
                    float2* p = reinterpret_cast<float2*>(Yf + o);
                    float2 x = *p;
                    x.x += v0;
                    x.y += v1;
                    *p = x;
                } else {
                    *reinterpret_cast<float2*>(Yf + o) = make_float2(v0, v1);
                }
            }
        }
}

// One extern "C" kernel per configuration, mm<ACC>_<BM>x<BN>x<BK>_<WM>x<WN>_s<ST>_b<MINB>, MINB the blocks per SM
// the registers are capped for. Keep in step with mmgemm.cpp.
#define MM_INST(ACC, BM, BN, BK, WM, WN, ST, MINB)                                                            \
    extern "C" __global__ void __launch_bounds__(WM * WN * 32, MINB)                                         \
        mm##ACC##_##BM##x##BN##x##BK##_##WM##x##WN##_s##ST##_b##MINB(                                        \
            const __half* __restrict__ X, const __half* __restrict__ W, const float* __restrict__ bias,     \
            __half* __restrict__ Yh, float* __restrict__ Yf, int M, int N, int K, int epi)                  \
    {                                                                                                        \
        mm_body<BM, BN, BK, WM, WN, ST, ACC>(X, W, bias, Yh, Yf, M, N, K, epi);                              \
    }

// gemm_bench defines MM_BENCH_ACC (0, 1 or 2) to compile every candidate of one accumulation. NVRTC takes a few
// seconds a kernel, so the backend compiles only the configurations mmgemm.cpp picks (the list below the #else).
#ifdef MM_BENCH_ACC
#define MM_SET(ACC)                        \
    MM_INST(ACC, 128, 128, 32, 2, 2, 3, 2) \
    MM_INST(ACC, 128, 128, 32, 2, 2, 4, 1) \
    MM_INST(ACC, 128, 128, 32, 2, 4, 3, 2) \
    MM_INST(ACC, 128, 128, 32, 4, 2, 3, 2) \
    MM_INST(ACC, 128, 128, 32, 2, 4, 4, 1) \
    MM_INST(ACC, 128, 128, 32, 4, 2, 4, 1) \
    MM_INST(ACC, 128, 128, 64, 2, 2, 3, 1) \
    MM_INST(ACC, 128, 128, 64, 4, 2, 3, 1) \
    MM_INST(ACC, 128, 256, 32, 2, 4, 3, 1) \
    MM_INST(ACC, 128, 256, 32, 2, 4, 4, 1) \
    MM_INST(ACC, 256, 128, 32, 4, 2, 3, 1) \
    MM_INST(ACC, 256, 128, 32, 4, 2, 4, 1) \
    MM_INST(ACC, 128, 64, 32, 2, 2, 4, 2)  \
    MM_INST(ACC, 64, 128, 32, 2, 2, 4, 2)  \
    MM_INST(ACC, 128, 64, 64, 2, 2, 3, 1)  \
    MM_INST(ACC, 64, 128, 64, 2, 2, 3, 1)
MM_SET(MM_BENCH_ACC)
#else
MM_INST(1, 128, 128, 32, 2, 2, 3, 2)
MM_INST(1, 128, 128, 32, 2, 4, 3, 2)
MM_INST(1, 128, 128, 32, 4, 2, 3, 2)
MM_INST(2, 128, 128, 64, 4, 2, 3, 1)
#endif
