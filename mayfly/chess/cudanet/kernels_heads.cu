// The end of MayFly's forward pass, CUDA C for NVRTC (see cudanet_spec.h).
//   cls_layernorm  the final layer norm of each CLS row (x row b*65) into cls f16 [B, 1024]
//   heads_post     _collect's arithmetic on the two heads' raw logits (cpp/netbackend.h)
// Launch configs are in kernels_heads_launch.h. Both kernels take any block size that is a multiple of 32 and at
// most 1024, and one block per batch row. Helpers are static and prefixed hd_ so this file can share a program with
// kernels_trunk.cu.
#include <cuda_fp16.h>

#define HD_D 1024          // width (kD)
#define HD_TOKENS 65       // tokens per position (kTokens)
#define HD_MOVES 4162      // policy rows (kMoves)
#define HD_OUT 8324        // logits per row: policy, then qvalue (2 * kMoves)

static __device__ __forceinline__ float hd_neg_inf() { return __int_as_float((int)0xff800000); }

static __device__ __forceinline__ float hd_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

static __device__ __forceinline__ float hd_warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Sum over the block, returned to every thread. sh holds 33 floats. The first barrier lets a previous call's
// readers finish before sh is written again.
static __device__ float hd_block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = hd_warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = lane < nw ? sh[lane] : 0.0f;
        t = hd_warp_sum(t);
        if (lane == 0) sh[32] = t;
    }
    __syncthreads();
    return sh[32];
}

// Max over the block, returned to every thread, by the same scheme as hd_block_sum.
static __device__ float hd_block_max(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = hd_warp_max(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = lane < nw ? sh[lane] : hd_neg_inf();
        t = hd_warp_max(t);
        if (lane == 0) sh[32] = t;
    }
    __syncthreads();
    return sh[32];
}

// torch's float sigmoid, 1 / (1 + exp(-x)), with accurate expf and IEEE division.
static __device__ __forceinline__ float hd_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// The final layer norm on the CLS rows. x f32 [B*65, 1024], g and bt f32 [1024], cls f16 [B, 1024].
// fp32 mean, then the biased variance from a second pass over (x - mean), rstd = rsqrtf(var + eps) as torch does.
extern "C" __global__ void cls_layernorm(const float* __restrict__ x, const float* __restrict__ g,
                                         const float* __restrict__ bt, __half* __restrict__ cls, int B, float eps) {
    __shared__ float sh[33];
    const int b = blockIdx.x;
    if (b >= B) return;  // the whole block leaves together, so the barriers below stay safe
    const float* row = x + (long long)b * HD_TOKENS * HD_D;

    float s = 0.0f;
    for (int i = threadIdx.x; i < HD_D; i += blockDim.x) s += row[i];
    const float mean = hd_block_sum(s, sh) / (float)HD_D;

    float v = 0.0f;
    for (int i = threadIdx.x; i < HD_D; i += blockDim.x) {
        const float d = row[i] - mean;
        v += d * d;
    }
    const float var = hd_block_sum(v, sh) / (float)HD_D;
    const float rstd = rsqrtf(var + eps);

    __half* out = cls + (long long)b * HD_D;
    for (int i = threadIdx.x; i < HD_D; i += blockDim.x)
        out[i] = __float2half_rn((row[i] - mean) * rstd * g[i] + bt[i]);
}

// _collect for one batch row per block. logits f32 [B, 8324] holds policy logits in columns 0..4161 and qvalue
// logits in 4162..8323. I int64 [B, wide] holds policy indices, M uint8 [B, wide] (nonzero = legal).
//   p_k = policy[I_k], q_k = sigmoid(qvalue[I_k])
//   w   = softmax of p over the legal slots (max subtracted, w_k = exp(p_k - max) / sum, as torch's softmax)
//   val = sum_k w_k * q_k * 2 - 1
//   pol = p_k, qv = q_k * 2 - 1 on legal slots, 0 on padding
// A row with no legal slot gets val = -1 (torch would give NaN, but the search never asks for one).
extern "C" __global__ void heads_post(const float* __restrict__ logits, const long long* __restrict__ I,
                                      const unsigned char* __restrict__ M, int B, int wide, float* __restrict__ pol,
                                      float* __restrict__ val, float* __restrict__ qv) {
    __shared__ float sh[33];
    const int b = blockIdx.x;
    if (b >= B) return;  // the whole block leaves together
    const float* lp = logits + (long long)b * HD_OUT;
    const float* lq = lp + HD_MOVES;
    const long long base = (long long)b * wide;
    const long long* Ib = I + base;
    const unsigned char* Mb = M + base;
    float* pb = pol + base;
    float* qb = qv + base;

    // pass 1: the per-move outputs and the largest legal logit
    float mx = hd_neg_inf();
    for (int k = threadIdx.x; k < wide; k += blockDim.x) {
        if (Mb[k]) {
            const long long j = Ib[k];
            const float p = lp[j];
            pb[k] = p;
            qb[k] = hd_sigmoid(lq[j]) * 2.0f - 1.0f;
            mx = fmaxf(mx, p);
        } else {
            pb[k] = 0.0f;
            qb[k] = 0.0f;
        }
    }
    mx = hd_block_max(mx, sh);

    // pass 2: the softmax denominator
    float s = 0.0f;
    for (int k = threadIdx.x; k < wide; k += blockDim.x)
        if (Mb[k]) s += expf(lp[Ib[k]] - mx);
    s = hd_block_sum(s, sh);

    // pass 3: sum of w * q (no legal slot means no term, so s is never 0 here). The product is rounded on its own
    // (no fused multiply add) like torch's w * qg before the sum.
    float acc = 0.0f;
    for (int k = threadIdx.x; k < wide; k += blockDim.x) {
        if (Mb[k]) {
            const long long j = Ib[k];
            const float w = expf(lp[j] - mx) / s;
            acc += __fmul_rn(w, hd_sigmoid(lq[j]));
        }
    }
    acc = hd_block_sum(acc, sh);
    if (threadIdx.x == 0) val[b] = acc * 2.0f - 1.0f;
}
