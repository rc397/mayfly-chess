// The hand-tuned CUDA forward pass of MayFly's network (see cudanet_spec.h).
// One CudaNet owns the weights on one device, every buffer for max_batch(), one stream and a CUDA graph per batch
// size (captured on its first use). evaluate takes host pointers and is synchronous. Calls are serialised.
// No CUDA header is needed to use this class.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudanet {

constexpr int kMaxWide = 256;     // widest I / M row accepted (chess has at most 218 legal moves)

class CudaNet {
public:
    // weights_path is the MFCNW001 file (export_cudanet.py). kernel_dir is where kernels_trunk.cu and kernels_heads.cu
    // live. Empty means env MAYFLY_CUDANET_SRC, else the directory of the module holding this code.
    // The largest batch is kMaxBatch, or env MAYFLY_CUDANET_MAX_BATCH (1..kMaxBatchLimit), read here.
    CudaNet(const std::string& weights_path, int device, const std::string& kernel_dir = "");
    ~CudaNet();
    CudaNet(const CudaNet&) = delete;
    CudaNet& operator=(const CudaNet&) = delete;

    // One batch, B in 1..max_batch(), wide in 1..kMaxWide. Inputs are ids [B*64], cast [B*4], hm [B], ep [B],
    // I [B*wide], M [B*wide]. Outputs are pol [B*wide] (policy logits at the legal moves), val [B], qv [B*wide].
    // Throws std::runtime_error on failure.
    void evaluate(int B, int wide, const int64_t* ids, const float* cast, const int64_t* hm, const int64_t* ep,
                  const int64_t* I, const uint8_t* M, float* pol, float* val, float* qv);

    // evaluate in two parts, for a caller that works while the card runs (the search's pipeline). begin copies
    // the inputs and queues the whole pass, and end waits for the oldest batch begun and copies its outputs out. The
    // same work as evaluate, so the same numbers. At most two batches are on the card. The next may be begun before
    // the last is ended, so the card never waits for the caller between them.
    void begin(int B, int wide, const int64_t* ids, const float* cast, const int64_t* hm, const int64_t* ep,
               const int64_t* I, const uint8_t* M);
    void end(float* pol, float* val, float* qv);

    // Per-batch timers of evaluate, begin and end (off by default, since they add events to the stream), summed
    // since they were switched on, in the order of eval_times_names() (host milliseconds, card milliseconds, counts).
    void eval_times_on(bool on);
    std::vector<double> eval_times();
    static const char* eval_times_names();

    const std::string& fingerprint() const;

    // the largest B evaluate and profile take (the buffers' size)
    int max_batch() const;

    // Milliseconds a pass by step at batch B, the mean of reps passes after one warm-up. Eager kernels with an event
    // after each step, then last the whole pass as the search runs it (the CUDA graph). The input is an empty board.
    // The steps are named by profile_names(), comma separated, in the same order.
    std::vector<double> profile(int B, int reps);
    static const char* profile_names();

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace cudanet
