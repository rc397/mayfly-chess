// The MayFly network behind netbackend.h's NetBackend, on LibTorch.
//
// The network is a TorchScript trace of ChessNet.forward returning (policy, qvalue), made by export_ts.py.
// On the card it is the "cuda" trace (bf16 copies of the matmul weights, traced under bf16 autocast, the same
// kernels mcts._replay_autocast_fast runs). On the CPU it is the fp32 "cpu" trace. The batch is never padded.
// On the card each batch size gets its own CUDA graph, captured the first time that size comes, all in one
// memory pool, as _replay_autocast does. After the forward, evaluate() runs _dispatch's value_from_q
// post-processing as torch ops on the same shapes the Python uses.
#pragma once
#include <map>
#include <memory>
#include <string>

#include <torch/script.h>

#include "netbackend.h"
// evaluate() hands back the policy cut to each row's legal moves (B*wide, gathered on the card) and reads
// everything back with one asynchronous copy into pinned memory. Reading back the full (B, 4162) policy
// blocked behind other processes' work on a shared card.

namespace mayfly {

class TorchNetBackend2 : public NetBackend {
public:
    // ts_path is the .ts file from export_ts.py, device is "cuda" or "cpu", and use_graph turns on CUDA graphs
    // on the card (only in a build with NET_GRAPH, which needs the CUDA runtime headers, else the eager trace).
    TorchNetBackend2(const std::string& ts_path, const std::string& device = "cuda", bool use_graph = true);
    ~TorchNetBackend2() override;
    void evaluate(const NetBatch& in, NetResult& out) override;
    // The forward alone (graph replay or the eager trace) on the inputs of the last evaluate(), timed
    // over `iters` passes with the card synchronised at the end, in milliseconds a pass.
    double bench_forward(int iters);

private:
    struct Graph;
    std::pair<at::Tensor, at::Tensor> forward(const at::Tensor& ids, const at::Tensor& cast,
                                              const at::Tensor& hm, const at::Tensor& ep);
    torch::jit::Module mod_;
    torch::Device dev_;
    bool graph_;
    std::map<int, std::unique_ptr<Graph>> graphs_;
    bool have_pool_ = false;
    std::pair<unsigned long long, unsigned long long> pool_{0, 0};
    at::Tensor last_ids_, last_cast_, last_hm_, last_ep_;
};

}  // namespace mayfly
