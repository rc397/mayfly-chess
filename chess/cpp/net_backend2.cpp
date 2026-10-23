// See net_backend2.h.
#include "net_backend2.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <torch/csrc/jit/runtime/graph_executor.h>

#include <torch/cuda.h>

#ifdef NET_GRAPH
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace mayfly {

struct TorchNetBackend2::Graph {
    at::Tensor ids, cast, hm, ep;   // static inputs, written before every replay
    at::Tensor pol, q;              // static outputs, rewritten by every replay
#ifdef NET_GRAPH
    std::unique_ptr<at::cuda::CUDAGraph> g;
#endif
};

TorchNetBackend2::TorchNetBackend2(const std::string& ts_path, const std::string& device, bool use_graph)
    : dev_(device), graph_(use_graph && device != "cpu") {
    // The trace runs as recorded, with no profiling executor, no fuser and no graph optimisation (a fused
    // kernel can round differently from the ones the Python runs).
    torch::jit::getProfilingMode() = false;
    torch::jit::getExecutorMode() = false;
    torch::jit::setGraphExecutorOptimize(false);
#ifndef NET_GRAPH
    graph_ = false;   // built without the CUDA runtime headers, so the eager trace on either device
#endif
    // No map location. A trace keeps the shape arithmetic of MultiheadAttention (head_dim = embed_dim //
    // num_heads, then int(head_dim)) on 0-dim CPU constants. Mapping the whole file to the card moves those
    // to the card too, so every int() is a device sync (and a CUDA graph capture fails on the first one).
    // Module::to moves the weights and leaves the constants where the trace put them.
    mod_ = torch::jit::load(ts_path);
    mod_.to(dev_);
    mod_.eval();
}

TorchNetBackend2::~TorchNetBackend2() {
    graphs_.clear();
}

std::pair<at::Tensor, at::Tensor> TorchNetBackend2::forward(const at::Tensor& ids, const at::Tensor& cast,
                                                           const at::Tensor& hm, const at::Tensor& ep) {
    if (!graph_) {
        auto r = mod_.forward({ids, cast, hm, ep}).toTuple();
        return {r->elements()[0].toTensor(), r->elements()[1].toTensor()};
    }
#ifdef NET_GRAPH
    const int B = static_cast<int>(ids.size(0));
    auto it = graphs_.find(B);
    if (it == graphs_.end()) {
        // As in _replay_autocast(_fast), static copies of the inputs, warm-up passes on a side stream outside
        // the capture (3 for the first graph, 1 after), then the capture into the shared pool.
        auto G = std::make_unique<Graph>();
        G->ids = ids.clone(); G->cast = cast.clone(); G->hm = hm.clone(); G->ep = ep.clone();
        if (!have_pool_) { pool_ = at::cuda::graph_pool_handle(); have_pool_ = true; }
        c10::cuda::device_synchronize();
        {
            c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromPool());
            for (int i = 0; i < (graphs_.empty() ? 3 : 1); ++i)
                mod_.forward({G->ids, G->cast, G->hm, G->ep});
            c10::cuda::device_synchronize();
        }
        c10::cuda::CUDACachingAllocator::emptyCache();
        G->g = std::make_unique<at::cuda::CUDAGraph>();
        try {
            c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromPool());
            G->g->capture_begin(pool_);
            try {
                auto r = mod_.forward({G->ids, G->cast, G->hm, G->ep}).toTuple();
                G->pol = r->elements()[0].toTensor();
                G->q = r->elements()[1].toTensor();
            } catch (...) {
                try { G->g->capture_end(); } catch (...) {}
                throw;
            }
            G->g->capture_end();
        } catch (const std::exception& e) {
            // As in _replay_autocast, a failed capture costs speed, never a move. Eager from now on, same numbers.
            std::fprintf(stderr, "net: CUDA graph capture failed, eager from now on: %.300s\n", e.what());
            graph_ = false;
            G.reset();
            graphs_.clear();
            c10::cuda::device_synchronize();
            auto r = mod_.forward({ids, cast, hm, ep}).toTuple();
            return {r->elements()[0].toTensor(), r->elements()[1].toTensor()};
        }
        c10::cuda::device_synchronize();
        it = graphs_.emplace(B, std::move(G)).first;
    }
    Graph& G = *it->second;
    G.ids.copy_(ids); G.cast.copy_(cast); G.hm.copy_(hm); G.ep.copy_(ep);
    G.g->replay();
    return {G.pol, G.q};
#else
    throw std::runtime_error("unreachable");
#endif
}

static bool net_log()
{
    static int on = -1;
    if (on < 0) {
        const char* e = std::getenv("MAYFLY_NET_LOG");
        on = (e && e[0] == '1') ? 1 : 0;
    }
    return on == 1;
}

void TorchNetBackend2::evaluate(const NetBatch& in, NetResult& out) {
    c10::InferenceMode guard;
    const auto t0 = std::chrono::steady_clock::now();
    const size_t graphs_before = graphs_.size();
    const int64_t B = in.B, W = in.wide;
    if (B <= 0 || W <= 0) throw std::runtime_error("empty batch");
    // One int64 staging buffer (ids, hm, ep, I, M) and one float buffer (cast), pinned when on the card
    // so each is one copy. The values are the ones _dispatch builds with torch.tensor.
    const bool cuda = dev_.is_cuda();
    auto li = at::TensorOptions().dtype(at::kLong).pinned_memory(cuda);
    auto lf = at::TensorOptions().dtype(at::kFloat).pinned_memory(cuda);
    const int64_t nI = B * 64 + 2 * B + 2 * B * W;
    at::Tensor hi = at::empty({nI}, li), hf = at::empty({B * 4}, lf);
    int64_t* p = hi.data_ptr<int64_t>();
    std::memcpy(p, in.ids.data(), sizeof(int64_t) * B * 64); p += B * 64;
    std::memcpy(p, in.hm.data(), sizeof(int64_t) * B); p += B;
    std::memcpy(p, in.ep.data(), sizeof(int64_t) * B); p += B;
    std::memcpy(p, in.I.data(), sizeof(int64_t) * B * W); p += B * W;
    for (int64_t k = 0; k < B * W; ++k) p[k] = in.M[k] ? 1 : 0;
    std::memcpy(hf.data_ptr<float>(), in.cast.data(), sizeof(float) * B * 4);
    at::Tensor di = hi.to(dev_, /*non_blocking=*/cuda), df = hf.to(dev_, cuda);
    at::Tensor ids = di.narrow(0, 0, B * 64).view({B, 64});
    at::Tensor hm = di.narrow(0, B * 64, B);
    at::Tensor ep = di.narrow(0, B * 65, B);
    at::Tensor It = di.narrow(0, B * 66, B * W).view({B, W});
    at::Tensor Mt = di.narrow(0, B * 66 + B * W, B * W).view({B, W}).ne(0);
    at::Tensor cast = df.view({B, 4});
    last_ids_ = ids; last_cast_ = cast; last_hm_ = hm; last_ep_ = ep;

    const auto t1 = std::chrono::steady_clock::now();
    auto [pol, qval] = forward(ids, cast, hm, ep);
    const auto t2 = std::chrono::steady_clock::now();

    // _dispatch and value_from_q (mcts.py), the same torch ops on the same shapes.
    at::Tensor q = at::sigmoid(qval.to(at::kFloat));
    at::Tensor pl = pol.to(at::kFloat);
    at::Tensor w = at::softmax(at::gather(pl, 1, It).masked_fill(Mt.logical_not(), -INFINITY), -1);
    at::Tensor qg = at::gather(q, 1, It);
    at::Tensor val = (w * qg).sum(1) * 2.0 - 1.0;
    at::Tensor qv = qg * 2.0 - 1.0;
    // One asynchronous read back of what the tree reads. The policy at each row's legal moves (gathered on the
    // card, the same floats a CPU slice of the full row gives), the value and the per-move q are copied into
    // pinned memory without blocking and waited on through this stream alone.
    at::Tensor pg = at::gather(pl, 1, It);
    at::Tensor all = at::cat({pg.reshape(-1), val.reshape(-1), qv.reshape(-1)});
    at::Tensor host = at::empty({all.numel()}, at::TensorOptions().dtype(at::kFloat).pinned_memory(cuda));
    host.copy_(all, /*non_blocking=*/cuda);
    if (cuda) c10::cuda::getCurrentCUDAStream().synchronize();
    if (net_log()) {
        const auto t3 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t3 - t0).count();
        if (ms > 150.0)
            std::fprintf(stderr, "net: slow evaluate B=%lld %.0f ms (inputs %.0f, forward launch %.0f%s, rest and wait %.0f)\n",
                         (long long)B, ms, std::chrono::duration<double, std::milli>(t1 - t0).count(),
                         std::chrono::duration<double, std::milli>(t2 - t1).count(),
                         graphs_.size() > graphs_before ? " with a capture" : "",
                         std::chrono::duration<double, std::milli>(t3 - t2).count());
    }
    const float* r = host.data_ptr<float>();
    out.policy.assign(r, r + B * W); r += B * W;
    out.val.assign(r, r + B); r += B;
    out.qv.assign(r, r + B * W);
}

double TorchNetBackend2::bench_forward(int iters) {
    c10::InferenceMode guard;
    if (!last_ids_.defined()) throw std::runtime_error("bench_forward before evaluate");
    forward(last_ids_, last_cast_, last_hm_, last_ep_);
    if (dev_.is_cuda()) torch::cuda::synchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) forward(last_ids_, last_cast_, last_hm_, last_ep_);
    if (dev_.is_cuda()) torch::cuda::synchronize();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
}

}  // namespace mayfly
