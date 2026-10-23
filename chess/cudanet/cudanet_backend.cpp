// See cudanet_backend.h. CudaNet (cudanet.h) behind mayfly2::Net and mayfly::NetBackend.
//
// cpp/build_rig.bat compiles this file with the other cudanet sources (every chess/cudanet/*.cpp except
// cudanet_capi.cpp, the dll's C ABI) and pyext.cpp with /DMAYFLY_CUDANET. No CUDA import library is linked, because
// cuda_dyn loads every CUDA library at runtime. kernels_trunk.cu and kernels_heads.cu sit beside the .pyd, since
// make_cuda_net passes an empty kernel_dir, which means the directory of the module that holds CudaNet.
//
// pyext.cpp makes a tree on the CUDA net when the `ts` path ends in ".cnw", so the Python call keeps its shape.
// mcts_cpp.py passes a .cnw only when the sealed setting cuda_net asks for it and the file's fingerprint (read from
// its JSON header without CUDA) matches the tree's weights. CudaNet does not give the LibTorch trace's numbers (FP16
// products with FP32 accumulation, its own kernels), so switching it on is a search change, not a speed fix.
//
// One process can hold two CUDA runtimes, torch's own cudart and the one cuda_dyn loads. Both sit on the driver's
// primary context of the device, so this works, but CudaNet uses its own stream and never torch's, and torch's
// caching allocator does not see CudaNet's memory (about 0.5 GB of FP16 weights plus buffers for max_batch()
// positions).
#include "cudanet_backend.h"

#include <cstring>
#include <deque>
#include <stdexcept>

#include "cudanet.h"
#include "cudanet_spec.h"

namespace {

// the batch's shape checks, shared by both adapters
void check_batch(int max_b, int B, int wide, size_t ids, size_t cast, size_t hm, size_t ep, size_t I, size_t M)
{
    if (B < 1 || B > max_b) throw std::runtime_error("cudanet: batch size out of range");
    if (wide < 1) throw std::runtime_error("cudanet: wide must be at least 1");
    const size_t b = (size_t)B, w = (size_t)wide;
    if (ids != b * 64 || cast != b * 4 || hm != b || ep != b || I != b * w || M != b * w)
        throw std::runtime_error("cudanet: batch arrays have the wrong sizes");
}

}  // namespace

namespace mayfly2 {

class CudaNetAdapter : public Net {
public:
    CudaNetAdapter(const std::string& path, int device) : net_(path, device) {}

    // CudaNet already cuts the policy to each row's legal moves (0 on padding), which is exactly search2's Out
    void evaluate(const Batch& in, Out& out) override
    {
        check_batch(net_.max_batch(), in.B, in.wide, in.ids.size(), in.cast.size(), in.hm.size(), in.ep.size(),
                    in.I.size(), in.M.size());
        const size_t B = (size_t)in.B, W = (size_t)in.wide;
        out.pol.resize(B * W);
        out.val.resize(B);
        out.qv.resize(B * W);
        net_.evaluate(in.B, in.wide, in.ids.data(), in.cast.data(), in.hm.data(), in.ep.data(), in.I.data(),
                      in.M.data(), out.pol.data(), out.val.data(), out.qv.data());
    }

    // the pipeline's halves. begin copies the inputs into pinned staging, so `in` may go once submit returns.
    void submit(const Batch& in) override
    {
        check_batch(net_.max_batch(), in.B, in.wide, in.ids.size(), in.cast.size(), in.hm.size(), in.ep.size(), in.I.size(),
                    in.M.size());
        net_.begin(in.B, in.wide, in.ids.data(), in.cast.data(), in.hm.data(), in.ep.data(), in.I.data(),
                   in.M.data());
        fly_.push_back(std::make_pair((size_t)in.B, (size_t)in.wide));
    }

    // the oldest batch submitted
    void collect(Out& out) override
    {
        if (fly_.empty()) throw std::runtime_error("cudanet: collect with no batch on the card");
        const size_t B = fly_.front().first, W = fly_.front().second;
        fly_.pop_front();
        out.pol.resize(B * W);
        out.val.resize(B);
        out.qv.resize(B * W);
        net_.end(out.pol.data(), out.val.data(), out.qv.data());
    }

    void timers_on(bool on) override { net_.eval_times_on(on); }

    std::vector<std::pair<std::string, double>> timers() override
    {
        const std::vector<double> v = net_.eval_times();
        const std::string names = cudanet::CudaNet::eval_times_names();
        std::vector<std::pair<std::string, double>> out;
        size_t at = 0;
        for (double x : v) {
            const size_t comma = names.find(',', at);
            out.emplace_back(names.substr(at, comma == std::string::npos ? std::string::npos : comma - at), x);
            at = comma == std::string::npos ? names.size() : comma + 1;
        }
        return out;
    }

    const std::string& fingerprint() const { return net_.fingerprint(); }

private:
    cudanet::CudaNet net_;
    std::deque<std::pair<size_t, size_t>> fly_;   // (B, wide) of each batch submitted and not yet collected
};

Net* make_cuda_net(const std::string& weights_path, int device)
{
    return new CudaNetAdapter(weights_path, device);
}

std::string cuda_net_fingerprint(Net* net)
{
    CudaNetAdapter* a = dynamic_cast<CudaNetAdapter*>(net);
    return a ? a->fingerprint() : std::string();
}

}  // namespace mayfly2

namespace mayfly {

class CudaBackend : public NetBackend {
public:
    CudaBackend(const std::string& path, int device) : net_(path, device) {}

    // the cut comes back from the card and the full rows are rebuilt from it (legal entries only, see cudanet_backend.h)
    void evaluate(const NetBatch& in, NetResult& out) override
    {
        check_batch(net_.max_batch(), in.B, in.wide, in.ids.size(), in.cast.size(), in.hm.size(), in.ep.size(),
                    in.I.size(), in.M.size());
        if (in.widths.size() != (size_t)in.B) throw std::runtime_error("cudanet: widths has the wrong size");
        const size_t B = (size_t)in.B, W = (size_t)in.wide;
        cut_.resize(B * W);
        out.val.resize(B);
        out.qv.resize(B * W);
        net_.evaluate(in.B, in.wide, in.ids.data(), in.cast.data(), in.hm.data(), in.ep.data(), in.I.data(),
                      in.M.data(), cut_.data(), out.val.data(), out.qv.data());
        out.policy.assign(B * (size_t)kMoves, 0.0f);
        for (size_t b = 0; b < B; b++) {
            const size_t n = (size_t)in.widths[b];
            if (n > W) throw std::runtime_error("cudanet: a row is wider than wide");
            float* row = out.policy.data() + b * (size_t)kMoves;
            for (size_t j = 0; j < n; j++) {
                const int64_t m = in.I[b * W + j];
                if (m < 0 || m >= kMoves) throw std::runtime_error("cudanet: a move index is out of range");
                row[m] = cut_[b * W + j];
            }
        }
    }

private:
    cudanet::CudaNet net_;
    std::vector<float> cut_;
};

NetBackend* make_cuda_backend(const std::string& weights_path, int device)
{
    return new CudaBackend(weights_path, device);
}

}  // namespace mayfly
