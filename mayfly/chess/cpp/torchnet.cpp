// See torchnet.h. Compiled with the LibTorch flags in build_rig.bat (torch 2.10).
// The backend is net_backend2 in this folder, with the policy cut on the card and one asynchronous read back.
#include "torchnet.h"

#include <stdexcept>

#include "net_backend2.h"

namespace mayfly2 {

class TorchNet : public Net {
public:
    TorchNet(const std::string& ts, const std::string& device, bool graph) : be_(ts, device, graph) {}

    // netbackend.h's NetBatch is search2.h's Batch field for field. The backend hands back the logits already
    // cut to each row's legal moves, which is what the tree reads.
    void evaluate(const Batch& in, Out& out) override
    {
        mayfly::NetBatch nb;
        nb.B = in.B;
        nb.wide = in.wide;
        nb.ids = in.ids;
        nb.cast = in.cast;
        nb.hm = in.hm;
        nb.ep = in.ep;
        nb.I = in.I;
        nb.M = in.M;
        nb.widths = in.widths;
        mayfly::NetResult r;
        be_.evaluate(nb, r);
        const size_t B = (size_t)in.B, W = (size_t)in.wide;
        if (r.policy.size() != B * W || r.val.size() != B || r.qv.size() != B * W)
            throw std::runtime_error("TorchNetBackend2 returned the wrong sizes");
        out.pol.swap(r.policy);          // already cut to the legal moves on the card
        out.val.swap(r.val);
        out.qv.swap(r.qv);
    }

private:
    mayfly::TorchNetBackend2 be_;
};

Net* make_torch_net(const std::string& ts_path, const std::string& device, bool graph)
{
    return new TorchNet(ts_path, device, graph);
}

}  // namespace mayfly2
