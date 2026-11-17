// The hand-tuned CUDA network (cudanet.h) behind the search's two network interfaces, mayfly2::Net (cpp/search2.h,
// the persistent tree) and mayfly::NetBackend (cpp/netbackend.h).
// No CUDA header here, so pyext.cpp can include it as it includes torchnet.h.
#pragma once
#include <string>

#include "../cpp/netbackend.h"
#include "../cpp/search2.h"

namespace mayfly2 {
// CudaNet on `device` from a .cnw weight file (export_cudanet.py), with the policy cut to each row's legal moves as
// search2 wants it (Out.pol is B*wide). Throws std::runtime_error when the file or a CUDA library cannot be loaded.
Net* make_cuda_net(const std::string& weights_path, int device);

// The .cnw file's fingerprint (mcts_cpp.fingerprint of the weights it was made from), for a caller holding a Net made
// above. Empty for any other Net.
std::string cuda_net_fingerprint(Net* net);
}  // namespace mayfly2

namespace mayfly {
// CudaNet behind NetBackend. NetResult.policy is the full B*4162, but CudaNet hands back only the logits at each
// row's legal moves, so this adapter scatters those into the full rows (policy[b*4162 + I[b*wide+j]] for the first
// widths[b] j) and leaves every other entry 0. That is all the search reads (it gathers at I, as _collect does), but
// it is not the torch backend's full row. A true full policy needs CudaNet to copy back the first kMoves columns of
// its logits buffer (cudanet_spec.h "logits"), one more device to host copy of B*4162 floats per batch.
NetBackend* make_cuda_backend(const std::string& weights_path, int device);
}  // namespace mayfly
