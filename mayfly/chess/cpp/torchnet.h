// The network in C++ for the C++ tree. No torch header here, so pyext.cpp compiles without LibTorch.
// torchnet.cpp holds the LibTorch side.
#pragma once
#include <string>

#include "search2.h"

namespace mayfly2 {
// TorchNetBackend2 (net_backend2.cpp, the TorchScript trace from export_ts.py with a CUDA graph per batch size and
// _dispatch's value_from_q post-processing as torch ops) behind Net, with the policy cut to each row's legal moves.
// Throws std::runtime_error when the trace cannot be loaded. Returns nullptr in a build without LibTorch.
Net* make_torch_net(const std::string& ts_path, const std::string& device, bool graph);
}  // namespace mayfly2
