// The one interface between the C++ search and the network.
//
// It mirrors the Python split. The search builds what MCTS._prepare builds (mcts.py), and the backend does
// everything _dispatch and _collect do on the card (the forward pass and the value-from-q post-processing)
// and hands back what _collect returns. The post-processing stays in the backend for bit-identity. val and
// qg are torch float32 ops (sigmoid, gather, masked softmax, sum) whose rounding depends on the kernels and
// the tensor shapes, so they run as torch ops on the shapes the Python uses, a (B, 4162) policy and q, and
// I and M padded to the batch's widest row (`wide`).
#pragma once
#include <cstdint>
#include <vector>

namespace mayfly {

constexpr int kMoves = 4162;   // policy rows: from*64+to (4096) + 66 underpromotions (chess/moves.py)

// One batch of unexpanded nodes, in the order the gather queued them (B <= 48, the root alone is B = 1).
struct NetBatch {
    int B = 0;
    int wide = 0;                    // max(widths), the width of I and M as _prepare pads them
    std::vector<int64_t> ids;        // B*64   board_to_ids (the side to move's frame), = mayfly_movegen.encode
    std::vector<float> cast;         // B*4    castling rights, own K-side, own Q-side, their K-side, their Q-side
    std::vector<int64_t> hm;         // B      min(halfmove clock, 100)
    std::vector<int64_t> ep;         // B      0, or the (mirrored) legal en passant square + 1
    std::vector<int64_t> I;          // B*wide each row's legal moves as policy indices, in move order, 0 padded
    std::vector<uint8_t> M;          // B*wide 1 on a legal move, 0 on padding
    std::vector<int> widths;         // B      legal moves per row
};

// What _collect returns, as float32.
//   policy = out["policy"].float()                                   (B, 4162) raw logits, no temperature
//   q      = sigmoid(out["qvalue"].float())                          (B, 4162)
//   w      = softmax(gather(policy, 1, I).masked_fill(~M, -inf), -1) (B, wide)
//   qg     = gather(q, 1, I)                                         (B, wide)
//   val    = (w * qg).sum(1) * 2 - 1                                 (B,)
//   qv     = qg * 2 - 1                                              (B, wide), only the first widths[b] read
struct NetResult {
    std::vector<float> policy;       // B*4162
    std::vector<float> val;          // B
    std::vector<float> qv;           // B*wide
};

class NetBackend {
public:
    virtual ~NetBackend() = default;
    // Fill `out` for `in`. Called once per batch, synchronously. Throw on failure.
    virtual void evaluate(const NetBatch& in, NetResult& out) = 0;
};

}  // namespace mayfly
