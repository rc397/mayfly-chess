// The weight file of cudanet_spec.h on the device.
// Parses the header, checks it against the spec (magic, numbers, every tensor with its dtype and shape, offsets,
// 256-byte alignment, sizes), reads the data and uploads it in one device allocation. Each tensor keeps its file
// offset from the data start, so every device pointer is 256-aligned like the file.
#pragma once
#include <cstddef>
#include <string>
#include <unordered_map>

namespace cudanet {

class Weights {
public:
    Weights(const std::string& path, int device);
    ~Weights();
    Weights(const Weights&) = delete;
    Weights& operator=(const Weights&) = delete;

    // device pointer of a tensor by its spec name ("piece", "L3.qkv_w", "heads_b", ...), throws for an unknown name
    const void* ptr(const std::string& name) const;

    int d = 0, layers = 0, heads = 0, ff = 0, tokens = 0, moves = 0;
    std::string fingerprint, source;
    int device = 0;
    size_t bytes = 0;        // size of the device allocation

private:
    void* base_ = nullptr;
    std::unordered_map<std::string, size_t> offset_;
};

}  // namespace cudanet
