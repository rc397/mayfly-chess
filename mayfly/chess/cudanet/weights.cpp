// Reads the weight file of cudanet_spec.h (written by export_cudanet.py) onto the device.
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuda_dyn.h"
#include "cudanet_spec.h"

namespace cudanet {
namespace {

constexpr uint64_t kAlign = 256;
constexpr char kMagic[] = "MFCNW001";
constexpr uint64_t kChunk = uint64_t(64) << 20;     // upload in 64 MB pieces, not one big host copy

[[noreturn]] void fail(const std::string& path, const std::string& why) {
    throw std::runtime_error("cudanet weights " + path + ": " + why);
}

uint64_t align_up(uint64_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

// ---------------------------------------------------------------------------------------------------- the JSON header
// Just enough JSON for the header's known shape, objects, arrays, strings, numbers (and true/false/null for safety).
struct Json {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json* get(const std::string& key) const {
        for (const auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}

    Json parse() {
        Json v = value(0);
        ws();
        if (i_ != s_.size()) err("trailing characters");
        return v;
    }

private:
    const std::string& s_;
    size_t i_ = 0;

    [[noreturn]] void err(const char* why) const {
        throw std::runtime_error(std::string("bad JSON header (") + why + ") at byte " + std::to_string(i_));
    }
    void ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
    }
    char peek() {
        ws();
        if (i_ >= s_.size()) err("unexpected end");
        return s_[i_];
    }
    void expect(char c) {
        if (peek() != c) err("unexpected character");
        ++i_;
    }
    bool word(const char* w) {
        const size_t n = std::strlen(w);
        if (s_.compare(i_, n, w) != 0) return false;
        i_ += n;
        return true;
    }

    Json value(int depth) {
        if (depth > 16) err("nested too deep");
        Json v;
        const char c = peek();
        if (c == '{') {
            v.kind = Json::Obj;
            ++i_;
            if (peek() == '}') { ++i_; return v; }
            for (;;) {
                if (peek() != '"') err("expected a key");
                std::string k = string();
                expect(':');
                v.obj.emplace_back(std::move(k), value(depth + 1));
                const char e = peek();
                ++i_;
                if (e == '}') break;
                if (e != ',') err("expected , or }");
            }
        } else if (c == '[') {
            v.kind = Json::Arr;
            ++i_;
            if (peek() == ']') { ++i_; return v; }
            for (;;) {
                v.arr.push_back(value(depth + 1));
                const char e = peek();
                ++i_;
                if (e == ']') break;
                if (e != ',') err("expected , or ]");
            }
        } else if (c == '"') {
            v.kind = Json::Str;
            v.str = string();
        } else if (word("true")) {
            v.kind = Json::Bool;
            v.b = true;
        } else if (word("false")) {
            v.kind = Json::Bool;
        } else if (word("null")) {
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            size_t j = i_;
            while (j < s_.size() && std::strchr("+-0123456789.eE", s_[j]) && s_[j] != '\0') ++j;
            const std::string t = s_.substr(i_, j - i_);
            char* end = nullptr;
            v.kind = Json::Num;
            v.num = std::strtod(t.c_str(), &end);
            if (end != t.c_str() + t.size()) err("bad number");
            i_ = j;
        } else {
            err("unexpected character");
        }
        return v;
    }

    unsigned hex4() {
        if (i_ + 4 > s_.size()) err("bad \\u escape");
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            const char h = s_[i_++];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= unsigned(h - '0');
            else if (h >= 'a' && h <= 'f') v |= unsigned(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') v |= unsigned(h - 'A' + 10);
            else err("bad \\u escape");
        }
        return v;
    }

    static void utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }

    std::string string() {
        ++i_;   // the opening quote
        std::string out;
        for (;;) {
            if (i_ >= s_.size()) err("unterminated string");
            const char c = s_[i_++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (i_ >= s_.size()) err("unterminated string");
            const char e = s_[i_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00) {      // a surrogate pair (Python's ensure_ascii writes them)
                        if (s_.compare(i_, 2, "\\u") != 0) err("lone surrogate");
                        i_ += 2;
                        const unsigned lo = hex4();
                        if (lo < 0xDC00 || lo >= 0xE000) err("bad surrogate pair");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(out, cp);
                    break;
                }
                default: err("bad escape");
            }
        }
    }
};

// ---------------------------------------------------------------------------------------------------- the spec
struct Want {
    std::string name;
    const char* dtype;
    std::vector<int64_t> shape;
};

// every tensor cudanet_spec.h lists, with its dtype and shape
std::vector<Want> spec_tensors() {
    const int64_t d = kD, ff = kFF, m = kMoves;
    std::vector<Want> w = {
        {"piece", "f32", {13, d}}, {"pos", "f32", {64, d}}, {"cls", "f32", {d}},
        {"cast_w", "f32", {d, 4}}, {"cast_b", "f32", {d}}, {"hm", "f32", {102, d}}, {"ep_mark", "f32", {d}},
    };
    for (int i = 0; i < kLayers; ++i) {
        const std::string p = "L" + std::to_string(i) + ".";
        w.push_back({p + "n1_g", "f32", {d}});
        w.push_back({p + "n1_b", "f32", {d}});
        w.push_back({p + "qkv_w", "f16", {3 * d, d}});
        w.push_back({p + "qkv_b", "f32", {3 * d}});
        w.push_back({p + "out_w", "f16", {d, d}});
        w.push_back({p + "out_b", "f32", {d}});
        w.push_back({p + "n2_g", "f32", {d}});
        w.push_back({p + "n2_b", "f32", {d}});
        w.push_back({p + "ff1_w", "f16", {ff, d}});
        w.push_back({p + "ff1_b", "f32", {ff}});
        w.push_back({p + "ff2_w", "f16", {d, ff}});
        w.push_back({p + "ff2_b", "f32", {d}});
    }
    w.push_back({"ln_g", "f32", {d}});
    w.push_back({"ln_b", "f32", {d}});
    w.push_back({"heads_w", "f16", {2 * m, d}});
    w.push_back({"heads_b", "f32", {2 * m}});
    return w;
}

// a whole non-negative number from the header, below 2^53 so the double holds it exactly
uint64_t whole(const std::string& path, const Json* v, const std::string& what) {
    if (!v || v->kind != Json::Num || v->num < 0 || v->num > 9.0e15 || v->num != std::floor(v->num))
        fail(path, "header field " + what + " missing or not a whole number");
    return uint64_t(v->num);
}

std::string text(const std::string& path, const Json* v, const std::string& what) {
    if (!v || v->kind != Json::Str) fail(path, "header field " + what + " missing or not a string");
    return v->str;
}

struct Entry {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset = 0, nbytes = 0;
};

}  // namespace

// ---------------------------------------------------------------------------------------------------- Weights
Weights::Weights(const std::string& path, int dev) : device(dev) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail(path, "cannot open");
    f.seekg(0, std::ios::end);
    const uint64_t fsize = uint64_t(f.tellg());
    f.seekg(0, std::ios::beg);

    // magic and header length
    unsigned char head[12];
    if (fsize < 12 || !f.read(reinterpret_cast<char*>(head), 12)) fail(path, "too short for a weight file");
    if (std::memcmp(head, kMagic, 8) != 0) fail(path, "bad magic (not a MFCNW001 weight file)");
    const uint64_t hlen = uint64_t(head[8]) | uint64_t(head[9]) << 8 | uint64_t(head[10]) << 16 |
                          uint64_t(head[11]) << 24;
    if (12 + hlen > fsize) fail(path, "header runs past the end of the file");
    std::string js(size_t(hlen), '\0');
    if (hlen && !f.read(&js[0], std::streamsize(hlen))) fail(path, "cannot read the header");
    Json h;
    try {
        h = Parser(js).parse();
    } catch (const std::exception& e) {
        fail(path, e.what());
    }
    if (h.kind != Json::Obj) fail(path, "header is not a JSON object");

    // the numbers must be the spec's
    if (whole(path, h.get("version"), "version") != 1) fail(path, "unknown version (this reader knows 1)");
    struct Num { const char* key; int* field; int want; };
    const Num nums[] = {{"d", &d, kD}, {"layers", &layers, kLayers}, {"heads", &heads, kHeads},
                        {"ff", &ff, kFF}, {"tokens", &tokens, kTokens}, {"moves", &moves, kMoves}};
    for (const Num& n : nums) {
        const uint64_t v = whole(path, h.get(n.key), n.key);
        if (v != uint64_t(n.want))
            fail(path, std::string(n.key) + " is " + std::to_string(v) + ", cudanet_spec.h wants " +
                           std::to_string(n.want));
        *n.field = int(v);
    }
    fingerprint = text(path, h.get("fingerprint"), "fingerprint");
    source = text(path, h.get("source"), "source");

    // the tensor table
    const Json* ts = h.get("tensors");
    if (!ts || ts->kind != Json::Arr) fail(path, "header has no tensors array");
    const uint64_t start = align_up(12 + hlen);           // the data start
    const uint64_t data = fsize > start ? fsize - start : 0;
    std::map<std::string, Entry> table;
    uint64_t end = 0;
    for (const Json& t : ts->arr) {
        if (t.kind != Json::Obj) fail(path, "a tensors entry is not an object");
        const std::string name = text(path, t.get("name"), "tensors[].name");
        Entry e;
        e.dtype = text(path, t.get("dtype"), name + ".dtype");
        uint64_t size = 0;
        if (e.dtype == "f16") size = 2;
        else if (e.dtype == "f32") size = 4;
        else fail(path, name + ": dtype " + e.dtype + " (f16 or f32 only)");
        const Json* sh = t.get("shape");
        if (!sh || sh->kind != Json::Arr) fail(path, name + ": no shape");
        uint64_t count = 1;
        for (const Json& s : sh->arr) {
            const uint64_t n = whole(path, &s, name + ".shape");
            if (n == 0 || n > (uint64_t(1) << 31)) fail(path, name + ": bad shape");
            count *= n;
            if (count > (uint64_t(1) << 40)) fail(path, name + ": shape too large");
            e.shape.push_back(int64_t(n));
        }
        e.offset = whole(path, t.get("offset"), name + ".offset");
        e.nbytes = whole(path, t.get("nbytes"), name + ".nbytes");
        if (e.nbytes != count * size) fail(path, name + ": nbytes does not match dtype and shape");
        if (e.offset % kAlign) fail(path, name + ": offset not 256-aligned");
        if (e.offset + e.nbytes > data) fail(path, name + ": runs past the end of the file");
        if (!table.emplace(name, std::move(e)).second) fail(path, name + ": listed twice");
        end = std::max(end, table[name].offset + table[name].nbytes);
    }
    // no two tensors may overlap
    std::vector<std::pair<uint64_t, uint64_t>> spans;
    for (const auto& kv : table) spans.emplace_back(kv.second.offset, kv.second.nbytes);
    std::sort(spans.begin(), spans.end());
    for (size_t i = 1; i < spans.size(); ++i)
        if (spans[i].first < spans[i - 1].first + spans[i - 1].second) fail(path, "two tensors overlap");
    // every spec tensor, with its dtype and shape
    for (const Want& w : spec_tensors()) {
        auto it = table.find(w.name);
        if (it == table.end()) fail(path, "missing tensor " + w.name);
        if (it->second.dtype != w.dtype) fail(path, w.name + ": dtype " + it->second.dtype + ", spec " + w.dtype);
        if (it->second.shape != w.shape) fail(path, w.name + ": shape differs from cudanet_spec.h");
    }

    // one device allocation, with the data region copied as it lies in the file, so each tensor keeps its offset
    bytes = size_t(align_up(end));
    Cuda& cu = cuda();
    check_rt(cu.cudaSetDevice(device), "cudaSetDevice (weights)");
    check_rt(cu.cudaMalloc(&base_, bytes), "cudaMalloc (weights)");
    try {
        std::vector<char> buf(size_t(std::min<uint64_t>(kChunk, bytes)));
        f.clear();
        f.seekg(std::streamoff(start), std::ios::beg);
        for (uint64_t pos = 0; pos < bytes; pos += kChunk) {
            const uint64_t n = std::min<uint64_t>(kChunk, bytes - pos);
            const uint64_t from_file = pos < end ? std::min<uint64_t>(n, end - pos) : 0;
            if (from_file && !f.read(buf.data(), std::streamsize(from_file))) fail(path, "cannot read the tensors");
            std::memset(buf.data() + from_file, 0, size_t(n - from_file));    // the padding after the last tensor
            check_rt(cu.cudaMemcpy(static_cast<char*>(base_) + pos, buf.data(), size_t(n), cudaMemcpyHostToDevice),
                     "cudaMemcpy (weights)");
        }
    } catch (...) {
        cu.cudaFree(base_);
        base_ = nullptr;
        throw;
    }
    for (const auto& kv : table) offset_.emplace(kv.first, size_t(kv.second.offset));
}

Weights::~Weights() {
    if (!base_) return;
    try {
        Cuda& cu = cuda();
        cu.cudaSetDevice(device);
        cu.cudaFree(base_);
    } catch (...) {
    }
}

const void* Weights::ptr(const std::string& name) const {
    auto it = offset_.find(name);
    if (it == offset_.end()) throw std::runtime_error("cudanet weights: no tensor named " + name);
    return static_cast<const char*>(base_) + it->second;
}

}  // namespace cudanet
