// See cudanet.h and cudanet_spec.h.
// The forward pass on one non-blocking stream: one pinned upload, the trunk and the heads product (a CUDA graph per
// batch size, captured on its first use), heads_post outside the graph (it depends on wide), one pinned download.
#include "cudanet.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "cudanet_spec.h"
#include "cuda_dyn.h"
#include "nvrtc_loader.h"
#include "kernels_trunk_launch.h"
#include "kernels_heads_launch.h"
#include "gemm.h"
#include "mmgemm.h"
#include "weights.h"

#include <cuda_fp16.h>

// a CUDA runtime call through the loaded DLL, thrown on failure
#define RT(call) check_rt(cuda().call, #call)

namespace cudanet {

namespace {

// the directory of the module (DLL or exe) this code is linked into
std::string module_dir()
{
#ifdef _WIN32
    HMODULE mod = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(reinterpret_cast<void*>(&module_dir)), &mod))
        return ".";
    std::vector<wchar_t> buf(MAX_PATH);
    for (;;) {
        const DWORD n = GetModuleFileNameW(mod, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) return ".";
        if (n < buf.size()) { buf.resize(n); break; }
        buf.resize(buf.size() * 2);   // truncated: try a longer buffer
    }
    std::wstring w(buf.begin(), buf.end());
    const size_t cut = w.find_last_of(L"\\/");
    w = cut == std::wstring::npos ? std::wstring(L".") : w.substr(0, cut);
    const int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    if (len <= 0) return ".";
    std::string s(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), &s[0], len, nullptr, nullptr);
    return s;
#else
    return ".";
#endif
}

std::string pick_kernel_dir(const std::string& given)
{
    if (!given.empty()) return given;
    const char* e = std::getenv("MAYFLY_CUDANET_SRC");
    if (e && e[0]) return e;
    return module_dir();
}

// the largest batch: env MAYFLY_CUDANET_MAX_BATCH (1..kMaxBatchLimit), else kMaxBatch
int pick_max_batch()
{
    const char* e = std::getenv("MAYFLY_CUDANET_MAX_BATCH");
    if (!e || !e[0]) return kMaxBatch;
    char* end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end == e || *end != 0 || v < 1 || v > kMaxBatchLimit)
        throw std::runtime_error(std::string("cudanet: MAYFLY_CUDANET_MAX_BATCH=") + e +
                                 " is not a whole number in 1.." + std::to_string(kMaxBatchLimit));
    return static_cast<int>(v);
}

std::string join(const std::string& dir, const char* file)
{
    if (dir.empty()) return file;
    const char last = dir.back();
    return (last == '\\' || last == '/') ? dir + file : dir + "\\" + file;
}

// Byte offsets of the inputs in the pinned staging block and in its device twin, one copy per batch.
// ids, hm, ep, cast and I start at offsets that depend on B alone, so a graph per B stays valid. Only M moves with
// wide, and M is read by heads_post, which runs outside the graph. Every offset is a multiple of 8.
struct InLayout {
    size_t ids, hm, ep, cast, I, M, total;
};

InLayout in_layout(int B, int wide)
{
    const size_t b = static_cast<size_t>(B), w = static_cast<size_t>(wide);
    InLayout l;
    l.ids = 0;
    l.hm = l.ids + sizeof(int64_t) * b * 64;
    l.ep = l.hm + sizeof(int64_t) * b;
    l.cast = l.ep + sizeof(int64_t) * b;
    l.I = l.cast + sizeof(float) * b * 4;
    l.M = l.I + sizeof(int64_t) * b * w;
    l.total = l.M + sizeof(uint8_t) * b * w;
    return l;
}

// the steps profile() times, in pass order, the names ending with the whole pass as a graph
enum ProfOp { kPEmbed, kPLn1, kPQkv, kPAtt, kPOut, kPLn2, kPFf1, kPGelu, kPFf2, kPCls, kPHeads, kProfOps };
const char* const kProfNames = "embed,ln1,qkv,attention,out,ln2,ff1,gelu,ff2,cls_ln,heads,graph_pass";
constexpr size_t kProfMarks = 2 + 8 * kLayers + 2;   // the start, embed, eight a layer, cls_ln, heads

// eval_times() slots are host milliseconds (inputs into staging, queueing the upload, the pass, heads_post and the
// download, waiting for the batch, outputs out), card milliseconds (upload, the pass, heads_post, download), then
// the batches and the eager passes among them
enum EtSlot {
    kEtStage, kEtUpQ, kEtPassQ, kEtTailQ, kEtWait, kEtOut, kEtUp, kEtPass, kEtPost, kEtDown, kEtBatches, kEtEager,
    kEtCount
};
const char* const kEtNames =
    "h_stage,h_queue_upload,h_queue_pass,h_queue_tail,h_wait,h_out,g_upload,g_pass,g_post,g_download,batches,eager";

// one layer's weights on the device
struct LayerW {
    const float *n1_g, *n1_b, *qkv_b, *out_b, *n2_g, *n2_b, *ff1_b, *ff2_b;
    const __half *qkv_w, *out_w, *ff1_w, *ff2_w;
};

}  // namespace

struct CudaNet::Impl {
    int device = 0;
    bool live = false;        // cuda() loaded: the destructor may call it
    bool use_graph = true;    // env MAYFLY_CUDANET_GRAPH=0 turns graphs off, and so does a failed capture
    bool acc16 = false;       // env MAYFLY_CUDANET_ACC16=1: the qkv and ff1 products accumulate in fp16
    bool acc16_all = false;   // env MAYFLY_CUDANET_ACC16=2: the out and ff2 products too (into r, added by the norms)
    bool att_mma = true;      // env MAYFLY_CUDANET_ATT=0: attention on the scalar kernel instead of tensor cores
    bool fuse = false;        // env MAYFLY_CUDANET_FUSE=1: kernels_fuse.cu (same bits, fewer passes over x)
    bool gelu_epi = false;    // env MAYFLY_CUDANET_GELU_EPI=1: GELU in the ff1 product's epilogue (tanh, not erf)
    // env MAYFLY_CUDANET_GEMM: kernels_gemm.cu instead of cuBLASLt for the products in MAYFLY_CUDANET_GEMM_OPS, with
    // GELU fused into ff1 and the out and ff2 residuals added straight into x (no r, no add in the norms).
    //   1  fp16 accumulation, default ops out,ff1,ff2 (qkv stays on cuBLASLt, which was a little faster there)
    //   2  fp16 within each k tile of 64, then fp32, for about fp32 accuracy and faster than cuBLASLt's fp32
    //      accumulation, default ops qkv,out,ff1,ff2
    int mm_acc = 0;
    int mm_min_b = 1;   // env MAYFLY_CUDANET_GEMM_MINB: cuBLASLt below this batch size (default 24 for 1, 4 for 2)
    enum { kMmQkv, kMmOut, kMmFf1, kMmFf2, kMmOps };
    std::array<bool, kMmOps> mm_use{};
    std::array<MmCfg, kMmOps> mm_cfg{};
    std::unique_ptr<MmGemm> mm;
    int max_batch = kMaxBatch;   // env MAYFLY_CUDANET_MAX_BATCH: the buffers' size in positions

    std::unique_ptr<Weights> w;
    std::unique_ptr<KernelModule> trunk, heads, fused;
    CUfunction k_embed = nullptr, k_ln = nullptr, k_add_ln = nullptr, k_add = nullptr, k_att = nullptr,
               k_att_mma = nullptr, k_gelu = nullptr, k_cls = nullptr, k_post = nullptr;
    CUfunction k_embed_ln = nullptr, k_add_ln_keep = nullptr, k_add2_ln = nullptr, k_cls_add2 = nullptr;   // fuse

    // embedding tables, the final norm and the heads
    const float *piece = nullptr, *pos = nullptr, *cls_tok = nullptr, *cast_w = nullptr, *cast_b = nullptr,
                *hm_emb = nullptr, *ep_mark = nullptr, *ln_g = nullptr, *ln_b = nullptr, *heads_b = nullptr;
    const __half* heads_w = nullptr;
    std::array<LayerW, kLayers> L{};

    cudaStream_t stream = nullptr;
    std::unique_ptr<Gemm> gemm;

    // device buffers, sized for max_batch (cudanet_spec.h "the buffers")
    float* x = nullptr;
    __half *h = nullptr, *qkv = nullptr, *att = nullptr, *ff = nullptr, *cls = nullptr;
    __half* r = nullptr;             // acc16_all: the out or ff2 product before it joins x
    __half* r2 = nullptr;            // acc16_all with fuse: the ff2 product (r keeps the out product)
    float* logits = nullptr;
    unsigned char* d_in = nullptr;   // the inputs, laid out by in_layout
    float* d_out = nullptr;          // pol [B*wide], val [B], qv [B*wide], back to back
    // pinned host staging in the same layouts, and a second pair for a batch queued behind another (begin/end)
    unsigned char* h_in = nullptr;
    float* h_out = nullptr;
    unsigned char* h_in2 = nullptr;
    float* h_out2 = nullptr;

    std::vector<cudaGraphExec_t> exec;   // by B, max_batch + 1 slots, null until captured
    std::mutex mu;

    // profile(): an event after each step of an eager pass, only while prof_on
    bool prof_on = false;
    std::vector<cudaEvent_t> prof_ev;   // kProfMarks, then two around the graph pass
    std::array<int, kProfMarks> prof_op{};
    size_t prof_n = 0;

    void mark(int op)
    {
        if (!prof_on) return;
        RT(cudaEventRecord(prof_ev[prof_n], stream));
        prof_op[prof_n++] = op;
    }

    // begin/end: the batches on the card, oldest first. A second one only when the search's pipeline queues its
    // next batch behind the last: it gets the other staging pair (slot) and waits on the first's done event.
    struct Fly {
        int B, wide, slot;
        bool ev;   // done_ev[slot] was recorded after its download
    };
    std::array<Fly, 2> fly{};
    int fly_n = 0;
    std::array<cudaEvent_t, 2> done_ev{};
    long long eager_n = 0;   // passes run eager (a batch size's first use, or graphs off)

    // with eval_times_on, host clocks and five events around each batch (before the upload, after it, after the pass,
    // after heads_post, after the download), one set a slot, summed into et by end
    bool et_on = false;
    std::array<std::array<cudaEvent_t, 5>, 2> et_ev{};
    std::array<double, kEtCount> et{};
    std::chrono::steady_clock::time_point et_t0;

    void et_mark(int slot, int i)
    {
        if (et_on) RT(cudaEventRecord(et_ev[static_cast<size_t>(slot)][static_cast<size_t>(i)], stream));
    }
    void et_add(int slot)   // host milliseconds since et_t0 into et[slot], then et_t0 = now
    {
        if (!et_on) return;
        const auto now = std::chrono::steady_clock::now();
        et[static_cast<size_t>(slot)] += std::chrono::duration<double, std::milli>(now - et_t0).count();
        et_t0 = now;
    }

    ~Impl();
    void init(const std::string& path, int dev, const std::string& kernel_dir);
    void check_numbers(const std::string& path) const;
    const float* f32(const std::string& name) const { return static_cast<const float*>(w->ptr(name)); }
    const __half* f16(const std::string& name) const { return static_cast<const __half*>(w->ptr(name)); }

    // a kernel on the stream, through cuLaunchKernel directly (nvrtc_loader.h's launch_kernel does the same).
    // Each argument is passed by value with the exact type the kernel takes.
    template <class... A>
    void run(CUfunction f, const LaunchCfg& c, A... a)
    {
        void* args[] = {const_cast<void*>(static_cast<const void*>(&a))...};
        check_cu(cuda().cuLaunchKernel(f, c.gx, c.gy, c.gz, c.bx, c.by, c.bz, c.shmem, stream, args, nullptr),
                 "cuLaunchKernel");
    }

    void check_batch(int B) const;
    void forward_kernels(int B);
    void capture(int B);
    void forward(int B);
    void begin_locked(int B, int wide, const int64_t* ids, const float* cast, const int64_t* hm, const int64_t* ep,
                      const int64_t* I, const uint8_t* M, bool queued);
    void end_locked(float* pol, float* val, float* qv);
    std::vector<double> profile(int B, int reps);
};

template <class T>
static void dalloc(T*& p, size_t count)
{
    void* v = nullptr;
    RT(cudaMalloc(&v, count * sizeof(T)));
    p = static_cast<T*>(v);
}

template <class T>
static void halloc(T*& p, size_t count)
{
    void* v = nullptr;
    RT(cudaMallocHost(&v, count * sizeof(T)));
    p = static_cast<T*>(v);
}

void CudaNet::Impl::check_numbers(const std::string& path) const
{
    if (w->d == kD && w->layers == kLayers && w->heads == kHeads && w->ff == kFF && w->tokens == kTokens &&
        w->moves == kMoves)
        return;
    char msg[768];
    std::snprintf(msg, sizeof(msg),
                  "cudanet: %s has d=%d layers=%d heads=%d ff=%d tokens=%d moves=%d; this build needs d=%d "
                  "layers=%d heads=%d ff=%d tokens=%d moves=%d",
                  path.c_str(), w->d, w->layers, w->heads, w->ff, w->tokens, w->moves, kD, kLayers, kHeads, kFF,
                  kTokens, kMoves);
    throw std::runtime_error(msg);
}

void CudaNet::Impl::init(const std::string& path, int dev, const std::string& kernel_dir)
{
    device = dev;
    max_batch = pick_max_batch();   // a bad value fails before anything is loaded
    exec.assign(static_cast<size_t>(max_batch) + 1, nullptr);
    cuda();   // loads the DLLs, throws naming the missing one
    live = true;
    RT(cudaSetDevice(device));
    RT(cudaFree(nullptr));   // the primary context exists and is current before the modules load into it

    // the weights, refused unless they are the spec's network
    w = std::make_unique<Weights>(path, device);
    check_numbers(path);
    piece = f32("piece");
    pos = f32("pos");
    cls_tok = f32("cls");
    cast_w = f32("cast_w");
    cast_b = f32("cast_b");
    hm_emb = f32("hm");
    ep_mark = f32("ep_mark");
    for (int i = 0; i < kLayers; ++i) {
        const std::string p = "L" + std::to_string(i) + ".";
        LayerW& l = L[static_cast<size_t>(i)];
        l.n1_g = f32(p + "n1_g");
        l.n1_b = f32(p + "n1_b");
        l.qkv_w = f16(p + "qkv_w");
        l.qkv_b = f32(p + "qkv_b");
        l.out_w = f16(p + "out_w");
        l.out_b = f32(p + "out_b");
        l.n2_g = f32(p + "n2_g");
        l.n2_b = f32(p + "n2_b");
        l.ff1_w = f16(p + "ff1_w");
        l.ff1_b = f32(p + "ff1_b");
        l.ff2_w = f16(p + "ff2_w");
        l.ff2_b = f32(p + "ff2_b");
    }
    ln_g = f32("ln_g");
    ln_b = f32("ln_b");
    heads_w = f16("heads_w");
    heads_b = f32("heads_b");

    // the kernels, compiled for this device
    const std::string dir = pick_kernel_dir(kernel_dir);
    trunk = std::make_unique<KernelModule>(read_text_file(join(dir, "kernels_trunk.cu")), "kernels_trunk", device);
    heads = std::make_unique<KernelModule>(read_text_file(join(dir, "kernels_heads.cu")), "kernels_heads", device);
    k_embed = trunk->get("embed");
    k_ln = trunk->get("layernorm_f32_f16");
    k_add_ln = trunk->get("add_layernorm_f32_f16");
    k_add = trunk->get("add_f16_f32");
    k_att = trunk->get("attention_f16");
    k_att_mma = trunk->get("attention_mma");
    k_gelu = trunk->get("gelu_f16");
    k_cls = heads->get("cls_layernorm");
    k_post = heads->get("heads_post");

    RT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    gemm = std::make_unique<Gemm>(stream);
    const char* mg = std::getenv("MAYFLY_CUDANET_GEMM");
    if (mg && (mg[0] == '1' || mg[0] == '2')) {
        mm_acc = mg[0] - '0';
        const char* ops = std::getenv("MAYFLY_CUDANET_GEMM_OPS");
        const std::string o = ops && ops[0] ? ops : (mm_acc == 2 ? "qkv,out,ff1,ff2" : "out,ff1,ff2");
        const char* names[kMmOps] = {"qkv", "out", "ff1", "ff2"};
        const int nk[kMmOps][2] = {{3 * kD, kD}, {kD, kD}, {kFF, kD}, {kD, kFF}};
        for (int i = 0; i < kMmOps; ++i) {
            mm_use[i] = o.find(names[i]) != std::string::npos;
            mm_cfg[i] = MmGemm::pick(nk[i][0], nk[i][1], mm_acc);
        }
        mm = std::make_unique<MmGemm>(dir, device, stream);
        // On an RTX 3070 with fp16 accumulation cuBLASLt was faster up to batch 16 and the hand-written products
        // from 32. In mode 2 they were faster than cuBLASLt's fp32 accumulation from 8.
        const char* mb = std::getenv("MAYFLY_CUDANET_GEMM_MINB");
        mm_min_b = mb && mb[0] ? std::atoi(mb) : (mm_acc == 2 ? 4 : 24);
    }

    // every buffer once, for the largest batch
    const size_t rows = static_cast<size_t>(max_batch) * kTokens;
    dalloc(x, rows * kD);
    dalloc(h, rows * kD);
    dalloc(qkv, rows * 3 * kD);
    dalloc(att, rows * kD);
    dalloc(ff, rows * kFF);
    dalloc(cls, static_cast<size_t>(max_batch) * kD);
    dalloc(r, rows * kD);
    dalloc(logits, static_cast<size_t>(max_batch) * 2 * kMoves);
    const size_t in_bytes = in_layout(max_batch, kMaxWide).total;
    const size_t out_floats = static_cast<size_t>(max_batch) * (2 * kMaxWide + 1);
    dalloc(d_in, in_bytes);
    dalloc(d_out, out_floats);
    halloc(h_in, in_bytes);
    halloc(h_out, out_floats);
    halloc(h_in2, in_bytes);
    halloc(h_out2, out_floats);
    for (auto& e : done_ev) RT(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));

    const char* g = std::getenv("MAYFLY_CUDANET_GRAPH");
    use_graph = !(g && g[0] == '0');
    const char* a16 = std::getenv("MAYFLY_CUDANET_ACC16");
    acc16 = a16 && (a16[0] == '1' || a16[0] == '2');
    acc16_all = a16 && a16[0] == '2';
    const char* am = std::getenv("MAYFLY_CUDANET_ATT");
    att_mma = !(am && am[0] == '0');

    const char* fu = std::getenv("MAYFLY_CUDANET_FUSE");
    fuse = fu && fu[0] == '1';
    if (fuse) {
        fused = std::make_unique<KernelModule>(read_text_file(join(dir, "kernels_fuse.cu")), "kernels_fuse", device);
        k_embed_ln = fused->get("embed_ln");
        k_add_ln_keep = fused->get("add_layernorm_keep");
        k_add2_ln = fused->get("add2_layernorm");
        k_cls_add2 = fused->get("cls_layernorm_add2");
        if (acc16_all) dalloc(r2, rows * kD);
    }
    const char* ge = std::getenv("MAYFLY_CUDANET_GELU_EPI");
    gelu_epi = ge && ge[0] == '1';
    if (gelu_epi) {
        // every batch size's ff1 plan now, so a missing algorithm turns the epilogue off for all of them
        const Mode mg = acc16 ? Mode::F16_GELU_BIAS_ACC16 : Mode::F16_GELU_BIAS;
        try {
            for (int b = 1; b <= max_batch; ++b) gemm->plan(b * kTokens, kFF, kD, mg);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "cudanet: no GELU epilogue (%.300s), the gelu kernel instead\n", ex.what());
            gelu_epi = false;
        }
    }
}

CudaNet::Impl::~Impl()
{
    if (!live) return;
    try {
        Cuda& c = cuda();
        c.cudaSetDevice(device);
        if (stream) c.cudaStreamSynchronize(stream);
        for (auto& e : exec)
            if (e) { c.cudaGraphExecDestroy(e); e = nullptr; }
        for (auto& e : prof_ev)
            if (e) c.cudaEventDestroy(e);
        for (auto& set : et_ev)
            for (auto& e : set)
                if (e) c.cudaEventDestroy(e);
        for (auto& e : done_ev)
            if (e) c.cudaEventDestroy(e);
        gemm.reset();
        if (stream) c.cudaStreamDestroy(stream);
        stream = nullptr;
        void* dev_bufs[] = {x, h, qkv, att, ff, cls, r, r2, logits, d_in, d_out};
        for (void* p : dev_bufs)
            if (p) c.cudaFree(p);
        if (h_in) c.cudaFreeHost(h_in);
        if (h_out) c.cudaFreeHost(h_out);
        if (h_in2) c.cudaFreeHost(h_in2);
        if (h_out2) c.cudaFreeHost(h_out2);
    } catch (...) {
        // never throw from a destructor
    }
    // the modules and the weights free themselves
}

void CudaNet::Impl::check_batch(int B) const
{
    if (B >= 1 && B <= max_batch) return;
    std::string msg = "cudanet: batch size " + std::to_string(B) + " outside 1.." + std::to_string(max_batch);
    if (B > max_batch && max_batch < kMaxBatchLimit) msg += " (env MAYFLY_CUDANET_MAX_BATCH sets the limit)";
    throw std::runtime_error(msg);
}

// The whole pass from the uploaded inputs to the logits, for B positions. Pointers depend on B alone.
void CudaNet::Impl::forward_kernels(int B)
{
    const int rows = B * kTokens;
    const float eps = kLnEps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kDh));
    const InLayout lay = in_layout(B, 1);   // ids..cast do not depend on wide
    const int64_t* ids = reinterpret_cast<const int64_t*>(d_in + lay.ids);
    const int64_t* hm = reinterpret_cast<const int64_t*>(d_in + lay.hm);
    const int64_t* ep = reinterpret_cast<const int64_t*>(d_in + lay.ep);
    const float* cast = reinterpret_cast<const float*>(d_in + lay.cast);

    mark(-1);
    if (fuse)   // the embedding and layer 0's first norm in one kernel
        run(k_embed_ln, launch::embed(B), ids, cast, hm, ep, piece, pos, cls_tok, cast_w, cast_b, hm_emb, ep_mark, x,
            L[0].n1_g, L[0].n1_b, h, B, eps);
    else
        run(k_embed, launch::embed(B), ids, cast, hm, ep, piece, pos, cls_tok, cast_w, cast_b, hm_emb, ep_mark, x, B);
    mark(kPEmbed);
    const Mode m16 = acc16 ? Mode::F16_BIAS_ACC16 : Mode::F16_BIAS;
    const Mode mff1 = !gelu_epi ? m16 : (acc16 ? Mode::F16_GELU_BIAS_ACC16 : Mode::F16_GELU_BIAS);
    // products still to be added into x, the out product in r (fuse keeps it there) and the ff2 product in r2 (fuse)
    // or r (no fuse). The gemm's residual epilogues add straight into x and leave nothing pending.
    bool pend_out = false, pend_ff2 = false;
    const bool use_mm = mm && B >= mm_min_b;   // per batch size, so each captured graph keeps its own choice
    __half* const rff2 = fuse ? r2 : r;
    bool first = true;
    for (const LayerW& l : L) {
        if (first && fuse)
            ;   // embed_ln wrote h
        else if (pend_out && pend_ff2)   // fuse: r and r2 both join x here
            run(k_add2_ln, launch::layernorm(rows), x, r, r2, l.n1_g, l.n1_b, h, rows, eps);
        else if (pend_out || pend_ff2)
            run(k_add_ln, launch::layernorm(rows), x, pend_out ? r : rff2, l.n1_g, l.n1_b, h, rows, kD, eps);
        else
            run(k_ln, launch::layernorm(rows), x, l.n1_g, l.n1_b, h, rows, kD, eps);
        pend_out = pend_ff2 = false;
        first = false;
        mark(kPLn1);
        if (use_mm && mm_use[kMmQkv])
            mm->run(mm_cfg[kMmQkv], h, l.qkv_w, l.qkv_b, qkv, nullptr, rows, 3 * kD, kD, MmEpi::Bias);
        else
            gemm->linear(h, l.qkv_w, l.qkv_b, qkv, rows, 3 * kD, kD, m16);
        mark(kPQkv);
        if (att_mma)
            run(k_att_mma, launch::attention_mma(B), qkv, att, B, scale);
        else
            run(k_att, launch::attention(B), qkv, att, B, scale);
        mark(kPAtt);
        if (use_mm && mm_use[kMmOut]) {
            mm->run(mm_cfg[kMmOut], att, l.out_w, l.out_b, nullptr, x, rows, kD, kD, MmEpi::BiasResidual);
            mark(kPOut);
            run(k_ln, launch::layernorm(rows), x, l.n2_g, l.n2_b, h, rows, kD, eps);
        } else if (acc16_all) {
            gemm->linear(att, l.out_w, l.out_b, r, rows, kD, kD, Mode::F16_BIAS_ACC16);
            mark(kPOut);
            if (fuse) {   // x + r in registers only, and the next first norm adds r again
                run(k_add_ln_keep, launch::layernorm(rows), x, r, l.n2_g, l.n2_b, h, rows, eps);
                pend_out = true;
            } else {
                run(k_add_ln, launch::layernorm(rows), x, r, l.n2_g, l.n2_b, h, rows, kD, eps);
            }
        } else {
            gemm->linear(att, l.out_w, l.out_b, x, rows, kD, kD, Mode::F32_BIAS_RESIDUAL);
            mark(kPOut);
            run(k_ln, launch::layernorm(rows), x, l.n2_g, l.n2_b, h, rows, kD, eps);
        }
        mark(kPLn2);
        if (use_mm && mm_use[kMmFf1]) {   // exact GELU in the epilogue
            mm->run(mm_cfg[kMmFf1], h, l.ff1_w, l.ff1_b, ff, nullptr, rows, kFF, kD, MmEpi::BiasGelu);
            mark(kPFf1);
        } else {
            gemm->linear(h, l.ff1_w, l.ff1_b, ff, rows, kFF, kD, mff1);
            mark(kPFf1);
            if (!gelu_epi) run(k_gelu, launch::gelu(rows * kFF), ff, rows * kFF);
        }
        mark(kPGelu);
        if (use_mm && mm_use[kMmFf2]) {
            mm->run(mm_cfg[kMmFf2], ff, l.ff2_w, l.ff2_b, nullptr, x, rows, kD, kFF, MmEpi::BiasResidual);
        } else if (acc16_all) {
            gemm->linear(ff, l.ff2_w, l.ff2_b, rff2, rows, kD, kFF, Mode::F16_BIAS_ACC16);
            pend_ff2 = true;
        } else {
            gemm->linear(ff, l.ff2_w, l.ff2_b, x, rows, kD, kFF, Mode::F32_BIAS_RESIDUAL);
        }
        mark(kPFf2);
    }
    if (pend_out && pend_ff2) {   // the last two products added for the CLS rows only
        run(k_cls_add2, launch::cls_layernorm(B), x, r, r2, ln_g, ln_b, cls, B, eps);
    } else {
        if (pend_out || pend_ff2)
            run(k_add, launch::add_f16_f32(rows * kD), x, pend_out ? r : rff2, rows * kD);
        run(k_cls, launch::cls_layernorm(B), x, ln_g, ln_b, cls, B, eps);
    }
    mark(kPCls);
    gemm->linear(cls, heads_w, heads_b, logits, B, 2 * kMoves, kD, Mode::F32_BIAS);
    mark(kPHeads);
}

std::vector<double> CudaNet::Impl::profile(int B, int reps)
{
    check_batch(B);
    if (reps < 1) reps = 1;
    std::lock_guard<std::mutex> lock(mu);
    RT(cudaSetDevice(device));
    // an empty board in every row, with ids, hm, ep and cast all zero
    RT(cudaMemsetAsync(d_in, 0, in_layout(B, 1).I, stream));
    if (prof_ev.empty()) {
        prof_ev.assign(kProfMarks + 2, nullptr);
        for (auto& e : prof_ev) RT(cudaEventCreateWithFlags(&e, cudaEventDefault));
    }
    forward_kernels(B);   // warm-up: plans cached, lazily loaded kernels loaded
    std::vector<double> ms(kProfOps + 1, 0.0);
    for (int r = 0; r < reps; ++r) {
        prof_n = 0;
        prof_on = true;
        try {
            forward_kernels(B);
        } catch (...) {
            prof_on = false;
            throw;
        }
        prof_on = false;
        RT(cudaEventSynchronize(prof_ev[prof_n - 1]));
        for (size_t i = 1; i < prof_n; ++i) {
            float t = 0.0f;
            RT(cudaEventElapsedTime(&t, prof_ev[i - 1], prof_ev[i]));
            ms[static_cast<size_t>(prof_op[i])] += t;
        }
    }
    // the whole pass as the search runs it (forward captures the graph on its first use of B)
    forward(B);
    cudaEvent_t a = prof_ev[kProfMarks], b = prof_ev[kProfMarks + 1];
    for (int r = 0; r < reps; ++r) {
        RT(cudaEventRecord(a, stream));
        forward(B);
        RT(cudaEventRecord(b, stream));
        RT(cudaEventSynchronize(b));
        float t = 0.0f;
        RT(cudaEventElapsedTime(&t, a, b));
        ms[kProfOps] += t;
    }
    for (double& v : ms) v /= reps;
    return ms;
}

// Record forward_kernels(B) as a graph. It runs nothing, since the eager pass before it already filled the buffers.
void CudaNet::Impl::capture(int B)
{
    RT(cudaStreamSynchronize(stream));
    RT(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    cudaGraph_t g = nullptr;
    try {
        forward_kernels(B);
    } catch (...) {
        cuda().cudaStreamEndCapture(stream, &g);   // ends the capture whatever went wrong
        if (g) cuda().cudaGraphDestroy(g);
        throw;
    }
    RT(cudaStreamEndCapture(stream, &g));
    cudaGraphExec_t e = nullptr;
    const cudaError_t err = cuda().cudaGraphInstantiate(&e, g, 0);
    cuda().cudaGraphDestroy(g);
    check_rt(err, "cudaGraphInstantiate");
    exec[static_cast<size_t>(B)] = e;
}

void CudaNet::Impl::forward(int B)
{
    cudaGraphExec_t e = exec[static_cast<size_t>(B)];
    if (e) {
        RT(cudaGraphLaunch(e, stream));
        return;
    }
    // the first use of B (or graphs off) runs eager, which gives this call's numbers and the warm-up a capture
    // needs (cuBLASLt plans cached, lazily loaded kernels loaded)
    eager_n++;
    forward_kernels(B);
    if (!use_graph) return;
    try {
        capture(B);
    } catch (const std::exception& ex) {
        // a capture that fails costs speed, never a move, so run eager from now on
        std::fprintf(stderr, "cudanet: CUDA graph capture failed at B=%d, eager from now on: %.300s\n", B, ex.what());
        use_graph = false;
        cuda().cudaGetLastError();
    }
}

// evaluate's first half queues on the stream the inputs into pinned staging, one upload, the pass, heads_post and
// one download. `queued` (begin, not evaluate) records a done event after the download, so end can wait for this
// batch alone while a later one is queued behind it. The caller holds mu.
void CudaNet::Impl::begin_locked(int B, int wide, const int64_t* ids, const float* cast, const int64_t* hm,
                                 const int64_t* ep, const int64_t* I, const uint8_t* M, bool queued)
{
    check_batch(B);
    if (wide < 1 || wide > kMaxWide)
        throw std::runtime_error("cudanet: wide " + std::to_string(wide) + " outside 1.." + std::to_string(kMaxWide));
    if (fly_n == 2) throw std::runtime_error("cudanet: begin with two batches already on the card");
    if (fly_n && !queued) throw std::runtime_error("cudanet: evaluate while a batch is on the card");
    const int slot = fly_n ? 1 - fly[0].slot : 0;
    unsigned char* hin = slot ? h_in2 : h_in;
    float* hout = slot ? h_out2 : h_out;

    RT(cudaSetDevice(device));   // the calling thread may not be the one that built us
    if (et_on) et_t0 = std::chrono::steady_clock::now();

    // the inputs into pinned staging, then one upload
    const size_t b = static_cast<size_t>(B), wd = static_cast<size_t>(wide);
    const InLayout lay = in_layout(B, wide);
    std::memcpy(hin + lay.ids, ids, sizeof(int64_t) * b * 64);
    std::memcpy(hin + lay.hm, hm, sizeof(int64_t) * b);
    std::memcpy(hin + lay.ep, ep, sizeof(int64_t) * b);
    std::memcpy(hin + lay.cast, cast, sizeof(float) * b * 4);
    std::memcpy(hin + lay.I, I, sizeof(int64_t) * b * wd);
    std::memcpy(hin + lay.M, M, sizeof(uint8_t) * b * wd);
    et_add(kEtStage);
    et_mark(slot, 0);
    RT(cudaMemcpyAsync(d_in, hin, lay.total, cudaMemcpyHostToDevice, stream));
    et_mark(slot, 1);
    et_add(kEtUpQ);

    const long long eager_before = eager_n;
    forward(B);
    et_mark(slot, 2);
    et_add(kEtPassQ);
    if (et_on) et[kEtEager] += static_cast<double>(eager_n - eager_before);

    // _collect's arithmetic at this call's wide, then one download
    const float* lg = logits;
    const int64_t* I_d = reinterpret_cast<const int64_t*>(d_in + lay.I);
    const uint8_t* M_d = reinterpret_cast<const uint8_t*>(d_in + lay.M);
    float* pol_d = d_out;
    float* val_d = d_out + b * wd;
    float* qv_d = val_d + b;
    run(k_post, launch::heads_post(B, wide), lg, I_d, M_d, B, wide, pol_d, val_d, qv_d);
    et_mark(slot, 3);
    const size_t n_out = b * (2 * wd + 1);
    RT(cudaMemcpyAsync(hout, d_out, sizeof(float) * n_out, cudaMemcpyDeviceToHost, stream));
    et_mark(slot, 4);
    if (queued) RT(cudaEventRecord(done_ev[static_cast<size_t>(slot)], stream));
    et_add(kEtTailQ);
    fly[static_cast<size_t>(fly_n++)] = Fly{B, wide, slot, queued};
}

// evaluate's second half waits for the oldest batch begin_locked queued (the whole stream when it is the only
// one), then copies its outputs out of pinned staging. The caller holds mu.
void CudaNet::Impl::end_locked(float* pol, float* val, float* qv)
{
    if (!fly_n) throw std::runtime_error("cudanet: end with no batch on the card");
    const Fly f = fly[0];
    fly[0] = fly[1];
    fly_n--;   // whatever happens below, the batch is off the card's books
    const size_t b = static_cast<size_t>(f.B), wd = static_cast<size_t>(f.wide);
    const float* hout = f.slot ? h_out2 : h_out;
    RT(cudaSetDevice(device));
    if (et_on) et_t0 = std::chrono::steady_clock::now();
    if (fly_n && f.ev)
        RT(cudaEventSynchronize(done_ev[static_cast<size_t>(f.slot)]));
    else
        RT(cudaStreamSynchronize(stream));
    et_add(kEtWait);

    std::memcpy(pol, hout, sizeof(float) * b * wd);
    std::memcpy(val, hout + b * wd, sizeof(float) * b);
    std::memcpy(qv, hout + b * wd + b, sizeof(float) * b * wd);
    et_add(kEtOut);
    if (et_on) {
        const auto& ev = et_ev[static_cast<size_t>(f.slot)];
        const int slots[4] = {kEtUp, kEtPass, kEtPost, kEtDown};
        for (int i = 0; i < 4; ++i) {
            float t = 0.0f;
            RT(cudaEventElapsedTime(&t, ev[static_cast<size_t>(i)], ev[static_cast<size_t>(i + 1)]));
            et[static_cast<size_t>(slots[i])] += t;
        }
        et[kEtBatches] += 1.0;
    }
}

CudaNet::CudaNet(const std::string& weights_path, int device, const std::string& kernel_dir)
    : p_(std::make_unique<Impl>())
{
    p_->init(weights_path, device, kernel_dir);   // on a throw, ~Impl frees what init got
}

CudaNet::~CudaNet() = default;

void CudaNet::evaluate(int B, int wide, const int64_t* ids, const float* cast, const int64_t* hm, const int64_t* ep,
                       const int64_t* I, const uint8_t* M, float* pol, float* val, float* qv)
{
    std::lock_guard<std::mutex> lock(p_->mu);
    p_->begin_locked(B, wide, ids, cast, hm, ep, I, M, false);
    p_->end_locked(pol, val, qv);
}

void CudaNet::begin(int B, int wide, const int64_t* ids, const float* cast, const int64_t* hm, const int64_t* ep,
                    const int64_t* I, const uint8_t* M)
{
    std::lock_guard<std::mutex> lock(p_->mu);
    p_->begin_locked(B, wide, ids, cast, hm, ep, I, M, true);
}

void CudaNet::end(float* pol, float* val, float* qv)
{
    std::lock_guard<std::mutex> lock(p_->mu);
    p_->end_locked(pol, val, qv);
}

void CudaNet::eval_times_on(bool on)
{
    std::lock_guard<std::mutex> lock(p_->mu);
    if (p_->fly_n) throw std::runtime_error("cudanet: eval_times_on while a batch is on the card");
    if (on && !p_->et_ev[0][0]) {
        RT(cudaSetDevice(p_->device));
        for (auto& set : p_->et_ev)
            for (auto& e : set) RT(cudaEventCreateWithFlags(&e, cudaEventDefault));
    }
    if (on && !p_->et_on) p_->et.fill(0.0);
    p_->et_on = on;
}

std::vector<double> CudaNet::eval_times()
{
    std::lock_guard<std::mutex> lock(p_->mu);
    return std::vector<double>(p_->et.begin(), p_->et.end());
}

const char* CudaNet::eval_times_names()
{
    return kEtNames;
}

const std::string& CudaNet::fingerprint() const
{
    return p_->w->fingerprint;
}

int CudaNet::max_batch() const
{
    return p_->max_batch;
}

std::vector<double> CudaNet::profile(int B, int reps)
{
    return p_->profile(B, reps);
}

const char* CudaNet::profile_names()
{
    return kProfNames;
}

}  // namespace cudanet
