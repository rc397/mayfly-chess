// The C++ tree as a persistent search driven from Python.
//
// The tree lives across searches, as mcts.py's does. begin() re-roots it on the position the game is in (the
// root, a child or a grandchild, MCTS._root_for), discount() is the ponder discount, and step() is one pass of
// _run_search's plain loop (gather up to a batch, one forward pass, expand and back up, give back the held
// collisions). The Python caller keeps the clock, the extensions and the stopping rules between steps and reads
// the root's statistics. The network is a callback. Net::evaluate gets the encoded batch and fills the policy
// logits cut to each row's legal moves, the value and the per-move q. The selection, backup, collisions, cache
// and LCB pick matched the Python bit for bit on 820 of 820 test searches.
#pragma once
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "chess_core.h"

namespace mayfly2 {
using namespace mayfly;

struct Settings {
    int batch = 48;
    double policy_temp = 0.889688;
    double q_n0 = 1.45229;
    double c1 = 1.0236, c2 = 1.07;
    double explore_growth = 0.205868, explore_base = 207.255;
    int forcing = 4;
    double forcing_after = 8.0;
    bool forcing_apart = true;
    int collide = 2;
    bool cache_progress = true;
    bool fill_once = true;
    int draw_plies = 100;
    bool lcb = true;
    double lcb_alpha = 0.00184487, lcb_floor = 0.0928919;
    bool prior_fallback = true;
    size_t cache_size = 400000;
    bool use_tb = false;             // ask Net::tb at a new child (MCTS._gather's _tb_value), up to tb_pieces
    int tb_pieces = 6;
    int pipeline = 0;                // MCTS's pipeline key, 0 off, 1 two batches of `batch` in flight, 2 two of half
    bool prof = false;               // time each part of a step into Tree::prof() (no effect on the search)
};

// One batch of unexpanded nodes, as MCTS._prepare builds it.
struct Batch {
    int B = 0, wide = 0;
    std::vector<int64_t> ids, hm, ep, I;
    std::vector<float> cast;
    std::vector<uint8_t> M;
    std::vector<int> widths;
};
// What the network hands back, logits cut to each row's legal moves (B*wide), val (B) and qv (B*wide).
struct Out {
    std::vector<float> pol, val, qv;
};
class Net {
public:
    virtual ~Net() = default;
    virtual void evaluate(const Batch& in, Out& out) = 0;   // throws on failure
    // The pipeline's two halves (Settings::pipeline). submit queues a batch and collect waits for the oldest one
    // queued and fills `out`, with at most two queued. By default submit keeps a copy and collect evaluates it, so
    // any Net runs the pipeline with the same numbers. A net on the card overrides both and runs while the tree works.
    virtual void submit(const Batch& in) { queued_.push_back(in); }
    virtual void collect(Out& out)
    {
        if (queued_.empty()) throw std::runtime_error("collect with no batch queued");
        Batch b = std::move(queued_.front());
        queued_.pop_front();
        evaluate(b, out);
    }
    // The network's own timers (Settings::prof), switched on once and read as (name, total) pairs. None by default.
    virtual void timers_on(bool on) { (void)on; }
    virtual std::vector<std::pair<std::string, double>> timers() { return {}; }
    // MCTS._tb_value for the position in `fen`. True with its value from the side to move, false for none.
    virtual bool tb(const std::string& fen, double& value) { (void)fen; (void)value; return false; }

protected:
    std::deque<Batch> queued_;
};

struct Node {
    Pos pos;
    Key key;
    std::vector<int> moves;
    std::vector<int64_t> IX;
    int flags = 0;
    std::vector<float> P, N, W, V2, F, X, Q0;
    std::vector<Node*> kids;
    std::vector<uint8_t> FM;
    bool fm_ready = false;
    bool has_term = false;
    double term = 0.0;
    double sum_n = 0.0;
    bool pending = false;
    double value = 0.0;
    double q0w = 1.0;
    bool is_expanded = false;
};

struct Step {
    Node* node;
    int a;
    double x;
    bool apart;
};
typedef std::vector<Step> Path;

struct Stats {
    long long evals = 0, batches = 0, cache_hits = 0, collisions = 0, forced = 0, forced_apart = 0;
    long long repetitions = 0, lcb_overrode = 0, discounted = 0, reused = 0, tablebase = 0;
};

// Settings::prof, seconds summed over steps since reset(), by part of a step
struct Prof {
    double descend = 0, leaf = 0;      // the walks (selection with virtual visits), the new leaves (movegen, cache)
    double encode = 0;                 // MCTS._prepare, the batch's arrays
    double net = 0;                    // the network in the plain loop, the whole synchronous evaluate
    double submit = 0, collect = 0;    // the network in the pipeline, queueing a batch and waiting for one
    double expand = 0;                 // expansion, backup, cache inserts, collisions given back
    double step = 0;                   // the whole of step()
    long long steps = 0;
};

class Tree {
public:
    Tree(Net* net, const Settings& s);
    ~Tree();
    void reset();                                      // a new game, with no tree, no cache and counters zeroed
    // Re-root on `root` (the game's position, with `history` its earlier positions, oldest first). With `reuse`, the
    // subtree from the last search is kept when the position is the old root, a child or a grandchild.
    // Returns the visits carried (the found root's sum_n, before any discount), 0 for a fresh root.
    double begin(const Pos& root, const std::vector<Pos>& history, bool reuse);
    void discount(double d, bool q0);                  // the ponder discount (_run_search, before the loop)
    bool needs_eval() const;                           // the root is unexpanded and not terminal
    void expand_root();                                // its one forward pass
    // One pass of the plain loop. Returns 0 after a batch, 1 when nothing was gathered (go on), 2 when the
    // tree is saturated (stop). deadline is in time.time() seconds, or < 0 for none.
    int step(double target, double deadline);
    // With the pipeline on (Settings::pipeline), finish() reads back and backs up the batch still on the card, as
    // MCTS._pipe_finish does when a search ends, and held() is its held collisions, which are not visits (MCTS._held).
    // Both do nothing with the pipeline off. begin(), expand_root() and reset() finish first.
    void finish();
    double held() const;
    bool in_flight() const { return fly_.live; }
    const Prof& prof() const { return pf_; }
    int choose();                                      // MCTS._choose (LCB), an index into root moves, or -1 for none
    std::vector<int> pv(int first, int depth) const;   // MCTS._pv, as move codes
    long long tree_nodes() const;                      // MCTS.tree_nodes(root)
    Node* root() const { return root_; }
    const Stats& stats() const { return st_; }
    size_t pool_size() const { return pool_.size(); }
    size_t cache_len() const { return cache_.size(); }

private:
    struct CacheKey {
        Key key;
        int hm;
        bool operator==(const CacheKey& o) const { return hm == o.hm && key == o.key; }
    };
    struct CacheKeyHash {
        size_t operator()(const CacheKey& c) const { return KeyHash()(c.key) * 31 + (size_t)c.hm; }
    };
    struct CacheEntry {
        std::vector<float> logits;
        double raw;
        std::vector<float> q0;
    };
    struct Pending {
        Path path;
        Node* kid;
        CacheKey key;
    };
    struct Flight {                   // the pipeline's batch on the card
        bool live = false;
        int wide = 0;
        std::vector<Pending> pending;
        std::vector<std::pair<Path, Node*>> collided;
    };

    Node* new_node(const Pos& p);
    void prepare(const std::vector<Node*>& nodes, Batch& in, int& wide);
    void evaluate(const std::vector<Node*>& nodes, Out& out, int& wide);
    int step_pipe(double target, double deadline);
    void complete(Flight& f, const Out& out, std::vector<std::pair<Path, Node*>>& later);
    void abandon(std::vector<Pending>& pending, size_t from, std::vector<std::pair<Path, Node*>>& collided);
    void drop_flight();
    double expand(Node* node, const float* logits, double value, const float* q0, int nq);
    bool descend(Node* root, Path& path, Node*& parent, int& a);
    void gather(Path& path, Node* parent, int a, std::vector<Pending>& pending, std::vector<Node*>& leaves);
    void backup(const Path& path, double leaf);
    void unwind(const Path& path);
    bool repeats(const Key& k, const Path& path) const;
    int lcb_pick(Node* root);
    void keep_only(Node* r);                           // drop every node not reachable from r
    bool tb_value(const Node* n, double& v);          // the tablebase probe, remembered by (key, clock)

    Net* net_;
    Settings s_;
    Stats st_;
    Node* root_ = nullptr;
    std::vector<std::unique_ptr<Node>> pool_;
    std::unordered_set<Key, KeyHash> history_;
    std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> cache_;
    std::unordered_map<CacheKey, double, CacheKeyHash> tb_;   // NaN when not in the tables
    std::vector<std::pair<Path, Node*>>* collided_ = nullptr;
    Flight fly_;
    Prof pf_;
    std::chrono::steady_clock::time_point pt_;   // Settings::prof's running clock
};

double t_quantile(double alpha, double df);

}  // namespace mayfly2
