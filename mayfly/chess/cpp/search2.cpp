// See search2.h. The float32 arithmetic follows the Python's order exactly. Build with MSVC /O2 /fp:precise.
#include "search2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace mayfly2 {

template <class T>
static int argmax_first(const std::vector<T>& v)
{
    int best = -1;
    for (int i = 0; i < (int)v.size(); i++) {
        if (v[i] != v[i]) return i;   // numpy returns the first NaN
        if (best < 0 || v[i] > v[best]) best = i;
    }
    return best;
}

// numpy's DOUBLE_pairwise_sum over a contiguous row (walk.c's `pairwise`)
static double pairwise(const double* a, size_t n)
{
    if (n < 8) {
        double res = -0.0;
        for (size_t i = 0; i < n; i++) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        size_t i;
        for (int j = 0; j < 8; j++) r[j] = a[j];
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

// Settings::prof's clock, adding the seconds since the last tick to `into`
typedef std::chrono::steady_clock PClock;
static inline void tick(PClock::time_point& t, double& into)
{
    const PClock::time_point now = PClock::now();
    into += std::chrono::duration<double>(now - t).count();
    t = now;
}

Tree::Tree(Net* net, const Settings& s) : net_(net), s_(s) { init_tables(); }
Tree::~Tree() = default;

void Tree::reset()
{
    drop_flight();
    root_ = nullptr;
    pool_.clear();
    cache_.clear();
    tb_.clear();
    history_.clear();
    st_ = Stats();
    pf_ = Prof();
}

// Every node reachable from r stays. The rest (the lines the game did not take) are freed, as Python's garbage
// collector frees them once the old root is dropped.
void Tree::keep_only(Node* r)
{
    std::unordered_set<Node*> live;
    std::vector<Node*> stack(1, r);
    while (!stack.empty()) {
        Node* n = stack.back();
        stack.pop_back();
        if (!live.insert(n).second) continue;
        for (Node* k : n->kids)
            if (k) stack.push_back(k);
    }
    std::vector<std::unique_ptr<Node>> kept;
    kept.reserve(live.size());
    for (auto& p : pool_)
        if (live.count(p.get())) kept.push_back(std::move(p));
    pool_.swap(kept);
}

// MCTS._root_for, then _run_search's root handling (the history keys, a terminal root made searchable)
double Tree::begin(const Pos& rootpos, const std::vector<Pos>& history, bool reuse)
{
    finish();
    history_.clear();
    for (const Pos& h : history) history_.insert(make_key(h));
    Node* found = nullptr;
    if (reuse && root_ && root_->is_expanded) {
        const Key key = make_key(rootpos);
        if (root_->key == key) found = root_;
        if (!found) {
            for (Node* kid : root_->kids) {
                if (!kid) continue;
                if (kid->key == key) { found = kid; break; }
                if (!kid->is_expanded) continue;
                for (Node* g : kid->kids)
                    if (g && g->key == key) { found = g; break; }
                if (found) break;
            }
        }
    }
    double carried = 0.0;
    if (found) {
        carried = found->sum_n;
        keep_only(found);
        root_ = found;
        st_.reused++;
    } else {
        pool_.clear();
        root_ = new_node(rootpos);
    }
    if (root_->has_term && !root_->moves.empty()) root_->has_term = false;
    return carried;
}

// The ponder discount. N, W and V2 are scaled in float32 (numpy's in-place multiply by a Python float, NEP 50),
// sum_n by Python's round (half to even), and the prior's weight with them under discount_q0.
void Tree::discount(double d, bool q0)
{
    if (!root_ || !root_->is_expanded) return;
    const float df = (float)d;
    for (float& x : root_->N) x = x * df;
    for (float& x : root_->W) x = x * df;
    for (float& x : root_->V2) x = x * df;
    root_->sum_n = std::nearbyint(root_->sum_n * d);
    if (q0 && d > 0.0) root_->q0w *= d;
    st_.discounted++;
}

bool Tree::needs_eval() const { return root_ && !root_->is_expanded && !root_->has_term; }

void Tree::expand_root()
{
    finish();
    if (s_.prof) pt_ = PClock::now();
    Out out;
    int wide;
    std::vector<Node*> one(1, root_);
    evaluate(one, out, wide);
    expand(root_, out.pol.data(), (double)out.val[0], out.qv.data(), (int)root_->moves.size());
}

// a FEN for python-chess (the raw en passant square, as the node's board holds it)
static std::string to_fen(const Pos& p)
{
    std::string s;
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            const int sq = rank * 8 + file;
            const u64 bb = 1ULL << sq;
            char c = 0;
            if (p.P & bb) c = 'p';
            else if (p.N & bb) c = 'n';
            else if (p.B & bb) c = 'b';
            else if (p.R & bb) c = 'r';
            else if (p.Q & bb) c = 'q';
            else if (p.K & bb) c = 'k';
            if (!c) { empty++; continue; }
            if (empty) { s += char('0' + empty); empty = 0; }
            if (p.co[1] & bb) c = (char)(c - 'a' + 'A');
            s += c;
        }
        if (empty) s += char('0' + empty);
        if (rank) s += '/';
    }
    s += p.turn ? " w " : " b ";
    std::string cr;
    const u64 c = p.castling;
    if (c & (1ULL << 7)) cr += 'K';
    if (c & (1ULL << 0)) cr += 'Q';
    if (c & (1ULL << 63)) cr += 'k';
    if (c & (1ULL << 56)) cr += 'q';
    s += cr.empty() ? "-" : cr;
    s += ' ';
    if (p.ep >= 0) { s += char('a' + p.ep % 8); s += char('1' + p.ep / 8); } else s += '-';
    s += ' ' + std::to_string(p.hm) + ' ' + std::to_string(p.fullmove);
    return s;
}

// MCTS._tb_value's own first tests (at most tb_pieces pieces, no castling rights), then the Python probe
bool Tree::tb_value(const Node* n, double& v)
{
    const Pos& p = n->pos;
    const u64 occ = p.co[0] | p.co[1];
    int pieces = 0;
    for (u64 b = occ; b; b &= b - 1) pieces++;
    if (pieces > s_.tb_pieces || p.castling) return false;
    const CacheKey ck{n->key, p.hm};
    auto it = tb_.find(ck);
    if (it != tb_.end()) {
        if (it->second != it->second) return false;
        v = it->second;
        return true;
    }
    double got = 0.0;
    const bool hit = net_->tb(to_fen(p), got);
    tb_[ck] = hit ? got : NAN;
    if (hit) v = got;
    return hit;
}

static double now_s()
{
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// One pass of _run_search's plain loop without the pipeline. Gather, one forward pass, expand and back up, give
// back the held collisions. A failed forward pass gives back the batch exactly (MCTS._abandon) and rethrows.
int Tree::step(double target, double deadline)
{
    if (s_.pipeline) return step_pipe(target, deadline);
    const bool pf = s_.prof;
    PClock::time_point t0;
    PClock::time_point& t = pt_;
    if (pf) { t0 = t = PClock::now(); pf_.steps++; }
    Node* root = root_;
    const bool hold = s_.collide == 2;
    const bool twice = !s_.fill_once;
    std::vector<Pending> pending;
    std::vector<Node*> leaves;
    std::vector<std::pair<Path, Node*>> collided;
    collided_ = s_.collide ? &collided : nullptr;
    int tries = 0;
    const double spent_before = root->sum_n;
    const long long served_before = st_.cache_hits;
    Path path;
    try {
        while ((int)leaves.size() < s_.batch && tries < 4 * s_.batch
               && root->sum_n + (twice ? (double)leaves.size() : 0.0) - (hold ? (double)collided.size() : 0.0) < target) {
            if (deadline >= 0.0 && tries && now_s() >= deadline) break;
            tries++;
            Node* parent = nullptr;
            int a = -1;
            if (!descend(root, path, parent, a)) { if (pf) tick(t, pf_.descend); continue; }
            if (pf) tick(t, pf_.descend);
            gather(path, parent, a, pending, leaves);
            if (pf) tick(t, pf_.leaf);
        }
    } catch (...) {
        // a failure in the walk is a bug, so the tree is dropped, as the Python drops it
        collided_ = nullptr;
        root_ = nullptr;
        pool_.clear();
        throw;
    }
    if (leaves.empty()) {
        for (auto& c : collided) unwind(c.first);
        collided.clear();
        collided_ = nullptr;
        if (pf) { tick(t, pf_.expand); tick(t0, pf_.step); }
        if (tries >= 4 * s_.batch
            && (!s_.cache_progress || st_.cache_hits == served_before || root->sum_n <= spent_before))
            return 2;
        return 1;
    }
    Out out;
    int wide = 0;
    size_t done = 0;
    try {
        evaluate(leaves, out, wide);
        for (size_t i = 0; i < pending.size(); i++) {
            Pending& pd = pending[i];
            Node* kid = pd.kid;
            kid->pending = false;
            const double v = (double)out.val[i];
            const float* row = &out.pol[i * (size_t)wide];
            const float* qrow = &out.qv[i * (size_t)wide];
            const int k = (int)kid->moves.size();
            backup(pd.path, expand(kid, row, v, qrow, k));
            done = i + 1;
            if (!hold) {
                std::vector<std::pair<Path, Node*>> keep;
                for (auto& c : collided) {
                    if (c.second == kid) backup(c.first, kid->value);
                    else keep.push_back(c);
                }
                collided.swap(keep);
            }
            if (cache_.size() < s_.cache_size) {
                CacheEntry e;
                e.logits.assign(row, row + k);
                e.raw = v;
                e.q0.assign(qrow, qrow + k);
                cache_[pd.key] = std::move(e);
            }
        }
    } catch (...) {
        // MCTS._abandon gives every held visit back and detaches the leaves never evaluated
        for (auto& c : collided) unwind(c.first);
        for (size_t i = done; i < pending.size(); i++) {
            Pending& pd = pending[i];
            pd.kid->pending = false;
            unwind(pd.path);
            const Step& last = pd.path.back();
            if (last.node->kids[last.a] == pd.kid) last.node->kids[last.a] = nullptr;
        }
        collided_ = nullptr;
        throw;
    }
    for (auto& c : collided) unwind(c.first);
    collided.clear();
    collided_ = nullptr;
    if (pf) { tick(t, pf_.expand); tick(t0, pf_.step); }
    return 0;
}

// MCTS._pipe_step, one cycle of the pipelined search (Settings::pipeline). With batch A on the card, gather
// batch B, send B, read A back, then expand and back up A while the card runs B. B goes to the card before A
// is read (the Python reads A first). The tree does the same things in the same order, but the card finds B
// queued behind A and never waits for the host between them. Returns 0 when a batch went to the card or A was
// backed up, else the plain loop's 1 or 2. Held collisions (collide 2) are given back when their batch returns.
// Under collide 1 a B walk that collided with one of A's leaves is backed up when A returns. A failure while
// gathering drops the tree, as the plain loop does, and one after it gives back every visit still waiting on
// the card, A's and B's.
int Tree::step_pipe(double target, double deadline)
{
    const bool pf = s_.prof;
    PClock::time_point t0;
    PClock::time_point& t = pt_;
    if (pf) { t0 = t = PClock::now(); pf_.steps++; }
    Node* root = root_;
    const bool hold = s_.collide == 2;
    const bool twice = !s_.fill_once;
    const int bsz = s_.pipeline == 1 ? s_.batch : std::max(1, s_.batch / 2);
    std::vector<Pending> pending;
    std::vector<Node*> leaves;
    std::vector<std::pair<Path, Node*>> collided;
    collided_ = s_.collide ? &collided : nullptr;
    int tries = 0;
    const double spent_before = root->sum_n;
    const long long served_before = st_.cache_hits;
    // A's held collisions sit on the root's count as B's would, and neither is a visit
    const size_t held = (fly_.live && hold) ? fly_.collided.size() : 0;
    const bool had_a = fly_.live;
    Path path;
    try {
        while ((int)leaves.size() < bsz && tries < 4 * bsz
               && root->sum_n + (twice ? (double)leaves.size() : 0.0)
                  - (hold ? (double)(collided.size() + held) : 0.0) < target) {
            if (deadline >= 0.0 && tries && now_s() >= deadline) break;
            tries++;
            Node* parent = nullptr;
            int a = -1;
            if (!descend(root, path, parent, a)) { if (pf) tick(t, pf_.descend); continue; }
            if (pf) tick(t, pf_.descend);
            gather(path, parent, a, pending, leaves);
            if (pf) tick(t, pf_.leaf);
        }
    } catch (...) {
        // a failure in the walk is a bug, so the tree is dropped, the card's batch with it
        collided_ = nullptr;
        drop_flight();
        root_ = nullptr;
        pool_.clear();
        throw;
    }
    collided_ = nullptr;
    bool a_open = had_a, sent = false;
    int queued = had_a ? 1 : 0;         // batches the net holds for us
    try {
        int wide = 0;
        if (!leaves.empty()) {
            Batch in;
            prepare(leaves, in, wide);
            if (pf) tick(t, pf_.encode);
            net_->submit(in);
            sent = true;
            queued++;
            st_.evals += in.B;
            st_.batches += 1;
            if (pf) tick(t, pf_.submit);
        }
        if (had_a) {
            Out out;
            queued--;
            net_->collect(out);
            const size_t B = fly_.pending.size(), W = (size_t)fly_.wide;
            if (out.pol.size() != B * W || out.val.size() != B || out.qv.size() != B * W)
                throw std::runtime_error("the network callback returned the wrong sizes");
            if (pf) tick(t, pf_.collect);
            a_open = false;             // complete() gives back whatever it cannot finish
            Flight a = std::move(fly_);
            fly_ = Flight();
            complete(a, out, collided);
            if (pf) tick(t, pf_.expand);
        }
        if (sent) {
            fly_.live = true;
            fly_.wide = wide;
            fly_.pending = std::move(pending);
            fly_.collided = std::move(collided);
        }
    } catch (...) {
        if (a_open) abandon(fly_.pending, 0, fly_.collided);
        fly_ = Flight();
        abandon(pending, 0, collided);
        for (; queued > 0; queued--) {
            try { Out junk; net_->collect(junk); } catch (...) {}   // what the card still holds is read and dropped
        }
        throw;
    }
    if (sent) {
        if (pf) tick(t0, pf_.step);
        return 0;
    }
    // nothing went to the card, so whatever collided has nothing left to wait for
    for (auto& c : collided) unwind(c.first);
    if (pf) { tick(t, pf_.expand); tick(t0, pf_.step); }
    if (had_a) return 0;                // A's leaves were backed up, which is progress
    // a cached leaf is work, not saturation (the plain loop's test)
    if (tries >= 4 * bsz && (!s_.cache_progress || st_.cache_hits == served_before || root->sum_n <= spent_before))
        return 2;
    return 1;
}

// MCTS._pipe_complete. Expand and back up a batch the card has answered, exactly as the plain loop does its own.
// `later` is the collided list of the batch gathered after it, whose walks may be waiting on this batch's leaves.
void Tree::complete(Flight& f, const Out& out, std::vector<std::pair<Path, Node*>>& later)
{
    const bool hold = s_.collide == 2;
    size_t done = 0;
    try {
        for (size_t i = 0; i < f.pending.size(); i++) {
            Pending& pd = f.pending[i];
            Node* kid = pd.kid;
            kid->pending = false;
            const double v = (double)out.val[i];
            const float* row = &out.pol[i * (size_t)f.wide];
            const float* qrow = &out.qv[i * (size_t)f.wide];
            const int k = (int)kid->moves.size();
            backup(pd.path, expand(kid, row, v, qrow, k));
            done = i + 1;
            if (!hold) {
                std::vector<std::pair<Path, Node*>>* lists[2] = {&f.collided, &later};
                for (auto* lst : lists) {
                    std::vector<std::pair<Path, Node*>> keep;
                    for (auto& c : *lst) {
                        if (c.second == kid) backup(c.first, kid->value);
                        else keep.push_back(c);
                    }
                    lst->swap(keep);
                }
            }
            if (cache_.size() < s_.cache_size) {
                CacheEntry e;
                e.logits.assign(row, row + k);
                e.raw = v;
                e.q0.assign(qrow, qrow + k);
                cache_[pd.key] = std::move(e);
            }
        }
    } catch (...) {
        abandon(f.pending, done, f.collided);
        throw;
    }
    for (auto& c : f.collided) unwind(c.first);
    f.collided.clear();
}

// MCTS._abandon. Every held visit goes back and the leaves never evaluated (from `from` on) are detached.
void Tree::abandon(std::vector<Pending>& pending, size_t from, std::vector<std::pair<Path, Node*>>& collided)
{
    for (auto& c : collided) unwind(c.first);
    collided.clear();
    for (size_t i = from; i < pending.size(); i++) {
        Pending& pd = pending[i];
        pd.kid->pending = false;
        unwind(pd.path);
        const Step& last = pd.path.back();
        if (last.node->kids[last.a] == pd.kid) last.node->kids[last.a] = nullptr;
    }
}

// MCTS._pipe_finish. The batch still on the card when the search ends has its visits counted already, so it is
// read back and backed up rather than thrown away.
void Tree::finish()
{
    if (!fly_.live) return;
    Flight a = std::move(fly_);
    fly_ = Flight();
    Out out;
    try {
        net_->collect(out);
        const size_t B = a.pending.size(), W = (size_t)a.wide;
        if (out.pol.size() != B * W || out.val.size() != B || out.qv.size() != B * W)
            throw std::runtime_error("the network callback returned the wrong sizes");
    } catch (...) {
        abandon(a.pending, 0, a.collided);
        throw;
    }
    std::vector<std::pair<Path, Node*>> none;
    complete(a, out, none);
}

double Tree::held() const
{
    return (fly_.live && s_.collide == 2) ? (double)fly_.collided.size() : 0.0;
}

// the batch on the card read and dropped, its nodes left as they are (reset frees them)
void Tree::drop_flight()
{
    if (!fly_.live) return;
    fly_ = Flight();
    try { Out junk; net_->collect(junk); } catch (...) {}
}

int Tree::choose()
{
    Node* root = root_;
    if (!root || root->moves.empty() || !root->is_expanded) return -1;
    float total = 0.0f;
    for (float x : root->N) total += x;
    if (s_.prior_fallback && total <= 0.0f) return argmax_first(root->P);
    if (s_.lcb) return lcb_pick(root);
    return argmax_first(root->N);
}

std::vector<int> Tree::pv(int first, int depth) const
{
    std::vector<int> out;
    Node* node = root_;
    while (node && node->is_expanded && node->sum_n > 0 && (int)out.size() < depth) {
        int a = (node == root_ && first >= 0) ? first : argmax_first(node->N);
        if (node->N[a] < 1.0f) break;
        out.push_back(node->moves[a]);
        node = node->kids[a];
    }
    return out;
}

long long Tree::tree_nodes() const
{
    if (!root_) return 0;
    if (root_->kids.empty()) return 1;
    long long t = 1;
    for (Node* k : root_->kids)
        if (k) t += 1 + (long long)k->sum_n;
    return t;
}

Node* Tree::new_node(const Pos& p)
{
    std::unique_ptr<Node> n(new Node());
    n->pos = p;
    n->key = make_key(p);
    if (!legal_moves(n->key, n->moves, n->IX, n->flags))
        throw std::runtime_error("move generator overflow (python falls back to python-chess here)");
    if (n->moves.empty()) {
        n->has_term = true;
        n->term = (n->flags & 1) ? -1.0 : 0.0;
    } else if ((n->flags & 2) || p.hm >= s_.draw_plies) {
        n->has_term = true;
        n->term = 0.0;
    }
    Node* r = n.get();
    pool_.push_back(std::move(n));
    return r;
}

// MCTS._prepare, the encoding and the legal-move index rows, padded to the widest row
void Tree::prepare(const std::vector<Node*>& nodes, Batch& in, int& wide)
{
    const int B = (int)nodes.size();
    in.B = B;
    wide = 0;
    for (Node* n : nodes) wide = std::max(wide, (int)n->moves.size());
    in.wide = wide;
    in.ids.assign((size_t)B * 64, 0);
    in.cast.assign((size_t)B * 4, 0.0f);
    in.hm.assign(B, 0);
    in.ep.assign(B, 0);
    in.I.assign((size_t)B * wide, 0);
    in.M.assign((size_t)B * wide, 0);
    in.widths.resize(B);
    for (int b = 0; b < B; b++) {
        Node* n = nodes[b];
        encode(n->key, n->pos.hm, &in.ids[(size_t)b * 64], &in.cast[(size_t)b * 4], &in.hm[b], &in.ep[b]);
        const int k = (int)n->moves.size();
        in.widths[b] = k;
        for (int i = 0; i < k; i++) {
            in.I[(size_t)b * wide + i] = n->IX[i];
            in.M[(size_t)b * wide + i] = 1;
        }
    }
}

// MCTS._prepare, _dispatch and _collect
void Tree::evaluate(const std::vector<Node*>& nodes, Out& out, int& wide)
{
    Batch in;
    prepare(nodes, in, wide);
    if (s_.prof) tick(pt_, pf_.encode);
    const int B = in.B;
    out = Out();
    net_->evaluate(in, out);
    if (s_.prof) tick(pt_, pf_.net);
    if (out.pol.size() != (size_t)B * wide || out.val.size() != (size_t)B || out.qv.size() != (size_t)B * wide)
        throw std::runtime_error("the network callback returned the wrong sizes");
    st_.evals += B;
    st_.batches += 1;
}

// MCTS._expand with walk.c's expand_rows (the float64 softmax prior, numpy's pairwise sums)
double Tree::expand(Node* node, const float* logits, double value, const float* q0, int nq)
{
    const size_t k = node->moves.size();
    std::vector<double> z(k);
    for (size_t i = 0; i < k; i++) z[i] = (double)logits[i];
    if (!std::isfinite(0.0 + pairwise(z.data(), k))) {
        // as numpy guards it, an unreadable logit gets no prior and a row with nothing readable is uniform
        bool bad = false, any = false;
        for (double v : z) if (std::isnan(v) || (std::isinf(v) && v > 0)) bad = true;
        if (bad) {
            for (double& v : z) if (!std::isfinite(v)) v = -INFINITY;
            for (double v : z) if (std::isfinite(v)) any = true;
            if (!any) for (double& v : z) v = 0.0;
        }
    }
    if (s_.policy_temp != 1.0)
        for (size_t i = 0; i < k; i++) z[i] /= s_.policy_temp;
    double mx = z[0];
    for (size_t i = 1; i < k; i++)
        if (z[i] > mx) mx = z[i];
#pragma loop(no_vector)
    for (size_t i = 0; i < k; i++) z[i] = std::exp(z[i] - mx);
    const double sum = 0.0 + pairwise(z.data(), k);
    node->P.resize(k);
    for (size_t i = 0; i < k; i++) node->P[i] = (float)(z[i] / sum);
    if (!std::isfinite(value)) value = 0.0;
    node->Q0.assign(q0, q0 + nq);
    for (float& q : node->Q0) {
        if (std::isnan(q)) q = 0.0f;
        else if (std::isinf(q)) q = q > 0 ? 1.0f : -1.0f;
    }
    node->N.assign(k, 0.0f);
    node->W.assign(k, 0.0f);
    node->V2.assign(k, 0.0f);
    node->F.assign(k, 0.0f);
    node->X.assign(k, 0.0f);
    node->kids.assign(k, nullptr);
    node->value = value;
    node->is_expanded = true;
    return value;
}

// walk.c's backup, W[a] += v - x, V2[a] += v*v, F[a] -= 1, X[a] -= x, the sign turning at every step
void Tree::backup(const Path& path, double leaf)
{
    double v = -leaf;
    for (size_t i = path.size(); i-- > 0;) {
        const Step& st = path[i];
        if (st.apart) { v = -v; continue; }
        Node* nd = st.node;
        const int a = st.a;
        const double x = st.x;
        nd->W[a] = nd->W[a] + (float)(v - x);
        nd->V2[a] = nd->V2[a] + (float)(v * v);
        nd->F[a] = nd->F[a] - 1.0f;
        nd->X[a] = nd->X[a] - (float)x;
        v = -v;
    }
}

// MCTS._unwind
void Tree::unwind(const Path& path)
{
    for (const Step& st : path) {
        Node* nd = st.node;
        if (st.apart) { nd->sum_n -= 1.0; continue; }
        const int a = st.a;
        const float x = (float)st.x;
        nd->N[a] = nd->N[a] - 1.0f;
        nd->W[a] = nd->W[a] - x;
        nd->sum_n -= 1.0;
        nd->F[a] = nd->F[a] - 1.0f;
        nd->X[a] = nd->X[a] - x;
    }
}

bool Tree::repeats(const Key& k, const Path& path) const
{
    if (history_.count(k)) return true;
    for (const Step& st : path)
        if (st.node->key == k) return true;
    return false;
}

// walk.c's descend, with _pick_fast's selection (puct_v with q_init, the virtual mean), the forcing guard at ply 1
// and forcing_apart. Returns true with (path, parent, a) at an unexpanded edge, false when the walk ended at
// a terminal (backed up) or a pending node (held or taken back).
bool Tree::descend(Node* root, Path& path, Node*& parent, int& a_out)
{
    path.clear();
    Node* node = root;
    const float c1f = (float)s_.c1, c2f = (float)(2.0 * s_.c2);
    const double eg = s_.explore_growth, eb = s_.explore_base;
    for (;;) {
        if (node->has_term) {
            backup(path, node->term);
            return false;
        }
        if (node->pending) {
            st_.collisions++;
            if (s_.collide && collided_) collided_->push_back(std::make_pair(path, node));
            else unwind(path);
            return false;
        }
        if (!node->is_expanded) throw std::runtime_error("descend: a live node that is not expanded");
        const size_t plen = path.size();
        const int k = (int)node->moves.size();
        float* n = node->N.data();
        float* w = node->W.data();
        const float* p = node->P.data();
        const float* q0 = node->Q0.data();
        const float* v2 = node->V2.data();
        const double n0 = s_.q_n0 * node->q0w;
        const double sum_n = node->sum_n;
        const float n0f = (float)n0;
        int a = -1;
        double xd = 0.0;
        bool forced = false;
        if (plen == 1 && s_.forcing > 0 && sum_n >= s_.forcing_after && k > 0) {
            if (!node->fm_ready) {
                if (!forcing_mask(node->key, node->FM)) throw std::runtime_error("forcing mask overflow");
                node->fm_ready = true;
            }
            const uint8_t* m = node->FM.data();
            const float lim = (float)s_.forcing;
            float least = 0.0f;
            for (int i = 0; i < k; i++)
                if (m[i] && n[i] < lim && (a < 0 || n[i] < least)) { a = i; least = n[i]; }
            if (a >= 0) {
                float t = n0f * q0[a];
                t = t + w[a];
                float den = n0f + n[a];
                xd = (double)(t / den);
                forced = true;
                st_.forced++;
            }
        }
        if (a < 0) {
            const double N = sum_n >= 1.0 ? sum_n : 1.0;
            const double grow = eg > 0.0 ? 1.0 + eg * std::log((N + eb) / eb) : 0.0;
            const float sqf = (float)std::sqrt(N), lgf = (float)std::log(N + 1.0), grf = (float)grow;
            const bool growing = grow > 0.0;
            int best = -1, first_nan = -1;
            float bestu = 0.0f, bestq = 0.0f, nanq = 0.0f;
            for (int i = 0; i < k; i++) {
                float t = n0f * q0[i];
                t = t + w[i];
                float den = n0f + n[i];
                float q = t / den;
                float sd;
                if (n[i] > 0.0f) {
                    float mean = w[i] / n[i];
                    float vv = v2[i] / n[i];
                    float mm = mean * mean;
                    vv = vv - mm;
                    vv = (vv < 0.0f) ? 0.0f : vv;
                    sd = std::sqrt(vv);
                } else {
                    sd = 1.0f;
                }
                float d = 1.0f + n[i];
                float t1 = c1f * p[i];
                t1 = t1 * sd;
                t1 = t1 * sqf;
                t1 = t1 / d;
                float t2 = c2f * p[i];
                t2 = t2 * lgf;
                t2 = t2 / d;
                float u = t1 + t2;
                if (growing) u = u * grf;
                u = u + q;
                if (u != u) {
                    if (first_nan < 0) { first_nan = i; nanq = q; }
                    continue;
                }
                if (best < 0 || u > bestu) { best = i; bestu = u; bestq = q; }
            }
            if (first_nan >= 0) { a = first_nan; xd = (double)nanq; }
            else { a = best; xd = (double)bestq; }
        }
        const float x = (float)xd;
        n[a] = n[a] + 1.0f;
        w[a] = w[a] + x;
        node->sum_n += 1.0;
        node->F[a] = node->F[a] + 1.0f;
        node->X[a] = node->X[a] + x;
        path.push_back(Step{node, a, xd, false});
        if (s_.forcing_apart && forced && plen + 1 == 2) {
            // kept apart from the root, so the root step's visit comes back out of its edge and root.sum_n keeps it
            Step& r = path[0];
            const float rxf = (float)r.x;
            Node* rn = r.node;
            rn->N[r.a] = rn->N[r.a] - 1.0f;
            rn->W[r.a] = rn->W[r.a] - rxf;
            rn->F[r.a] = rn->F[r.a] - 1.0f;
            rn->X[r.a] = rn->X[r.a] - rxf;
            r.apart = true;
            st_.forced_apart++;
        }
        Node* kid = node->kids[a];
        if (!kid) {
            parent = node;
            a_out = a;
            return true;
        }
        node = kid;
    }
}

// MCTS._gather without the graph and the tablebases
void Tree::gather(Path& path, Node* parent, int a, std::vector<Pending>& pending, std::vector<Node*>& leaves)
{
    Pos b = parent->pos;
    push(b, parent->moves[a]);
    Node* kid = new_node(b);
    parent->kids[a] = kid;
    if (!kid->has_term && repeats(kid->key, path)) {
        kid->has_term = true;
        kid->term = 0.0;
        st_.repetitions++;
    }
    if (!kid->has_term && s_.use_tb) {
        double tv;
        if (tb_value(kid, tv)) {
            kid->has_term = true;
            kid->term = tv;
            st_.tablebase++;
        }
    }
    if (kid->has_term) {
        backup(path, kid->term);
        return;
    }
    CacheKey ck{kid->key, std::min(kid->pos.hm, 100)};
    auto it = cache_.find(ck);
    if (it != cache_.end()) {
        st_.cache_hits++;
        const CacheEntry& e = it->second;
        backup(path, expand(kid, e.logits.data(), e.raw, e.q0.data(), (int)e.q0.size()));
        return;
    }
    kid->pending = true;
    pending.push_back(Pending{path, kid, ck});
    leaves.push_back(kid);
}


int Tree::lcb_pick(Node* root)
{
    const int k = (int)root->moves.size();
    std::vector<double> n(k);
    double nmax = 0.0;
    for (int i = 0; i < k; i++) n[i] = (double)root->N[i];
    const int top = argmax_first(n);
    nmax = n[top];
    const double floor = std::max(2.0, s_.lcb_floor * nmax);
    std::vector<int> c;
    for (int i = 0; i < k; i++)
        if (n[i] >= floor) c.push_back(i);
    if ((int)c.size() < 2) return top;
    std::vector<double> lcb(c.size());
    for (size_t j = 0; j < c.size(); j++) {
        const double nn = n[c[j]];
        const double mean = (double)root->W[c[j]] / nn;
        double var = (double)root->V2[c[j]] - nn * mean * mean;
        var = (var < 0.0) ? 0.0 : var;
        var = var / (nn - 1.0);
        lcb[j] = mean - t_quantile(s_.lcb_alpha, nn - 1.0) * std::sqrt(var / nn);
    }
    const int best = c[argmax_first(lcb)];
    if (best != top) st_.lcb_overrode++;
    return best;
}

// ------------------------------------------------------------------ Student-t quantile (scipy t.isf)
// The regularized incomplete beta by Lentz's continued fraction, then a bracketed Newton solve on the upper
// tail. Not scipy's (Boost's) code, so it can differ from it in the last bits. It only ranks the root moves'
// lower bounds, where a difference that small cannot change which is largest unless two are tied to 1e-13.
static double betacf(double a, double b, double x)
{
    const double EPS = 1e-16, FPMIN = 1e-300;
    double qab = a + b, qap = a + 1.0, qam = a - 1.0, c = 1.0, d = 1.0 - qab * x / qap;
    if (std::fabs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= 200000; m++) {
        const int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if (std::fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if (std::fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < EPS) break;
    }
    return h;
}

static double betai(double a, double b, double x)
{
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    const double lbt = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log1p(-x);
    const double bt = std::exp(lbt);
    if (x < (a + 1.0) / (a + b + 2.0)) return bt * betacf(a, b, x) / a;
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

static double t_sf(double t, double df)   // P(T > t), t >= 0
{
    return 0.5 * betai(0.5 * df, 0.5, df / (df + t * t));
}

static double t_pdf(double t, double df)
{
    return std::exp(std::lgamma(0.5 * (df + 1.0)) - std::lgamma(0.5 * df) - 0.5 * std::log(df * 3.14159265358979323846)
                    - 0.5 * (df + 1.0) * std::log1p(t * t / df));
}

double t_quantile(double alpha, double df)
{
    if (!(df > 0.0) || !(alpha > 0.0) || !(alpha < 0.5)) return NAN;
    if (df == 1.0) return 1.0 / std::tan(3.14159265358979323846 * alpha);
    if (df == 2.0) return (1.0 - 2.0 * alpha) / std::sqrt(2.0 * alpha * (1.0 - alpha));
    double lo = 0.0, hi = 4.0;
    while (t_sf(hi, df) > alpha) hi *= 2.0;
    double t = 0.5 * (lo + hi);
    for (int it = 0; it < 200; it++) {
        const double f = t_sf(t, df) - alpha;   // decreasing in t
        if (f > 0) lo = t; else hi = t;
        double tn = t + f / t_pdf(t, df);
        if (!(tn > lo && tn < hi)) tn = 0.5 * (lo + hi);
        if (std::fabs(tn - t) <= 1e-15 * t) { t = tn; break; }
        t = tn;
    }
    return t;
}

}  // namespace mayfly2
