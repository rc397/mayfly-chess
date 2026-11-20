// search.h: Midge's alpha-beta search and the veto. Written from scratch, 17 Nov 2026.
// PVS with iterative deepening and aspiration windows, a transposition table, quiescence, null move, late move
// reductions, futility and late move pruning, SEE; killers, history and counter moves for ordering.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>
#include <vector>
#include "chess.h"
#include "nnue.h"

const int INF = 32000, MATE = 31000, MATE_BOUND = MATE - 512, MAXPLY = 128;
enum { B_UPPER = 1, B_LOWER = 2, B_EXACT = 3 };
enum { EV_NNUE = 0, EV_HAND = 1, EV_MATERIAL = 2 };

struct TTData { int16_t score, eval; Move move; uint8_t depth, bg; };   // bg = bound | generation << 2
struct TTEntry { U64 keyx, data; };                                     // keyx = key ^ data

struct TTable {
  TTEntry *t = nullptr;
  U64 mask = 0;
  uint8_t gen = 0;
  // TTWays (19 Nov 2026; idea known from Stockfish, our own implementation). 1: one entry a slot, as before.
  // 4: four neighbouring entries (one cache line) share a slot; a new position takes the place of the entry
  // worth least, by its depth less four for every search since it was last used.
  int ways = 1;
  void release() { if (t) ::operator delete[](t, std::align_val_t(64)); t = nullptr; }
  ~TTable() { release(); }
  void resize(size_t mb) {
    size_t n = 1;
    while (n * 2 * sizeof(TTEntry) <= (mb << 20)) n *= 2;
    release();
    // not enough memory: half, and again. Lines of 64 bytes, so four entries sit in one cache line.
    while (!(t = (TTEntry *)::operator new[](n * sizeof(TTEntry), std::align_val_t(64), std::nothrow)) && n > 1024) n /= 2;
    mask = n - 1; clear();
  }
  int clear_threads = 1;   // set from the Threads option: a table of many GB is cleared in parts
  void clear() {
    size_t n = mask + 1;
    int th = n >= (size_t(1) << 24) ? std::min(clear_threads, 32) : 1;   // 256 MB and more
    if (th <= 1) { memset((void *)t, 0, n * sizeof(TTEntry)); return; }
    std::vector<std::thread> ts;
    for (int i = 0; i < th; i++) {
      size_t a = n / th * i, b = i == th - 1 ? n : n / th * (i + 1);
      ts.emplace_back([this, a, b] { memset((void *)(t + a), 0, (b - a) * sizeof(TTEntry)); });
    }
    for (auto &x : ts) x.join();
  }
  // entries of the current search among the first 1000, as the UCI hashfull number
  int hashfull() const {
    int c = 0;
    for (size_t i = 0; i < 1000 && i <= mask; i++) {
      TTData d; U64 dd = t[i].data; memcpy(&d, &dd, 8);
      c += dd != 0 && (d.bg >> 2) == (gen & 63);
    }
    return c;
  }
  void prefetch(U64 k) const { __builtin_prefetch(&t[k & mask & ~U64(ways - 1)]); }   // the entry a new position will ask for
  TTEntry *probe4(U64 k, bool &hit, TTData &d) {
    TTEntry *g = &t[k & mask & ~U64(3)], *victim = g;
    int least = 1 << 30;
    for (int i = 0; i < 4; i++) {
      U64 kx = g[i].keyx, dd = g[i].data;
      TTData x;
      memcpy(&x, &dd, 8);
      if (dd && (kx ^ dd) == k) {
        if ((x.bg >> 2) != (gen & 63)) {   // used again: it counts as this search's
          x.bg = uint8_t((x.bg & 3) | ((gen & 63) << 2));
          memcpy(&dd, &x, 8);
          g[i].data = dd; g[i].keyx = k ^ dd;
        }
        hit = true; d = x;
        return &g[i];
      }
      int keep = dd ? int(x.depth) - 4 * int((gen - (x.bg >> 2)) & 63) : -100000;
      if (keep < least) { least = keep; victim = &g[i]; }
    }
    hit = false;
    U64 dd = victim->data;
    memcpy(&d, &dd, 8);
    return victim;
  }
  TTEntry *probe(U64 k, bool &hit, TTData &d) {
    if (ways == 4) return probe4(k, hit, d);
    TTEntry *e = &t[k & mask];
    U64 kx = e->keyx, dd = e->data;
    hit = (kx ^ dd) == k;
    memcpy(&d, &dd, 8);
    return e;
  }
  void store(TTEntry *e, U64 k, int score, int eval, Move m, int depth, int bound) {
    U64 kx = e->keyx, dd = e->data;
    TTData o;
    memcpy(&o, &dd, 8);
    U64 ok = kx ^ dd;
    bool same = ok == k;
    // an entry of this search that is much deeper stays, unless the new one is exact
    // (with four entries a slot: only the same position's own deeper entry; another position's was chosen to go)
    if (dd && (o.bg >> 2) == (gen & 63) && depth + 4 < o.depth && bound != B_EXACT && (ways == 1 || same)) return;
    TTData n;
    n.move = (!same || m) ? m : o.move;
    n.score = int16_t(score); n.eval = int16_t(eval); n.depth = uint8_t(std::max(depth, 0));
    n.bg = uint8_t(bound | ((gen & 63) << 2));
    U64 nd;
    memcpy(&nd, &n, 8);
    e->data = nd; e->keyx = k ^ nd;
  }
};

inline int to_tt(int s, int ply) { return s >= MATE_BOUND ? s + ply : s <= -MATE_BOUND ? s - ply : s; }
inline int from_tt(int s, int ply) { return s >= MATE_BOUND ? s - ply : s <= -MATE_BOUND ? s + ply : s; }

const int SEEV[6] = {100, 320, 330, 500, 900, 10000};

struct Limits {
  int depth = MAXPLY - 1;
  U64 nodes = 0;
  int64_t movetime = 0, wtime = -1, btime = -1, winc = 0, binc = 0;
  int movestogo = 0;
  bool infinite = false, ponder = false;
  std::vector<Move> searchmoves;
};

struct Search {
  Position *pos = nullptr;
  nnue::AccStack *acc = new nnue::AccStack;
  TTable *tt = nullptr;   // shared by all threads of one search
  int eval_mode = EV_NNUE, qchecks = 0, singular = 1, contw = 1, corrw = 0, timescale = 160;
  int rfp_m = 80, fut_b = 100, fut_m = 90, lmp_b = 3;   // pruning margins (UCI options)
  // search test switches, off (0) unless a match kept them (SeeQuiet 50 was kept: +22 Elo, +8 to +35, 800 games)
  int nmp = 1;       // NullMove: the null-move cut (off only for tests)
  int nmpv = 0;      // NmpVerify: from this depth a null-move cutoff is checked by a search without null moves
  int nmp_min_ply = 0;   // inside a verification search: no null move above this ply
  int seeq = 50;     // SeeQuiet: at shallow depth skip a quiet move that loses more than seeq x depth by exchange
  int tmstab = 0;    // TmStab: think shorter when the best move has stayed the same, longer when it changed
  int singx = 0;     // SingularX: two plies when the table move is far ahead; one less when another move is close
  int caph = 0;      // CapHist: captures that caused cutoffs before are tried earlier
  int probcut = 0;   // ProbCut: a good capture that still beats beta + probcut at reduced depth cuts the node
  int chist[12][64][6] = {};   // capture history: moving piece, target square, captured type
  // ---- 19 Nov 2026: ideas known from Stockfish (notes/stockfish_gap_report.md), our own implementation and our
  // own first numbers; every one is off at 0 and the search is then the same as before ----
  int histp = 0;     // HistPrune: at depth 4 and less a plain quiet move whose history is below -histp x depth is skipped
  int phist = 0;     // PawnHist: quiet moves also get a history by pawn structure (piece, target square)
  int cont4 = 0;     // Cont4: the continuation history also looks four plies back (at half weight)
  int prevb = 0;     // PriorBonus: when a node fails low, the quiet move that led to it is rewarded
  int lmrx = 0;      // LmrX, one bit each: 1 one more when the table move is a capture; 2 one more once three of
                     // this node's moves were refuted by a cutoff; 4 after a reduced search beats alpha the full
                     // search goes a ply deeper or shallower by how far it beat it; 8 one more when the static
                     // score is not improving; 16 losing captures are reduced too (half the table's value)
  int corrx = 0;     // CorrX: percent of the mean of four corrections (pawns, white's pieces, black's pieces, the
                     // last two moves), each the running mean of search result minus the network's own score
  int ttfirst = 0;   // TtFirst (our own, 19 Nov 2026): the table's move is searched before the other moves get their
                     // ordering scores; a cutoff there saves the scoring, and the others are scored with fresher history
  int tmx = 0;       // TmX: thinking time by the best move's share of the nodes, its stability and a falling score
  std::vector<int> ph;          // pawn history [pawn key & 1023][piece][to], made when the switch is on
  int cnp[2][2][16384] = {};    // corrections by the pieces of white and of black, for each side to move
  std::vector<int> ccorr;       // corrections by the last two moves (piece and target of each)
  int rawsev[MAXPLY + 2] = {};  // the network's own score at each ply, before any correction
  int cuts[MAXPLY + 4] = {};    // cutoffs so far among the children of the node one ply up
  bool wasq[MAXPLY + 2] = {};   // was the move played at this ply quiet
  U64 rootn[64][64] = {};       // nodes spent under each root move in this search (main thread)
  int last_score = 0; bool have_last = false;   // the score this side's last search ended with
  void prep() {
    if (phist && ph.empty()) ph.assign(1024 * 768, 0);
    if (corrx && ccorr.empty()) ccorr.assign(768 * 768, 0);
  }
  int &phv(int pc, int to) { return ph[((pos->pkey & 1023) * 12 + pc) * 64 + to]; }
  int *ccp(int ply) {
    if (ply < 2 || !played[ply - 1] || !played[ply - 2]) return nullptr;
    return &ccorr[(mpc[ply - 1] * 64 + mto(played[ply - 1])) * 768 + mpc[ply - 2] * 64 + mto(played[ply - 2])];
  }
  // CorrX: the mean of the four tables, in centipawns, times corrx percent
  int correction(int ply) {
    int us = pos->side, sum = corr[us][pos->pkey & 16383] + cnp[us][0][pos->npkey[0] & 16383] + cnp[us][1][pos->npkey[1] & 16383], n = 3;
    if (int *c = ccp(ply)) { sum += *c; n++; }
    return sum / n * corrx / 25600;
  }
  int corr[2][16384] = {};   // search result minus static eval, by pawn structure, in 1/256 cp
  Move excl[MAXPLY + 2] = {};   // a move left out of this node (the singular test)
  bool silent = false;
  int rootdepth = 1;     // the depth of the current iteration
  std::atomic<bool> stop{false};
  std::atomic<U64> nodes{0};             // written by its own thread only; the main thread reads the helpers'
  U64 node_limit = 0;
  std::vector<Search *> *team = nullptr;   // the helper searches, for the node total
  int team_n = 0;
  U64 all_nodes() const { U64 n = nodes; for (int i = 0; team && i < team_n; i++) n += (*team)[i]->nodes; return n; }
  int seldepth = 0;
  int64_t soft_ms = 0, hard_ms = 0;
  int overhead = 50;                     // MoveOverhead: ms kept back for the move's way to the GUI
  std::atomic<bool> pondering{false};    // go ponder: no clock until ponderhit
  std::atomic<int64_t> start_us{0};      // when the clock of this search started
  bool timed = false;
  // Every table below starts at zero. A helper thread's Search is made on the heap, where memory is not zero by
  // itself: a release build with 2 threads read tables that were never set and crashed (found 21 Nov 2026).
  Move killers[MAXPLY + 2][2] = {};
  int hist[2][64][64] = {};
  // continuation history: (piece and target of the move 1 or 2 plies back) -> (piece and target of this move)
  std::vector<int> cont = std::vector<int>(12 * 64 * 12 * 64, 0);
  int mpc[MAXPLY + 2] = {};
  int &ch(int ppc, int pto, int pc, int to) { return cont[((ppc * 64 + pto) * 12 + pc) * 64 + to]; }
  int qhist(int ply, Move m, int pc) {
    int h = hist[pos->side][mfrom(m)][mto(m)];
    if (contw)
      for (int k = 1; k <= 2; k++)
        if (ply >= k && played[ply - k]) h += ch(mpc[ply - k], mto(played[ply - k]), pc, mto(m));
    if (cont4 && ply >= 4 && played[ply - 4]) h += ch(mpc[ply - 4], mto(played[ply - 4]), pc, mto(m)) / 2;
    if (phist) h += phv(pc, mto(m));
    return h;
  }
  Move counter[64][64] = {};
  Move pv[MAXPLY + 1][MAXPLY + 1] = {};
  int pvlen[MAXPLY + 1] = {};
  int sev[MAXPLY + 2] = {};
  Move played[MAXPLY + 2] = {};
  int lmr[64][64] = {};
  std::vector<Move> rootonly;
  int multipv = 1;               // MultiPV: how many root moves the main thread reports at each depth
  std::vector<Move> rootskip;    // root moves left out of this search (the better lines, for MultiPV)
  bool root_limited() const { return !rootonly.empty() || !rootskip.empty(); }
  Move root_best = NO_MOVE;
  int root_score = 0, done_depth = 0;

  // late move reduction table: base + ln(depth) ln(move number) / div
  void init_lmr(double base, double div) {
    for (int d = 0; d < 64; d++)
      for (int m = 0; m < 64; m++) lmr[d][m] = (d && m) ? int(base + std::log(d) * std::log(m) / div) : 0;
  }
  int start_depth = 1;   // helpers start deeper to spread the work
  bool helper = false;
  Search *lead = nullptr;   // a helper's main search
  // a new game: the move ordering tables start empty
  void new_game() {
    memset(hist, 0, sizeof hist); memset(counter, 0, sizeof counter); std::fill(cont.begin(), cont.end(), 0);
    memset(corr, 0, sizeof corr); memset(chist, 0, sizeof chist);
    std::fill(ph.begin(), ph.end(), 0); memset(cnp, 0, sizeof cnp); std::fill(ccorr.begin(), ccorr.end(), 0); have_last = false;
  }
  explicit Search(TTable *shared = nullptr) {
    tt = shared ? shared : new TTable;
    if (!shared) tt->resize(64);
    init_lmr(0.8, 2.3);
    memset(hist, 0, sizeof hist);
  }
  static int64_t now_us() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
  int64_t elapsed() const { return (now_us() - start_us) / 1000; }
  // the opponent played the expected move: the clock starts now
  void ponderhit() { start_us = now_us(); pondering = false; }
  // depth 1 is never cut short, so there is always a searched move to play
  bool halt() const { return stop && rootdepth > 1; }
  // set by a program that uses Midge as a library: when the flag it points at goes up, the look ends
  const std::atomic<bool> *hurry = nullptr;
  // a second flag for the library: it ends a look only once the look has searched hurry2_min nodes
  const std::atomic<bool> *hurry2 = nullptr;
  U64 hurry2_min = 0;
  bool out_of_time() const { return timed && !pondering && elapsed() >= hard_ms; }
  void check_time() {
    if (lead) {
      // a helper watches the main search's clock and stop flag. With more threads than CPUs the main thread
      // can be off the CPU when its time runs out; whichever thread is running then ends the search for all.
      if (lead->stop || lead->out_of_time()) { lead->stop = true; stop = true; }
      return;
    }
    if (out_of_time() || (node_limit && nodes >= node_limit)) stop = true;
    if (hurry && hurry->load(std::memory_order_relaxed)) stop = true;
    if (hurry2 && nodes >= hurry2_min && hurry2->load(std::memory_order_relaxed)) stop = true;
  }
  int evaluate() {
    // the network, always; another evaluation only where a DEV build or the veto asked for it by name
    int e = eval_mode == EV_NNUE ? nnue::evaluate(*pos, *acc)
            : eval_mode == EV_MATERIAL ? handeval::material(*pos) : handeval::evaluate(*pos);
    e = std::clamp(e, -30000, 30000);        // inside int16 and below the mate scores
    return e * (200 - pos->rule50) / 200;   // drift toward a draw as the fifty-move count grows
  }
  void do_move(Move m) {
    {
      // the table line of the next position is asked for before the move is made, from a guess of its key that is
      // right for plain moves and captures (the others get theirs after the move, as before); no result changes
      const Position &p = *pos;
      int f = mfrom(m), t = mto(m), pc = p.board[f];
      U64 k = p.key ^ Zpsq[pc][f] ^ Zpsq[pc][t] ^ Zside;
      if (p.board[t] != NO_PIECE) k ^= Zpsq[p.board[t]][t];
      tt->prefetch(k);
    }
    pos->make(m); acc->computed[pos->sp] = false;
    tt->prefetch(pos->key);
    nodes.store(nodes.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);   // no lock: one writer
  }
  void undo() { pos->unmake(); }
  // Is the side to move in check? The node that made the move has just worked it out for this very position
  // (to know whether the move gives check); the answer is kept with the position's key and used again here.
  U64 chk_key = 0;
  bool chk_val = false;
  bool in_chk() const { return pos->key == chk_key ? chk_val : pos->in_check(); }

  // static exchange: material the side to move gains with m if both sides keep taking with their cheapest piece
  int see(Move m) const {
    const Position &p = *pos;
    int f = mfrom(m), t = mto(m), fl = mflag(m);
    if (fl == F_KCASTLE || fl == F_QCASTLE) return 0;
    int gain[40], d = 0;
    U64 occ = p.occ[2];
    int victim = fl == F_EP ? PAWN : (p.board[t] != NO_PIECE ? pc_type(p.board[t]) : -1);
    gain[0] = victim >= 0 ? SEEV[victim] : 0;
    int onsq = pc_type(p.board[f]);
    if (fl & 8) { gain[0] += SEEV[promo_type(m)] - SEEV[PAWN]; onsq = promo_type(m); }
    occ ^= sqbb(f);
    if (fl == F_EP) occ ^= sqbb(t + (p.side == WHITE ? -8 : 8));
    U64 diag = p.bb[BISHOP] | p.bb[6 + BISHOP] | p.bb[QUEEN] | p.bb[6 + QUEEN];
    U64 orth = p.bb[ROOK] | p.bb[6 + ROOK] | p.bb[QUEEN] | p.bb[6 + QUEEN];
    U64 att = p.attackers_to(t, occ) & occ;
    int side = p.side ^ 1;
    while (true) {
      U64 mine = att & p.occ[side];
      if (!mine) break;
      int pt = PAWN, sq = -1;
      for (; pt <= KING; pt++) { U64 b = mine & p.bb[pc_make(side, pt)]; if (b) { sq = lsb(b); break; } }
      if (pt == KING && (att & p.occ[side ^ 1] & ~sqbb(sq))) break;   // the king cannot take into a defended square
      d++;
      gain[d] = SEEV[onsq] - gain[d - 1];
      if (std::max(-gain[d - 1], gain[d]) < 0) break;
      onsq = pt;
      occ ^= sqbb(sq);
      att |= (bishop_att(t, occ) & diag) | (rook_att(t, occ) & orth);
      att &= occ;
      side ^= 1;
      if (d >= 38) break;
    }
    while (d > 0) { gain[d - 1] = -std::max(-gain[d - 1], gain[d]); d--; }
    return gain[0];
  }

  void score_moves(Move *ml, int *sc, int n, Move ttm, int ply) {
    const Position &p = *pos;
    Move cm = ply > 0 && played[ply - 1] ? counter[mfrom(played[ply - 1])][mto(played[ply - 1])] : NO_MOVE;
    for (int i = 0; i < n; i++) {
      Move m = ml[i];
      if (m == ttm) { sc[i] = 30000000; continue; }
      if (is_cap(m)) {
        int v = mflag(m) == F_EP ? PAWN : pc_type(p.board[mto(m)]);
        int s = see(m);
        sc[i] = (s >= 0 ? 20000000 : -1000000) + SEEV[v] * 16 - pc_type(p.board[mfrom(m)]);
        if (caph) sc[i] += chist[p.board[mfrom(m)]][mto(m)][v] / 8;
      } else if (is_promo(m)) sc[i] = promo_type(m) == QUEEN ? 19000000 : -2000000;
      else if (m == killers[ply][0]) sc[i] = 15000000;
      else if (m == killers[ply][1]) sc[i] = 14000000;
      else if (m == cm) sc[i] = 13000000;
      else sc[i] = qhist(ply, m, p.board[mfrom(m)]);
    }
  }
  static Move pick(Move *ml, int *sc, int n, int i) {
    int b = i;
    for (int j = i + 1; j < n; j++) if (sc[j] > sc[b]) b = j;
    std::swap(ml[i], ml[b]); std::swap(sc[i], sc[b]);
    return ml[i];
  }
  // the fifty-move count is full: a draw, unless the side to move is already checkmated
  bool mated_now() { Move t[256]; return pos->in_check() && gen_moves(*pos, t, GEN_ALL) == 0; }
  bool non_pawn(int c) const { return (pos->occ[c] & ~pos->pieces(c, PAWN) & ~pos->pieces(c, KING)) != 0; }

  int qsearch(int alpha, int beta, int ply, int qd) {
    pvlen[ply] = ply;
    if ((nodes & 1023) == 0) check_time();
    if (halt()) return 0;
    if (ply > seldepth) seldepth = ply;
    if (ply >= MAXPLY) return evaluate();
    if (pos->rule50 >= 100) return mated_now() ? -MATE + ply : 0;
    if (pos->insufficient()) return 0;
    bool inchk = in_chk(), pvnode = beta - alpha > 1;
    bool hit;
    TTData td;
    TTEntry *e = tt->probe(pos->key, hit, td);
    if (hit && !pvnode) {
      int s = from_tt(td.score, ply), b = td.bg & 3;
      if (b == B_EXACT || (b == B_LOWER && s >= beta) || (b == B_UPPER && s <= alpha)) return s;
    }
    int best = -INF, ev = -INF, oalpha = alpha;
    if (!inchk) {
      if (qlazy && !hit && std::abs(beta) < MATE_BOUND && std::abs(alpha) < MATE_BOUND) {
        int me = handeval::material(*pos);
        if (me - qlazy >= beta) return me - qlazy;              // far above the window: the stand-pat cut
        if (me + qlazy + SEEV[QUEEN] <= alpha) return me + qlazy;   // far below it even after winning a queen
      }
      ev = hit ? td.eval : evaluate();
      best = ev;
      if (best >= beta) return best;
      if (best > alpha) alpha = best;
      if (qmax && qd >= qmax) return best;
    }
    Move ml[256]; int sc[256];
    bool checks = !inchk && qchecks && qd == 0;
    int n = gen_moves(*pos, ml, inchk || checks ? GEN_ALL : GEN_NOISY);
    score_moves(ml, sc, n, hit ? td.move : NO_MOVE, ply);
    Move bm = NO_MOVE;
    int legal = 0;
    for (int i = 0; i < n; i++) {
      Move m = pick(ml, sc, n, i);
      bool noisy = is_cap(m) || (is_promo(m) && promo_type(m) == QUEEN);
      if (!inchk) {
        if (!noisy && !checks) continue;
        if (is_promo(m) && promo_type(m) != QUEEN) continue;
        if (noisy && sc[i] < 0 && sc[i] != 30000000) continue;          // a losing capture by SEE
        if (is_cap(m) && !is_promo(m)) {
          int v = mflag(m) == F_EP ? PAWN : pc_type(pos->board[mto(m)]);
          if (ev + SEEV[v] + qs_delta <= alpha) continue;                    // delta pruning
        }
      }
      do_move(m);
      if (!inchk && !noisy && !pos->in_check()) { undo(); continue; }   // the quiet move must give check
      legal++;
      int s = -qsearch(-beta, -alpha, ply + 1, qd + 1);
      undo();
      if (halt()) return 0;
      if (s > best) {
        best = s;
        if (s > alpha) {
          alpha = s; bm = m;
          pv[ply][ply] = m;
          for (int j = ply + 1; j < pvlen[ply + 1]; j++) pv[ply][j] = pv[ply + 1][j];
          pvlen[ply] = pvlen[ply + 1];
          if (s >= beta) break;
        }
      }
    }
    if (inchk && !legal) return -MATE + ply;
    tt->store(e, pos->key, to_tt(best, ply), ev == -INF ? 0 : ev, bm, 0, best >= beta ? B_LOWER : best > oalpha ? B_EXACT : B_UPPER);
    return best;
  }

  int search(int alpha, int beta, int depth, int ply, bool cutnode) {
    bool pvnode = beta - alpha > 1, root = ply == 0;
    pvlen[ply] = ply;
    if ((nodes & 1023) == 0) check_time();
    if (halt()) return 0;
    if (ply >= MAXPLY) return evaluate();
    cuts[ply + 1] = 0;
    bool inchk = in_chk();
    if (inchk && ply < MAXPLY / 2) depth++;
    if (depth <= 0) return qsearch(alpha, beta, ply, 0);
    if (ply > seldepth) seldepth = ply;
    if (!root) {
      if (pos->rule50 >= 100) return mated_now() ? -MATE + ply : 0;
      if (pos->repeated() || pos->insufficient()) return 0;
      alpha = std::max(alpha, -MATE + ply);
      beta = std::min(beta, MATE - ply - 1);
      if (alpha >= beta) return alpha;
    }
    bool hit;
    TTData td;
    TTEntry *e = tt->probe(pos->key, hit, td);
    Move ttm = hit ? td.move : NO_MOVE;
    int tts = hit ? from_tt(td.score, ply) : 0, tb = hit ? (td.bg & 3) : 0;
    Move ex = excl[ply];
    if (!pvnode && hit && td.depth >= depth && !ex) {
      if (tb == B_EXACT || (tb == B_LOWER && tts >= beta) || (tb == B_UPPER && tts <= alpha)) return tts;
    }
    int ev;
    int rawev = 0;
    if (inchk) ev = sev[ply] = -INF;
    else {
      rawev = hit ? td.eval : evaluate();
      rawsev[ply] = rawev;
      ev = sev[ply] = rawev + (corrx ? correction(ply) : corrw ? corr[pos->side][pos->pkey & 16383] / 256 : 0);
      if (hit && (tb == B_EXACT || (tb == B_LOWER && tts > ev) || (tb == B_UPPER && tts < ev))) ev = tts;
    }
    bool improving = !inchk && ply >= 2 && sev[ply] > sev[ply - 2];
    if (!pvnode && !inchk && !ex) {
      if (razor && depth <= 2 && ev + razor * depth <= alpha && std::abs(alpha) < MATE_BOUND) return qsearch(alpha, beta, ply, 0);
      if (depth <= rfp_d && ev - rfp_m * (depth - improving) >= beta && std::abs(beta) < MATE_BOUND) return ev;
      if (nmp && ply >= nmp_min_ply && depth >= 3 && ev >= beta && sev[ply] >= beta && ply > 0 && played[ply - 1] != NO_MOVE && non_pawn(pos->side)) {
        int R = nmp_r + depth / nmp_div + std::min((ev - beta) / nmp_evdiv, 3);
        pos->make_null(); acc->computed[pos->sp] = false; played[ply] = NO_MOVE;
        int s = -search(-beta, -beta + 1, depth - 1 - R, ply + 1, !cutnode);
        pos->unmake_null();
        if (halt()) return 0;
        if (s >= beta) {
          if (!nmpv || depth < nmpv || nmp_min_ply) return s >= MATE_BOUND ? beta : s;
          // deep enough to matter: the same reduced depth again, with no null move in its upper three quarters
          nmp_min_ply = ply + std::max(1, 3 * (depth - 1 - R) / 4);
          int v = search(beta - 1, beta, depth - 1 - R, ply, false);
          nmp_min_ply = 0;
          if (halt()) return 0;
          if (v >= beta) return s >= MATE_BOUND ? beta : s;
        }
      }
    }
    if (iir && depth >= iir && ttm == NO_MOVE && !inchk && !ex) depth--;   // no table move: one ply shallower
    // ProbCut: a capture that does not lose material and still beats beta by a margin at reduced depth
    if (probcut && !pvnode && !inchk && !ex && depth >= 5 && std::abs(beta) < MATE_BOUND &&
        !(hit && td.depth >= depth - 3 && tts < beta + probcut)) {
      int pb = beta + probcut;
      Move pl[256]; int ps[256];
      int pn = gen_moves(*pos, pl, GEN_NOISY);
      score_moves(pl, ps, pn, NO_MOVE, ply);
      for (int i = 0; i < pn; i++) {
        Move m = pick(pl, ps, pn, i);
        if (ps[i] < 0) break;   // losing captures and under-promotions are not tried
        played[ply] = m; mpc[ply] = pos->board[mfrom(m)]; wasq[ply] = false;
        do_move(m);
        int v = -qsearch(-pb, -pb + 1, ply + 1, 0);
        if (v >= pb) v = -search(-pb, -pb + 1, depth - 4, ply + 1, !cutnode);
        undo();
        if (halt()) return 0;
        if (v >= pb) { tt->store(e, pos->key, to_tt(v, ply), rawev, m, depth - 3, B_LOWER); return v; }
      }
    }
    if (!ttm && !ex && depth >= iir_d && (pvnode || cutnode)) depth--;   // no hint from the table: a cheaper look first

    Move ml[256]; int sc[256];
    int n = gen_moves(*pos, ml, GEN_ALL);
    bool lazy = false;   // TtFirst: the table's move stands first and the others are not scored yet
    if (ttfirst && ttm != NO_MOVE)
      for (int i = 0; i < n; i++)
        if (ml[i] == ttm) { std::swap(ml[0], ml[i]); sc[0] = 30000000; lazy = true; break; }
    if (!lazy) score_moves(ml, sc, n, ttm, ply);
    int best = -INF, oalpha = alpha, legal = 0, searched = 0, nq = 0;
    Move bm = NO_MOVE, quiets[64], caps[32];
    int nc = 0;
    for (int i = 0; i < n; i++) {
      if (lazy && i == 1) { score_moves(ml + 1, sc + 1, n - 1, ttm, ply); lazy = false; }
      Move m = lazy ? ml[0] : pick(ml, sc, n, i);
      if (root && !rootonly.empty() && std::find(rootonly.begin(), rootonly.end(), m) == rootonly.end()) continue;
      if (root && !rootskip.empty() && std::find(rootskip.begin(), rootskip.end(), m) != rootskip.end()) continue;
      if (m == ex) continue;
      bool quiet = !is_cap(m) && !is_promo(m);
      legal++;
      if (!root && best > -MATE_BOUND && !inchk) {
        if (quiet) {
          if (depth <= lmp_d && nq >= lmp_b + depth * depth / (improving ? 1 : 2)) continue;   // late move pruning
          if (histp && depth <= 4 && m != ttm && sc[i] < 13000000 && sc[i] < -histp * depth) continue;   // a plain quiet move with a bad history
          if (seeq && depth <= seeq_d && m != ttm && see(m) < -seeq * depth) continue;          // the quiet move loses material
        } else if (depth <= badcap_d && sc[i] < 0 && see(m) < -badcap_m * depth) continue;         // a bad capture, shallow
      }
      // singular extension: if every other move fails well below the table's score, the table move gets one more ply
      int ext = 0;
      if (singular && !root && m == ttm && !ex && depth >= sing_d && hit && td.depth >= depth - 3 && (tb & B_LOWER) && std::abs(tts) < MATE_BOUND) {
        int sb = tts - sing_m * depth;
        excl[ply] = m;
        int sv = search(sb - 1, sb, (depth - 1) / 2, ply, cutnode);
        excl[ply] = NO_MOVE;
        if (halt()) return 0;
        if (sv < sb) ext = (singx && !pvnode && sv < sb - 40 && ply < 2 * rootdepth) ? 2 : 1;
        else if (sb >= beta) return sb;   // several moves beat beta: cut here
        else if (singx && tts >= beta) ext = -1;   // a second move is close and the table move beats beta: look less
      }
      int hv = quiet ? qhist(ply, m, pos->board[mfrom(m)]) : 0;
      played[ply] = m; mpc[ply] = pos->board[mfrom(m)]; wasq[ply] = quiet;
      U64 before = root ? nodes.load(std::memory_order_relaxed) : 0;
      do_move(m);
      bool gives = pos->in_check();
      chk_key = pos->key; chk_val = gives;
      if (!root && quiet && !gives && !inchk && best > -MATE_BOUND && depth <= fut_d && sev[ply] + fut_b + fut_m * depth <= alpha && m != ttm) {
        undo(); if (nq < 64) quiets[nq++] = m; continue;                                // futility
      }
      int nd = depth - 1 + ext, s;
      if (searched == 0) s = -search(-beta, -alpha, nd, ply + 1, false);
      else {
        int r = 0;
        bool badcap = (lmrx & 16) && !quiet && sc[i] < 0;   // LmrX 16: a capture that loses material by exchange
        if (depth >= 3 && searched >= 1 + 2 * pvnode && (quiet || badcap) && !gives) {
          r = lmr[std::min(depth, 63)][std::min(searched, 63)];
          if (badcap) r /= 2;
          r += !pvnode + cutnode - (m == killers[ply][0] || m == killers[ply][1]);
          r -= hv / ((contw ? lmr_hdiv : lmr_hdiv / 2) + (phist ? 4000 : 0) + (cont4 ? 2000 : 0));
          if (lmrx & 1) r += ttm != NO_MOVE && is_cap(ttm);
          if (lmrx & 2) r += cuts[ply + 1] >= 3;
          if (lmrx & 8) r += !improving && !pvnode;
          r = std::clamp(r, 0, nd - 1);
        }
        s = -search(-alpha - 1, -alpha, nd - r, ply + 1, true);
        if (s > alpha && r > 0) {
          // LmrX 4: far above the best so far, the full search goes one ply deeper; barely above it, one shallower
          int full = nd + ((lmrx & 4) ? (s > best + 50 + 2 * nd) - (s < best + 8) : 0);
          if (full > nd - r) s = -search(-alpha - 1, -alpha, full, ply + 1, !cutnode);
        }
        if (s > alpha && s < beta) s = -search(-beta, -alpha, nd, ply + 1, false);
      }
      undo();
      searched++;
      if (root) rootn[mfrom(m)][mto(m)] += nodes.load(std::memory_order_relaxed) - before;
      if (halt()) return 0;
      if (s > best) {
        best = s;
        if (s > alpha) {
          alpha = s; bm = m;
          pv[ply][ply] = m;
          for (int j = ply + 1; j < pvlen[ply + 1]; j++) pv[ply][j] = pv[ply + 1][j];
          pvlen[ply] = pvlen[ply + 1];
          if (root) { root_best = m; root_score = s; }
          if (s >= beta) {
            cuts[ply]++;
            if (quiet) {
              if (killers[ply][0] != m) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
              int bonus = std::min(depth * depth * hb_mul, hb_max), us = pos->side;
              auto upd = [&](Move q, int b) { int &h = hist[us][mfrom(q)][mto(q)]; h += b - h * std::abs(b) / 16384; };
              upd(m, bonus);
              for (int j = 0; j < nq; j++) upd(quiets[j], -bonus);
              if (contw) {
                auto cupd = [&](Move q, int b) {
                  for (int k = 1; k <= 2; k++)
                    if (ply >= k && played[ply - k]) { int &h = ch(mpc[ply - k], mto(played[ply - k]), pos->board[mfrom(q)], mto(q)); h += b - h * std::abs(b) / 16384; }
                  if (cont4 && ply >= 4 && played[ply - 4]) { int &h = ch(mpc[ply - 4], mto(played[ply - 4]), pos->board[mfrom(q)], mto(q)); h += b - h * std::abs(b) / 16384; }
                };
                cupd(m, bonus);
                for (int j = 0; j < nq; j++) cupd(quiets[j], -bonus);
              }
              if (phist) {
                auto pupd = [&](Move q, int b) { int &h = phv(pos->board[mfrom(q)], mto(q)); h += b - h * std::abs(b) / 16384; };
                pupd(m, bonus);
                for (int j = 0; j < nq; j++) pupd(quiets[j], -bonus);
              }
              if (ply > 0 && played[ply - 1]) counter[mfrom(played[ply - 1])][mto(played[ply - 1])] = m;
            }
            if (caph) {
              int cb = std::min(depth * depth * hb_mul, hb_max);
              auto capupd = [&](Move q, int b) {
                int v = mflag(q) == F_EP ? PAWN : pc_type(pos->board[mto(q)]);
                int &h = chist[pos->board[mfrom(q)]][mto(q)][v]; h += b - h * std::abs(b) / 16384;
              };
              if (is_cap(m)) capupd(m, cb);
              for (int j = 0; j < nc; j++) capupd(caps[j], -cb);
            }
            break;
          }
        }
      }
      if (quiet && nq < 64) quiets[nq++] = m;
      else if (is_cap(m) && nc < 32) caps[nc++] = m;
    }
    if (!legal) return ex ? alpha : root && root_limited() ? 0 : (inchk ? -MATE + ply : 0);
    if (best == -INF) return alpha;   // every move was pruned (cannot happen before one is searched)
    int bound = best >= beta ? B_LOWER : best > oalpha ? B_EXACT : B_UPPER;
    // PriorBonus: nothing here reached alpha, so the quiet move that led here was a good one for the other side
    if (prevb && bound == B_UPPER && depth >= 3 && ply > 0 && !ex && played[ply - 1] && wasq[ply - 1]) {
      Move pm = played[ply - 1];
      int b = std::min(depth * depth * 8, 800), &h = hist[pos->side ^ 1][mfrom(pm)][mto(pm)];
      h += b - h * b / 16384;
    }
    // CorrX: each of the four tables moves toward (search result minus the network's own score)
    if (corrx && !inchk && !ex && (!bm || !is_cap(bm)) && std::abs(best) < MATE_BOUND &&
        !(bound == B_LOWER && best <= rawsev[ply]) && !(bound == B_UPPER && best >= rawsev[ply])) {
      int us = pos->side, w = std::min(depth + 1, 16), goal = std::clamp(best - rawsev[ply], -400, 400) * 256;
      auto mv = [&](int &c) { c += (goal - c) * w / 256; };
      mv(corr[us][pos->pkey & 16383]); mv(cnp[us][0][pos->npkey[0] & 16383]); mv(cnp[us][1][pos->npkey[1] & 16383]);
      if (int *c = ccp(ply)) mv(*c);
    }
    if (corrw && !corrx && !inchk && !ex && (!bm || !is_cap(bm)) && std::abs(best) < MATE_BOUND &&
        !(bound == B_LOWER && best <= sev[ply]) && !(bound == B_UPPER && best >= sev[ply])) {
      int &cc = corr[pos->side][pos->pkey & 16383], w = std::min(depth + 1, 16);
      cc += (std::clamp(best - sev[ply], -400, 400) * 256 - cc) * w / 256;
    }
    if (!ex && !(root && root_limited())) tt->store(e, pos->key, to_tt(best, ply), inchk ? 0 : rawev, bm ? bm : ttm, depth,
             best >= beta ? B_LOWER : best > oalpha ? B_EXACT : B_UPPER);
    return best;
  }

  static std::string score_str(int s) {
    if (s >= MATE_BOUND) return "mate " + std::to_string((MATE - s + 1) / 2);
    if (s <= -MATE_BOUND) return "mate -" + std::to_string((MATE + s) / 2);
    return "cp " + std::to_string(s);
  }

  // one whole search: the limits, then iterative deepening; returns the best move
  Move think(const Limits &L) { set_limits(L); return iterate(L); }

  // the clock and the limits of a new search. The main search sets them before its helpers are woken,
  // because the helpers read its clock.
  void set_limits(const Limits &L) {
    start_us = now_us();
    nodes = 0; node_limit = L.nodes; timed = false;
    int64_t mine = pos->side == WHITE ? L.wtime : L.btime, inc = pos->side == WHITE ? L.winc : L.binc;
    if (L.movetime > 0) { timed = true; soft_ms = hard_ms = std::max<int64_t>(1, L.movetime - 20); }
    else if (mine >= 0 && !L.infinite) {
      timed = true;
      int mtg = L.movestogo > 0 ? std::min(L.movestogo, 40) : tm_mtg;
      int64_t avail = std::max<int64_t>(1, mine - overhead);
      soft_ms = std::min(avail, (avail / mtg + inc * 3 / 4) * timescale / 100);   // TimeScale: percent of the usual share
      hard_ms = std::min(avail, std::max(soft_ms, avail / 4) > soft_ms * 4 ? soft_ms * 4 : std::max(soft_ms, avail / 4));
      hard_ms = std::min(hard_ms, avail);
    }
  }

  // iterative deepening from the limits already set
  Move iterate(const Limits &L) {
    if (!helper) tt->gen++;
    acc->reset();
    prep();
    memset(killers, 0, sizeof killers);
    memset(rootn, 0, sizeof rootn);
    for (auto &a : hist) for (auto &b : a) for (auto &h : b) h /= 2;
    for (auto &h : cont) h /= 2;
    for (auto &h : ph) h /= 2;
    rootonly = L.searchmoves;
    Move ml[256];
    int n = gen_moves(*pos, ml, GEN_ALL);
    if (!n) return NO_MOVE;
    Move best = rootonly.empty() ? ml[0] : rootonly[0];
    int prev = 0, stable = 0, score1 = 0, score2 = 0;
    Move last_best = NO_MOVE;
    done_depth = 0;
    for (int d = std::min(start_depth, L.depth); d <= L.depth; d++) {
      seldepth = 0; root_best = NO_MOVE; rootdepth = d;
      int delta = asp_delta, a = -INF, b = INF, s;
      if (d >= 5) { a = std::max(prev - delta, -INF); b = std::min(prev + delta, INF); }
      while (true) {
        s = search(a, b, d, 0, false);
        if (halt()) break;
        if (s <= a) { b = (a + b) / 2; a = std::max(s - delta, -INF); delta += delta; }
        else if (s >= b) { b = std::min(s + delta, INF); delta += delta; }
        else break;
      }
      if (halt()) { if (root_best) best = root_best; break; }
      prev = s; done_depth = d;
      if (pvlen[0] > 0 && pv[0][0]) best = pv[0][0];
      if (!silent) {
        int64_t ms = elapsed();
        U64 an = all_nodes();
        printf("info depth %d seldepth %d%s score %s nodes %llu nps %llu hashfull %d time %lld pv", d, seldepth,
               multipv > 1 ? " multipv 1" : "", score_str(s).c_str(),
               (unsigned long long)an, (unsigned long long)(an * 1000 / std::max<int64_t>(ms, 1)), tt->hashfull(), (long long)ms);
        for (int j = 0; j < pvlen[0]; j++) printf(" %s", move_str(pv[0][j]).c_str());
        printf("\n");
        fflush(stdout);
      }
      // MultiPV: the next best root moves at this depth, each found with the better ones left out. The first
      // line above is the search's own result and is not touched by this.
      if (multipv > 1 && !helper && !silent && !stop) {
        rootskip.clear();
        Move line = best;
        for (int k = 2; k <= multipv && (int)rootskip.size() + 1 < n; k++) {
          rootskip.push_back(line);
          root_best = NO_MOVE;
          int s2 = search(-INF, INF, d, 0, false);
          if (halt() || pvlen[0] <= 0 || !pv[0][0]) break;
          line = pv[0][0];
          int64_t ms = elapsed();
          U64 an = all_nodes();
          printf("info depth %d seldepth %d multipv %d score %s nodes %llu nps %llu time %lld pv", d, seldepth, k,
                 score_str(s2).c_str(), (unsigned long long)an, (unsigned long long)(an * 1000 / std::max<int64_t>(ms, 1)),
                 (long long)ms);
          for (int j = 0; j < pvlen[0]; j++) printf(" %s", move_str(pv[0][j]).c_str());
          printf("\n");
          fflush(stdout);
        }
        rootskip.clear();
      }
      root_score = s;
      if (stop) break;
      stable = best == last_best ? stable + 1 : 0;
      last_best = best;
      // TmStab: no new depth starts after this share of the soft time: 130% of the usual half right after the
      // best move changed, 9 points less for each depth it stays, down to 67%
      int64_t share = tmstab ? soft_ms / 2 * std::max(67, 130 - 9 * stable) / 100 : soft_ms * tm_share / 100;
      if (tmx && best) {
        // TmX (our own rule and numbers): the usual half of the soft time, times
        //   1.45 less the best move's share of this search's nodes (a move that took nearly all of them is settled),
        //   1.2 right after the best move changed, 0.85 once it has held four depths,
        //   and up to 1.6 when the score has fallen (against this side's last search, or two depths ago)
        U64 all = nodes.load(std::memory_order_relaxed);
        double eff = all ? double(rootn[mfrom(best)][mto(best)]) / double(all) : 0.5;
        double f = (1.45 - eff) * (stable == 0 ? 1.2 : stable >= 4 ? 0.85 : 1.0);
        int ref = have_last ? last_score : (d >= 3 ? score2 : s);
        if (std::abs(s) < MATE_BOUND && std::abs(ref) < MATE_BOUND) f *= 1.0 + std::clamp(ref - s, 0, 60) / 100.0;
        share = int64_t(double(soft_ms * tm_share / 100) * std::clamp(f, 0.5, 2.2));
      }
      score2 = score1; score1 = s;
      if (timed && !pondering && L.movetime <= 0 && elapsed() >= share) break;   // a fixed move time is used in full
      if (node_limit && nodes >= node_limit) break;
      if (std::abs(s) >= MATE_BOUND && d >= 8 && !L.infinite && timed && !pondering) break;
    }
    if (done_depth > 0) { last_score = root_score; have_last = true; }
    return best;
  }

  // A short look for another program (midge_lib.cpp): iterative deepening to `depth`, stopped near `cap` nodes.
  // The move ordering tables are left as they are between calls. Returns the score of the last finished depth;
  // quick_static holds the network's own evaluation of the position.
  int quick_static = 0;
  // qlazy (19 Nov 2026, for the short looks; 0 = off): in quiescence, a position whose material count is this many
  // centipawns beyond the window is cut on the count alone, without asking the network.
  int qlazy = 0;
  // Options for quick(), set by midge_lib (19 Nov 2026); both off by default, so nothing else changes:
  //   quick_asp     aspiration windows from depth 5, as the normal search uses them (fewer nodes for each depth)
  //   quick_stable  stop once the best move has stayed the same for this many depths in a row and the score has
  //                 moved by no more than 12 cp over them (from depth 6); 0 = off
  bool quick_asp = false;
  int quick_stable = 0;
  int quick_stable_d = 6, quick_stable_cp = 12;   // the stable stop's first depth and its score band
  int quick_asp_delta = 18;                        // the first half-width of quick()'s aspiration window
  // More settings for the short looks (19 Nov 2026), each at the engine's own value unless the library sets it:
  //   rfp_d   the deepest depth the static cut above beta is tried at       nmp_r, nmp_div  the null move's reduction
  //   razor   0 off; at depth 1 and 2, a static score this far (times depth) below alpha goes straight to quiescence
  //   iir     0 off; from this depth a node with no table move is searched one ply shallower
  //   qmax    0 off; quiescence stops taking after this many plies of captures (unless in check)
  int rfp_d = 7, nmp_r = 3, nmp_div = 4, razor = 0, iir = 0, qmax = 0;
  // The search's other fixed numbers as fields (19 Nov 2026), each at the value the code had, for the tuner:
  int asp_delta = 18;     // the first half-width of the aspiration window
  int nmp_evdiv = 200;    // null move: one more ply of reduction for each this many centipawns above beta (up to 3)
  int lmp_d = 5;          // late move pruning up to this depth
  int seeq_d = 6;         // SeeQuiet up to this depth
  int badcap_m = 100, badcap_d = 5;   // a capture losing more than badcap_m x depth is skipped up to badcap_d
  int fut_d = 6;          // futility up to this depth
  int sing_d = 8, sing_m = 2;   // singular test from this depth; its bar is the table score less sing_m x depth
  int lmr_hdiv = 12000;   // history points for one ply less reduction
  int hb_mul = 16, hb_max = 1600;   // history bonus: depth x depth x hb_mul, at most hb_max
  int qs_delta = 180;     // quiescence: a capture that cannot bring the score within this of alpha is skipped
  int iir_d = 4;          // without a table move, pv and cut nodes from this depth are searched one ply shallower
  int tm_share = 50;      // no new depth starts after this percent of the soft time
  int tm_mtg = 28;        // moves the remaining time is spread over when the GUI gives none
  int quick(int depth, U64 cap) {
    start_us = now_us();
    nodes = 0; node_limit = cap; timed = false; stop = false; pondering = false;
    acc->reset();
    prep();
    rootonly.clear(); rootskip.clear();
    memset(killers, 0, sizeof killers);
    quick_static = evaluate();
    int s = quick_static;
    done_depth = 0;
    Move last_best = NO_MOVE;
    int same = 0, ref_score = 0;
    for (int d = 1; d <= depth; d++) {
      seldepth = 0; root_best = NO_MOVE; rootdepth = d;
      int r;
      if (quick_asp && d >= 5 && std::abs(s) < MATE_BOUND) {
        int delta = quick_asp_delta, a = std::max(s - delta, -INF), b = std::min(s + delta, INF);
        while (true) {
          r = search(a, b, d, 0, false);
          if (halt()) break;
          if (r <= a) { b = (a + b) / 2; a = std::max(r - delta, -INF); delta += delta; }
          else if (r >= b) { b = std::min(r + delta, INF); delta += delta; }
          else break;
        }
      } else {
        r = search(-INF, INF, d, 0, false);
      }
      if (halt()) break;
      s = r; done_depth = d;
      if (nodes >= cap) break;
      if (quick_stable > 0) {
        Move bm = pvlen[0] > 0 ? pv[0][0] : root_best;
        if (bm != NO_MOVE && bm == last_best && std::abs(s - ref_score) <= quick_stable_cp) same++;
        else { same = 0; ref_score = s; }
        last_best = bm;
        if (d >= quick_stable_d && same >= quick_stable) break;
      }
    }
    return s;
  }

  // ---------------- veto: does a chosen move lose material by force? ----------------
  int veto_eval = EV_MATERIAL;
  U64 veto_nodes_cap = 4000000;
  int veval() {
    return veto_eval == EV_NNUE && nnue::loaded ? nnue::evaluate(*pos, *acc) : veto_eval == EV_HAND ? handeval::evaluate(*pos) : handeval::material(*pos);
  }
  // captures, queen promotions and checks for d more plies (every evasion when in check), then captures only;
  // the side to move may always stand pat unless in check
  int fsearch(int alpha, int beta, int ply, int d) {
    if (nodes >= veto_nodes_cap) { stop = true; return 0; }
    if (ply >= MAXPLY) return veval();
    if (pos->rule50 >= 100) return mated_now() ? -MATE + ply : 0;
    if (pos->repeated() || pos->insufficient()) return 0;
    bool inchk = pos->in_check();
    int best = -INF;
    if (!inchk) { int sp = veval(); if (sp >= beta) return sp; best = sp; if (sp > alpha) alpha = sp; }
    Move ml[256]; int sc[256];
    int n = gen_moves(*pos, ml, inchk || d > 0 ? GEN_ALL : GEN_NOISY);
    score_moves(ml, sc, n, NO_MOVE, std::min(ply, MAXPLY));
    int legal = 0;
    for (int i = 0; i < n; i++) {
      Move m = pick(ml, sc, n, i);
      bool noisy = is_cap(m) || (is_promo(m) && promo_type(m) == QUEEN);
      if (!inchk && is_promo(m) && promo_type(m) != QUEEN) continue;
      do_move(m);
      if (!inchk && !noisy && !pos->in_check()) { undo(); continue; }
      legal++;
      int s = -fsearch(-beta, -alpha, ply + 1, d - 1);
      undo();
      if (stop) return 0;
      if (s > best) { best = s; if (s > alpha) { alpha = s; if (s >= beta) break; } }
    }
    if (inchk && !legal) return -MATE + ply;
    return best;
  }
  struct VetoResult { Move best; int vbest, vmove; bool flag, complete; U64 nodes; };
  // forcing mode: each root move, then the forcing search to depth d; full mode: the ordinary search to depth d
  VetoResult veto(Move mv, int d, bool full, int margin) {
    VetoResult r{NO_MOVE, -INF, -INF, false, true, 0};
    start_us = now_us();
    stop = false; nodes = 0; timed = false; node_limit = full ? veto_nodes_cap : 0; rootdepth = 2;
    acc->reset();
    prep();
    Move ml[256];
    int n = gen_moves(*pos, ml, GEN_ALL);
    int save_eval = eval_mode;
    if (full) eval_mode = veto_eval;
    for (int i = 0; i < n; i++) {
      do_move(ml[i]);
      int s;
      if (full) { tt->gen++; played[0] = ml[i]; s = -search(-INF, INF, d - 1, 1, false); }
      else s = -fsearch(-INF, INF, 1, d - 1);
      undo();
      if (stop) { r.complete = false; break; }
      if (s > r.vbest) { r.vbest = s; r.best = ml[i]; }
      if (ml[i] == mv) r.vmove = s;
    }
    eval_mode = save_eval;
    r.nodes = nodes;
    r.flag = r.complete && r.vmove > -INF && r.vbest - r.vmove >= margin && r.vmove <= -margin / 2;
    return r;
  }
};

// The numbers a tuner may set, with their bounds (a DEV build lists each as the UCI option T_<name>).
struct Tunable { const char *name; int Search::*p; int lo, hi; };
inline const Tunable TUNABLES[] = {
    {"AspDelta", &Search::asp_delta, 4, 120},     {"NmpR", &Search::nmp_r, 1, 6},
    {"NmpDiv", &Search::nmp_div, 2, 12},          {"NmpEvDiv", &Search::nmp_evdiv, 50, 800},
    {"RfpDepth", &Search::rfp_d, 2, 14},          {"RfpMargin", &Search::rfp_m, 20, 300},
    {"LmpBase", &Search::lmp_b, 1, 12},           {"LmpDepth", &Search::lmp_d, 2, 12},
    {"SeeQuiet", &Search::seeq, 0, 300},          {"SeeQuietDepth", &Search::seeq_d, 2, 12},
    {"BadCapMargin", &Search::badcap_m, 20, 400}, {"BadCapDepth", &Search::badcap_d, 2, 12},
    {"FutBase", &Search::fut_b, 0, 400},          {"FutMul", &Search::fut_m, 20, 300},
    {"FutDepth", &Search::fut_d, 2, 12},          {"SingDepth", &Search::sing_d, 4, 14},
    {"SingMargin", &Search::sing_m, 1, 8},        {"LmrHistDiv", &Search::lmr_hdiv, 2000, 40000},
    {"HistBonusMul", &Search::hb_mul, 2, 64},     {"HistBonusMax", &Search::hb_max, 200, 8000},
    {"QsDelta", &Search::qs_delta, 40, 600},      {"IirDepth", &Search::iir_d, 2, 12},
    {"TmShare", &Search::tm_share, 20, 100},      {"TmMtg", &Search::tm_mtg, 10, 60},
    {"TimeScale", &Search::timescale, 20, 400},   {"HistPrune", &Search::histp, 0, 20000},
    {"CorrX", &Search::corrx, 0, 300},            {"ProbCut", &Search::probcut, 0, 600},
};
