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
  void resize(size_t mb) {
    size_t n = 1;
    while (n * 2 * sizeof(TTEntry) <= (mb << 20)) n *= 2;
    delete[] t; t = nullptr;
    while (!(t = new (std::nothrow) TTEntry[n]) && n > 1024) n /= 2;   // not enough memory: half, and again
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
  TTEntry *probe(U64 k, bool &hit, TTData &d) {
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
    if (dd && (o.bg >> 2) == (gen & 63) && depth + 4 < o.depth && bound != B_EXACT) return;
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
  Move killers[MAXPLY + 2][2];
  int hist[2][64][64];
  // continuation history: (piece and target of the move 1 or 2 plies back) -> (piece and target of this move)
  std::vector<int> cont = std::vector<int>(12 * 64 * 12 * 64, 0);
  int mpc[MAXPLY + 2];
  int &ch(int ppc, int pto, int pc, int to) { return cont[((ppc * 64 + pto) * 12 + pc) * 64 + to]; }
  int qhist(int ply, Move m, int pc) {
    int h = hist[pos->side][mfrom(m)][mto(m)];
    if (contw)
      for (int k = 1; k <= 2; k++)
        if (ply >= k && played[ply - k]) h += ch(mpc[ply - k], mto(played[ply - k]), pc, mto(m));
    return h;
  }
  Move counter[64][64];
  Move pv[MAXPLY + 1][MAXPLY + 1];
  int pvlen[MAXPLY + 1];
  int sev[MAXPLY + 2];
  Move played[MAXPLY + 2];
  int lmr[64][64];
  std::vector<Move> rootonly;
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
  bool out_of_time() const { return timed && !pondering && elapsed() >= hard_ms; }
  void check_time() {
    if (lead) {
      // a helper watches the main search's clock and stop flag. With more threads than CPUs the main thread
      // can be off the CPU when its time runs out; whichever thread is running then ends the search for all.
      if (lead->stop || lead->out_of_time()) { lead->stop = true; stop = true; }
      return;
    }
    if (out_of_time() || (node_limit && nodes >= node_limit)) stop = true;
  }
  int evaluate() {
    int e = eval_mode == EV_NNUE && nnue::loaded ? nnue::evaluate(*pos, *acc)
            : eval_mode == EV_MATERIAL ? handeval::material(*pos) : handeval::evaluate(*pos);
    e = std::clamp(e, -30000, 30000);        // inside int16 and below the mate scores
    return e * (200 - pos->rule50) / 200;   // drift toward a draw as the fifty-move count grows
  }
  void do_move(Move m) {
    pos->make(m); acc->computed[pos->sp] = false;
    nodes.store(nodes.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);   // no lock: one writer
  }
  void undo() { pos->unmake(); }

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
    bool inchk = pos->in_check(), pvnode = beta - alpha > 1;
    bool hit;
    TTData td;
    TTEntry *e = tt->probe(pos->key, hit, td);
    if (hit && !pvnode) {
      int s = from_tt(td.score, ply), b = td.bg & 3;
      if (b == B_EXACT || (b == B_LOWER && s >= beta) || (b == B_UPPER && s <= alpha)) return s;
    }
    int best = -INF, ev = -INF, oalpha = alpha;
    if (!inchk) {
      ev = hit ? td.eval : evaluate();
      best = ev;
      if (best >= beta) return best;
      if (best > alpha) alpha = best;
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
          if (ev + SEEV[v] + 180 <= alpha) continue;                    // delta pruning
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
    bool inchk = pos->in_check();
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
      ev = sev[ply] = rawev + (corrw ? corr[pos->side][pos->pkey & 16383] / 256 : 0);
      if (hit && (tb == B_EXACT || (tb == B_LOWER && tts > ev) || (tb == B_UPPER && tts < ev))) ev = tts;
    }
    bool improving = !inchk && ply >= 2 && sev[ply] > sev[ply - 2];
    if (!pvnode && !inchk && !ex) {
      if (depth <= 7 && ev - rfp_m * (depth - improving) >= beta && std::abs(beta) < MATE_BOUND) return ev;
      if (nmp && ply >= nmp_min_ply && depth >= 3 && ev >= beta && sev[ply] >= beta && ply > 0 && played[ply - 1] != NO_MOVE && non_pawn(pos->side)) {
        int R = 3 + depth / 4 + std::min((ev - beta) / 200, 3);
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
        played[ply] = m; mpc[ply] = pos->board[mfrom(m)];
        do_move(m);
        int v = -qsearch(-pb, -pb + 1, ply + 1, 0);
        if (v >= pb) v = -search(-pb, -pb + 1, depth - 4, ply + 1, !cutnode);
        undo();
        if (halt()) return 0;
        if (v >= pb) { tt->store(e, pos->key, to_tt(v, ply), rawev, m, depth - 3, B_LOWER); return v; }
      }
    }
    if (!ttm && !ex && depth >= 4 && (pvnode || cutnode)) depth--;   // no hint from the table: a cheaper look first

    Move ml[256]; int sc[256];
    int n = gen_moves(*pos, ml, GEN_ALL);
    score_moves(ml, sc, n, ttm, ply);
    int best = -INF, oalpha = alpha, legal = 0, searched = 0, nq = 0;
    Move bm = NO_MOVE, quiets[64], caps[32];
    int nc = 0;
    for (int i = 0; i < n; i++) {
      Move m = pick(ml, sc, n, i);
      if (root && !rootonly.empty() && std::find(rootonly.begin(), rootonly.end(), m) == rootonly.end()) continue;
      if (m == ex) continue;
      bool quiet = !is_cap(m) && !is_promo(m);
      legal++;
      if (!root && best > -MATE_BOUND && !inchk) {
        if (quiet) {
          if (depth <= 5 && nq >= lmp_b + depth * depth / (improving ? 1 : 2)) continue;   // late move pruning
          if (seeq && depth <= 6 && m != ttm && see(m) < -seeq * depth) continue;          // the quiet move loses material
        } else if (depth <= 5 && sc[i] < 0 && see(m) < -100 * depth) continue;         // a bad capture, shallow
      }
      // singular extension: if every other move fails well below the table's score, the table move gets one more ply
      int ext = 0;
      if (singular && !root && m == ttm && !ex && depth >= 8 && hit && td.depth >= depth - 3 && (tb & B_LOWER) && std::abs(tts) < MATE_BOUND) {
        int sb = tts - 2 * depth;
        excl[ply] = m;
        int sv = search(sb - 1, sb, (depth - 1) / 2, ply, cutnode);
        excl[ply] = NO_MOVE;
        if (halt()) return 0;
        if (sv < sb) ext = (singx && !pvnode && sv < sb - 40 && ply < 2 * rootdepth) ? 2 : 1;
        else if (sb >= beta) return sb;   // several moves beat beta: cut here
        else if (singx && tts >= beta) ext = -1;   // a second move is close and the table move beats beta: look less
      }
      int hv = quiet ? qhist(ply, m, pos->board[mfrom(m)]) : 0;
      played[ply] = m; mpc[ply] = pos->board[mfrom(m)];
      do_move(m);
      bool gives = pos->in_check();
      if (!root && quiet && !gives && !inchk && best > -MATE_BOUND && depth <= 6 && sev[ply] + fut_b + fut_m * depth <= alpha && m != ttm) {
        undo(); if (nq < 64) quiets[nq++] = m; continue;                                // futility
      }
      int nd = depth - 1 + ext, s;
      if (searched == 0) s = -search(-beta, -alpha, nd, ply + 1, false);
      else {
        int r = 0;
        if (depth >= 3 && searched >= 1 + 2 * pvnode && quiet && !gives) {
          r = lmr[std::min(depth, 63)][std::min(searched, 63)];
          r += !pvnode + cutnode - (m == killers[ply][0] || m == killers[ply][1]);
          r -= hv / (contw ? 12000 : 6000);
          r = std::clamp(r, 0, nd - 1);
        }
        s = -search(-alpha - 1, -alpha, nd - r, ply + 1, true);
        if (s > alpha && r > 0) s = -search(-alpha - 1, -alpha, nd, ply + 1, !cutnode);
        if (s > alpha && s < beta) s = -search(-beta, -alpha, nd, ply + 1, false);
      }
      undo();
      searched++;
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
            if (quiet) {
              if (killers[ply][0] != m) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
              int bonus = std::min(depth * depth * 16, 1600), us = pos->side;
              auto upd = [&](Move q, int b) { int &h = hist[us][mfrom(q)][mto(q)]; h += b - h * std::abs(b) / 16384; };
              upd(m, bonus);
              for (int j = 0; j < nq; j++) upd(quiets[j], -bonus);
              if (contw) {
                auto cupd = [&](Move q, int b) {
                  for (int k = 1; k <= 2; k++)
                    if (ply >= k && played[ply - k]) { int &h = ch(mpc[ply - k], mto(played[ply - k]), pos->board[mfrom(q)], mto(q)); h += b - h * std::abs(b) / 16384; }
                };
                cupd(m, bonus);
                for (int j = 0; j < nq; j++) cupd(quiets[j], -bonus);
              }
              if (ply > 0 && played[ply - 1]) counter[mfrom(played[ply - 1])][mto(played[ply - 1])] = m;
            }
            if (caph) {
              int cb = std::min(depth * depth * 16, 1600);
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
    if (!legal) return ex ? alpha : root && !rootonly.empty() ? 0 : (inchk ? -MATE + ply : 0);
    if (best == -INF) return alpha;   // every move was pruned (cannot happen before one is searched)
    int bound = best >= beta ? B_LOWER : best > oalpha ? B_EXACT : B_UPPER;
    if (corrw && !inchk && !ex && (!bm || !is_cap(bm)) && std::abs(best) < MATE_BOUND &&
        !(bound == B_LOWER && best <= sev[ply]) && !(bound == B_UPPER && best >= sev[ply])) {
      int &cc = corr[pos->side][pos->pkey & 16383], w = std::min(depth + 1, 16);
      cc += (std::clamp(best - sev[ply], -400, 400) * 256 - cc) * w / 256;
    }
    if (!ex && !(root && !rootonly.empty())) tt->store(e, pos->key, to_tt(best, ply), inchk ? 0 : rawev, bm ? bm : ttm, depth,
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
      int mtg = L.movestogo > 0 ? std::min(L.movestogo, 40) : 28;
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
    memset(killers, 0, sizeof killers);
    for (auto &a : hist) for (auto &b : a) for (auto &h : b) h /= 2;
    for (auto &h : cont) h /= 2;
    rootonly = L.searchmoves;
    Move ml[256];
    int n = gen_moves(*pos, ml, GEN_ALL);
    if (!n) return NO_MOVE;
    Move best = rootonly.empty() ? ml[0] : rootonly[0];
    int prev = 0, stable = 0;
    Move last_best = NO_MOVE;
    done_depth = 0;
    for (int d = std::min(start_depth, L.depth); d <= L.depth; d++) {
      seldepth = 0; root_best = NO_MOVE; rootdepth = d;
      int delta = 18, a = -INF, b = INF, s;
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
        printf("info depth %d seldepth %d score %s nodes %llu nps %llu hashfull %d time %lld pv", d, seldepth, score_str(s).c_str(),
               (unsigned long long)an, (unsigned long long)(an * 1000 / std::max<int64_t>(ms, 1)), tt->hashfull(), (long long)ms);
        for (int j = 0; j < pvlen[0]; j++) printf(" %s", move_str(pv[0][j]).c_str());
        printf("\n");
        fflush(stdout);
      }
      root_score = s;
      if (stop) break;
      stable = best == last_best ? stable + 1 : 0;
      last_best = best;
      // TmStab: no new depth starts after this share of the soft time: 130% of the usual half right after the
      // best move changed, 9 points less for each depth it stays, down to 67%
      int64_t share = tmstab ? soft_ms / 2 * std::max(67, 130 - 9 * stable) / 100 : soft_ms / 2;
      if (timed && !pondering && L.movetime <= 0 && elapsed() >= share) break;   // a fixed move time is used in full
      if (node_limit && nodes >= node_limit) break;
      if (std::abs(s) >= MATE_BOUND && d >= 8 && !L.infinite && timed && !pondering) break;
    }
    return best;
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
