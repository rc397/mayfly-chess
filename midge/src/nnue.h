// nnue.h: Midge's network. 768 inputs per side (own or their colour, piece type, square seen from that side), one
// shared layer of H to two accumulators kept per move, squared clipped ReLU, 2H -> 1 with output buckets by piece
// count. int16 weights: feature layer at scale QA, output at scale QB. Written from scratch, 17 Nov 2026.
#pragma once
#include <cmath>
#include <fstream>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include "chess.h"

namespace nnue {
#ifndef MIDGE_H
#define MIDGE_H 256
#endif
#ifndef MIDGE_KB
#define MIDGE_KB 1
#endif
// build with -DMIDGE_H=512 for a wider net, -DMIDGE_KB=4 for king buckets
const int H = MIDGE_H, NB = 8, QA = 255, QB = 64, KB = MIDGE_KB, NIN = 768 * KB;

struct Net {
  alignas(64) int16_t w1[NIN][H];
  alignas(64) int16_t b1[H];
  alignas(64) int16_t w2[NB][2 * H];
  int32_t b2[NB];
  float cp_scale;   // output units to centipawns
};
inline Net *net = nullptr;
inline bool loaded = false;

struct alignas(64) Acc { int16_t v[2][H]; };
// one per search thread: accumulators indexed by the position's move count
struct AccStack {
  Acc accs[MAXSP + 1];
  bool computed[MAXSP + 1];
  void reset() { memset(computed, 0, sizeof computed); }   // a new search or position: nothing below is trusted
};

// king class for perspective c with its king on ks: bucket * 2 + mirrored (always 0 when KB == 1)
inline int kclass(int c, int ks) {
  if (KB == 1) return 0;
  int s = c == WHITE ? ks : ks ^ 56, f = s & 7, r = s >> 3, fl = f >= 4;
  if (fl) f = 7 - f;
  int b = r == 0 ? (f <= 1 ? 0 : 1) : r == 1 ? 2 : 3;
  return b * 2 + fl;
}
inline int feat(int c, int pc, int sq, int kc) {
  int s = (c == WHITE ? sq : sq ^ 56) ^ ((kc & 1) ? 7 : 0);
  return (kc >> 1) * 768 + (pc_color(pc) != c) * 384 + pc_type(pc) * 64 + s;
}

// file: "MIDG", version, H, NB, QA, QB (int32 each), cp_scale (float), then w1, b1, w2 (int16), b2 (int32)
inline bool load(const std::string &path, std::string &err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { err = "cannot open " + path; return false; }
  char mg[4]; int32_t hd[5], kb = 1; float sc;
  f.read(mg, 4); f.read((char *)hd, sizeof hd);
  if (hd[0] == 2) f.read((char *)&kb, 4);   // version 2 carries the king bucket count
  f.read((char *)&sc, 4);
  if (memcmp(mg, "MIDG", 4) || (hd[0] != 1 && hd[0] != 2) || hd[1] != H || hd[2] != NB || hd[3] != QA || hd[4] != QB || kb != KB) {
    err = "header does not match this build (H " + std::to_string(H) + ", KB " + std::to_string(KB) + ")"; return false;
  }
  if (!net) net = new Net;
  f.read((char *)net->w1, sizeof net->w1); f.read((char *)net->b1, sizeof net->b1);
  f.read((char *)net->w2, sizeof net->w2); f.read((char *)net->b2, sizeof net->b2);
  if (!f) { err = "file too short"; return false; }
  net->cp_scale = sc;
  loaded = true;
  return true;
}

inline void add_row(int16_t *a, int fi) { const int16_t *w = net->w1[fi]; for (int i = 0; i < H; i++) a[i] += w[i]; }
inline void sub_row(int16_t *a, int fi) { const int16_t *w = net->w1[fi]; for (int i = 0; i < H; i++) a[i] -= w[i]; }

inline void refresh_one(const Position &p, int16_t *a, int c) {
  int kc = kclass(c, p.king_sq(c));
  memcpy(a, net->b1, sizeof net->b1);
  U64 b = p.occ[2];
  while (b) { int s = pop_lsb(b); add_row(a, feat(c, p.board[s], s, kc)); }
}
inline void refresh(const Position &p, Acc &a) { refresh_one(p, a.v[0], 0); refresh_one(p, a.v[1], 1); }

// bring the accumulator at p.sp up to date from the nearest computed one below (lazy, so pruned moves cost nothing);
// a side whose king changed bucket or mirror on the way is rebuilt from the board
inline Acc &update(const Position &p, AccStack &st) {
  Acc *accs = st.accs;
  bool *computed = st.computed;
  int top = p.sp, j = top;
  while (j > 0 && !computed[j] && top - j < 24) j--;
  if (!computed[j]) { refresh(p, accs[top]); computed[top] = true; return accs[top]; }
  if (j == top) return accs[top];
  for (int c = 0; c < 2; c++) {
    int kc = kclass(c, p.king_sq(c)), king = pc_make(c, KING);
    bool rebuild = false;
    for (int i = j + 1; i <= top && KB > 1 && !rebuild; i++) {
      const Dirty &d = p.dirty[i];
      for (int k = 0; k < d.n; k++)
        if (d.pc[k] == king && (kclass(c, d.from[k]) != kc || kclass(c, d.to[k]) != kc)) rebuild = true;
    }
    int16_t *a = accs[top].v[c];
    if (rebuild) { refresh_one(p, a, c); continue; }
    memcpy(a, accs[j].v[c], sizeof(int16_t) * H);
    for (int i = j + 1; i <= top; i++) {
      const Dirty &d = p.dirty[i];
      for (int k = 0; k < d.n; k++) {
        if (d.from[k] >= 0) sub_row(a, feat(c, d.pc[k], d.from[k], kc));
        if (d.to[k] >= 0) add_row(a, feat(c, d.pc[k], d.to[k], kc));
      }
    }
  }
  computed[top] = true;
  return accs[top];
}

inline int64_t screlu_dot(const int16_t *a, const int16_t *w) {
#if defined(__AVX2__)
  __m256i s = _mm256_setzero_si256(), z = _mm256_setzero_si256(), q = _mm256_set1_epi16(QA);
  for (int i = 0; i < H; i += 16) {
    __m256i v = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i *)(a + i)), z), q);
    __m256i vw = _mm256_mullo_epi16(v, _mm256_load_si256((const __m256i *)(w + i)));   // |w| <= 127 keeps this in int16
    s = _mm256_add_epi32(s, _mm256_madd_epi16(vw, v));
  }
  alignas(32) int32_t l[8];
  _mm256_store_si256((__m256i *)l, s);
  int64_t t = 0;
  for (int i = 0; i < 8; i++) t += l[i];
  return t;
#else
  int64_t t = 0;
  for (int i = 0; i < H; i++) { int v = a[i] < 0 ? 0 : a[i] > QA ? QA : a[i]; t += (int64_t)v * v * w[i]; }
  return t;
#endif
}

// side to move's score in centipawns
inline int evaluate(const Position &p, AccStack &st) {
  Acc &a = update(p, st);
  int b = (popcnt(p.occ[2]) - 2) / 4;
  if (b > NB - 1) b = NB - 1;
  int64_t s = screlu_dot(a.v[p.side], net->w2[b]) + screlu_dot(a.v[p.side ^ 1], net->w2[b] + H);
  double y = double(s / QA + net->b2[b]) / (double(QA) * QB);
  return int(std::lround(y * net->cp_scale));
}
}  // namespace nnue

// ---------------- hand-made stand-in eval (material plus simple square tables), for tests and the veto ----------------
namespace handeval {
const int VAL[6] = {100, 310, 330, 500, 900, 0};
inline int center(int s) { int f = s & 7, r = s >> 3; int df = f < 4 ? 3 - f : f - 4, dr = r < 4 ? 3 - r : r - 4; return 6 - df - dr; }
inline int material(const Position &p) {
  int s = 0;
  for (int t = PAWN; t < KING; t++) s += VAL[t] * (popcnt(p.pieces(p.side, t)) - popcnt(p.pieces(p.side ^ 1, t)));
  return s;
}
inline int evaluate(const Position &p) {
  int mg = 0, eg = 0, phase = 0;
  for (int c = 0; c < 2; c++) {
    int sg = c == p.side ? 1 : -1;
    for (int t = PAWN; t <= KING; t++) {
      U64 b = p.pieces(c, t);
      while (b) {
        int s = pop_lsb(b), rr = c == WHITE ? (s >> 3) : 7 - (s >> 3), cn = center(s);
        int m = VAL[t], e = VAL[t];
        if (t == PAWN) { m += rr * 4 + (cn > 3 ? 8 : 0); e += rr * rr * 3; }
        else if (t == KNIGHT) { m += cn * 6 - 15; e += cn * 4 - 10; phase += 1; }
        else if (t == BISHOP) { m += cn * 3; e += cn * 2; phase += 1; }
        else if (t == ROOK) { m += rr == 6 ? 15 : 0; phase += 2; }
        else if (t == QUEEN) { m += cn; e += cn * 2; phase += 4; }
        else { m += (rr == 0 ? 20 : -10 * rr) - cn * 3; e += cn * 6; }
        mg += sg * m; eg += sg * e;
      }
    }
  }
  if (phase > 24) phase = 24;
  return (mg * phase + eg * (24 - phase)) / 24 + 10;
}
}  // namespace handeval
