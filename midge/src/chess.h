// chess.h: board, attack tables and legal move generation for Midge (MayFly's alpha-beta second brain).
// Written from scratch, 17 Nov 2026. Squares: a1 = 0 ... h8 = 63. Piece = colour * 6 + type, 12 = empty.
#pragma once
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

typedef uint64_t U64;
typedef uint16_t Move;

enum { WHITE = 0, BLACK = 1 };
enum { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING };
const int NO_PIECE = 12;
const Move NO_MOVE = 0;
// move = from | to << 6 | flag << 12; promotion piece = KNIGHT + (flag & 3)
enum { F_QUIET = 0, F_DOUBLE = 1, F_KCASTLE = 2, F_QCASTLE = 3, F_CAPTURE = 4, F_EP = 5, F_PROMO = 8, F_PCAPTURE = 12 };

inline int mfrom(Move m) { return m & 63; }
inline int mto(Move m) { return (m >> 6) & 63; }
inline int mflag(Move m) { return m >> 12; }
inline Move mk(int f, int t, int fl) { return Move(f | (t << 6) | (fl << 12)); }
inline bool is_cap(Move m) { return (mflag(m) & 4) != 0; }
inline bool is_promo(Move m) { return (mflag(m) & 8) != 0; }
inline int promo_type(Move m) { return KNIGHT + (mflag(m) & 3); }
inline int pc_make(int c, int t) { return c * 6 + t; }
inline int pc_type(int p) { return p % 6; }
inline int pc_color(int p) { return p / 6; }

inline int lsb(U64 b) { return std::countr_zero(b); }
inline int popcnt(U64 b) { return std::popcount(b); }
inline int pop_lsb(U64 &b) { int s = lsb(b); b &= b - 1; return s; }
inline U64 sqbb(int s) { return 1ULL << s; }

// ---------------- attack tables ----------------
inline U64 KnightAtt[64], KingAtt[64], PawnAtt[2][64], Between[64][64], LineBB[64][64];
struct Magic { U64 mask, magic; U64 *att; int shift; };
inline Magic RookM[64], BishopM[64];
inline U64 RookTab[102400], BishopTab[5248];
inline U64 Zpsq[12][64], Zside, Zcastle[16], Zep[8];
inline int CastleMask[64];

inline U64 rook_att(int s, U64 occ) { const Magic &m = RookM[s]; return m.att[((occ & m.mask) * m.magic) >> m.shift]; }
inline U64 bishop_att(int s, U64 occ) { const Magic &m = BishopM[s]; return m.att[((occ & m.mask) * m.magic) >> m.shift]; }

namespace tables {
inline U64 rng = 0x2545F4914F6CDD1DULL;
inline U64 rnd() { rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; return rng * 2685821657736338717ULL; }
inline bool on(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }
const int RD[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
const int BD[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
// ray walk: attacks with blockers, or the relevant-occupancy mask (edges left out)
inline U64 slide(int sq, U64 occ, const int d[4][2], bool mask) {
  U64 a = 0;
  for (int k = 0; k < 4; k++) {
    int f = (sq & 7) + d[k][0], r = (sq >> 3) + d[k][1];
    while (on(f, r)) {
      if (mask && !on(f + d[k][0], r + d[k][1])) break;
      a |= sqbb(r * 8 + f);
      if (!mask && (occ & sqbb(r * 8 + f))) break;
      f += d[k][0]; r += d[k][1];
    }
  }
  return a;
}
// our own magic search: random sparse numbers until every occupancy subset maps without a bad collision
inline void find_magic(int sq, const int d[4][2], Magic &m, U64 *table) {
  static U64 occ[4096], ref[4096];
  static int used[4096], epoch = 0;
  m.mask = slide(sq, 0, d, true);
  int bits = popcnt(m.mask), n = 1 << bits;
  m.shift = 64 - bits; m.att = table;
  U64 sub = 0;
  for (int i = 0; i < n; i++) { occ[i] = sub; ref[i] = slide(sq, sub, d, false); sub = (sub - m.mask) & m.mask; }
  for (;;) {
    U64 mg = rnd() & rnd() & rnd();
    if (popcnt((m.mask * mg) >> 56) < 6) continue;
    epoch++;
    bool ok = true;
    for (int i = 0; i < n && ok; i++) {
      int idx = int((occ[i] * mg) >> m.shift);
      if (used[idx] != epoch) { used[idx] = epoch; table[idx] = ref[i]; }
      else if (table[idx] != ref[i]) ok = false;
    }
    if (ok) { m.magic = mg; return; }
  }
}
inline void init() {
  for (int s = 0; s < 64; s++) {
    int f = s & 7, r = s >> 3;
    const int kn[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    KnightAtt[s] = KingAtt[s] = 0;
    for (auto &d : kn) if (on(f + d[0], r + d[1])) KnightAtt[s] |= sqbb((r + d[1]) * 8 + f + d[0]);
    for (int df = -1; df <= 1; df++)
      for (int dr = -1; dr <= 1; dr++)
        if ((df || dr) && on(f + df, r + dr)) KingAtt[s] |= sqbb((r + dr) * 8 + f + df);
    PawnAtt[WHITE][s] = PawnAtt[BLACK][s] = 0;
    if (r < 7) { if (f > 0) PawnAtt[WHITE][s] |= sqbb(s + 7); if (f < 7) PawnAtt[WHITE][s] |= sqbb(s + 9); }
    if (r > 0) { if (f > 0) PawnAtt[BLACK][s] |= sqbb(s - 9); if (f < 7) PawnAtt[BLACK][s] |= sqbb(s - 7); }
  }
  U64 *rt = RookTab, *bt = BishopTab;
  for (int s = 0; s < 64; s++) {
    find_magic(s, RD, RookM[s], rt); rt += 1ULL << (64 - RookM[s].shift);
    find_magic(s, BD, BishopM[s], bt); bt += 1ULL << (64 - BishopM[s].shift);
  }
  // between and line masks
  for (int a = 0; a < 64; a++)
    for (int b = 0; b < 64; b++) {
      Between[a][b] = LineBB[a][b] = 0;
      if (a == b) continue;
      if (rook_att(a, 0) & sqbb(b)) {
        Between[a][b] = rook_att(a, sqbb(b)) & rook_att(b, sqbb(a));
        LineBB[a][b] = (rook_att(a, 0) & rook_att(b, 0)) | sqbb(a) | sqbb(b);
      } else if (bishop_att(a, 0) & sqbb(b)) {
        Between[a][b] = bishop_att(a, sqbb(b)) & bishop_att(b, sqbb(a));
        LineBB[a][b] = (bishop_att(a, 0) & bishop_att(b, 0)) | sqbb(a) | sqbb(b);
      }
    }
  for (int p = 0; p < 12; p++) for (int s = 0; s < 64; s++) Zpsq[p][s] = rnd();
  Zside = rnd();
  for (int i = 0; i < 16; i++) Zcastle[i] = rnd();
  for (int i = 0; i < 8; i++) Zep[i] = rnd();
  for (int s = 0; s < 64; s++) CastleMask[s] = 15;
  CastleMask[4] = 15 ^ 3; CastleMask[7] = 15 ^ 1; CastleMask[0] = 15 ^ 2;
  CastleMask[60] = 15 ^ 12; CastleMask[63] = 15 ^ 4; CastleMask[56] = 15 ^ 8;
}
}  // namespace tables

// ---------------- position ----------------
const int MAXSP = 4096;           // moves since the position was set (game plus search)
struct Undo { int captured, castle, ep, rule50; U64 key; Move move; };
struct Dirty { int n; int pc[3], from[3], to[3]; };  // piece changes of one move, for the network; -1 = none

struct Position {
  U64 bb[12], occ[3];
  int board[64];
  int side, castle, ep, rule50, sp;
  U64 key, pkey;   // pkey: pawns only
  U64 npkey[2] = {0, 0};   // each colour's pieces other than pawns, the king too (for the static-score corrections)
  Undo st[MAXSP];
  U64 keys[MAXSP];
  Dirty dirty[MAXSP];

  void put(int s, int p) { board[s] = p; bb[p] |= sqbb(s); occ[pc_color(p)] |= sqbb(s); occ[2] |= sqbb(s); key ^= Zpsq[p][s]; if (pc_type(p) == PAWN) pkey ^= Zpsq[p][s]; else npkey[pc_color(p)] ^= Zpsq[p][s]; }
  void remove(int s) { int p = board[s]; board[s] = NO_PIECE; bb[p] ^= sqbb(s); occ[pc_color(p)] ^= sqbb(s); occ[2] ^= sqbb(s); key ^= Zpsq[p][s]; if (pc_type(p) == PAWN) pkey ^= Zpsq[p][s]; else npkey[pc_color(p)] ^= Zpsq[p][s]; }
  void move_pc(int f, int t) { int p = board[f]; remove(f); put(t, p); }
  U64 pieces(int c, int t) const { return bb[pc_make(c, t)]; }
  int king_sq(int c) const { return lsb(bb[pc_make(c, KING)]); }

  U64 attackers_to(int s, U64 o) const {
    return (PawnAtt[BLACK][s] & bb[pc_make(WHITE, PAWN)]) | (PawnAtt[WHITE][s] & bb[pc_make(BLACK, PAWN)]) |
           (KnightAtt[s] & (bb[KNIGHT] | bb[6 + KNIGHT])) | (KingAtt[s] & (bb[KING] | bb[6 + KING])) |
           (bishop_att(s, o) & (bb[BISHOP] | bb[6 + BISHOP] | bb[QUEEN] | bb[6 + QUEEN])) |
           (rook_att(s, o) & (bb[ROOK] | bb[6 + ROOK] | bb[QUEEN] | bb[6 + QUEEN]));
  }
  bool attacked(int s, int by, U64 o) const {
    if (PawnAtt[by ^ 1][s] & pieces(by, PAWN)) return true;
    if (KnightAtt[s] & pieces(by, KNIGHT)) return true;
    if (KingAtt[s] & pieces(by, KING)) return true;
    if (bishop_att(s, o) & (pieces(by, BISHOP) | pieces(by, QUEEN))) return true;
    if (rook_att(s, o) & (pieces(by, ROOK) | pieces(by, QUEEN))) return true;
    return false;
  }
  bool in_check() const { return attacked(king_sq(side), side ^ 1, occ[2]); }

  bool set_fen(const std::string &fen) {
    std::istringstream is(fen);
    std::string b, s, c, e;
    int hm = 0, fm = 1;
    is >> b >> s >> c >> e;
    if (!(is >> hm)) hm = 0;
    if (!(is >> fm)) fm = 1;
    memset(bb, 0, sizeof bb); memset(occ, 0, sizeof occ);
    for (int i = 0; i < 64; i++) board[i] = NO_PIECE;
    key = 0; pkey = 0; npkey[0] = npkey[1] = 0;
    int r = 7, f = 0;
    for (char ch : b) {
      if (ch == '/') { r--; f = 0; continue; }
      if (ch >= '1' && ch <= '8') { f += ch - '0'; continue; }
      const char *pcs = "PNBRQKpnbrqk";
      const char *q = strchr(pcs, ch);
      if (!q || f > 7 || r < 0) return false;
      put(r * 8 + f, int(q - pcs));
      f++;
    }
    side = (s == "b") ? BLACK : WHITE;
    castle = 0;
    for (char ch : c) castle |= ch == 'K' ? 1 : ch == 'Q' ? 2 : ch == 'k' ? 4 : ch == 'q' ? 8 : 0;
    ep = -1;
    if (e.size() == 2 && e[0] >= 'a' && e[0] <= 'h') {
      int es = (e[1] - '1') * 8 + (e[0] - 'a');
      if (PawnAtt[side ^ 1][es] & pieces(side, PAWN)) ep = es;   // only when a capture is possible
    }
    rule50 = hm; sp = 0;
    if (side == BLACK) key ^= Zside;
    key ^= Zcastle[castle];
    if (ep >= 0) key ^= Zep[ep & 7];
    return popcnt(pieces(WHITE, KING)) == 1 && popcnt(pieces(BLACK, KING)) == 1;
  }
  std::string fen() const {
    std::string o;
    for (int r = 7; r >= 0; r--) {
      int empty = 0;
      for (int f = 0; f < 8; f++) {
        int p = board[r * 8 + f];
        if (p == NO_PIECE) { empty++; continue; }
        if (empty) { o += char('0' + empty); empty = 0; }
        o += "PNBRQKpnbrqk"[p];
      }
      if (empty) o += char('0' + empty);
      if (r) o += '/';
    }
    o += side == WHITE ? " w " : " b ";
    std::string c;
    if (castle & 1) c += 'K';
    if (castle & 2) c += 'Q';
    if (castle & 4) c += 'k';
    if (castle & 8) c += 'q';
    o += c.empty() ? "-" : c;
    o += ' ';
    if (ep >= 0) { o += char('a' + (ep & 7)); o += char('1' + (ep >> 3)); } else o += '-';
    o += ' ' + std::to_string(rule50) + " 1";
    return o;
  }

  void make(Move m) {
    Undo &u = st[sp];
    u.captured = NO_PIECE; u.castle = castle; u.ep = ep; u.rule50 = rule50; u.key = key; u.move = m;
    keys[sp] = key;
    sp++;
    Dirty &d = dirty[sp];
    d.n = 0;
    int f = mfrom(m), t = mto(m), fl = mflag(m), p = board[f], us = side, them = us ^ 1;
    rule50++;
    if (ep >= 0) { key ^= Zep[ep & 7]; ep = -1; }
    if (fl == F_EP) {
      int cs = t + (us == WHITE ? -8 : 8);
      u.captured = board[cs];
      d.pc[d.n] = board[cs]; d.from[d.n] = cs; d.to[d.n++] = -1;
      remove(cs);
    } else if (fl & 4) {
      u.captured = board[t];
      d.pc[d.n] = board[t]; d.from[d.n] = t; d.to[d.n++] = -1;
      remove(t);
    }
    if (u.captured != NO_PIECE) rule50 = 0;
    if (fl & 8) {
      int np = pc_make(us, promo_type(m));
      remove(f); put(t, np);
      d.pc[d.n] = p; d.from[d.n] = f; d.to[d.n++] = -1;
      d.pc[d.n] = np; d.from[d.n] = -1; d.to[d.n++] = t;
    } else {
      move_pc(f, t);
      d.pc[d.n] = p; d.from[d.n] = f; d.to[d.n++] = t;
    }
    if (pc_type(p) == PAWN) {
      rule50 = 0;
      if (fl == F_DOUBLE) {
        int e = (f + t) / 2;
        if (PawnAtt[us][e] & pieces(them, PAWN)) { ep = e; key ^= Zep[e & 7]; }
      }
    }
    if (fl == F_KCASTLE || fl == F_QCASTLE) {
      int rf = fl == F_KCASTLE ? f + 3 : f - 4, rt = fl == F_KCASTLE ? f + 1 : f - 1;
      d.pc[d.n] = board[rf]; d.from[d.n] = rf; d.to[d.n++] = rt;
      move_pc(rf, rt);
    }
    key ^= Zcastle[castle];
    castle &= CastleMask[f] & CastleMask[t];
    key ^= Zcastle[castle];
    side = them;
    key ^= Zside;
  }
  void unmake() {
    sp--;
    Undo &u = st[sp];
    Move m = u.move;
    int f = mfrom(m), t = mto(m), fl = mflag(m);
    side ^= 1;
    int us = side;
    if (fl & 8) { remove(t); put(f, pc_make(us, PAWN)); }
    else move_pc(t, f);
    if (fl == F_KCASTLE || fl == F_QCASTLE) {
      int rf = fl == F_KCASTLE ? f + 3 : f - 4, rt = fl == F_KCASTLE ? f + 1 : f - 1;
      move_pc(rt, rf);
    }
    if (u.captured != NO_PIECE) put(fl == F_EP ? t + (us == WHITE ? -8 : 8) : t, u.captured);
    castle = u.castle; ep = u.ep; rule50 = u.rule50; key = u.key;
  }
  void make_null() {
    Undo &u = st[sp];
    u.captured = NO_PIECE; u.castle = castle; u.ep = ep; u.rule50 = rule50; u.key = key; u.move = NO_MOVE;
    keys[sp] = key;
    sp++;
    dirty[sp].n = 0;
    if (ep >= 0) { key ^= Zep[ep & 7]; ep = -1; }
    rule50++;
    side ^= 1; key ^= Zside;
  }
  void unmake_null() {
    sp--;
    Undo &u = st[sp];
    side ^= 1;
    ep = u.ep; rule50 = u.rule50; key = u.key;
  }
  // a repeat of an earlier position since the last irreversible move
  bool repeated() const {
    int back = rule50 < sp ? rule50 : sp;
    for (int i = 4; i <= back; i += 2)
      if (keys[sp - i] == key) return true;
    return false;
  }
  bool insufficient() const {
    if (bb[PAWN] | bb[6 + PAWN] | bb[ROOK] | bb[6 + ROOK] | bb[QUEEN] | bb[6 + QUEEN]) return false;
    return popcnt(occ[2]) <= 3;   // king and at most one minor piece against a bare king
  }
};

// ---------------- legal move generation ----------------
enum { GEN_ALL = 0, GEN_NOISY = 1 };   // noisy = captures and queen promotions

inline int gen_moves(const Position &p, Move *list, int mode) {
  int n = 0, us = p.side, them = us ^ 1, ks = p.king_sq(us);
  U64 own = p.occ[us], opp = p.occ[them], all = p.occ[2];
  U64 checkers = p.attackers_to(ks, all) & opp;
  // king
  U64 ka = KingAtt[ks] & ~own;
  if (mode == GEN_NOISY) ka &= opp;
  while (ka) {
    int t = pop_lsb(ka);
    if (!p.attacked(t, them, all ^ sqbb(ks))) list[n++] = mk(ks, t, (opp & sqbb(t)) ? F_CAPTURE : F_QUIET);
  }
  if (popcnt(checkers) > 1) return n;
  U64 target = checkers ? (checkers | Between[ks][lsb(checkers)]) : ~own;
  if (mode == GEN_NOISY) target &= opp | (checkers ? 0 : 0);
  U64 pinned = 0;
  U64 snipers = ((rook_att(ks, 0) & (p.pieces(them, ROOK) | p.pieces(them, QUEEN))) |
                 (bishop_att(ks, 0) & (p.pieces(them, BISHOP) | p.pieces(them, QUEEN))));
  while (snipers) {
    int s = pop_lsb(snipers);
    U64 b = Between[ks][s] & all;
    if (popcnt(b) == 1 && (b & own)) pinned |= b;
  }
  // castling
  if (mode == GEN_ALL && !checkers) {
    if (us == WHITE) {
      if ((p.castle & 1) && !(all & 0x60ULL) && !p.attacked(5, them, all) && !p.attacked(6, them, all)) list[n++] = mk(4, 6, F_KCASTLE);
      if ((p.castle & 2) && !(all & 0x0EULL) && !p.attacked(3, them, all) && !p.attacked(2, them, all)) list[n++] = mk(4, 2, F_QCASTLE);
    } else {
      if ((p.castle & 4) && !(all & (0x60ULL << 56)) && !p.attacked(61, them, all) && !p.attacked(62, them, all)) list[n++] = mk(60, 62, F_KCASTLE);
      if ((p.castle & 8) && !(all & (0x0EULL << 56)) && !p.attacked(59, them, all) && !p.attacked(58, them, all)) list[n++] = mk(60, 58, F_QCASTLE);
    }
  }
  // knights, bishops, rooks, queens
  for (int t = KNIGHT; t <= QUEEN; t++) {
    U64 b = p.pieces(us, t);
    if (t == KNIGHT) b &= ~pinned;
    while (b) {
      int f = pop_lsb(b);
      U64 a = t == KNIGHT ? KnightAtt[f] : t == BISHOP ? bishop_att(f, all) : t == ROOK ? rook_att(f, all) : (bishop_att(f, all) | rook_att(f, all));
      a &= target;
      if (pinned & sqbb(f)) a &= LineBB[ks][f];
      while (a) { int to = pop_lsb(a); list[n++] = mk(f, to, (opp & sqbb(to)) ? F_CAPTURE : F_QUIET); }
    }
  }
  // pawns
  U64 tgt_all = checkers ? (checkers | Between[ks][lsb(checkers)]) : ~0ULL;
  int fwd = us == WHITE ? 8 : -8;
  U64 pw = p.pieces(us, PAWN);
  while (pw) {
    int f = pop_lsb(pw);
    U64 allow = (pinned & sqbb(f)) ? LineBB[ks][f] : ~0ULL;
    int r = f >> 3;
    bool promo = us == WHITE ? r == 6 : r == 1;
    int t = f + fwd;
    if (!(all & sqbb(t)) && (sqbb(t) & tgt_all & allow)) {
      if (promo) {
        list[n++] = mk(f, t, F_PROMO | 3);
        if (mode == GEN_ALL) { list[n++] = mk(f, t, F_PROMO | 0); list[n++] = mk(f, t, F_PROMO | 1); list[n++] = mk(f, t, F_PROMO | 2); }
      } else if (mode == GEN_ALL) list[n++] = mk(f, t, F_QUIET);
    }
    if (mode == GEN_ALL && (us == WHITE ? r == 1 : r == 6) && !(all & sqbb(t))) {
      int t2 = t + fwd;
      if (!(all & sqbb(t2)) && (sqbb(t2) & tgt_all & allow)) list[n++] = mk(f, t2, F_DOUBLE);
    }
    U64 caps = PawnAtt[us][f] & opp & tgt_all & allow;
    while (caps) {
      int to = pop_lsb(caps);
      if (promo) {
        list[n++] = mk(f, to, F_PCAPTURE | 3);
        if (mode == GEN_ALL) { list[n++] = mk(f, to, F_PCAPTURE | 0); list[n++] = mk(f, to, F_PCAPTURE | 1); list[n++] = mk(f, to, F_PCAPTURE | 2); }
      } else list[n++] = mk(f, to, F_CAPTURE);
    }
    if (p.ep >= 0 && (PawnAtt[us][f] & sqbb(p.ep))) {
      int cs = p.ep - fwd;
      U64 o2 = (all ^ sqbb(f) ^ sqbb(cs)) | sqbb(p.ep);
      // the full test after the capture covers pins, checks and the two-pawn rank case
      U64 att = (rook_att(ks, o2) & (p.pieces(them, ROOK) | p.pieces(them, QUEEN))) |
                (bishop_att(ks, o2) & (p.pieces(them, BISHOP) | p.pieces(them, QUEEN))) |
                (KnightAtt[ks] & p.pieces(them, KNIGHT)) | (PawnAtt[us][ks] & p.pieces(them, PAWN) & ~sqbb(cs));
      if (!att) list[n++] = mk(f, p.ep, F_EP);
    }
  }
  return n;
}

inline std::string move_str(Move m) {
  if (!m) return "0000";
  std::string s;
  s += char('a' + (mfrom(m) & 7)); s += char('1' + (mfrom(m) >> 3));
  s += char('a' + (mto(m) & 7)); s += char('1' + (mto(m) >> 3));
  if (is_promo(m)) s += "nbrq"[mflag(m) & 3];
  return s;
}
inline Move parse_move(const Position &p, const std::string &s) {
  Move list[256];
  int n = gen_moves(p, list, GEN_ALL);
  for (int i = 0; i < n; i++) if (move_str(list[i]) == s) return list[i];
  return NO_MOVE;
}

inline U64 perft(Position &p, int depth) {
  Move list[256];
  int n = gen_moves(p, list, GEN_ALL);
  if (depth <= 1) return depth == 1 ? n : 1;
  U64 total = 0;
  for (int i = 0; i < n; i++) { p.make(list[i]); total += perft(p, depth - 1); p.unmake(); }
  return total;
}
