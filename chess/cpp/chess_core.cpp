// See chess_core.h. The generator below follows native/movegen.c function for function.
// GPL-3.0-or-later, derived from python-chess 1.11 through movegen.c.
#include "chess_core.h"

#include <cstring>
#include <sstream>
#ifdef _MSC_VER
#include <intrin.h>
static __inline int msb64(uint64_t b) { unsigned long i; _BitScanReverse64(&i, b); return (int)i; }
static __inline int popc(uint64_t b) { return (int)__popcnt64(b); }
#else
static inline int msb64(uint64_t b) { return 63 - __builtin_clzll(b); }
static inline int popc(uint64_t b) { return __builtin_popcountll(b); }
#endif

namespace mayfly {

#define CAP 1024
#define BB(s) (1ULL << (s))
#define ALL 0xFFFFFFFFFFFFFFFFULL
#define RANK_1 0xFFULL
#define RANK_8 (0xFFULL << 56)
#define RANK(r) (0xFFULL << (8 * (r)))
#define FILE_C 0x0404040404040404ULL
#define FILE_D 0x0808080808080808ULL
#define FILE_F 0x2020202020202020ULL
#define FILE_G 0x4040404040404040ULL
#define LIGHT 0x55AA55AA55AA55AAULL
#define DARK 0xAA55AA55AA55AA55ULL

static u64 KNIGHT[64], KING[64], PAWN[2][64], RAYS[64][64];
static u64 RANKE[64], FILEE[64], DIAGE[64];
static int64_t IDXTAB[2][6 * 4096];
static bool READY = false;

static u64 slide(int sq, u64 occ, const int* d, int nd)
{
    u64 a = 0;
    int f0 = sq & 7, r0 = sq >> 3;
    for (int k = 0; k < nd; k++) {
        int f = f0 + d[2 * k], r = r0 + d[2 * k + 1];
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
            int s = r * 8 + f;
            a |= BB(s);
            if (occ & BB(s)) break;
            f += d[2 * k]; r += d[2 * k + 1];
        }
    }
    return a;
}
static const int DR[4] = {1, 0, -1, 0}, DF[4] = {0, 1, 0, -1};
static const int DD[8] = {1, 1, 1, -1, -1, 1, -1, -1};
static __inline u64 rank_att(int sq, u64 occ) { return slide(sq, occ, DR, 2); }
static __inline u64 file_att(int sq, u64 occ) { return slide(sq, occ, DF, 2); }
static __inline u64 rook_att(int sq, u64 occ) { return rank_att(sq, occ) | file_att(sq, occ); }
static __inline u64 diag_att(int sq, u64 occ) { return slide(sq, occ, DD, 4); }
static __inline u64 between(int a, int b)
{
    u64 bb = RAYS[a][b] & ((ALL << a) ^ (ALL << b));
    return bb & (bb - 1);
}

struct S {
    u64 P, N, Bi, R, Q, K, co[2], occ, castle;
    int us, ep;
    int n;
    int overflow;
    int code[CAP];
};

static __inline void emit(S* s, int f, int t, int promo)
{
    if (s->n < CAP) s->code[s->n++] = promo * 4096 + f * 64 + t;
    else s->overflow = 1;
}

static __inline u64 attackers(const S* s, int color, int sq, u64 occ)
{
    u64 qr = s->Q | s->R, qb = s->Q | s->Bi;
    u64 a = (KING[sq] & s->K) | (KNIGHT[sq] & s->N) | (rook_att(sq, occ) & qr) | (diag_att(sq, occ) & qb)
            | (PAWN[!color][sq] & s->P);
    return a & s->co[color];
}

static __inline u64 attacks_mask(const S* s, int sq)
{
    u64 b = BB(sq);
    if (b & s->N) return KNIGHT[sq];
    if (b & s->K) return KING[sq];
    u64 a = 0;
    if ((b & s->Bi) || (b & s->Q)) a = diag_att(sq, s->occ);
    if ((b & s->R) || (b & s->Q)) a |= rook_att(sq, s->occ);
    return a;
}

static void pseudo_ep(S* s, u64 from_mask, u64 to_mask)
{
    if (s->ep <= 0 || !(BB(s->ep) & to_mask)) return;
    if (BB(s->ep) & s->occ) return;
    u64 cap = s->P & s->co[s->us] & from_mask & PAWN[!s->us][s->ep] & RANK(s->us ? 4 : 3);
    while (cap) { int c = msb64(cap); emit(s, c, s->ep, 0); cap ^= BB(c); }
}

static int attacked_for_king(const S* s, u64 path, u64 occ)
{
    while (path) { int q = msb64(path); if (attackers(s, !s->us, q, occ)) return 1; path ^= BB(q); }
    return 0;
}

static void castling(S* s, u64 from_mask, u64 to_mask)
{
    u64 back = s->us ? RANK_1 : RANK_8;
    u64 king = s->co[s->us] & s->K & back & from_mask;
    king &= (~king + 1);
    if (!king) return;
    u64 bc = FILE_C & back, bd = FILE_D & back, bf = FILE_F & back, bg = FILE_G & back;
    u64 cand = s->castle & back & to_mask;
    int ks = msb64(king);
    while (cand) {
        int c = msb64(cand);
        u64 rook = BB(c);
        int a_side = rook < king;
        u64 kto = a_side ? bc : bg, rto = a_side ? bd : bf;
        u64 kpath = between(ks, msb64(kto)), rpath = between(c, msb64(rto));
        if (!(((s->occ ^ king ^ rook) & (kpath | rpath | kto | rto))
              || attacked_for_king(s, kpath | king, s->occ ^ king)
              || attacked_for_king(s, kto, s->occ ^ king ^ rook ^ rto))) {
            int to = c;
            if (ks == 4 && (s->K & BB(4))) { if (c == 7) to = 6; else if (c == 0) to = 2; }
            else if (ks == 60 && (s->K & BB(60))) { if (c == 63) to = 62; else if (c == 56) to = 58; }
            emit(s, ks, to, 0);
        }
        cand ^= BB(c);
    }
}

static void pseudo(S* s, u64 from_mask, u64 to_mask)
{
    u64 ours = s->co[s->us];
    u64 np = ours & ~s->P & from_mask;
    while (np) {
        int f = msb64(np);
        u64 m = attacks_mask(s, f) & ~ours & to_mask;
        while (m) { int t = msb64(m); emit(s, f, t, 0); m ^= BB(t); }
        np ^= BB(f);
    }
    if (from_mask & s->K) castling(s, from_mask, to_mask);
    u64 pawns = s->P & ours & from_mask;
    if (!pawns) return;
    u64 cap = pawns;
    while (cap) {
        int f = msb64(cap);
        u64 tg = PAWN[s->us][f] & s->co[!s->us] & to_mask;
        while (tg) {
            int t = msb64(tg);
            if ((t >> 3) == 0 || (t >> 3) == 7) { emit(s, f, t, 5); emit(s, f, t, 4); emit(s, f, t, 3); emit(s, f, t, 2); }
            else emit(s, f, t, 0);
            tg ^= BB(t);
        }
        cap ^= BB(f);
    }
    u64 single, dbl;
    if (s->us) {
        single = (pawns << 8) & ~s->occ;
        dbl = (single << 8) & ~s->occ & (RANK(2) | RANK(3));
    } else {
        single = (pawns >> 8) & ~s->occ;
        dbl = (single >> 8) & ~s->occ & (RANK(5) | RANK(4));
    }
    single &= to_mask; dbl &= to_mask;
    while (single) {
        int t = msb64(single), f = t + (s->us ? -8 : 8);
        if ((t >> 3) == 0 || (t >> 3) == 7) { emit(s, f, t, 5); emit(s, f, t, 4); emit(s, f, t, 3); emit(s, f, t, 2); }
        else emit(s, f, t, 0);
        single ^= BB(t);
    }
    while (dbl) { int t = msb64(dbl); emit(s, t + (s->us ? -16 : 16), t, 0); dbl ^= BB(t); }
    if (s->ep > 0) pseudo_ep(s, from_mask, to_mask);
}

static void evasions(S* s, int king, u64 checkers)
{
    u64 sliders = checkers & (s->Bi | s->R | s->Q), attacked = 0, x = sliders;
    while (x) { int c = msb64(x); attacked |= RAYS[king][c] & ~BB(c); x ^= BB(c); }
    u64 km = KING[king] & ~s->co[s->us] & ~attacked;
    while (km) { int t = msb64(km); emit(s, king, t, 0); km ^= BB(t); }
    int checker = msb64(checkers);
    if (BB(checker) == checkers) {
        u64 target = between(king, checker) | checkers;
        pseudo(s, ~s->K, target);
        if (s->ep > 0 && !(BB(s->ep) & target)) {
            int last_double = s->ep + (s->us ? -8 : 8);
            if (last_double == checker) pseudo_ep(s, ALL, ALL);
        }
    }
}

static u64 pin_mask(const S* s, int sq)
{
    u64 km = s->co[s->us] & s->K;
    if (!km) return ALL;
    int king = msb64(km);
    u64 sm = BB(sq);
    const u64* tabs[3] = {FILEE, RANKE, DIAGE};
    u64 sl[3] = {s->R | s->Q, s->R | s->Q, s->Bi | s->Q};
    for (int i = 0; i < 3; i++) {
        u64 rays = tabs[i][king];
        if (rays & sm) {
            u64 sn = rays & sl[i] & s->co[!s->us];
            while (sn) {
                int x = msb64(sn);
                if ((between(x, king) & (s->occ | sm)) == sm) return RAYS[king][x];
                sn ^= BB(x);
            }
            break;
        }
    }
    return ALL;
}

static int ep_skewered(const S* s, int king, int capturer)
{
    int last_double = s->ep + (s->us ? -8 : 8);
    u64 occ = (s->occ & ~BB(last_double) & ~BB(capturer)) | BB(s->ep);
    u64 h = s->co[!s->us] & (s->R | s->Q);
    if (rank_att(king, occ) & h) return 1;
    u64 d = s->co[!s->us] & (s->Bi | s->Q);
    if (diag_att(king, occ) & d) return 1;
    return 0;
}

static int is_safe(const S* s, int king, u64 blockers, int code)
{
    int f = (code >> 6) & 63, t = code & 63;
    if (f == king) {
        int diff = (f & 7) - (t & 7);
        if (diff > 1 || diff < -1 || (s->R & s->co[s->us] & BB(t))) return 1;
        return attackers(s, !s->us, t, s->occ) == 0;
    }
    int d = t - f; if (d < 0) d = -d;
    if (s->ep == t && (s->P & BB(f)) && (d == 7 || d == 9) && !(s->occ & BB(t)))
        return (pin_mask(s, f) & BB(t)) && !ep_skewered(s, king, f);
    return !(blockers & BB(f)) || (RAYS[f][t] & BB(king));
}

static void legal_codes(S* s, int* check)
{
    u64 km = s->K & s->co[s->us];
    *check = 0;
    if (!km) { pseudo(s, ALL, ALL); return; }
    int king = msb64(km);
    u64 rq = s->R | s->Q, bq = s->Bi | s->Q;
    u64 sn = ((RANKE[king] & rq) | (FILEE[king] & rq) | (DIAGE[king] & bq)) & s->co[!s->us], blockers = 0;
    while (sn) {
        int x = msb64(sn);
        u64 b = between(king, x) & s->occ;
        if (b && BB(msb64(b)) == b) blockers |= b;
        sn ^= BB(x);
    }
    blockers &= s->co[s->us];
    u64 checkers = attackers(s, !s->us, king, s->occ);
    static thread_local S tmp;
    tmp = *s; tmp.n = 0; tmp.overflow = 0;
    if (checkers) { *check = 1; evasions(&tmp, king, checkers); }
    else pseudo(&tmp, ALL, ALL);
    s->n = 0;
    s->overflow = tmp.overflow;
    for (int i = 0; i < tmp.n; i++)
        if (is_safe(s, king, blockers, tmp.code[i])) s->code[s->n++] = tmp.code[i];
}

static int insufficient_color(const S* s, int c)
{
    u64 mine = s->co[c], theirs = s->co[!c];
    if (mine & (s->P | s->R | s->Q)) return 0;
    if (mine & s->N) return popc(mine) <= 2 && !(theirs & ~s->K & ~s->Q);
    if (mine & s->Bi) {
        int same = !(s->Bi & DARK) || !(s->Bi & LIGHT);
        return same && !s->P && !s->N;
    }
    return 1;
}

static void from_key(const Key& k, S* s)
{
    s->P = k.P; s->N = k.N; s->Bi = k.B; s->R = k.R; s->Q = k.Q; s->K = k.K;
    s->co[1] = k.W; s->co[0] = k.Bk; s->occ = k.W | k.Bk;
    s->us = k.turn ? 1 : 0; s->castle = k.castle; s->ep = k.ep;
    s->n = 0; s->overflow = 0;
}

static void from_pos(const Pos& p, u64 castle, int ep, S* s)
{
    s->P = p.P; s->N = p.N; s->Bi = p.B; s->R = p.R; s->Q = p.Q; s->K = p.K;
    s->co[1] = p.co[1]; s->co[0] = p.co[0]; s->occ = p.co[0] | p.co[1];
    s->us = p.turn; s->castle = castle; s->ep = ep;
    s->n = 0; s->overflow = 0;
}

static int gives_check(const S* s, int code)
{
    int f = (code >> 6) & 63, t = code & 63, promo = code >> 12, us = s->us;
    u64 P = s->P, N = s->N, Bi = s->Bi, R = s->R, Q = s->Q, K = s->K, co[2] = {s->co[0], s->co[1]};
    u64 fb = BB(f), tb = BB(t);
    int castle = (K & fb) && (((f & 7) - (t & 7) > 1) || ((t & 7) - (f & 7) > 1) || (R & co[us] & tb));
    int ep = (s->ep == t) && (P & fb) && (t - f == 7 || t - f == 9 || f - t == 7 || f - t == 9) && !(s->occ & tb);
    u64* pcs[6] = {&P, &N, &Bi, &R, &Q, &K};
    int moved = -1;
    for (int i = 0; i < 6; i++) if (*pcs[i] & fb) { moved = i; *pcs[i] &= ~fb; }
    co[us] &= ~fb;
    if (castle) {
        int back = us ? 0 : 56, aside = (t & 7) < (f & 7);
        int rook_from = aside ? back + 0 : back + 7;
        if (R & co[us] & tb) rook_from = t;
        R &= ~BB(rook_from); co[us] &= ~BB(rook_from);
        int kto = back + (aside ? 2 : 6), rto = back + (aside ? 3 : 5);
        K |= BB(kto); R |= BB(rto); co[us] |= BB(kto) | BB(rto);
    } else {
        for (int i = 0; i < 6; i++) *pcs[i] &= ~tb;
        co[!us] &= ~tb;
        if (ep) { int cap = t + (us ? -8 : 8); P &= ~BB(cap); co[!us] &= ~BB(cap); }
        int put = promo ? promo - 1 : moved;
        *pcs[put] |= tb; co[us] |= tb;
    }
    u64 theirK = K & co[!us];
    if (!theirK) return 0;
    int ks = msb64(theirK);
    u64 occ = co[0] | co[1];
    u64 a = (KING[ks] & K) | (KNIGHT[ks] & N) | (rook_att(ks, occ) & (Q | R)) | (diag_att(ks, occ) & (Q | Bi))
            | (PAWN[!us][ks] & P);
    return (a & co[us]) != 0;
}

// ------------------------------------------------------------------ tables (python-chess's definitions)
static int sqdist(int a, int b)
{
    int df = (a & 7) - (b & 7), dr = (a >> 3) - (b >> 3);
    if (df < 0) df = -df;
    if (dr < 0) dr = -dr;
    return df > dr ? df : dr;
}
static u64 step_attacks(int sq, const int* deltas, int n)
{
    u64 a = 0;
    for (int i = 0; i < n; i++) {
        int t = sq + deltas[i];
        if (t >= 0 && t < 64 && sqdist(t, sq) <= 2) a |= BB(t);
    }
    return a;
}

int64_t move_to_idx(int code, bool flip)
{
    int f = (code >> 6) & 63, t = code & 63, promo = code >> 12;
    if (flip) { f ^= 56; t ^= 56; }
    if (promo && promo != 5) {
        // PROMO_IDX in chess/moves.py, with f on the seventh rank, t in (f+7, f+8, f+9) and piece in (N, B, R)
        int n = 4096;
        for (int ff = 48; ff < 56; ff++)
            for (int dt = 7; dt <= 9; dt++) {
                int tt = ff + dt;
                int dfile = (tt % 8) - (ff % 8);
                if (!(tt >= 56 && tt <= 63) || dfile > 1 || dfile < -1) continue;
                for (int p = 2; p <= 4; p++) {
                    if (ff == f && tt == t && p == promo) return n;
                    n++;
                }
            }
        return f * 64 + t;
    }
    return f * 64 + t;
}

void init_tables()
{
    if (READY) return;
    static const int KN[8] = {17, 15, 10, 6, -17, -15, -10, -6}, KG[8] = {9, 8, 7, 1, -9, -8, -7, -1};
    static const int PB[2] = {-7, -9}, PW[2] = {7, 9};
    for (int sq = 0; sq < 64; sq++) {
        KNIGHT[sq] = step_attacks(sq, KN, 8);
        KING[sq] = step_attacks(sq, KG, 8);
        PAWN[0][sq] = step_attacks(sq, PB, 2);
        PAWN[1][sq] = step_attacks(sq, PW, 2);
        RANKE[sq] = rank_att(sq, 0); FILEE[sq] = file_att(sq, 0); DIAGE[sq] = diag_att(sq, 0);
    }
    for (int a = 0; a < 64; a++)
        for (int b = 0; b < 64; b++) {
            u64 bb = BB(b);
            if (DIAGE[a] & bb) RAYS[a][b] = (DIAGE[a] & DIAGE[b]) | BB(a) | bb;
            else if (RANKE[a] & bb) RAYS[a][b] = RANKE[a] | BB(a);
            else if (FILEE[a] & bb) RAYS[a][b] = FILEE[a] | BB(a);
            else RAYS[a][b] = 0;
        }
    for (int c = 0; c < 6 * 4096; c++) {
        IDXTAB[0][c] = move_to_idx(c, false);
        IDXTAB[1][c] = move_to_idx(c, true);
    }
    READY = true;
}

// ------------------------------------------------------------------ positions
u64 clean_castling(const Pos& p)
{
    u64 castling = p.castling & p.R;
    u64 wc = castling & RANK_1 & p.co[1], bc = castling & RANK_8 & p.co[0];
    wc &= (BB(0) | BB(7));
    bc &= (BB(56) | BB(63));
    if (!(p.co[1] & p.K & BB(4))) wc = 0;
    if (!(p.co[0] & p.K & BB(60))) bc = 0;
    return wc | bc;
}

Key make_key(const Pos& p)
{
    Key k;
    k.P = p.P; k.N = p.N; k.B = p.B; k.R = p.R; k.Q = p.Q; k.K = p.K; k.W = p.co[1]; k.Bk = p.co[0];
    k.turn = p.turn; k.castle = clean_castling(p); k.ep = -1;
    if (p.ep >= 0) {
        // has_legal_en_passant: any legal move that captures en passant
        static thread_local S s;
        from_pos(p, k.castle, p.ep, &s);
        int check;
        legal_codes(&s, &check);
        for (int i = 0; i < s.n; i++) {
            int f = (s.code[i] >> 6) & 63, t = s.code[i] & 63, d = t - f;
            if (d < 0) d = -d;
            if (t == p.ep && (p.P & BB(f)) && (d == 7 || d == 9) && !(s.occ & BB(t))) { k.ep = p.ep; break; }
        }
    }
    return k;
}

static int piece_at(const Pos& p, int sq)
{
    u64 b = BB(sq);
    if (p.P & b) return 1;
    if (p.N & b) return 2;
    if (p.B & b) return 3;
    if (p.R & b) return 4;
    if (p.Q & b) return 5;
    if (p.K & b) return 6;
    return 0;
}
static void remove_at(Pos& p, int sq)
{
    u64 m = ~BB(sq);
    p.P &= m; p.N &= m; p.B &= m; p.R &= m; p.Q &= m; p.K &= m; p.co[0] &= m; p.co[1] &= m;
}
static void set_at(Pos& p, int sq, int pt, int color)
{
    remove_at(p, sq);
    u64 b = BB(sq);
    switch (pt) {
        case 1: p.P |= b; break;
        case 2: p.N |= b; break;
        case 3: p.B |= b; break;
        case 4: p.R |= b; break;
        case 5: p.Q |= b; break;
        case 6: p.K |= b; break;
    }
    p.co[color] |= b;
}

void push(Pos& p, int code)
{
    int f = (code >> 6) & 63, t = code & 63, promo = code >> 12;
    // _to_chess960
    if (f == 4 && (p.K & BB(4))) {
        if (t == 6 && !(p.R & BB(6))) t = 7;
        else if (t == 2 && !(p.R & BB(2))) t = 0;
    } else if (f == 60 && (p.K & BB(60))) {
        if (t == 62 && !(p.R & BB(62))) t = 63;
        else if (t == 58 && !(p.R & BB(58))) t = 56;
    }
    p.castling = clean_castling(p);
    int ep_square = p.ep;
    p.ep = -1;
    p.hm += 1;
    if (p.turn == 0) p.fullmove += 1;
    u64 from_bb = BB(f), to_bb = BB(t);
    u64 touched = from_bb ^ to_bb;
    if ((touched & p.P) || (touched & p.co[!p.turn])) p.hm = 0;
    int pt = piece_at(p, f);
    remove_at(p, f);
    int captured = piece_at(p, t);
    p.castling &= ~to_bb & ~from_bb;
    if (pt == 6) p.castling &= p.turn ? ~RANK_1 : ~RANK_8;
    if (pt == 1) {
        int diff = t - f;
        if (diff == 16 && (f >> 3) == 1) p.ep = f + 8;
        else if (diff == -16 && (f >> 3) == 6) p.ep = f - 8;
        else if (t == ep_square && (diff == 7 || diff == 9 || diff == -7 || diff == -9) && !captured) {
            remove_at(p, ep_square + (p.turn ? -8 : 8));
        }
    }
    if (promo) pt = promo;
    bool castle = pt == 6 && (p.co[p.turn] & to_bb);
    if (castle) {
        bool a_side = (t & 7) < (f & 7);
        remove_at(p, f);
        remove_at(p, t);
        int back = p.turn ? 0 : 56;
        if (a_side) { set_at(p, back + 2, 6, p.turn); set_at(p, back + 3, 4, p.turn); }
        else { set_at(p, back + 6, 6, p.turn); set_at(p, back + 5, 4, p.turn); }
    } else {
        set_at(p, t, pt, p.turn);
    }
    p.turn = !p.turn;
}

bool parse_fen(const std::string& fen, Pos& out)
{
    std::istringstream in(fen);
    std::string board, turn, cast, ep;
    int hm = 0, fm = 1;
    if (!(in >> board >> turn)) return false;
    if (!(in >> cast)) cast = "-";
    if (!(in >> ep)) ep = "-";
    if (!(in >> hm)) hm = 0;
    if (!(in >> fm)) fm = 1;
    Pos p;
    int rank = 7, file = 0;
    for (char c : board) {
        if (c == '/') { rank--; file = 0; continue; }
        if (c >= '1' && c <= '8') { file += c - '0'; continue; }
        const char* pcs = "pnbrqk";
        const char* q = strchr(pcs, c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
        if (!q || rank < 0 || file > 7) return false;
        set_at(p, rank * 8 + file, (int)(q - pcs) + 1, (c >= 'A' && c <= 'Z') ? 1 : 0);
        file++;
    }
    if (turn != "w" && turn != "b") return false;
    p.turn = turn == "w" ? 1 : 0;
    p.castling = 0;
    if (cast != "-")
        for (char c : cast) {
            if (c == 'K') p.castling |= BB(7);
            else if (c == 'Q') p.castling |= BB(0);
            else if (c == 'k') p.castling |= BB(63);
            else if (c == 'q') p.castling |= BB(56);
        }
    p.ep = -1;
    if (ep != "-" && ep.size() == 2) p.ep = (ep[1] - '1') * 8 + (ep[0] - 'a');
    p.hm = hm;
    p.fullmove = fm;
    out = p;
    return true;
}

bool legal_moves(const Key& k, std::vector<int>& codes, std::vector<int64_t>& idx, int& flags)
{
    static thread_local S s;
    from_key(k, &s);
    int check;
    legal_codes(&s, &check);
    if (s.overflow) return false;
    flags = check | ((insufficient_color(&s, 1) && insufficient_color(&s, 0)) << 1);
    codes.assign(s.code, s.code + s.n);
    idx.resize(s.n);
    const int64_t* tab = IDXTAB[s.us ? 0 : 1];
    for (int i = 0; i < s.n; i++) idx[i] = tab[s.code[i]];
    return true;
}

bool forcing_mask(const Key& k, std::vector<uint8_t>& out)
{
    static thread_local S s;
    from_key(k, &s);
    int check;
    legal_codes(&s, &check);
    if (s.overflow) return false;
    out.resize(s.n);
    for (int i = 0; i < s.n; i++) {
        int f = (s.code[i] >> 6) & 63, t = s.code[i] & 63, d = t - f;
        int cap = ((BB(f) ^ BB(t)) & s.co[!s.us]) != 0
                  || (s.ep == t && (s.P & BB(f)) && (d == 7 || d == 9 || d == -7 || d == -9) && !(s.occ & BB(t)));
        out[i] = (uint8_t)(cap || gives_check(&s, s.code[i]));
    }
    return true;
}

void encode(const Key& k, int halfmove, int64_t* row, float* cast, int64_t* hm, int64_t* ep)
{
    S s;
    from_key(k, &s);
    int flip = !s.us, mirror = flip ? 56 : 0;
    for (int i = 0; i < 64; i++) row[i] = 0;
    u64* pc[6] = {&s.P, &s.N, &s.Bi, &s.R, &s.Q, &s.K};
    for (int t = 0; t < 6; t++) {
        u64 x = *pc[t];
        while (x) {
            int sq = msb64(x);
            row[sq ^ mirror] = (t + 1) + ((s.co[s.us] & BB(sq)) ? 0 : 6);
            x ^= BB(sq);
        }
    }
    for (int side = 0; side < 2; side++) {
        int color = side == 0 ? s.us : !s.us;
        u64 back = color ? RANK_1 : RANK_8;
        u64 km = s.K & s.co[color] & back, cr = s.castle & back;
        int ks = 0, qs = 0;
        while (km && cr) {
            u64 rook = cr & (~cr + 1);
            if (rook > km) ks = 1;
            if (rook < km) qs = 1;
            cr &= cr - 1;
        }
        cast[2 * side] = (float)ks; cast[2 * side + 1] = (float)qs;
    }
    *hm = halfmove < 100 ? halfmove : 100;
    *ep = s.ep < 0 ? 0 : ((s.ep ^ mirror) + 1);
}

std::string uci(int code)
{
    int f = (code >> 6) & 63, t = code & 63, promo = code >> 12;
    std::string s;
    s += (char)('a' + (f & 7)); s += (char)('1' + (f >> 3));
    s += (char)('a' + (t & 7)); s += (char)('1' + (t >> 3));
    if (promo) s += " pnbrqk"[promo];
    return s;
}

int parse_uci(const Pos& p, const std::string& u)
{
    std::vector<int> codes;
    std::vector<int64_t> idx;
    int flags;
    if (!legal_moves(make_key(p), codes, idx, flags)) return -1;
    for (int c : codes)
        if (uci(c) == u) return c;
    return -1;
}

bool is_checkmate(const Pos& p)
{
    std::vector<int> codes;
    std::vector<int64_t> idx;
    int flags;
    if (!legal_moves(make_key(p), codes, idx, flags)) return false;
    return codes.empty() && (flags & 1);
}

}  // namespace mayfly
