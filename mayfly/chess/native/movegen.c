/* MayFly's node creation in C (timing_fixes, v5). python-chess's legal move generation, ported line by
   line so the same moves come back in the same order, read straight off the transposition key every Node
   already holds. Checked against python-chess on 725,055 positions with 0 mismatches in moves, order,
   index rows, flags or forcing mask.

   Derived from python-chess 1.11 (Niklas Fiekas and contributors, GPL-3.0-or-later), namely
   generate_legal_moves, _slider_blockers, _is_safe, _ep_skewered, generate_castling_moves,
   is_insufficient_material and gives_check. This file is therefore GPL-3.0-or-later.

       key = (pawns, knights, bishops, rooks, queens, kings, white, black, turn, clean_castling, legal_ep)
   legal(key)   -> (moves, idx, flags): a list of INTERNED chess.Move objects in python-chess's order, an
                   int64 ndarray of moves.move_to_idx(m, turn == BLACK), and flags (bit 0 in check, bit 1
                   is_insufficient_material)
   forcing(key) -> bytes, one 0/1 per legal move: board.is_capture(m) or board.gives_check(m)
   encode(keys, hms) -> (ids, cast, hm, ep), what encode.encode_keys returns
   count(key)   -> the number of legal moves (benchmarks)
   A position whose pseudo-legal moves would overflow the buffer raises OverflowError, and mcts.py falls
   back to python-chess for that node.
   Using the key's legal en passant square in place of the board's raw ep_square gives the same legal
   moves, because a raw square with no legal capture only ever yields pseudo-legal captures that _is_safe
   drops. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdint.h>
#ifdef _MSC_VER
#include <intrin.h>
static __inline int msb64(uint64_t b) { unsigned long i; _BitScanReverse64(&i, b); return (int)i; }
static __inline int popc(uint64_t b) { return (int)__popcnt64(b); }
#else
static inline int msb64(uint64_t b) { return 63 - __builtin_clzll(b); }
static inline int popc(uint64_t b) { return __builtin_popcountll(b); }
#endif
typedef uint64_t u64;
#define CAP 1024                           /* far above any position's pseudo-legal moves */
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
static u64 RANKE[64], FILEE[64], DIAGE[64];        /* empty-board attack sets, = BB_*_ATTACKS[sq][0] */
static PyObject *MOVETAB = NULL;                   /* list of 6*4096 interned Moves: promo*4096 + f*64 + t */
static int64_t IDXTAB[2][6 * 4096];                /* [flip][promo*4096 + f*64 + t] -> move_to_idx */
static int READY = 0;

static u64 slide(int sq, u64 occ, const int *d, int nd)
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

typedef struct {
    u64 P, N, Bi, R, Q, K, co[2], occ, castle;
    int us, ep;                        /* us: 1 white, 0 black; ep: -1 none */
    int n;
    int overflow;
    int code[CAP];
} S;

static __inline void emit(S *s, int f, int t, int promo)
{
    if (s->n < CAP) s->code[s->n++] = promo * 4096 + f * 64 + t;
    else s->overflow = 1;
}

static __inline u64 attackers(const S *s, int color, int sq, u64 occ)
{
    u64 qr = s->Q | s->R, qb = s->Q | s->Bi;
    u64 a = (KING[sq] & s->K) | (KNIGHT[sq] & s->N) | (rook_att(sq, occ) & qr) | (diag_att(sq, occ) & qb)
            | (PAWN[!color][sq] & s->P);
    return a & s->co[color];
}

static __inline u64 attacks_mask(const S *s, int sq)
{
    u64 b = BB(sq);
    if (b & s->N) return KNIGHT[sq];
    if (b & s->K) return KING[sq];
    u64 a = 0;
    if ((b & s->Bi) || (b & s->Q)) a = diag_att(sq, s->occ);
    if ((b & s->R) || (b & s->Q)) a |= rook_att(sq, s->occ);
    return a;
}

static void pseudo_ep(S *s, u64 from_mask, u64 to_mask)
{
    if (s->ep <= 0 || !(BB(s->ep) & to_mask)) return;     /* python: `if not self.ep_square` (0 is falsy) */
    if (BB(s->ep) & s->occ) return;
    u64 cap = s->P & s->co[s->us] & from_mask & PAWN[!s->us][s->ep] & RANK(s->us ? 4 : 3);
    while (cap) { int c = msb64(cap); emit(s, c, s->ep, 0); cap ^= BB(c); }
}

static int attacked_for_king(const S *s, u64 path, u64 occ)
{
    while (path) { int q = msb64(path); if (attackers(s, !s->us, q, occ)) return 1; path ^= BB(q); }
    return 0;
}

static void castling(S *s, u64 from_mask, u64 to_mask)
{
    u64 back = s->us ? RANK_1 : RANK_8;
    u64 king = s->co[s->us] & s->K & back & from_mask;   /* ~promoted: a king is never promoted */
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
            int to = c;                                    /* _from_chess960(False, king, candidate) */
            if (ks == 4 && (s->K & BB(4))) { if (c == 7) to = 6; else if (c == 0) to = 2; }
            else if (ks == 60 && (s->K & BB(60))) { if (c == 63) to = 62; else if (c == 56) to = 58; }
            emit(s, ks, to, 0);
        }
        cand ^= BB(c);
    }
}

static void pseudo(S *s, u64 from_mask, u64 to_mask)
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

static void evasions(S *s, int king, u64 checkers)
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

static u64 pin_mask(const S *s, int sq)
{
    u64 km = s->co[s->us] & s->K;
    if (!km) return ALL;
    int king = msb64(km);
    u64 sm = BB(sq);
    const u64 *tabs[3] = {FILEE, RANKE, DIAGE};
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

static int ep_skewered(const S *s, int king, int capturer)
{
    int last_double = s->ep + (s->us ? -8 : 8);
    u64 occ = (s->occ & ~BB(last_double) & ~BB(capturer)) | BB(s->ep);
    u64 h = s->co[!s->us] & (s->R | s->Q);
    if (rank_att(king, occ) & h) return 1;
    u64 d = s->co[!s->us] & (s->Bi | s->Q);
    if (diag_att(king, occ) & d) return 1;
    return 0;
}

static int is_safe(const S *s, int king, u64 blockers, int code)
{
    int f = (code >> 6) & 63, t = code & 63;
    if (f == king) {
        /* is_castling: a king move two files wide or onto our own rook */
        int diff = (f & 7) - (t & 7);
        if (diff > 1 || diff < -1 || (s->R & s->co[s->us] & BB(t))) return 1;
        return attackers(s, !s->us, t, s->occ) == 0;
    }
    int d = t - f; if (d < 0) d = -d;
    if (s->ep == t && (s->P & BB(f)) && (d == 7 || d == 9) && !(s->occ & BB(t)))
        return (pin_mask(s, f) & BB(t)) && !ep_skewered(s, king, f);
    return !(blockers & BB(f)) || (RAYS[f][t] & BB(king));
}

static void legal_codes(S *s, int *check)
{
    u64 km = s->K & s->co[s->us];
    *check = 0;
    if (!km) { pseudo(s, ALL, ALL); return; }
    int king = msb64(km);
    /* _slider_blockers */
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
    S tmp; tmp = *s; tmp.n = 0; tmp.overflow = 0;   /* pseudo-legal moves into tmp, filtered into s */
    if (checkers) { *check = 1; evasions(&tmp, king, checkers); }
    else pseudo(&tmp, ALL, ALL);
    s->n = 0;
    s->overflow = tmp.overflow;
    for (int i = 0; i < tmp.n; i++)
        if (is_safe(s, king, blockers, tmp.code[i])) s->code[s->n++] = tmp.code[i];
}

static int insufficient_color(const S *s, int c)
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

static int read_key(PyObject *key, S *s)
{
    if (!PyTuple_Check(key) || PyTuple_GET_SIZE(key) != 11) { PyErr_SetString(PyExc_TypeError, "key"); return -1; }
    u64 v[10];
    for (int i = 0; i < 10; i++) {
        v[i] = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(key, i));
        if (v[i] == (u64)-1 && PyErr_Occurred()) return -1;
    }
    s->P = v[0]; s->N = v[1]; s->Bi = v[2]; s->R = v[3]; s->Q = v[4]; s->K = v[5];
    s->co[1] = v[6]; s->co[0] = v[7]; s->occ = v[6] | v[7];
    s->us = v[8] ? 1 : 0; s->castle = v[9];
    PyObject *e = PyTuple_GET_ITEM(key, 10);
    s->ep = (e == Py_None) ? -1 : (int)PyLong_AsLong(e);
    if (s->ep == -1 && PyErr_Occurred()) return -1;
    s->n = 0;
    s->overflow = 0;
    return 0;
}

static PyObject *legal(PyObject *self, PyObject *key)
{
    if (!READY) { PyErr_SetString(PyExc_RuntimeError, "init first"); return NULL; }
    S s;
    if (read_key(key, &s) != 0) return NULL;
    int check;
    legal_codes(&s, &check);
    if (s.overflow) { PyErr_SetString(PyExc_OverflowError, "too many pseudo-legal moves"); return NULL; }
    int flags = check | ((insufficient_color(&s, 1) && insufficient_color(&s, 0)) << 1);
    PyObject *lst = PyList_New(s.n);
    if (!lst) return NULL;
    npy_intp dim = s.n;
    PyObject *arr = PyArray_SimpleNew(1, &dim, NPY_INT64);
    if (!arr) { Py_DECREF(lst); return NULL; }
    int64_t *ix = (int64_t *)PyArray_DATA((PyArrayObject *)arr);
    const int64_t *tab = IDXTAB[s.us ? 0 : 1];
    for (int i = 0; i < s.n; i++) {
        PyObject *m = PyList_GET_ITEM(MOVETAB, s.code[i]);
        Py_INCREF(m);
        PyList_SET_ITEM(lst, i, m);
        ix[i] = tab[s.code[i]];
    }
    return Py_BuildValue("(NNi)", lst, arr, flags);
}

/* forcing(key) -> bytes, one 0/1 per legal move in legal()'s order, set for board.is_capture(m) or
   board.gives_check(m), which is what _pick and _pick_fast build node.FM from. gives_check is push + is_check, so the move is made on the
   bitboards (castling's rook, en passant's pawn, the promotion) and the other king's attackers are read. */
static int gives_check(const S *s, int code)
{
    int f = (code >> 6) & 63, t = code & 63, promo = code >> 12, us = s->us;
    u64 P = s->P, N = s->N, Bi = s->Bi, R = s->R, Q = s->Q, K = s->K, co[2] = {s->co[0], s->co[1]};
    u64 fb = BB(f), tb = BB(t);
    int castle = (K & fb) && (((f & 7) - (t & 7) > 1) || ((t & 7) - (f & 7) > 1) || (R & co[us] & tb));
    int ep = (s->ep == t) && (P & fb) && (t - f == 7 || t - f == 9 || f - t == 7 || f - t == 9) && !(s->occ & tb);
    u64 *pcs[6] = {&P, &N, &Bi, &R, &Q, &K};
    int moved = -1;
    for (int i = 0; i < 6; i++) if (*pcs[i] & fb) { moved = i; *pcs[i] &= ~fb; }
    co[us] &= ~fb;
    if (castle) {
        int back = us ? 0 : 56, aside = (t & 7) < (f & 7);
        /* python-chess moves king e1->g1 (or onto the rook square in 960 notation); either way the king lands
           on g/c and the rook on f/d of the back rank */
        int rook_from = aside ? back + 0 : back + 7;
        if (R & co[us] & tb) rook_from = t;
        R &= ~BB(rook_from); co[us] &= ~BB(rook_from);
        int kto = back + (aside ? 2 : 6), rto = back + (aside ? 3 : 5);
        K |= BB(kto); R |= BB(rto); co[us] |= BB(kto) | BB(rto);
    } else {
        for (int i = 0; i < 6; i++) *pcs[i] &= ~tb;     /* the captured piece, if any */
        co[!us] &= ~tb;
        if (ep) { int cap = t + (us ? -8 : 8); P &= ~BB(cap); co[!us] &= ~BB(cap); }
        int put = promo ? promo - 1 : moved;            /* piece types 2..5 -> index 1..4 */
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

static PyObject *forcing(PyObject *self, PyObject *key)
{
    S s;
    if (read_key(key, &s) != 0) return NULL;
    int check;
    legal_codes(&s, &check);
    if (s.overflow) { PyErr_SetString(PyExc_OverflowError, "too many pseudo-legal moves"); return NULL; }
    char buf[CAP];
    for (int i = 0; i < s.n; i++) {
        int f = (s.code[i] >> 6) & 63, t = s.code[i] & 63, d = t - f;
        int cap = ((BB(f) ^ BB(t)) & s.co[!s.us]) != 0
                  || (s.ep == t && (s.P & BB(f)) && (d == 7 || d == 9 || d == -7 || d == -9) && !(s.occ & BB(t)));
        buf[i] = (char)(cap || gives_check(&s, s.code[i]));
    }
    return PyBytes_FromStringAndSize(buf, s.n);
}

/* encode(keys, hms) -> (ids (B,64) int64, cast (B,4) float32, hm (B,) int64, ep (B,) int64), what
   encode.encode_keys returns (board_to_ids + clock_and_ep of every board), written straight into the arrays */
static PyObject *encode(PyObject *self, PyObject *args)
{
    PyObject *keys, *hms;
    if (!PyArg_ParseTuple(args, "OO", &keys, &hms)) return NULL;
    if (!PyList_Check(keys) || !PyList_Check(hms) || PyList_GET_SIZE(keys) != PyList_GET_SIZE(hms)) {
        PyErr_SetString(PyExc_TypeError, "encode(list of keys, list of clocks)"); return NULL;
    }
    npy_intp B = PyList_GET_SIZE(keys), d2[2] = {B, 64}, d4[2] = {B, 4}, d1[1] = {B};
    PyObject *ids = PyArray_ZEROS(2, d2, NPY_INT64, 0), *cast = PyArray_SimpleNew(2, d4, NPY_FLOAT32);
    PyObject *hm = PyArray_SimpleNew(1, d1, NPY_INT64), *ep = PyArray_SimpleNew(1, d1, NPY_INT64);
    if (!ids || !cast || !hm || !ep) goto fail;
    int64_t *Iv = (int64_t *)PyArray_DATA((PyArrayObject *)ids), *Hv = (int64_t *)PyArray_DATA((PyArrayObject *)hm), *Ev = (int64_t *)PyArray_DATA((PyArrayObject *)ep);
    float *Cv = (float *)PyArray_DATA((PyArrayObject *)cast);
    for (npy_intp b = 0; b < B; b++) {
        S s;
        if (read_key(PyList_GET_ITEM(keys, b), &s) != 0) goto fail;
        int flip = !s.us, mirror = flip ? 56 : 0;
        u64 *pc[6] = {&s.P, &s.N, &s.Bi, &s.R, &s.Q, &s.K};
        int64_t *row = Iv + 64 * b;
        for (int t = 0; t < 6; t++) {
            u64 x = *pc[t];
            while (x) {
                int sq = msb64(x);
                row[sq ^ mirror] = (t + 1) + ((s.co[s.us] & BB(sq)) ? 0 : 6);
                x ^= BB(sq);
            }
        }
        for (int side = 0; side < 2; side++) {          /* own, then theirs */
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
            Cv[4 * b + 2 * side] = (float)ks; Cv[4 * b + 2 * side + 1] = (float)qs;
        }
        long long h = PyLong_AsLongLong(PyList_GET_ITEM(hms, b));
        if (h == -1 && PyErr_Occurred()) goto fail;
        Hv[b] = h < 100 ? h : 100;
        Ev[b] = s.ep < 0 ? 0 : ((s.ep ^ mirror) + 1);
    }
    return Py_BuildValue("(NNNN)", ids, cast, hm, ep);
fail:
    Py_XDECREF(ids); Py_XDECREF(cast); Py_XDECREF(hm); Py_XDECREF(ep);
    return NULL;
}

/* codes only, for benchmarking the generator without the Python objects */
static PyObject *count(PyObject *self, PyObject *key)
{
    S s;
    if (read_key(key, &s) != 0) return NULL;
    int check;
    legal_codes(&s, &check);
    if (s.overflow) { PyErr_SetString(PyExc_OverflowError, "too many pseudo-legal moves"); return NULL; }
    return PyLong_FromLong(s.n);
}

static int fill_u64(PyObject *seq, u64 *dst, Py_ssize_t n)
{
    if (!PyList_Check(seq) || PyList_GET_SIZE(seq) != n) { PyErr_SetString(PyExc_TypeError, "table size"); return -1; }
    for (Py_ssize_t i = 0; i < n; i++) {
        dst[i] = PyLong_AsUnsignedLongLong(PyList_GET_ITEM(seq, i));
        if (dst[i] == (u64)-1 && PyErr_Occurred()) return -1;
    }
    return 0;
}

/* init(movetab, idx_white, idx_black, knight, king, pawn_black, pawn_white, rays_flat) */
static PyObject *init(PyObject *self, PyObject *args)
{
    PyObject *mt, *iw, *ib, *kn, *kg, *p0, *p1, *ry;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &mt, &iw, &ib, &kn, &kg, &p0, &p1, &ry)) return NULL;
    if (!PyList_Check(mt) || PyList_GET_SIZE(mt) != 6 * 4096) { PyErr_SetString(PyExc_TypeError, "movetab"); return NULL; }
    for (int i = 0; i < 6 * 4096; i++) {
        IDXTAB[0][i] = PyLong_AsLongLong(PyList_GET_ITEM(iw, i));
        IDXTAB[1][i] = PyLong_AsLongLong(PyList_GET_ITEM(ib, i));
    }
    if (PyErr_Occurred()) return NULL;
    if (fill_u64(kn, KNIGHT, 64) || fill_u64(kg, KING, 64) || fill_u64(p0, PAWN[0], 64) || fill_u64(p1, PAWN[1], 64)
        || fill_u64(ry, &RAYS[0][0], 4096)) return NULL;
    for (int sq = 0; sq < 64; sq++) { RANKE[sq] = rank_att(sq, 0); FILEE[sq] = file_att(sq, 0); DIAGE[sq] = diag_att(sq, 0); }
    Py_XDECREF(MOVETAB); Py_INCREF(mt); MOVETAB = mt;
    READY = 1;
    Py_RETURN_NONE;
}

static PyMethodDef M[] = {
    {"init", init, METH_VARARGS, NULL},
    {"legal", legal, METH_O, NULL},
    {"count", count, METH_O, NULL},
    {"forcing", forcing, METH_O, NULL},
    {"encode", encode, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};
static struct PyModuleDef MOD = {PyModuleDef_HEAD_INIT, "mayfly_movegen", NULL, -1, M};
PyMODINIT_FUNC PyInit_mayfly_movegen(void) { import_array(); return PyModule_Create(&MOD); }
