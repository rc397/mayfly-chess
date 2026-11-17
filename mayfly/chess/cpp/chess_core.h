// Positions, python-chess's legal move order, the transposition key and the network encoding.
//
// The move generator is native/movegen.c (python-chess 1.11's generate_legal_moves ported line by line,
// checked there on 725,055 positions) without its Python glue. Like movegen.c it is derived from python-chess,
// so this file and chess_core.cpp are GPL-3.0-or-later. push() follows python-chess's Board.push for standard
// chess (castling rights, en passant square, halfmove clock).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace mayfly {

typedef uint64_t u64;

// python-chess's Board._transposition_key(), the tuple
// (pawns, knights, bishops, rooks, queens, kings, white, black, turn, clean_castling_rights, legal ep or None)
struct Key {
    u64 P, N, B, R, Q, K, W, Bk, castle;
    int turn;   // 1 white, 0 black
    int ep;     // -1 for None
    bool operator==(const Key& o) const {
        return P == o.P && N == o.N && B == o.B && R == o.R && Q == o.Q && K == o.K && W == o.W && Bk == o.Bk
               && castle == o.castle && turn == o.turn && ep == o.ep;
    }
};
struct KeyHash {
    size_t operator()(const Key& k) const {
        u64 h = 1469598103934665603ULL;
        const u64 v[11] = {k.P, k.N, k.B, k.R, k.Q, k.K, k.W, k.Bk, k.castle, (u64)k.turn, (u64)(int64_t)k.ep};
        for (u64 x : v) { h ^= x + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2); h *= 1099511628211ULL; }
        return (size_t)h;
    }
};

// A python-chess Board without its move stack.
struct Pos {
    u64 P = 0, N = 0, B = 0, R = 0, Q = 0, K = 0;
    u64 co[2] = {0, 0};   // co[1] white, co[0] black
    int turn = 1;         // 1 white, 0 black
    u64 castling = 0;     // raw castling_rights (rook squares)
    int ep = -1;          // raw ep_square, -1 for None
    int hm = 0;           // halfmove clock
    int fullmove = 1;
};

// A move is coded as in movegen, promotion * 4096 + from * 64 + to (promotion 0, or 2..5 = N B R Q).
// Castling is the king's two-square move, as in python-chess's standard UCI.
void init_tables();                                   // once, before anything else
bool parse_fen(const std::string& fen, Pos& out);     // false on a malformed FEN
u64 clean_castling(const Pos& p);
Key make_key(const Pos& p);                           // the transposition key (legal ep only)
void push(Pos& p, int code);                          // python-chess Board.push
// Legal moves in python-chess's order, read off the key. Flags bit 0 is in check, bit 1 insufficient material.
// Returns false on an overflow, where Python falls back to python-chess (never seen in practice).
bool legal_moves(const Key& k, std::vector<int>& codes, std::vector<int64_t>& idx, int& flags);
// One 0/1 per legal move, set when board.is_capture(m) or board.gives_check(m)
bool forcing_mask(const Key& k, std::vector<uint8_t>& out);
// mayfly_movegen.encode for one position, filling ids[64], cast[4], hm (capped at 100) and ep
void encode(const Key& k, int halfmove, int64_t* ids, float* cast, int64_t* hm, int64_t* ep);
int64_t move_to_idx(int code, bool flip);
std::string uci(int code);
int parse_uci(const Pos& p, const std::string& s);    // the legal move with that UCI, or -1
bool is_checkmate(const Pos& p);

}  // namespace mayfly
