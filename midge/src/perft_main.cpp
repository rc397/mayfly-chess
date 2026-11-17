// perft_main.cpp: move generation check. "perft_main FEN DEPTH" prints the node count and the time;
// "perft_main -" reads lines "FEN;DEPTH;EXPECTED" from stdin and prints PASS or FAIL for each.
#include <chrono>
#include <iostream>
#include "chess.h"

// walk the tree and check the incremental key against a key rebuilt from the FEN at every node
static int hash_bad = 0;
static void hash_walk(Position &p, int depth) {
  static Position q;
  q.set_fen(p.fen());
  if (q.key != p.key) hash_bad++;
  if (!depth) return;
  Move list[256];
  int n = gen_moves(p, list, GEN_ALL);
  for (int i = 0; i < n; i++) { p.make(list[i]); hash_walk(p, depth - 1); p.unmake(); }
}

int main(int argc, char **argv) {
  tables::init();
  static Position p;
  if (argc >= 4 && std::string(argv[1]) == "-h") {
    p.set_fen(argv[2]);
    hash_walk(p, atoi(argv[3]));
    printf("hash mismatches %d\n", hash_bad);
    return hash_bad != 0;
  }
  if (argc >= 3) {
    p.set_fen(argv[1]);
    auto t0 = std::chrono::steady_clock::now();
    U64 n = perft(p, atoi(argv[2]));
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("%llu nodes %.3f s %.1f Mnps\n", (unsigned long long)n, s, n / s / 1e6);
    return 0;
  }
  std::string line;
  int bad = 0, all = 0;
  while (std::getline(std::cin, line)) {
    size_t a = line.find(';'), b = line.find(';', a + 1);
    if (a == std::string::npos || b == std::string::npos) continue;
    std::string fen = line.substr(0, a);
    int d = std::stoi(line.substr(a + 1, b - a - 1));
    unsigned long long want = std::stoull(line.substr(b + 1));
    if (!p.set_fen(fen)) { printf("BADFEN %s\n", fen.c_str()); bad++; continue; }
    U64 key0 = p.key;
    unsigned long long got = perft(p, d);
    all++;
    bool ok = got == want && p.key == key0 && p.sp == 0;
    if (!ok) bad++;
    printf("%s %s d%d got %llu want %llu\n", ok ? "PASS" : "FAIL", fen.c_str(), d, got, want);
  }
  printf("%d of %d passed\n", all - bad, all);
  return bad != 0;
}
