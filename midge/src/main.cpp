// main.cpp: Midge's UCI front end, plus perft, bench, eval and veto commands. Written from scratch, 17 Nov 2026.
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "chess.h"
#include "nnue.h"
#include "search.h"

// the version shown in "id name": the release build sets it, any other build says dev
#ifndef MIDGE_VERSION
#define MIDGE_VERSION "dev"
#endif
static Position pos;
static Search S;
static std::thread worker;
static int nthreads = 1;
static std::vector<Search *> helpers;   // threads 2 and up: same table, own board and histories
// The helper threads are started once, by the Threads option, and sleep between searches. Each builds its own
// Search, so the memory of many threads is set up side by side and not one after another.
static std::vector<std::thread> pool;
static std::mutex pool_m;
static std::condition_variable pool_wake, pool_idle;
static std::vector<char> pool_go;   // set for each helper that must start a search
static int pool_busy = 0;       // helpers still searching
static int pool_building = 0;   // helpers whose memory is still being set up
static int pool_active = 0;     // helpers that take part in the next search
static int pool_games = 0;      // goes up by one for every ucinewgame; a helper then empties its own tables
static bool pool_quit = false;
static Limits pool_limits;
static Position pool_pos;          // the root position of the search, copied before the main search moves on it
static void copy_settings(Search &d, const Search &s) {
  d.eval_mode = s.eval_mode; d.qchecks = s.qchecks; d.singular = s.singular; d.contw = s.contw; d.corrw = s.corrw;
  d.timescale = s.timescale; d.rfp_m = s.rfp_m; d.fut_b = s.fut_b; d.fut_m = s.fut_m; d.lmp_b = s.lmp_b;
  d.overhead = s.overhead; d.nmp = s.nmp; d.nmpv = s.nmpv;
  d.seeq = s.seeq; d.tmstab = s.tmstab; d.singx = s.singx; d.caph = s.caph; d.probcut = s.probcut;
  memcpy(d.lmr, s.lmr, sizeof d.lmr);
}
static const char *START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// waits until no search runs and every helper is built and asleep
static void join() {
  if (worker.joinable()) worker.join();
  std::unique_lock<std::mutex> lk(pool_m);
  pool_idle.wait(lk, [] { return pool_busy == 0 && pool_building == 0; });
}

// the life of helper number i
static void helper_loop(int i) {
  Search *h = new Search(S.tt);
  h->pos = new Position; h->silent = true; h->helper = true; h->lead = &S; h->start_depth = 1 + (i % 2);
  std::unique_lock<std::mutex> lk(pool_m);
  helpers[i] = h;
  if (--pool_building == 0) pool_idle.notify_all();
  int games = pool_games;
  for (;;) {
    pool_wake.wait(lk, [&] { return pool_quit || pool_go[i]; });
    if (pool_quit) return;
    pool_go[i] = 0;
    Limits L = pool_limits;
    bool fresh = games != pool_games;
    games = pool_games;
    lk.unlock();
    if (fresh) h->new_game();
    *h->pos = pool_pos;
    copy_settings(*h, S);
    h->think(L);
    lk.lock();
    if (--pool_busy == 0) pool_idle.notify_all();
  }
}

// the Threads option: more helper threads are started when it grows; spare ones stay asleep when it shrinks
static void set_threads(int n) {
  nthreads = n;
  std::lock_guard<std::mutex> lk(pool_m);
  pool_active = n - 1;
  if ((int)helpers.size() < n - 1) { helpers.resize(n - 1, nullptr); pool_go.resize(n - 1, 0); }
  while ((int)pool.size() < n - 1) {
    pool_building++;
    pool.emplace_back(helper_loop, (int)pool.size());
  }
}

// wakes the helpers for a search with the main search's limits already set
static void start_helpers(const Limits &L) {
  std::lock_guard<std::mutex> lk(pool_m);
  for (int i = 0; i < pool_active; i++) { helpers[i]->nodes = 0; helpers[i]->stop = false; pool_go[i] = 1; }
  pool_pos = pos;
  pool_limits = L;
  pool_limits.movetime = 0; pool_limits.wtime = pool_limits.btime = -1; pool_limits.nodes = 0; pool_limits.infinite = true;
  pool_busy += pool_active;
  pool_wake.notify_all();
}

static void stop_pool() {
  join();
  { std::lock_guard<std::mutex> lk(pool_m); pool_quit = true; }
  pool_wake.notify_all();
  for (auto &t : pool) t.join();
}

static void set_position(std::istringstream &is) {
  std::string tok, fen;
  is >> tok;
  if (tok == "startpos") { fen = START; is >> tok; }
  else if (tok == "fen") {
    while (is >> tok && tok != "moves") fen += tok + " ";
  } else return;
  if (!pos.set_fen(fen)) pos.set_fen(START);   // a bad FEN: the start position, not a broken board
  if (tok == "moves")
    while (is >> tok) {
      Move m = parse_move(pos, tok);
      if (!m) break;
      pos.make(m);
      // a game near the end of the position stack: start again from here (older repeats are forgotten)
      if (pos.sp > MAXSP - 400) { std::string f = pos.fen(); pos.set_fen(f); }
    }
}

static void go(std::istringstream &is) {
  Limits L;
  std::string tok;
  while (is >> tok) {
    if (tok == "depth") is >> L.depth;
    else if (tok == "nodes") is >> L.nodes;
    else if (tok == "movetime") is >> L.movetime;
    else if (tok == "wtime") is >> L.wtime;
    else if (tok == "btime") is >> L.btime;
    else if (tok == "winc") is >> L.winc;
    else if (tok == "binc") is >> L.binc;
    else if (tok == "movestogo") is >> L.movestogo;
    else if (tok == "infinite") L.infinite = true;
    else if (tok == "ponder") L.ponder = true;
    else if (tok == "searchmoves") {
      while (is >> tok) { Move m = parse_move(pos, tok); if (m) L.searchmoves.push_back(m); }
    }
  }
  L.depth = std::clamp(L.depth, 1, MAXPLY - 1);
  S.stop = false;            // cleared here, before the thread starts, so an early stop is not lost
  S.pondering = L.ponder;
  worker = std::thread([L] {
    S.team = &helpers; S.team_n = nthreads - 1;
    S.set_limits(L);
    start_helpers(L);
    Move m = S.iterate(L);
    // a ponder search that ran out of depth still waits for ponderhit or stop before it answers
    while (S.pondering && !S.stop) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    for (int i = 0; i < nthreads - 1; i++) helpers[i]->stop = true;
    // the answer goes out at once; the helpers are waited for by the next command, not by the clock
    printf("bestmove %s\n", move_str(m).c_str());
    fflush(stdout);
  });
}

static void bench(int depth) {
  const char *fens[] = {
      START,
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
      "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
      "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
      "2r3k1/pp3ppp/4p3/3p4/3P4/2P1PN2/P4PPP/2R3K1 w - - 0 20",
      "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
  };
  U64 total = 0;
  auto t0 = std::chrono::steady_clock::now();
  S.silent = true;
  for (const char *f : fens) {
    pos.set_fen(f);
    S.tt->clear();
    S.stop = false;
    Limits L; L.depth = depth;
    Move m = S.think(L);
    total += S.nodes;
    printf("%-72s depth %d best %s score %d nodes %llu\n", f, S.done_depth, move_str(m).c_str(), S.root_score, (unsigned long long)S.nodes);
  }
  S.silent = false;
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("bench: %llu nodes %.2f s %llu nps\n", (unsigned long long)total, s, (unsigned long long)(total / std::max(s, 1e-9)));
  fflush(stdout);
}

int main(int argc, char **argv) {
  tables::init();
  pos.set_fen(START);
  S.pos = &pos;
  // a net named midge_net.bin beside the exe (or in the working folder) is loaded at start
  {
    std::string self = argc > 0 ? argv[0] : "", err;
    size_t cut = self.find_last_of("/\\");
    std::string dir = cut == std::string::npos ? "" : self.substr(0, cut + 1);
    if (!nnue::load(dir + "midge_net.bin", err) && !dir.empty()) nnue::load("midge_net.bin", err);
  }
  std::string line;
  if (argc > 1 && std::string(argv[1]) == "bench") { bench(argc > 2 ? atoi(argv[2]) : 12); return 0; }
  while (std::getline(std::cin, line)) {
    std::istringstream is(line);
    std::string cmd;
    is >> cmd;
    if (cmd == "uci") {
      printf("id name Midge %s\nid author Ryan Chen\n", MIDGE_VERSION);
      printf("option name Threads type spin default 1 min 1 max 1024\n");
      printf("option name Hash type spin default 64 min 1 max 1048576\n");
      printf("option name Ponder type check default false\n");
      printf("option name MoveOverhead type spin default 50 min 0 max 5000\n");
      printf("option name EvalFile type string default <none>\n");
#ifdef MIDGE_DEV
      // test builds only: the switches the A/B matches flip
      printf("option name Eval type combo default nnue var nnue var hand var material\n");
      printf("option name QChecks type check default false\n");
      printf("option name Singular type check default true\n");
      printf("option name ContHist type check default true\n");
      printf("option name CorrHist type check default false\n");
      printf("option name TimeScale type spin default 160 min 20 max 400\n");
      printf("option name LmrDiv type spin default 230 min 100 max 500\n");
      printf("option name RfpMargin type spin default 80 min 20 max 300\n");
      printf("option name FutBase type spin default 100 min 0 max 400\n");
      printf("option name FutMul type spin default 90 min 20 max 300\n");
      printf("option name LmpBase type spin default 3 min 1 max 12\n");
      printf("option name NullMove type check default true\n");
      printf("option name NmpVerify type spin default 0 min 0 max 64\n");
      printf("option name SeeQuiet type spin default 50 min 0 max 300\n");
      printf("option name TmStab type check default false\n");
      printf("option name SingularX type check default false\n");
      printf("option name CapHist type check default false\n");
      printf("option name ProbCut type spin default 0 min 0 max 600\n");
#endif
      printf("info string net %s\n", nnue::loaded ? "loaded" : "NOT loaded (set EvalFile, or put midge_net.bin beside the exe)");
      printf("uciok\n");
    } else if (cmd == "isready") {
      // ready means the helper threads of a new Threads value are built; a running search is not waited for
      { std::unique_lock<std::mutex> lk(pool_m); pool_idle.wait(lk, [] { return pool_building == 0; }); }
      printf("readyok\n");
    }
    else if (cmd == "ucinewgame") {
      join(); S.tt->clear(); S.new_game();
      std::lock_guard<std::mutex> lk(pool_m);
      pool_games++;
    }
    else if (cmd == "setoption") {
      std::string tok, name, value;
      is >> tok;
      while (is >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
      std::getline(is, value);
      if (!value.empty() && value[0] == ' ') value.erase(0, 1);
      join();
      if (name == "Threads") {
        set_threads(std::clamp(atoi(value.c_str()), 1, 1024));
        S.tt->clear_threads = nthreads;
      } else if (name == "Hash") {
        S.tt->resize(std::clamp(atoi(value.c_str()), 1, 1048576));
        printf("info string hash table %llu MB\n", (unsigned long long)((S.tt->mask + 1) * sizeof(TTEntry) >> 20));
      }
      else if (name == "EvalFile") {
        std::string err;
        if (nnue::load(value, err)) printf("info string loaded %s\n", value.c_str());
        else printf("info string EvalFile failed: %s\n", err.c_str());
      } else if (name == "Eval") S.eval_mode = value == "material" ? EV_MATERIAL : value == "hand" ? EV_HAND : EV_NNUE;
      else if (name == "QChecks") S.qchecks = value == "true";
      else if (name == "Singular") S.singular = value == "true";
      else if (name == "ContHist") S.contw = value == "true";
      else if (name == "CorrHist") S.corrw = value == "true";
      else if (name == "LmrDiv") S.init_lmr(0.8, std::clamp(atoi(value.c_str()), 100, 500) / 100.0);
      else if (name == "RfpMargin") S.rfp_m = atoi(value.c_str());
      else if (name == "FutBase") S.fut_b = atoi(value.c_str());
      else if (name == "FutMul") S.fut_m = atoi(value.c_str());
      else if (name == "LmpBase") S.lmp_b = atoi(value.c_str());
      else if (name == "SeeQuiet") S.seeq = atoi(value.c_str());
      else if (name == "NullMove") S.nmp = value == "true";
      else if (name == "NmpVerify") S.nmpv = atoi(value.c_str());
      else if (name == "TmStab") S.tmstab = value == "true";
      else if (name == "SingularX") S.singx = value == "true";
      else if (name == "CapHist") S.caph = value == "true";
      else if (name == "ProbCut") S.probcut = atoi(value.c_str());
      else if (name == "MoveOverhead") S.overhead = std::clamp(atoi(value.c_str()), 0, 5000);
      else if (name == "TimeScale") S.timescale = std::clamp(atoi(value.c_str()), 20, 400);
    } else if (cmd == "position") { join(); set_position(is); }
    else if (cmd == "go") { join(); S.pos = &pos; go(is); }
    else if (cmd == "stop") { S.stop = true; join(); }
    else if (cmd == "ponderhit") S.ponderhit();
    else if (cmd == "quit") { S.stop = true; break; }
    else if (cmd == "d") printf("%s\n", pos.fen().c_str());
    else if (cmd == "evaltest" && nnue::loaded) {
      // random games: the incremental eval after every move (and back down after unmakes) against a fresh one
      int games = 200, bad = 0, checks = 0; is >> games;
      static nnue::AccStack inc, fresh;
      U64 r = 0x9E3779B97F4A7C15ULL;
      for (int g = 0; g < games; g++) {
        pos.set_fen(START); inc.reset();
        nnue::evaluate(pos, inc);
        for (int ply = 0; ply < 160; ply++) {
          Move ml[256]; int n = gen_moves(pos, ml, GEN_ALL);
          if (!n) break;
          r ^= r << 13; r ^= r >> 7; r ^= r << 17;
          pos.make(ml[r % n]); inc.computed[pos.sp] = false;
          if (r % 3) {   // sometimes skip an eval, so the lazy path covers several moves at once
            fresh.reset(); checks++;
            if (nnue::evaluate(pos, inc) != nnue::evaluate(pos, fresh)) bad++;
          }
          if (r % 7 == 0 && pos.sp > 1) {   // step back one move and check again
            pos.unmake(); fresh.reset(); checks++;
            if (nnue::evaluate(pos, inc) != nnue::evaluate(pos, fresh)) bad++;
          }
        }
      }
      printf("evaltest: %d games, %d checks, %d mismatches\n", games, checks, bad);
      pos.set_fen(START);
    }
    else if (cmd == "acctest" && nnue::loaded) {
      // every position of a full move tree: the lazily updated accumulators against a fresh build
      join();
      int depth = 3; is >> depth;
      static nnue::AccStack inc, fresh;
      U64 nodes = 0, checks = 0, bad = 0, castles = 0, eps = 0, promos = 0, kings = 0, r = 88172645463325252ULL;
      int maxabs = 0;
      auto walk = [&](auto &&self, int d) -> void {
        Move ml[256]; int n = gen_moves(pos, ml, GEN_ALL);
        for (int i = 0; i < n; i++) {
          Move m = ml[i]; int fl = mflag(m);
          kings += pc_type(pos.board[mfrom(m)]) == KING;
          pos.make(m); inc.computed[pos.sp] = false;
          nodes++; castles += fl == F_KCASTLE || fl == F_QCASTLE; eps += fl == F_EP; promos += (fl & 8) != 0;
          r ^= r << 13; r ^= r >> 7; r ^= r << 17;
          if (d <= 1 || r % 3) {   // every leaf, and two inner nodes in three, so some updates span several moves
            fresh.reset(); checks++;
            int a = nnue::evaluate(pos, inc), b = nnue::evaluate(pos, fresh);
            if (a != b) bad++;
            maxabs = std::max(maxabs, std::abs(b));
          }
          if (d > 1) self(self, d - 1);
          pos.unmake();
        }
      };
      const char *fens[] = {START,
          "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
          "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
          "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
          "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
          "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
          "r3k2r/8/8/3pP3/3Pp3/8/8/R3K2R w KQkq d6 0 1",
          "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"};
      for (const char *f : fens) { pos.set_fen(f); inc.reset(); walk(walk, depth); }
      printf("acctest: depth %d, %llu positions, %llu checks, %llu mismatches; castlings %llu, en passant %llu, promotions %llu, king moves %llu; largest eval %d\n",
             depth, (unsigned long long)nodes, (unsigned long long)checks, (unsigned long long)bad, (unsigned long long)castles,
             (unsigned long long)eps, (unsigned long long)promos, (unsigned long long)kings, maxabs);
      pos.set_fen(START);
    }
    else if (cmd == "perft") {
      int d = 1; is >> d;
      Move ml[256]; int n = gen_moves(pos, ml, GEN_ALL); U64 t = 0;
      for (int i = 0; i < n; i++) { pos.make(ml[i]); U64 c = d > 1 ? perft(pos, d - 1) : 1; pos.unmake(); t += c; printf("%s: %llu\n", move_str(ml[i]).c_str(), (unsigned long long)c); }
      printf("total %llu\n", (unsigned long long)t);
    } else if (cmd == "bench") { join(); int d = 12; is >> d; S.pos = &pos; bench(d); pos.set_fen(START); }
    else if (cmd == "eval") {
      join(); S.pos = &pos; S.acc->reset();
      printf("material %d hand %d", handeval::material(pos), handeval::evaluate(pos));
      if (nnue::loaded) printf(" nnue %d", nnue::evaluate(pos, *S.acc));
      printf("\n");
    } else if (cmd == "veto") {
      // veto move M [depth D] [mode forcing|full] [eval material|hand|nnue] [margin T]
      join(); S.pos = &pos;
      std::string tok, ms; int d = 6, margin = 200; bool full = false; int ev = EV_MATERIAL;
      while (is >> tok) {
        if (tok == "move") is >> ms;
        else if (tok == "depth") is >> d;
        else if (tok == "margin") is >> margin;
        else if (tok == "mode") { is >> tok; full = tok == "full"; }
        else if (tok == "eval") { is >> tok; ev = tok == "nnue" ? EV_NNUE : tok == "hand" ? EV_HAND : EV_MATERIAL; }
      }
      Move mv = parse_move(pos, ms);
      if (!mv) { printf("info string veto: %s is not legal here\n", ms.c_str()); fflush(stdout); continue; }
      S.veto_eval = ev;
      auto r = S.veto(mv, std::max(d, 1), full, margin);
      printf("info veto move %s value %d best %s bestvalue %d drop %d flag %d complete %d nodes %llu\n", ms.c_str(), r.vmove,
             move_str(r.best).c_str(), r.vbest, r.vbest - r.vmove, int(r.flag), int(r.complete), (unsigned long long)r.nodes);
    }
    fflush(stdout);
  }
  S.stop = true;   // end of input or quit: do not keep searching
  stop_pool();
  return 0;
}
