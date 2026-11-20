# Midge

Midge is a UCI chess engine: an alpha-beta search with a small NNUE-style network, written from scratch in C++.
It is the small, fast sibling of [MayFly](../mayfly), which is a transformer network
with a tree search on a GPU. Midge runs on the CPU only.

Author: Ryan Chen. Licence: GPL-3.0-or-later (see [LICENSE.md](LICENSE.md)).

This is version 0.2, the first published version. What is tested and what is not is listed at the end.

## What is in this folder

| file | what it is |
|---|---|
| `src/`, `build.sh` | the full source and the build script |
| `nets/NETS.md` | what every network was trained on |
| `CHANGES.md`, `LICENSE.md` | the history and the licence |

The executables and the network file are not in the repository. They are attached to the Midge 0.2 release on
the [releases page](https://github.com/rc397/mayfly-chess/releases):

| file | what it is |
|---|---|
| `midge-0.2-x86-64-v3.exe` | Windows, for CPUs with AVX2 and BMI2 (most CPUs since 2015) |
| `midge-0.2-x86-64.exe` | Windows, for any 64-bit CPU (slower network code) |
| `midge-0.2-x86-64-v3` | Linux, AVX2 and BMI2, one static file |
| `midge_net.bin` | the network; the engine loads it by itself when it sits beside the executable |

## Using it

Point any UCI host (a GUI such as Cute Chess, Arena or Banksia, or a match runner such as cutechess-cli) at the
executable. Keep `midge_net.bin` in the same folder. After `uci` the engine prints `info string net loaded`;
if it prints `net NOT loaded`, the file is missing and the engine falls back to a weak hand-made evaluation.
That is how the 0.2 executables behave. The source here has changed since: a build made with `NET=` carries
the network inside the executable, and a build that has no usable network prints an error and stops instead
of playing weakly (see CHANGES.md).

UCI options:

| option | default | meaning |
|---|---|---|
| `Threads` | 1 | search threads, 1 to 1024 (they share one table) |
| `Hash` | 64 | table size in MB, 1 to 1048576; the engine takes less if the machine cannot give it |
| `Ponder` | false | think on the opponent's time (`go ponder`, `ponderhit`) |
| `MoveOverhead` | 50 | milliseconds kept back on every move for the way to the host; raise it on a slow link |
| `EvalFile` | (none) | another network file; it must match the build (512 wide, no king buckets) |

Not supported yet: tablebases, MultiPV, Chess960.

## Strength so far

All matches: one thread each, start positions from a set of 200 openings with every opening played from both
sides, no adjudication, no tablebases, played on one laptop (Intel i7-12700H) with several games at a time. The
interval is the 95% interval over opening pairs. These are a few matches on one machine; they are not
rating-list numbers.

| against | time | games | wins-draws-losses | score | Elo difference (95%) |
|---|---|---|---|---|---|
| Critter 1.6a | 180 s + 3 s | 100 | 75-21-4 | 85.5% | +308 (+243 to +397) |
| Stockfish 19 limited to UCI_Elo 3190 | 10 s + 0.1 s | 100 | 26-52-22 | 52.0% | +14 (-39 to +68) |
| Stockfish 19 at full strength, 25,000 nodes a move | Midge on 10 s + 0.1 s | 100 | 1-40-59 | 21.0% | -230 (-282 to -186) |
| MayFly v6.1 on an RTX 5070 Ti | 60 s + 1 s, Midge on 4 threads | 24 | 4-14-6 | 45.8% | -29 (-126 to +64) |
| MayFly v6.1 on an RTX 5070 Ti | 180 s + 2 s, Midge on 4 threads | 24 | 3-16-5 | 45.8% | -29 (-115 to +54) |

The Critter match in full: Critter 1.6a 64-bit with its own book off, tablebases off and its session file off;
Hash 64 MB for both; 8 games at a time; 79 games ended in checkmate, 11 by threefold repetition, 6 by
insufficient material, 3 by the fifty-move rule and 1 at the cap of 400 plies (a draw); no game was lost on time.
The two Stockfish rows were measured just before the last search change of this version (worth +22 Elo, +8 to
+35, in self-play). A much older engine at a long time control and a limited Stockfish at a short one are
different yardsticks, so the rows should not be added up. The MayFly rows are the exception to the conditions
above: 12 openings, each from both sides, on another machine, with Stockfish judging a game only if it reaches
240 plies. A third match at 300 s + 3 s is running.

### MayFly and Midge together

The question behind Midge is whether MayFly plays better with Midge checking its moves. Midge searches on spare
CPU cores while MayFly thinks, and steps in only when MayFly's chosen move looks clearly worse to it. Duo 3 tries
the other way round: Midge never overrules MayFly, it only suggests a move before MayFly searches. All three tests
play the pair against MayFly v6.1 alone, 60 s + 1 s, on one machine (RTX 5070 Ti, i7-14700F).

| version | how Midge takes part | games | wins-draws-losses | score | Elo difference (95%) |
|---|---|---|---|---|---|
| duo 1 | MayFly's move gives up 0.20 or more of Midge's win chance | 11 | 1-9-1 | 50.0% | stopped early |
| duo 2 | the same at 0.10, plus clearly won positions and time trouble | 100 | 5-89-6 | 49.5% | -3 (-26 to +19) |
| duo 3 | a hint, never an override: its best move gets a quarter of MayFly's first guesses | 100 | 5-93-2 | 51.5% | +10 (-8 to +28) |

In duo 2 Midge changed MayFly's move on 131 of 6,858 moves (1.9%), and its checks added about 20 ms a move on
average. The time trouble rule never triggered. In duo 3 Midge looked at each position first, 150,000 nodes on
one thread (about 110 ms, taken from MayFly's clock). Its move was not the first choice of MayFly's network on
52% of moves, and in those MayFly went on to play it about a third of the time. So far the pair is not
measurably stronger than MayFly alone. The intervals are over single games; over pairs of games from the same
opening, duo 3's is -5 to +26.

## Threads and time

Measured with version 0.2 on the same laptop (6 fast cores with two threads each and 8 slow cores, 20 logical
CPUs), nothing else running.

Speed by threads: six middlegame and endgame positions, 3 s each, a 256 MB table, the engine at high priority.

| threads | nodes a second | against 1 thread |
|---|---|---|
| 1 | 1.00 million | 1.00 |
| 2 | 1.87 million | 1.88 |
| 4 | 3.87 million | 3.88 |
| 8 | 7.83 million | 7.85 |
| 16 | 8.42 million | 8.44 |

The laptop has 6 fast cores, so beyond 8 threads the extra ones land on slow cores or share a fast one. The same
run at normal priority gave 0.92, 1.73, 3.71, 6.58 and 6.59 million. More nodes is not the same as more
strength: the time to reach depth 20 on the same positions was 19.0 s with 1 thread and 24.2, 11.4, 8.4 and
10.4 s with 2, 4, 8 and 16 (one run each; a search with several threads does not repeat exactly). In games,
4 threads against 1 scored 67.0% over 100 games at 10 s + 0.1 s (+123 Elo, +84 to +165), measured with an
earlier build of this version. Other thread counts have not been played.

Strength against time: version 0.2 against itself, one side on 10 s + 0.1 s and the other on 5 s + 0.05 s,
100 games: the side with twice the time scored 26 wins, 70 draws, 4 losses, 61.0%, +78 Elo (+39 to +119). No
other pair of time controls has been played.

## How it works

- Board: bitboards with magic lookups (the magics are searched at start), fully legal move generation, checked by
  perft on the six standard positions and on 300 random positions.
- Search: iterative deepening, principal variation search with aspiration windows, a transposition table shared
  by all threads, quiescence search, null move, late move reductions, futility and late move pruning, static
  exchange evaluation, singular extensions, killer, counter move, history and continuation history ordering.
- Threads: every thread searches the same position on its own board and shares the table. The helper threads
  are started once and sleep between moves, and any thread that is running can end the search when the time is
  up, so the answer does not wait for a thread that has lost its CPU.
- Evaluation: a network with 768 inputs for each side (piece, colour, square, seen from that side), one shared
  layer of 512 units for each side, squared clipped ReLU, and one output chosen from 8 by the number of pieces.
  The first layer is updated move by move. Integer arithmetic, AVX2 where the CPU has it.
- Training data: 300,461,680 positions from Leela Chess Zero self-play games, each labelled with the value
  Leela's own search gave it. Positions where Leela's best move was a capture were left out. No other engine's
  code or network weights are used anywhere; Stockfish and other engines appear only as test opponents.
  Leela Chess Zero's training data is made available under the Open Database License (ODbL) 1.0 (see
  nets/NETS.md).
- The source also holds a small `veto` command (does a given move lose material by force?). It is used by an
  experiment that pairs Midge with MayFly and plays no part in Midge's own games.

## Building

You need g++ 13 or later (or clang++), on Linux or on Windows under MSYS2.

    H=512 VERSION=0.2 ./build.sh                 # for CPUs with AVX2 and BMI2
    H=512 VERSION=0.2 ARCH=x86-64 ./build.sh     # for any 64-bit CPU
    build/midge bench 12                         # prints a node count; it must match the one in CHANGES.md

Put the network file beside the executable as `midge_net.bin`. To build the network into the executable
instead, name it in the build: `NET=path/to/midge_net.bin H=512 ./build.sh` (the release network is
`midge_net.bin` from the 0.2 download).

## Tested, and not tested

Tested on Windows: perft; the network's move-by-move updates against a fresh build of them (0 mismatches over
213,691 checks); 40 UCI behaviour tests (clocks down to 1 ms, stop and quit at any time, pondering, very long
games, bad input); more than 10,000 complete games with every move checked legal; and the stress set below.

The stress set, the engine against itself, run on a test build of this source (the same code compiled for the
test machine, with the search test switches listed as options):

| games | time | threads a side | table | games at a time | lost on time | crashes or illegal moves |
|---|---|---|---|---|---|---|
| 300 | 0.5 s + 0.01 s | 1 | 1 MB | 4 | 0 | 0 |
| 100 | 1 s + 0.02 s | 8 | 16 MB | 2 | 0 | 0 |
| 200 | 2 s, no increment | 1 | 64 MB | 4 | 0 | 0 |
| 20 | 1 s + 0.05 s | 64 (on 20 logical CPUs) | 64 MB | 1 | 0 | 0 |

The last row is the one an earlier build failed: it lost all 20 games on time (see CHANGES.md).

The release executables themselves (`midge-0.2-x86-64-v3.exe` unless it says otherwise): the bench count on both
Windows files; the 100 games of the time test above; and 30 moves with 64 threads on a 1 s + 0.05 s clock, of
which the slowest took 165 ms.

Tested on Linux (Ubuntu 24.04): the static build gives the same bench count (723006), 0 mismatches in the network checks
(213,691 and 37,754 checks), perft 12 of 12 and 300 of 300, 37 of 37 UCI tests with at most two threads, and 12
games with two threads a side at 2 s + 0.05 s without a time loss or a crash. It was tested on a shared machine
at low priority, so only one and two threads were tried.

Not tested, so do not rely on it: tables larger than a few GB, time controls longer than 180 s + 3 s, Linux with
more than two threads, the plain x86-64 build in games, and any machine with more than 20 logical CPUs.
