# Midge: changes

Every strength number is from self-play at short time controls on one laptop (one thread, 200-position opening
book, each opening from both sides) with its 95% interval over opening pairs, unless it says otherwise.

## 0.2 (17 November 2026)

Check of a build: `midge bench 12` with the release network prints 723006 nodes.

Against other engines (one thread each, Hash 64 MB, no tablebases, every opening from both sides)
- Critter 1.6a, 180 s + 3 s, its own book off, 100 games: 75 wins, 21 draws, 4 losses, 85.5%,
  +308 Elo (+243 to +397).
- Stockfish 19 limited to UCI_Elo 3190, 10 s + 0.1 s, 100 games: 26-52-22, 52.0%, +14 (-39 to +68). Measured
  before the static exchange rule below went in.

Speed and time (the release build on an Intel i7-12700H)
- Nodes a second at 1, 2, 4, 8 and 16 threads: 1.00, 1.87, 3.87, 7.83 and 8.42 million (six positions, 3 s
  each, a 256 MB table).
- Twice the time (10 s + 0.1 s against 5 s + 0.05 s, 100 games): 26-70-4, 61.0%, +78 Elo (+39 to +119).

Search
- Singular extensions: +45 Elo (+25 to +66), 400 games at 5 s + 0.05 s.
- Continuation history (the moves one and two plies back) in move ordering and in the reduction rule:
  +27 (+7 to +48), 400 games.
- At shallow depth a quiet move that loses material by static exchange is skipped: +22 (+8 to +35), 800 games.
- More of the clock is used on each move (soft limit x1.6): +20 (+6 to +34), 800 games.
- Several search threads sharing one table (option Threads): 4 threads against 1 is +123 (+84 to +165),
  100 games at 10 s + 0.1 s.
- Tried and left out, because the games did not show a gain: quiet checks in quiescence, correction history,
  capture history, a second singular extension, a best-move stability rule for the clock, and other values of
  the reduction, futility and reverse futility margins.

Network
- 512 units a side (was 256), trained on 300 million positions (was 86 million). From 86 to 172 million
  positions was worth +74 (+40 to +109); from 172 to 300 million +12 (-16 to +41), so more of the same data has
  stopped paying.

Engine and protocol
- Pondering (`go ponder`, `ponderhit`), the options Ponder, MoveOverhead and Hash up to 1,048,576 MB.
- The network file `midge_net.bin` beside the executable is loaded at start.
- Node counts and speed in the info line cover all threads; `hashfull` is reported.
- A Linux build, and a build for CPUs without AVX2.

Fixes (found by a bug check with 40 UCI tests; none of them changes play at a fixed depth)
- With almost no time on the clock the engine played the first move it generated without searching it. It now
  always finishes a depth 1 search.
- Checkmate delivered on the move that completes the fifty-move count was scored as a draw.
- `stop` sent right after `go infinite` could be lost, and the engine then never answered.
- The engine kept searching after its input was closed.
- A game of more than about 3,800 plies overran a fixed array.
- A search limited to some root moves (`searchmoves`) stored its result in the table as if it were the full
  search.
- With several threads, every thread advanced the table's age.
- With more threads than the machine has CPUs the answer could come long after the time was up: at 64 threads
  on 20 logical CPUs a move with a limit of 237 ms took up to 1,119 ms, and all 20 test games at 1 s + 0.05 s
  were lost on time. The helper threads now stay alive between moves, every thread watches the clock, and the
  move is sent before the helpers are waited for. After the fix the slowest of 60 such moves took 241 ms.
  The same 20 games played again: none lost on time.
- `ucinewgame` now also empties the helper threads' move ordering tables.

## 0.1 (17 November 2026, not published)

The first version that played complete games: legal move generation checked by perft, a principal variation
search with a transposition table, quiescence, null move, late move reductions and futility pruning, and the
first network (256 units a side, 86 million positions). It scored 48.3% (-12 Elo, -64 to +40) over 60 games at
10 s + 0.1 s against Stockfish 19 limited to UCI_Elo 3190.
