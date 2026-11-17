# Midge

Midge is our second engine, written from scratch in C++. Where MayFly is a large network searched on a graphics
card, Midge is a small, fast alpha-beta engine on the CPU: a 768x2 to 512 NNUE network (SCReLU, eight output
buckets) evaluated in int16 with SIMD, inside a principal variation search. One thread of a laptop i7-12700H
searches about 650,000 positions a second. Its first release, 0.2, is being prepared, and its source comes into
this repository with it. We are also testing MayFly and Midge together, with Midge checking MayFly's moves.

Early results, before the release:

| Opponent | Conditions | Games | Midge W-D-L | Score | Elo, 95% interval |
|---|---|---|---|---|---|
| Critter 1.6a | 180 s + 3 s, 1 thread each, 64 MB hash, no book, no tablebases | 100 | 75-21-4 | 85.5% | +308 (+243 to +397) |
| MayFly v6.1 | 60 s + 1 s, Midge on 4 threads, MayFly on an RTX 5070 Ti | 24 | 4-14-6 | 45.8% | -29 (-126 to +64) |

The Critter games start from 50 openings, each played with both colours. Against MayFly, 12 openings are each played
with both colours, and Stockfish judges a game only if it reaches 240 plies. Longer time controls against MayFly are
still running.

The source code and the first release come into this folder with Midge 0.2.
