# Midge networks

The engine's code and the network's structure are our own. A network learns from positions that a stronger
player has already judged; where those judgements come from is stated here for every released network.

Leela Chess Zero's training data, which these networks are trained on, is made available under the Open
Database License (ODbL) 1.0, and its individual contents under the Database Contents License (DbCL) 1.0. The
data comes from the Leela Chess Zero project (https://lczero.org).

## midge_v5h512.bin (the network of 0.2)

- Structure: 768 inputs for each side (own or their colour, piece type, square, seen from that side), one
  shared layer of 512 units for each side, squared clipped ReLU, one output chosen from 8 by the number of
  pieces on the board. About 400,000 parameters, stored as 16-bit integers.
- Positions: 300,461,680 positions from Leela Chess Zero self-play training games, sampled from 700 files of
  them. Positions where Leela's best move was a capture were left out, so the network judges quiet positions
  and the search resolves the captures.
- Labels: the value Leela's own search gave each position in those games (the value of its best move after
  the search, the field best_q of Leela's training data), turned into a win chance. No game results and no
  other engine's evaluation are used.
- Training: our own trainer (PyTorch), 400,000 steps of 16,384 positions, squared error on the win chance.
  Loss on held-out positions (files not used for training): 0.01203.
- Check: the engine's integer evaluation agrees with the trainer's floating-point evaluation to 3 centipawns
  on average over held-out positions, and a position mirrored with colours swapped gets exactly the same
  value.

No code, network file or weights of any other engine are used. Stockfish and other engines appear only as
opponents in test matches.
