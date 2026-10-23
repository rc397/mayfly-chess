# Architecture

MayFly is a network that scores every move of a position, wrapped in a batched tree search. This page describes both
as they stand in the code.

## The network

The network is a transformer encoder (`chess/chess_net.py`, class `ChessNet`) with about 146.1 million parameters.

| part | value |
|---|---|
| layers | 16, pre-norm |
| width | 1024 |
| attention heads | 16, 64 dimensions each |
| feed-forward | 2048 wide, GELU |
| dropout | none |
| tokens | 65, a summary token and the 64 squares |
| moves | 4,162 outputs, 64 x 64 from-to pairs plus 66 underpromotions |

### Input

The board is always seen from the side to move. When Black is to move the position is mirrored, so the network sees
every position as if it were White. Colour itself is not an input.

Each square token is a piece embedding (empty, or one of six pieces for either side) plus a learned embedding of the
square. En passant is a learned vector added to the token of the square that can be captured on. The summary token
carries the castling rights, through a linear layer from four bits, and the halfmove clock as one of 102 learned
buckets (0 to 100 plies, plus one for a clock the data did not record). Buckets let the network learn that a clock
of 99 means something quite different from a clock of 98.

A move's index is `from * 64 + to` in the side to move's frame. Promotions to a knight, bishop or rook get 66 extra
indices after the first 4,096, so two promotions to the same square never share a target. The numbering lives in
`chess/moves.py` and the board encoding in `chess/encode.py`.

### Heads

#### Heads on the summary token

These read the summary token after the final layer norm.

- **Policy.** One logit for each of the 4,162 moves. The search's priors are a softmax over the legal ones.
- **Action value.** Another 4,162 logits. A sigmoid turns each into the win probability of playing that move. This is
  the head the action-value data trains, and the search builds its leaf values from it.
- **Value.** One logit for the position. It has been frozen since v4 and the search does not read it.
- **Auxiliary heads.** Material, king safety, pawn structure, mobility and the bishop pair, used only as extra
  training signal.
- **Moves left.** Plies until the game ends, from a small two-layer network. Trained, and switched off in the sealed
  searches.

#### Heads on the square tokens

- **Attention policy and attention action value.** A move is scored from its two squares, a query from the
  from-square against a key from the to-square in 256 dimensions, in the style of Leela's attention policy. They
  train on the same targets as the summary heads. The sealed searches read the summary heads.

#### Distributional heads

Distributional heads give the action value of a move and the value of the position as 64-bucket distributions,
trained against HL-Gauss targets. The move-conditioned one runs a move embedding and the summary vector through a
small network (about two million parameters) rather than widening the output layer. The search does not read them,
and the value distribution learns from the trunk without sending gradient back into it.

### What the search reads

The search's forward pass computes the trunk, the final layer norm and the policy and action-value heads, then
reduces them over the legal moves.

```
prior(a) = softmax(policy / policy_temp)[a]            over the legal moves
q(a)     = sigmoid(action_value[a])                    win probability after move a
value    = 2 * sum over a of softmax(policy)[a] * q(a) - 1
```

`policy_temp` reaches the priors only. The value always uses the plain softmax.

## The search

The search is batched PUCT, written from the published AlphaZero description. `chess/mcts.py` is the reference
implementation in Python, with its three hottest loops in C (`chess/native`). `chess/cpp` is a C++ port of the same
tree. On 414 test searches, with tree reuse and tablebases on, the port chose the same moves with the same statistics
as the Python, and it ran 1.32 to 1.49 times the visits a second.

### One pass

1. One thread walks the tree and collects up to 48 unexpanded leaves.
2. The leaves go through the network in one batched forward pass. A transposition cache in front of the network
   answers positions it has already evaluated.
3. Every leaf is expanded and its value is backed up.

Batching is the main source of speed. With an early network the search ran 1,608 nodes a second at a batch of 64
against 86 at a batch of one.

### Selection and backup

- **UCB-V selection** (`puct_v`). PUCT with a variance term and a range term added to the exploration bonus.
- **q_init.** An edge that has not been visited yet is priced at the action-value head's estimate for that move,
  and a visited edge blends that estimate with its backed-up values as `(n0 * Q0 + W) / (n0 + N)`.
- **Virtual mean.** A visit still waiting for the network counts as the edge's own current mean, the rule lc0
  uses, rather than as a loss, so a value is not dragged down while its visit waits.
- **Held collisions.** A walk that reaches a leaf already in the batch is held and not counted as a visit.
- **Forcing.** Once a root candidate has 8 visits, every capture and check in reply to it gets 4 real visits, so
  short tactical replies get looked at. With `forcing_apart` those visits still teach the reply node but
  stay out of the root edge's own statistics.

### Choosing the move

The move played is the one with the best lower confidence bound on its mean, a Student-t bound as in Leela Zero. A
move needs both a good mean and enough visits, and a candidate needs a minimum share of the leader's visits to be
considered at all.

### Tree reuse, pondering and rules

After a move, the subtree under the move played becomes the next root. Its carried visits count at a quarter
(`ponder_discount`), and each search adds its full budget of new visits on top. The engine ponders on the
opponent's time, and the tree is bounded by memory (`PONDER_MEM_GB`, 12 GB by default) rather than by visits.

Each node keeps its position key, so a line that repeats a position scores a draw. The fifty-move rule and a
100-ply draw are applied in the tree. With Syzygy tablebases present (`chess/syzygy.py`), the tree reads results
only right after a capture or pawn move, when the fifty-move counter is zero, and the root uses distance to zeroing
to convert won endings.

### Time

`chess/timectl.py` gives each move a slice of the remaining time plus most of the increment, capped so one move can
never lose the game on time. Inside the search, `tm_extend` keeps thinking while the top two moves are too close to
separate and `tm_stop` ends early when the leader is clearly ahead. In games of a minute or less with no increment,
`tm_short_sudden` plans each move at the network's real cost per call.

### Sealed settings

Every released version is frozen together with its search, 84 settings for v5 and 88 for v6. A version always plays the settings
it was sealed with, and new behaviour goes behind new settings that default to off. `chess/golden/` holds the sealed
settings of v5 and v6, and `chess/selftest.py` checks that every version pins every setting.

### What changed results

Most rows below are matches of one network against itself with one setting changed. Self-play numbers like these
overstate gains against other engines.

| change | measured |
|---|---|
| virtual mean | +512 Elo at 200 visits, 200 games |
| lower confidence bound pick | +196 at 200 visits, +258 at 400 visits, 200 games each |
| held collisions | +282 on a 30 second clock, +392 on a 60 second clock |
| q_init with held collisions | +78 at fixed visits, +135 on a 30 second clock |
| forcing_apart | +8.75 points against Stockfish Skill 10 over 200 games, about +70 Elo |
| compiled pick | +98 at a fixed time a move over 500 games, from x1.39 the visits a second |

Graph search over transpositions, proof propagation, power-mean backups, rescaling the leaf value, a capture search
at the leaf, wider priors and entropy-driven time all measured within noise or worse. The leaf value is noisy, and
the changes that helped removed a distortion in how visits were counted or priced rather than reweighting values.

## Compiled parts and backends

- **chess/native** builds three CPython extensions. `pick.c` is the selection arithmetic, `movegen.c` the legal
  moves, encoding and forcing masks, and `walk.c` the descent, unwind and backup. They are compiled with
  `/fp:precise` so their float32 results match the numpy code bit for bit. Without them the numpy code runs.
- **CUDA graphs.** The Python search can capture the bf16 forward pass as a CUDA graph per batch size, which took a
  pass from about 40 ms to 11 to 17 ms.
- **chess/cpp** builds `mayfly_cppsearch`, the C++ tree. It runs the network through a TorchScript trace in LibTorch
  (a CUDA graph per batch size), through the hand-written CUDA backend ([cuda_backend.md](cuda_backend.md)), or by
  calling back into the Python network. A setting the C++ tree does not cover, a missing build or a network file that
  does not match falls back to the Python search or network, never to a different search.
