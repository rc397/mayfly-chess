# Training

One network learns from two kinds of record in the same forward pass. `chess/train_av.py` is the trainer on a GPU and
`chess/train_jax.py` the same trainer in JAX for TPUs.

## Records

- **An action value** is a position, one move and the win probability of playing that move. It is packed into 40
  bytes (`chess/pack_av.py`) and read by `AVLoader` (`chess/data_av.py`).
- **A position** is a board with a target distribution over its best moves, a value and some auxiliary labels. It is
  packed into 112 bytes (`chess/pack.py`) and read by `PackedLoader` (`chess/data_packed.py`).

Shards are written under `SHARDS` (by default `data/shards`, see `chess/paths.py`), and the trainer picks a source by
its file prefix. The holdout split hashes the position itself, the same way for every source, because ChessBench and
the Lichess evaluations both come from Lichess games and share positions. The validation set is frozen on disk.

## Sources

| prefix | source | what a record holds | amount trained on | script |
|---|---|---|---|---|
| `av_` | ChessBench | every legal move of positions from Lichess games, scored by Stockfish 16 at 50 ms a move (about depth 13) | 12.0 billion action values, about 481 GB packed | `chessbench.py` |
| `lz_`, `lc_` | Leela Chess Zero training data, run test91 | the visit distribution of Leela's search cut to its top five moves, with the game value and plies left | about 5.2 billion position records, Lichess below included | `lc0_convert.py`, `lc0_pool.py` |
| `lz_` | Lichess evaluation database | Stockfish lines at a median of about 25 million nodes, three to five moves a position, as policy targets | about 39 million positions | `fetch_lichess_evals.py` |
| `lv_` | the same Lichess evaluations | each scored move as its own action value | 37.7 million positions, 143 million action values | `lichess_to_av.py` |
| `dp_` | self-play | MayFly's own positions, the top moves scored by Stockfish | tens of millions of rows | `selfplay.py`, `deep_label.py` |
| `dh_` | deep relabels | positions MayFly misplayed, relabelled at depth 26 or more | optional slice | `hard_positions.py`, `deep_label.py` |
| `fs_` | Fishtest games | Stockfish's own long games, one move a position at its own evaluation | optional, not in the released mix | `fishtest_to_av.py` |
| `st_` | search targets | MayFly's own search values at its self-play positions | optional, not in the released mix | `pack_selfplay.py` |

Some facts about the data.

- Cutting Leela's distributions to five moves drops 17% of the visit mass on average (the median is 8%). Stockfish's
  best move is inside Leela's top five in 96% of positions.
- Centipawns become a win probability by ChessBench's logistic curve for every source, and a mate counts as certain,
  so all action values sit on one scale.
- Lichess deep evaluations are about a thousand times the search of a ChessBench label per position.

### Self-play labels

`selfplay.py` plays MayFly against itself at 400 visits a move and keeps the positions, not the results. The first
20 plies sample moves from the visit counts, a quarter of the root prior is replaced by Dirichlet noise (total
concentration 10.83, AlphaZero's form), and games start from both the initial position and positions from the corpus.
`deep_label.py` then scores the top moves of each position with Stockfish under MultiPV (`MULTIPV`, ten by default,
0 for every legal move). Labelling only the top moves buys depth. In a test against 60 second labels, the top five at
15 seconds a position reached depth 24.7 and matched the reference's best move in 15 of 24 positions, while every
legal move at the same cost reached depth 17.8 and matched 12.

## The mix

A step takes `AV_BSZ` action values and `POL_BSZ` positions and runs them through the trunk together, each loss
applied to its own slice. For v5 and v6 a step was 64 action values and 64 positions, and the action values were split
41 ChessBench, 13 self-play and 10 Lichess deep. `DP_SHARE`, `LV_SHARE`, `DH_SHARE`, `FS_SHARE` and `ST_SHARE`
set the other shares, and whatever they leave comes from ChessBench.

## Losses

- **Action value.** Binary cross entropy between the action-value logit of the one move a record names and its win
  probability. The other 4,161 outputs get no gradient from that record, so moves nobody asked about are never
  taught to be bad.
- **Policy.** Cross entropy against the target distribution, restricted to legal moves (`MASK_LEGAL=1`). Before the
  mask, a quarter of the policy's mass sat on illegal moves.
- **Value.** Binary cross entropy with weight `W_V`. It is 0 from v4 on, which leaves the head out of the loss.
- **Auxiliary heads.** Squared error and cross entropy on five positional labels, weight 0.2.
- **Attention heads.** The same targets as the summary heads, weight `W_ATT`. For their first steps they learn on
  a detached trunk, so a random head cannot shake it.
- **Distributional heads.** Cross entropy against HL-Gauss targets over 64 buckets. The value distribution never
  sends gradient into the trunk.
- **Moves left.** Huber loss on plies divided by 100, weight 0.1, on rows that carry the label.

After every step the policy head is recentred by subtracting its mean row. In exact arithmetic that changes nothing,
but bf16 logits otherwise drift to a large common offset where the top moves tie. Before recentring, v3's median
logit was about -299 and 44% of positions had a tie for the top prior.

## Optimiser and schedule

- AdamW, weight decay 0.05 on matrices and none on vectors, beta2 0.999.
- Gradients clipped to norm 1, and a step is skipped when the norm is not finite or above 50.
- Warmup over 12,000 steps. The rate halves at each version freeze (24, 48, 96 and 192 training hours), a rate
  proportional to one over time, and then holds at 3e-5.
- No dropout.

Versions are frozen at training-hour marks (`VERSION_AT`) with their metrics. A release candidate is a cooled copy
of the run whose rate falls along a half cosine to 1% of the base rate (`COOL_FROM`, `COOL_STEPS`, `COOL_FLOOR`).
For v5 the cooldown ran about 455,000 steps and moved top-1 accuracy on legal moves from 46.6% to 51.4% and the
action-value loss from 0.4707 to 0.4612.

### Precision and speed

The default is bf16 autocast. `PRECISION=fp16acc` (fp16 with fp16 accumulation and a gradient scaler) together with
`COMPILE=1` (torch.compile) trained 1.48 times as fast on an RTX 5070 Ti, 1,133 against 763 positions a second, with
a loss closer to float32 than bf16 gives. FP8 through torchao measured slower on the same card.

## The TPU trainer

`train_jax.py` is the GPU trainer's model, losses, AdamW, schedule and checkpoint format, written in JAX and compiled
once. One process drives every chip with the batch sharded across them, and loader processes feed it whole batches.
Matrix products run in bf16 with float32 accumulation, while weights, optimizer moments, softmaxes, layer norms and
losses stay in float32.

- Against the PyTorch step on the same batch, the loss was identical, every gradient agreed within 1.2e-6 and a whole
  update within 1.3e-5.
- Checkpoints are read and written in `train_av.py`'s format, so a GPU run resumes from a TPU run and back.
- The first save comes at step 40, so a save that cannot work shows within minutes, then every 30 minutes and at the
  end. `XLA_COOL_MIN` cools a copy at the end of a run.
- A JAX training bench (`tpu_bench.py`) ran the full network at 11,814 positions a second on a TPU v5e-8 at a global
  batch of 1,024, about 43% of peak, and about 13 times one RTX 5070 Ti with fp16 accumulation and compile.

`train_xla.py` runs the PyTorch trainer through PyTorch/XLA and is kept as a fallback. It traces every step in
Python, which makes it host-bound, so JAX is tried first. `chessnet_jax.py` is the forward pass alone in JAX, used
for self-play on TPUs.
