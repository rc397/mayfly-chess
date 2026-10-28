# MayFly

**[Contributors and how to help train MayFly](https://rc397.github.io/mayfly-chess/)**

MayFly is a chess engine built on a transformer network of about 146 million parameters. The network reads a position
as 64 square tokens and a summary token, and predicts a policy over moves and the win probability of every legal
move. It plays through a batched PUCT tree search.

- The network was trained on about 12 billion action values (Stockfish's win probability for each legal move of a
  position), about 5 billion positions with target move distributions, and positions from its own self-play
  labelled by Stockfish.
- The search is written in Python with compiled C parts, and has a C++ port that makes the same choices.
- A hand-written CUDA backend runs the network for the search. Its kernels are compiled at run time by NVRTC and its
  matrix products run in cuBLASLt. An optional TensorRT path builds the same forward pass for comparison.
- The trainer is PyTorch on a GPU, with a JAX version for TPUs.

MayFly speaks UCI, so any chess GUI or match runner can use it, and it comes with a small board that runs in a browser.

## Acknowledgements

MayFly's training is supported by an AI Builder Grant from [CloudRift](https://cloudrift.ai), which provides GPU
cloud compute.

## Help train MayFly

MayFly learns from its own games and from deep Stockfish analysis, and you can lend it free compute on Kaggle. Open
the [volunteer notebook](https://www.kaggle.com/code/rc7530/mayfly-selfplay-volunteer), click Copy & Edit, then Save
Version. With a TPU (Settings > Accelerator) it plays MayFly self-play; without one it labels hard positions with
Stockfish. Everyone who helps is on the [contributors page](https://rc397.github.io/mayfly-chess/).

On Windows you can also lend your own graphics card. Download `MayFlyVolunteer.exe` from the
[latest release](https://github.com/rc397/mayfly-chess/releases/latest), open it, pick a name and press Run. It works on
NVIDIA, AMD and Intel graphics, and its source is in [chess/volunteer](chess/volunteer).

## Results

These are measured results for v6, the latest released version (24 October 2026), and for the CUDA backend. The
research notes they come from are summarised in [docs/](docs/).

### Against Stockfish

Stockfish 18 at 100 ms a move with 2 threads and a 256 MB hash, 24 games for each cell. A level is held when MayFly
scores more than half the points.

| MayFly v6 | highest level held | first level failed |
|---|---|---|
| network alone, no search | Skill 8, 54.2% | Skill 9, 43.8% |
| 400 visits a move | Skill 13, 60.4% | Skill 14, 47.9% |
| 1,600 visits a move | Skill 19, 54.2% | Skill 20, 27.1% |
| 6,400 visits a move | Skill 19, 66.7% | Skill 20, 45.8% |

v5 held the same level at 6,400 visits (Skill 19, 75.0%) but scored 22.9% at Skill 20, and held Skill 10 at 400
visits and Skill 18 at 1,600.

### Other measurements of v6

- 503 rated Lichess puzzles: a puzzle rating of 1,993 with no search, 2,866 at 400 visits and 3,043 at 1,600 (v5:
  1,927 and 2,830). A puzzle rating measures how hard puzzles are for people and is not a playing rating.
- 300 moves from opening book positions, kept within 30 centipawns of Stockfish's best: 94.7% with no search, 96.0%
  at 400 visits and 97.3% at 1,600.

### The CUDA backend

- On an RTX 5070 Ti, the same C++ search ran 1,600 visits in 517 ms on the CUDA backend (fp16 accumulation in two
  of the four products) against 823 ms on a LibTorch TorchScript trace, 1.59 times as fast.
- In a match at 60 seconds plus 1 a move with pondering, with the same search and network on both sides, the CUDA
  backend scored +39 =157 -3 over 199 games against the LibTorch trace. That is about +64 Elo (95% interval +43 to
  +85), from 6,294 visits a move against 3,994 in the same time.
- On an RTX 3070, its fastest setting evaluated a batch of 48 positions in 23.73 ms against 23.94 ms for TensorRT
  10.16, and a single position in 1.56 ms against 2.92 ms. Its accurate setting, close to fp32, is 12 to 15% behind
  TensorRT at large batches.
- Its default setting matches a float32 reference to a largest policy logit error of 0.0052, against 0.088 for the
  bf16 PyTorch pass.

[docs/cuda_backend.md](docs/cuda_backend.md) has the full tables.

## Repository layout

```
chess/                     the engine, the trainer, the data tools and the measurement scripts
  chess_net.py             the network
  encode.py, moves.py      board encoding and move numbering
  mcts.py                  the tree search (Python reference)
  mcts_cpp.py              runs a search on the C++ tree
  timectl.py               time management
  uci.py                   the UCI engine
  play_web.py              a board in the browser
  train_av.py              the trainer (PyTorch)
  train_jax.py             the trainer in JAX for TPUs (train_xla.py is a PyTorch/XLA fallback)
  data_av.py, data_packed.py, pack.py, pack_av.py    record formats and loaders
  chessbench.py, fetch_lichess_evals.py, lichess_to_av.py, lc0_convert.py, lc0_pool.py    data conversion
  selfplay.py, deep_label.py, hard_positions.py      self-play and Stockfish labelling
  kaggle_selfplay.py       the volunteer notebook (self-play on a TPU or GPU, labelling on a CPU)
  kaggle_volunteer_collect.py, contributors_page.py  collecting volunteers' runs and the contributors page
  elo_vs_sf.py, clock_match.py, search_match.py, round_robin.py    matches and ladders
  tactics_suite.py, lichess_puzzles.py, opening_suite.py, blunder_suite.py    position suites
  harness.py, selftest.py  tests, among them that sealed versions still play the same moves
  paths.py                 where data lives
  native/                  C modules for the Python search (selection, move generation, tree walk)
  cpp/                     the C++ search tree and its LibTorch network backend
  cudanet/                 the hand-written CUDA backend
  trtnet/                  the TensorRT path
  golden/                  sealed search settings of v5 and v6
_abq/                      a queue that runs A/B matches from a frozen copy of the code
docs/                      architecture, training and CUDA backend notes
```

## Requirements

- Python 3.13, the version the code and its native modules were built and tested with.
- The packages in `requirements.txt`, with a CUDA build of PyTorch. An NVIDIA GPU is expected for play and training.
- Stockfish, for labelling and for the measurement scripts. Most scripts read its path from
  `chess/stockfish_path.txt`, a one-line file (`python chess/get_stockfish.py` downloads a Windows build and writes
  it), and some read the `SF` environment variable.
- Optional, Syzygy tablebases. `python chess/syzygy.py` fetches the six-piece set into `data/syzygy`
  (`SYZYGY_DIR`).
- Optional, Windows with Visual Studio 2022 for the native modules, the C++ search and the CUDA backend. Each of them
  has a Python fallback, so the engine runs without them, more slowly.

```
pip install -r requirements.txt
```

### Where data goes

Everything the code reads or writes lives under one folder, set by `MAYFLY_DATA` and by default `data/` next to the
repository (`chess/paths.py`). Checkpoints go to `data/ckpt`, frozen versions to `data/versions`, training shards to
`data/shards` and self-play to `data/selfplay`. A script that has its own variable for a path (`CKPT_DIR`, `SHARDS`,
`VERSION_DIR`, `SELFPLAY_DIR` and others) still reads it first.

Trained weights are not part of the repository. A version is a checkpoint such as `data/versions/mayfly_v6.pt`,
listed in `data/versions/versions.json` with the search settings it was sealed with.

## Building the native parts

All of these are optional and use MSVC.

The compiled selection, move generator and tree walk for the Python search build as three extensions, which go
beside the Python files in `chess/`.

```
cd chess\native
python setup.py build_ext --inplace
move mayfly_*.pyd ..
```

The C++ tree without LibTorch, with the network still run from Python, builds the same way. Put the module in
`chess/` or point `CPPSEARCH_DIR` at its folder.

```
cd chess\cpp
python setup.py build_ext --inplace
```

The C++ tree with the network in C++ (LibTorch with CUDA graphs) and the CUDA backend builds with
`chess\cpp\build_rig.bat`. It reads four environment variables, `TORCH_KIT` (the installed torch's `include`, `lib`
and `cuda\include`), `CUDA_KIT` (the CUDA 12 headers), `PY` (the CPython install with `include` and `libs`) and `OUT`
(the output folder, `build` by default). The module loads only beside the torch version it was linked against. The
LibTorch path needs a TorchScript trace of the network next to the checkpoint.

```
python -c "import sys; sys.path.insert(0, 'chess'); import mcts_cpp; mcts_cpp.export_trace('data/versions/mayfly_v6.pt')"
```

The CUDA backend has its own build script and weight file format, described in
[docs/cuda_backend.md](docs/cuda_backend.md).

## Playing

```
python chess/uci.py
```

Then type `uci`, `isready`, `position startpos` and `go nodes 400`, or point a GUI or match runner at it. `VERSION`
picks the file in `data/versions` (for example `VERSION=mayfly_v6.pt`), and each version plays the search it was
sealed with. Under a clock the time manager decides the visits, and `VISITS` is only used when no clock or node count
is given. `go wtime ... btime ...`, `go movetime`, `go nodes`, `go infinite` and pondering are supported.

```
python chess/play_web.py
```

This serves a board at http://localhost:8000. `VISITS=0` plays the network alone.

## Training

Training reads packed shards from `data/shards`. The data scripts download and convert each source, and most of
them resume where they stopped.

```
python chess/chessbench.py            # ChessBench action values, av_ shards
python chess/fetch_lichess_evals.py   # Lichess evaluations as positions, lz_ shards
python chess/lichess_to_av.py         # the same evaluations as action values, lv_ shards
python chess/lc0_pool.py              # Leela training data as positions, lz_ shards
```

The sources are large. Ten billion ChessBench action values take about 800 GB of download, the 12 billion used here
pack into about 480 GB, and Leela's training archives run to terabytes.

```
python chess/train_av.py
```

The trainer's settings are environment variables, listed at the top of `chess/train_av.py`. It writes checkpoints to
`data/ckpt` and freezes versions into `data/versions` at set training hours. `chess/cooldown.py` makes a release
candidate by cooling a copy of the run. `chess/train_jax.py` takes the same settings on a TPU host and reads and
writes the same checkpoint format, so a run can move between a GPU and a TPU.

Self-play data comes from `chess/selfplay.py` (positions from MayFly's own games) and `chess/deep_label.py` (Stockfish
scores for their top moves), written as `dp_` shards.

[docs/training.md](docs/training.md) describes the data mix, the losses and the schedule used for v5 and v6, and
[docs/architecture.md](docs/architecture.md) the network and the search.

## Data sources and their licences

| source | used for | licence |
|---|---|---|
| [ChessBench](https://github.com/google-deepmind/searchless_chess) from Google DeepMind | action values (`av_`) | the dataset is CC BY 4.0, except the parts from lichess.org, which are CC0. The code is Apache 2.0 |
| [Lichess open database](https://database.lichess.org/), the evaluations and the puzzles | positions (`lz_`), action values (`lv_`), puzzle measurements | CC0 |
| [Leela Chess Zero training data](https://storage.lczero.org/files/training_data/) | positions (`lz_`, `lc_`) | [ODbL 1.0 for the collection, DbCL 1.0 for its contents](https://lczero.org/blog/2021/06/the-importance-of-open-data/) |
| [Stockfish](https://github.com/official-stockfish/Stockfish) | labels for self-play and relabelled positions, the opponent in measurements | GPL 3 |
| [Fishtest games](https://huggingface.co/datasets/official-stockfish/fishtest_pgns), optional | `fs_` action values, not in the released mix | LGPL 3.0 as listed on the dataset page |
| [Syzygy tablebases](https://tablebase.lichess.ovh/tables/standard/), optional | exact endgame results during play | no licence stated on the mirror |

What these mean for you.

- ChessBench is CC BY 4.0. If you share the data or anything derived from it, credit Google DeepMind and cite the
  paper that introduced it (https://arxiv.org/abs/2402.04494).
- The Leela data is under the ODbL. If you publish a database made from it, such as converted shards, it must credit
  the source and stay under the ODbL.
- The Lichess exports are CC0 and carry no conditions.
- Stockfish runs as a separate program and is not included here. If you distribute a Stockfish binary, its GPL
  applies to it.

## Third-party code in this repository

- `chess/encoder.cc` and `chess/trainingdata.cc` come from Leela Chess Zero, copyright The LCZero Authors, under the
  GPL version 3 or later with an additional permission for linking NVIDIA libraries. They are kept as the reference
  for Leela's training data format.
- Dependencies installed separately keep their own licences. python-chess is GPL 3. PyTorch, JAX and the other Python
  packages have their own permissive licences, and CUDA, cuBLASLt, NVRTC and TensorRT are under NVIDIA's licences.

## Licence

Copyright (C) 2026 rc397

MayFly is free software. You can redistribute it and modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License or (at your option) any later version.
See [LICENSE](LICENSE).

### Additional permission under GNU GPL version 3 section 7

If you modify this program, or any covered work, by linking or combining it with NVIDIA Corporation's libraries from
the NVIDIA CUDA Toolkit, including cuBLAS, cuBLASLt and NVRTC, or with NVIDIA TensorRT (or a modified version of
those libraries), containing parts covered by the terms of the respective license agreement, the licensors of this
program grant you additional permission to convey the resulting work.
