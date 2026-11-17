# MayFly and Midge

Two chess engines, both our own code.

| Folder | Engine | What it is |
|---|---|---|
| [mayfly](mayfly) | MayFly | A transformer of about 146 million parameters, searched with a batched PUCT tree on the GPU. It plays on Lichess. |
| [midge](midge) | Midge | A small, fast alpha-beta engine with an NNUE network, written from scratch in C++. It runs on the CPU. |

Each folder has its own README with results, how to build it and how to play it.

**[Contributors and how to help train MayFly](https://rc397.github.io/mayfly-chess/)**

## Latest results

- MayFly v6.1 is the current release ([results](mayfly/README.md#results), [download](https://github.com/rc397/mayfly-chess/releases/latest)).
- Midge 0.2 is the first version of Midge, with its full source in [midge](midge): 75 wins, 21 draws and 4 losses
  in 100 games against Critter 1.6a at 180 s + 3 s (85.5%, +308 Elo, 95% interval +243 to +397), one thread each
  ([results, speed and what was tested](midge/README.md), [changes](midge/CHANGES.md)).

## Acknowledgements

MayFly's training is supported by an AI Builder Grant from [CloudRift](https://cloudrift.ai), which provides GPU
cloud compute.

## License

GPL-3.0, see [LICENSE](LICENSE).
