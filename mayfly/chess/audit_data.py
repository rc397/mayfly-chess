"""Check the rich corpus for split leakage and opening bias.

The split takes every twentieth line, not every twentieth position, and positions repeat (openings
above all), so the same board can sit on both sides of the split. This counts how many held-out
positions also appear in training and where in the game the positions come from. audit_memorise.py
then scores seen and unseen positions apart.

    python chess/audit_data.py [N_LINES]
"""
import os, sys, glob, json, random
from collections import Counter
import chess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
RICH = os.path.join(HERE, "rich")
HOLDOUT = 20


def board_key(fen):
    """Position identity ignoring move counters, which is what 'the same position' means."""
    return " ".join(fen.split()[:4])


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(RICH, "*.jsonl")))
    if not files:
        raise SystemExit("no rich/*.jsonl found")
    print(f"{len(files)} shards, reading up to {N:,} lines\n", flush=True)

    train, val = set(), []
    plies, read = Counter(), 0
    rng = random.Random(0)
    rng.shuffle(files)

    for fp in files:
        if read >= N:
            break
        with open(fp, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if read >= N:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                read += 1
                k = board_key(rec["fen"])
                mv = int(rec["fen"].split()[-1])
                plies[min(mv, 80)] += 1
                if i % HOLDOUT == 0:
                    val.append((k, rec["fen"]))
                else:
                    train.add(k)

    seen = [v for v in val if v[0] in train]
    print(f"lines read              {read:,}")
    print(f"unique positions        {len(train | {v[0] for v in val}):,}  "
          f"({100*len(train | {v[0] for v in val})/read:.1f}% of lines)")
    print(f"validation positions    {len(val):,}")
    print(f"ALSO in the train slice {len(seen):,}  ({100*len(seen)/max(len(val),1):.1f}%)")
    print()

    tot = sum(plies.values())
    buckets = [("opening, moves 1-10", range(1, 11)), ("moves 11-20", range(11, 21)),
               ("moves 21-40", range(21, 41)), ("moves 41+", range(41, 81))]
    print("where in the game the positions come from")
    for label, rng_ in buckets:
        c = sum(plies[m] for m in rng_)
        print(f"  {label:22s} {100*c/max(tot,1):5.1f}%")

    json.dump({"read": read, "val": len(val), "val_seen_in_train": len(seen),
               "leak_rate": len(seen) / max(len(val), 1),
               "ply_hist": {str(k): v for k, v in sorted(plies.items())}},
              open(os.path.join(HERE, "audit_data.json"), "w"), indent=1)
    print("\nwrote audit_data.json")
