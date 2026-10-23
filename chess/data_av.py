"""Batch loader for the 40-byte action-value shards, on the same train and validation split as the rest.

The holdout hashes the position exactly as the packed loader does, with the same FNV constant and
castling term. The same board appears in several corpora drawn from Lichess games, and a different
hash would put it in training for one and validation for another.

    from data_av import AVLoader
    tr = AVLoader(shard_dir, batch=128, split="train")
    ids, cast, hm, ep, mv, q = tr.next()
"""
import os, glob, queue, threading, time
import numpy as np
import torch
from pack_av import DT_AV, REC_AV, unpack_ids, batch_tensors_av
from pack import ShardReader, shard_cap
from moves import N_MOVES


def board_hash_av(rows):
    """The position's 64-bit hash before the holdout's modulo, FNV over the unpacked board with the
    castling byte folded in. Every legal move of one position shares one value."""
    ids = unpack_ids(rows["ids4"]).astype(np.uint64)
    h = np.zeros(len(rows), np.uint64)
    for i in range(64):
        h = (h ^ ids[:, i]) * np.uint64(0x100000001B3)
    # pack.split_is_val folds in the castling flags as the dot product of four zero-or-one bytes
    # with 1, 2, 4, 8. Those bits are already packed into one byte here, and that byte holds
    # exactly that sum, so the two agree without any unpacking.
    h ^= rows["cast"].astype(np.uint64)
    return h


def split_is_val_av(rows, holdout=64):
    """The same hash pack.split_is_val computes, over the unpacked board."""
    return (board_hash_av(rows) % np.uint64(holdout)) == 0


class AVLoader:
    # how long next() waits before checking the filler is alive, a class attribute so tests can shorten it
    POLL = 30.0

    def __init__(self, shard_dir, batch=128, split="train", device="cuda", holdout=64,
                 seed=0, refresh=600, prefix="av", depth=6, pick=16, block=1024,
                 pool_size=16_000, val_size=50_000, thin=1, down_keys=None, down_weight=1.0):
        self.shard_dir, self.refresh, self.prefix = shard_dir, refresh, prefix
        # down_keys is a sorted .npy of board_hash_av values. A training row whose position is in it
        # is kept with probability down_weight, so those positions train less often, for example
        # self-play positions reached with a faulty search. Validation ignores it.
        self.down_weight = float(down_weight)
        self.down = (np.load(down_keys).astype(np.uint64) if down_keys and self.down_weight < 1.0
                     and split != "val" else None)
        if self.down is not None:
            self.down = np.unique(self.down) if len(self.down) else None
        self.batch, self.split, self.holdout = batch, split, holdout
        self.want_val = split == "val"
        self.pick, self.block, self.pool_size = pick, block, pool_size
        # thin is for shards that are not shuffled. ChessBench is globally shuffled, so thin stays 1.
        # deep_label writes every legal move of a position back to back, so a run of 1,024 rows from a
        # dp shard holds only about 33 positions. With thin=k a block reads k times the run, still one
        # sequential read, and keeps a random `block` of its rows, reaching k times as many positions.
        self.thin = max(1, int(thin))
        self.pool, self.pos = None, 0
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.paths = self._list()
        if not self.paths:
            raise SystemExit(f"no {prefix}_*.bin in {shard_dir}")
        self._reopen()
        self.last_refresh = time.time()
        self.fixed = self._build_fixed(val_size) if self.want_val else None
        if self.want_val:
            print(f"[av loader] validation frozen at {len(self.fixed):,} action values", flush=True)
        self.q = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.error = None       # what stopped the filler, if anything did
        self.fails = 0
        self.max_fails = int(os.environ.get("LOADER_RETRIES", 30))
        self.t = threading.Thread(target=self._fill, daemon=True)
        self.t.start()

    def _list(self):
        return sorted(glob.glob(os.path.join(self.shard_dir, f"{self.prefix}_*.bin")))

    def _reopen(self):
        self.paths = self._list()
        self.mm = [ShardReader(p, DT_AV) for p in self.paths]   # read on demand, see pack.ShardReader
        self.paths, self.mm = shard_cap(self.paths, self.mm)      # a frozen corpus under SHARD_CAP
        sizes = [len(m) for m in self.mm]
        self.cum = np.concatenate([[0], np.cumsum(sizes)])
        self.total = int(self.cum[-1])

    def __len__(self):
        return self.total

    def describe(self):
        return (f"{len(self.paths)} shards [{self.prefix}], {self.total:,} action values, "
                f"{self.total * REC_AV / 1e9:.1f} GB, split={self.split}"
                + (f", {len(self.down):,} positions at weight {self.down_weight:g}"
                   if self.down is not None else ""))

    def _blocks(self, block=2048):
        """Pooled rows, read as a few sequential runs instead of hundreds of scattered ones.

        Random rows across hundreds of shards mean hundreds of page faults per batch, which stalls on
        disk. A batch instead reads a contiguous run from each of a handful of shards, and rows are
        subsampled from the pool so a batch still spans several shards.
        """
        live = [i for i in range(len(self.mm)) if len(self.mm[i])]
        if not live:
            return []
        out = []
        # Training weighs each shard by its rows, with replacement, so a row in a young shard is drawn
        # no more often than one in a full shard. max(rows, block) gives every row the same chance per
        # pick, thinned or not. Validation keeps its old rule so its set stays the same.
        k = min(self.pick, len(live))
        if self.want_val:
            pick = self.rng.choice(live, k, replace=False)
        else:
            w = np.array([max(len(self.mm[i]), block) for i in live], np.float64)
            pick = self.rng.choice(live, k, replace=True, p=w / w.sum())
        for s in pick:
            m = self.mm[int(s)]
            span = min(block * self.thin, len(m))
            # a run may end on the last row, as in data_packed, but the validation draw keeps the old
            # range so the set it rebuilds at every start stays the same
            j = int(self.rng.integers(0, max(len(m) - span, 1) if self.want_val
                                     else len(m) - span + 1))
            run = np.asarray(m[j:j + span])
            # only when thinning, so thin=1 draws exactly the stream it always drew
            if self.thin > 1 and len(run) > block:
                run = run[self.rng.choice(len(run), block, replace=False)]
            # only with down_keys, so without them the stream is exactly the one it always drew
            if self.down is not None and len(run):
                h = board_hash_av(run)
                at = np.minimum(np.searchsorted(self.down, h), len(self.down) - 1)
                hit = self.down[at] == h
                if hit.any():
                    run = run[~hit | (self.rng.random(len(run)) < self.down_weight)]
            out.append(run)
        return out

    def down_share(self, rows):
        """The fraction of these rows whose position is in down_keys (0 without them)."""
        if self.down is None or not len(rows):
            return 0.0
        h = board_hash_av(rows)
        at = np.minimum(np.searchsorted(self.down, h), len(self.down) - 1)
        return float(np.mean(self.down[at] == h))

    def _refill(self):
        """Read one pool and serve many batches from it.

        A standard shuffle buffer. The pool is read once, shuffled and consumed a batch at a time, so
        one read serves over a hundred steps. It is safe because the pool comes from several shards
        at once, and `thin` makes that hold for dp shards, which are grouped by position.
        """
        out, got = [], 0
        for _ in range(8):
            for rows in self._blocks(block=self.block):
                rows = rows[~split_is_val_av(rows, self.holdout)]
                if len(rows):
                    out.append(rows)
                    got += len(rows)
            if got >= self.pool_size:
                break
        pool = np.concatenate(out) if out else np.zeros(0, DT_AV)
        self.pool = pool[self.rng.permutation(len(pool))]
        self.pos = 0

    def _draw(self):
        if self.want_val:
            i = self.rng.choice(len(self.fixed), min(self.batch, len(self.fixed)), replace=False)
            return self.fixed[i]
        if self.pool is None or self.pos + self.batch > len(self.pool):
            self._refill()
        rows = self.pool[self.pos:self.pos + self.batch]
        self.pos += self.batch
        return rows

    def _build_fixed(self, n):
        """The validation set, built once and held in memory.

        Resampling would make the number incomparable between iterations as the corpus grows, and a
        one in 64 holdout reads 64 times what it keeps. Fixing it costs a few megabytes.
        """
        out, got = [], 0
        for _ in range(400):
            for rows in self._blocks(block=8192):
                rows = rows[split_is_val_av(rows, self.holdout)]
                if len(rows):
                    out.append(rows)
                    got += len(rows)
            if got >= n:
                break
        fixed = np.concatenate(out) if out else np.zeros(0, DT_AV)
        return fixed[:n]

    def _fill(self):
        while not self.stop.is_set():
            try:
                if (self.refresh and not self.want_val
                        and time.time() - self.last_refresh > self.refresh):
                    before = self.total
                    self._reopen()
                    self.last_refresh = time.time()
                    if self.total > before:
                        print(f"[av loader] corpus grew by {self.total - before:,} to "
                              f"{self.total:,}", flush=True)
                ids, cast, mv, q, hm, ep = batch_tensors_av(self._draw())
                t = lambda a, d: torch.as_tensor(a).to(self.device, dtype=d, non_blocking=True)
                item = (t(ids, torch.long), t(cast, torch.float32),
                        t(hm, torch.long), t(ep, torch.long),
                        t(mv, torch.long), t(q, torch.float32))
                # a full queue waits with the batch in hand rather than drawing past it
                while not self.stop.is_set():
                    try:
                        self.q.put(item, timeout=30)
                        break
                    except queue.Full:
                        continue
            except Exception as e:
                # retry transient faults such as a brief out of memory, but give up after
                # max_fails so the consumer does not wait forever
                self.fails += 1
                print(f"[av loader] {type(e).__name__}: {e} ({self.fails} in a row)", flush=True)
                if self.fails >= self.max_fails:
                    self.error = e
                    print(f"[av loader] giving up after {self.fails} failures in a row", flush=True)
                    return
                time.sleep(2)
            else:
                self.fails = 0

    def next(self):
        """The next batch. If the filler has given up, this raises what stopped it instead of waiting."""
        while True:
            try:
                return self.q.get(timeout=self.POLL)
            except queue.Empty:
                if self.error is not None:
                    raise RuntimeError("the action-value loader gave up") from self.error
                if self.stop.is_set():
                    raise RuntimeError("the loader was closed while a batch was still wanted")
                if not self.t.is_alive():
                    raise RuntimeError("the action-value loader thread is gone")

    def close(self):
        self.stop.set()


if __name__ == "__main__":
    import sys
    import paths
    d = sys.argv[1] if len(sys.argv) > 1 else paths.SHARDS

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = AVLoader(d, batch=128, split="train", device=dev, refresh=0)
    va = AVLoader(d, batch=128, split="val", device=dev, seed=1, refresh=0)
    print(tr.describe())
    print(va.describe())

    for _ in range(8):
        tr.next()
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time(); N = 60
    for _ in range(N):
        b = tr.next()
    if dev == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / N * 1000
    print(f"\nbatch of 128: {ms:.2f} ms ({128 / ms * 1000:,.0f} action values/s)")
    ids, cast, hm, ep, mv, q = b
    print(f"shapes  ids {tuple(ids.shape)}  cast {tuple(cast.shape)}  "
          f"mv {tuple(mv.shape)}  q {tuple(q.shape)}")
    print(f"ids range {int(ids.min())} to {int(ids.max())}, "
          f"mv range {int(mv.min())} to {int(mv.max())}, "
          f"q range {float(q.min()):.3f} to {float(q.max()):.3f} mean {float(q.mean()):.3f}")
    print(f"clock range {int(hm.min())} to {int(hm.max())} "
          f"({100*float((hm == 101).float().mean()):.1f}% unknown), "
          f"en passant set on {100*float((ep > 0).float().mean()):.2f}%")

    def keyset(loader, n):
        s = set()
        for _ in range(n):
            rows = loader._draw()
            ids = unpack_ids(rows["ids4"])
            for r, i in zip(rows, ids):
                s.add(i.tobytes() + bytes([r["cast"]]))
        return s
    a, c = keyset(tr, 10), keyset(va, 10)
    print(f"\ntrain positions {len(a):,}, val {len(c):,}, overlap {len(a & c)}")
    assert not (a & c), "a position appears on both sides of the split"
    print("no position appears on both sides of the split")
    tr.close(); va.close()
