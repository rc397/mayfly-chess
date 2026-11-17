"""Batch loader for the 112-byte packed policy shards, feeding the GPU from a background thread.

Records are memory-mapped and batched with vectorised numpy, and a thread keeps a queue of batches
already on the GPU. The train and validation split keys on the position, so a repeated board always
lands on the same side. Records are already from the side to move's point of view, so nothing is
flipped at training time.

    from data_packed import PackedLoader
    tr = PackedLoader(shard_dir, batch=128, split="train")
    ids, cast, hm, ep, policy, value, material, king, pawns, mob, bishop, aux = tr.next()
"""
import os, glob, queue, threading
import numpy as np
import torch
from pack import DT, REC, ShardReader, batch_tensors, shard_cap, split_is_val
from moves import N_MOVES

# prefixes of this record, lz Lichess deep Stockfish, lc Leela, gm tournament games, sp self-play.
# The av action-value record belongs to data_av.
PACKED = ("lz", "lc", "gm", "sp")


def _mtime(path):
    """A shard's modification time, or 0 for one a converter moved away since the listing."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


class PackedLoader:
    # how long next() waits before checking the filler is alive, a class attribute so tests can shorten it
    POLL = 30.0

    def __init__(self, shard_dir, batch=128, split="train", device="cuda", holdout=64,
                 seed=0, depth=6, temp=100.0, refresh=300, sources=None,
                 pick=16, block=1024, pool_size=16_000, val_size=50_000, legal=False,
                 val_cache=None, mleft=False, need_flag=0, min_mtime=None):
        # A memmap fixes the file length when opened, so shards are reopened every `refresh` seconds
        # to pick up records still being written. Zero disables it for a finished corpus. `sources`
        # picks which teachers (shard prefixes) to mix.
        self.shard_dir, self.refresh = shard_dir, refresh
        self.sources = tuple(sources) if sources else None
        # mleft appends (plies left, labelled or not) to every batch. need_flag keeps only records with
        # those flag bits (8 means the record carries plies left), and min_mtime only shards written
        # since then, since older records carry no plies left.
        self.mleft, self.need_flag, self.min_mtime = mleft, int(need_flag), min_mtime
        self.paths = self._list()
        if not self.paths:
            raise SystemExit(f"no shards in {shard_dir}")
        self._reopen()
        self.last_refresh = __import__("time").time()
        self.batch, self.split, self.holdout, self.temp = batch, split, holdout, temp
        self.pick, self.block, self.pool_size = pick, block, pool_size
        # legal-move masks are computed on the filler thread, overlapped with the training step
        self.legal = legal
        self.pool, self.pos = None, 0
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.want_val = split == "val"
        # The validation set is saved to val_cache the first time and read back after, so a restart
        # on a grown corpus does not silently change the yardstick.
        self.fixed = None
        if self.want_val and val_cache and os.path.exists(val_cache):
            self.fixed = np.load(val_cache)
            print(f"[loader] validation read back from {os.path.basename(val_cache)}", flush=True)
        elif self.want_val:
            self.fixed = self._build_fixed(val_size)
            # an empty set is not saved, or it would be read back on every restart and score nothing
            if val_cache and len(self.fixed):
                tmp = val_cache + ".tmp.npy"
                np.save(tmp, self.fixed)
                os.replace(tmp, val_cache)
                print(f"[loader] validation saved to {os.path.basename(val_cache)}", flush=True)
        if self.want_val:
            print(f"[loader] validation frozen at {len(self.fixed):,} positions", flush=True)
        self.q = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.error = None       # what killed the filler, if anything did
        self.t = threading.Thread(target=self._fill, daemon=True)
        self.t.start()

    def _list(self):
        p = sorted(glob.glob(os.path.join(self.shard_dir, "*.bin")))
        want = self.sources or PACKED
        # the 40-byte action-value shards share the folder, so keep to this record's prefixes
        p = [x for x in p if os.path.basename(x).split("_")[0] in want]
        if self.min_mtime is not None:
            p = [x for x in p if _mtime(x) >= self.min_mtime]
        return p

    def _reopen(self):
        """Re-map every shard, picking up records appended since the last time."""
        self.paths = self._list()
        self.mm = [ShardReader(p) for p in self.paths]   # read on demand, see pack.ShardReader
        self.paths, self.mm = shard_cap(self.paths, self.mm)      # a frozen corpus under SHARD_CAP
        # where each shard's flagged records begin, which is a suffix since converters append
        self.starts = [self._flag_start(m) for m in self.mm] if self.need_flag else None
        self.counts = np.array([len(m) for m in self.mm], np.int64)
        self.total = int(self.counts.sum())
        self.cum = np.concatenate([[0], np.cumsum(self.counts)])

    def __len__(self):
        return self.total

    def describe(self):
        src = ",".join(self.sources) if self.sources else "all"
        return (f"{len(self.paths)} shards [{src}], {self.total:,} records, "
                f"{self.total * REC / 1e9:.1f} GB, split={self.split}")

    def _flag_start(self, m):
        """First index from which every record has need_flag, found by bisection on the suffix."""
        n = len(m)
        has = lambda i: bool(int(m[i]["flags"]) & self.need_flag)
        if n == 0 or not has(n - 1):
            return n
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if has(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _blocks(self, block=2048):
        """Pooled rows, read as a few sequential runs instead of hundreds of scattered ones.

        Random rows across hundreds of shards mean hundreds of page faults per batch, which stalls on
        disk. A batch instead reads a contiguous run from each of a handful of shards, and rows are
        subsampled from the pool so a batch still spans several shards.
        """
        first = self.starts if self.starts is not None else [0] * len(self.mm)
        live = [i for i in range(len(self.mm)) if len(self.mm[i]) > first[i]]
        if not live:
            return []
        k = min(self.pick, len(live))
        out = []
        # Training weighs each shard by the rows it can give, with replacement, so young shards are
        # not over-drawn. Validation keeps its old rule so a set drawn again is the same set.
        if self.want_val:
            pick = self.rng.choice(live, k, replace=False)
        else:
            w = np.array([max(len(self.mm[i]) - first[i], block) for i in live], np.float64)
            pick = self.rng.choice(live, k, replace=True, p=w / w.sum())
        for s in pick:
            m = self.mm[int(s)]
            lo = first[int(s)]
            span = min(block, len(m) - lo)
            j = lo + int(self.rng.integers(0, len(m) - lo - span + 1))
            rows = np.asarray(m[j:j + span])
            if self.need_flag:
                rows = rows[(rows["flags"] & self.need_flag) > 0]
            out.append(rows)
        return out

    def _refill(self):
        """Read one pool and serve many batches from it.

        A standard shuffle buffer. The pool is read once, shuffled and consumed a batch at a time, so
        one read serves over a hundred steps. It is safe because the pool comes from several shards.
        """
        out, got = [], 0
        for _ in range(8):
            for rows in self._blocks(block=self.block):
                rows = rows[~split_is_val(rows, self.holdout)]
                if len(rows):
                    out.append(rows)
                    got += len(rows)
            if got >= self.pool_size:
                break
        pool = np.concatenate(out) if out else np.zeros(0, DT)
        self.pool = pool[self.rng.permutation(len(pool))]
        self.pos = 0

    def _draw(self):
        """A batch from the right side of the split, drawn from a pool that mixes every source."""
        if self.want_val:
            return self._from_fixed()
        if self.pool is None or self.pos + self.batch > len(self.pool):
            self._refill()
        rows = self.pool[self.pos:self.pos + self.batch]
        self.pos += self.batch
        return rows

    def _build_fixed(self, n):
        """The validation set, built once and held in memory.

        Resampling would make the number incomparable between iterations as the corpus grows, and a
        one in 64 holdout reads 64 times what it keeps. Fixing it costs a few tens of megabytes.
        """
        out, got = [], 0
        for _ in range(400):
            for rows in self._blocks(block=8192):
                rows = rows[split_is_val(rows, self.holdout)]
                if len(rows):
                    out.append(rows)
                    got += len(rows)
            if got >= n:
                break
        fixed = np.concatenate(out) if out else np.zeros(0, DT)
        return fixed[:n]

    def _from_fixed(self):
        i = self.rng.choice(len(self.fixed), min(self.batch, len(self.fixed)), replace=False)
        return self.fixed[i]

    def _fill(self):
        import time as _t
        while not self.stop.is_set():
            try:
                if (self.refresh and not self.want_val
                        and _t.time() - self.last_refresh > self.refresh):
                    before = self.total
                    self._reopen()
                    self.last_refresh = _t.time()
                    if self.total > before:
                        print(f"[loader] corpus grew by {self.total - before:,} to "
                              f"{self.total:,}", flush=True)
                b = batch_tensors(self._draw(), N_MOVES, self.temp, self.legal)
                t = lambda a, d: torch.as_tensor(a).to(self.device, dtype=d, non_blocking=True)
                item = ((t(b["ids"], torch.long), t(b["cast"], torch.float),
                            t(b["hm"], torch.long), t(b["ep"], torch.long),
                            t(b["policy"], torch.float), t(b["value"], torch.float),
                            t(b["material"], torch.float), t(b["king"], torch.float),
                            t(b["pawns"], torch.float), t(b["mob"], torch.float),
                            t(b["bishop"], torch.float), t(b["aux"], torch.float))
                           + ((t(b["legal"], torch.bool),) if self.legal else ())
                           + ((t(b["mleft"], torch.float), t(b["mlv"], torch.float))
                              if self.mleft else ()))
                # a full queue waits with the batch in hand, so DATA_SEED replays the same stream
                while not self.stop.is_set():
                    try:
                        self.q.put(item, timeout=5)
                        break
                    except queue.Full:
                        continue
            except Exception as e:
                if self.stop.is_set():
                    return
                # kept for next() to raise, since an exception in a thread alone would leave the
                # consumer waiting forever
                self.error = e
                raise

    def next(self):
        """The next batch. If the filler thread has died, this raises what killed it instead of waiting."""
        while True:
            try:
                return self.q.get(timeout=self.POLL)
            except queue.Empty:
                if self.error is not None:
                    raise RuntimeError("the loader thread died and took the run with it")                         from self.error
                if self.stop.is_set():
                    raise RuntimeError("the loader was closed while a batch was still wanted")
                if not self.t.is_alive():
                    raise RuntimeError("the loader thread is gone and left no exception")

    def close(self):
        self.stop.set()


if __name__ == "__main__":
    import sys, time
    import paths
    d = sys.argv[1] if len(sys.argv) > 1 else paths.under("test")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = PackedLoader(d, batch=128, split="train", device=dev, seed=1)
    va = PackedLoader(d, batch=128, split="val", device=dev, seed=2)
    print(tr.describe())

    for _ in range(8):
        tr.next()
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.time(); N = 60
    for _ in range(N):
        b = tr.next()
    torch.cuda.synchronize() if dev == "cuda" else None
    ms = (time.time() - t0) / N * 1000
    print(f"\nbatch of 128: {ms:.2f} ms  ({128/ms*1000:,.0f} positions/s)")
    print(f"the JSON loader was 16.90 ms, so {16.9/ms:.1f}x faster")

    ids, cast, pol, val = b[0], b[1], b[4], b[5]
    print(f"\nshapes  ids {tuple(ids.shape)}  policy {tuple(pol.shape)}  value {tuple(val.shape)}")
    print(f"policy rows sum {float(pol.sum(1).min()):.4f} to {float(pol.sum(1).max()):.4f}")
    print(f"ids range {int(ids.min())} to {int(ids.max())}")

    # the two splits must not overlap

    def keyset(loader, n):
        s = set()
        for _ in range(n):
            rows = loader._draw()
            for r in rows:
                s.add(r["ids"].tobytes() + r["cast"].tobytes())
        return s
    a, c = keyset(tr, 12), keyset(va, 12)
    print(f"\ntrain positions sampled {len(a):,}, val {len(c):,}, overlap {len(a & c)}")
    assert not (a & c), "a position appears in both splits"
    print("no position appears on both sides of the split")
    tr.close(); va.close()
