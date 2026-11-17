"""Find out why the search's batches of 48 run part empty.

The collection loop in search stops when the batch is full, when the visit target is reached, or after
4 * batch tries (the saturation cap of 192). A try can also end without filling a slot, through a
collision, a terminal, a repetition, a tablebase hit or a cache hit. _descend is called once per try,
so counting its calls between two _evaluate calls gives the tries for each batch without touching the
search code. The second half times the forward pass against batch size, to see whether fuller batches
would be nearly free. GPU timings are inflated if a trainer shares the card, but the fill counts are not.

    python chess/batch_why.py [POSITIONS] [VISITS]
"""
import os, sys, glob, random, time, collections
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

E = os.environ.get
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
VISITS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PGN_DIR = E("PGN", os.path.join(HERE, "twic"))


def sample(n, seed=3):
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
    rng.shuffle(files)
    out = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            while len(out) < n:
                g = chess.pgn.read_game(fh)
                if g is None:
                    break
                mv = list(g.mainline_moves())
                if len(mv) < 30:
                    continue
                b = g.board()
                for m in mv[:rng.randrange(14, 40)]:
                    b.push(m)
                if not b.is_game_over():
                    out.append(b.fen())
        if len(out) >= n:
            break
    return out[:n]


if __name__ == "__main__":
    import numpy as np
    import torch
    from mcts import MCTS
    from play_web import load_version
    from chess_net import board_to_ids, clock_and_ep

    clock = time.perf_counter
    fens = sample(N_POS)
    model, label, _ = load_version(E("VERSION", "mayfly_v1.pt"))
    dev = str(next(model.parameters()).device)
    print(f"{len(fens)} positions, {VISITS} visits, batch 48, {label} on {dev}", flush=True)

    class Why(MCTS):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.tries = 0
            self.miss = 0
            self.rows = []          # (leaves, tries, collisions, cache, term, tb, rep)
            self.mark = None

        def _descend(self, root):
            self.tries += 1
            got = super()._descend(root)
            if got is None:
                self.miss += 1
            return got

        def _snap(self):
            s = self.stats
            return (s.get("cache_hits", 0), s.get("tablebase", 0), s.get("repetitions", 0))

        def _evaluate(self, nodes):
            now = self._snap()
            base = self.mark or (0, 0, 0)
            self.rows.append((len(nodes), self.tries, self.miss,
                              now[0] - base[0], now[1] - base[1], now[2] - base[2]))
            self.mark = now
            self.tries = 0
            self.miss = 0
            return super()._evaluate(nodes)

    rows = []
    for fen in fens:
        t = Why(model, dev, batch=48, cache_size=20000)
        t.search(chess.Board(fen), visits=VISITS, reuse=False)
        rows += t.rows

    rows = [r for r in rows if r[1] > 0]           # drop the root's own single evaluation
    n = len(rows)
    leaves = sum(r[0] for r in rows)
    tries = sum(r[1] for r in rows)
    coll = sum(r[2] for r in rows)
    cache = sum(r[3] for r in rows)
    tb = sum(r[4] for r in rows)
    rep = sum(r[5] for r in rows)
    other = max(tries - leaves - coll - cache - tb - rep, 0)

    print("")
    print(f"  {n:,} collection loops, {leaves:,} leaves from {tries:,} descents, "
          f"mean fill {leaves/n:.1f} of 48")
    print("  where a descent went:")
    for name, c, note in (
            ("filled a slot", leaves, "a real unexpanded position for the card"),
            ("collision", coll, "arrived at a node already in this batch, unwound"),
            ("eval-cache hit", cache, "position already evaluated, expanded from the cache"),
            ("tablebase", tb, "exact result under 7 pieces"),
            ("repetition", rep, "scored a draw by the history rule"),
            ("terminal or other", other, "checkmate, stalemate, the 75-move rule")):
        print(f"    {name:<19} {c:>8,}  {100*c/max(tries,1):>5.1f}%  {note}")

    capped = sum(1 for r in rows if r[1] >= 4 * 48)
    full = sum(1 for r in rows if r[0] >= 48)
    print("")
    print("  why each collection loop stopped:")
    print(f"    filled to 48         {full:>6,}  {100*full/n:>5.1f}%")
    print(f"    hit the 192-try cap  {capped:>6,}  {100*capped/n:>5.1f}%  "
          f"<- the cap, not the batch, is deciding")
    print(f"    reached the target   {n-full-capped:>6,}  {100*(n-full-capped)/n:>5.1f}%  "
          f"the tail of a search")

    print("")
    print(f"  a {VISITS}-visit search costs {n/len(fens):.1f} forward passes where "
          f"{VISITS/48:.1f} would do, so {100*(n/len(fens))/(VISITS/48)-100:.0f}% more than the "
          f"visit count needs")

    # does a bigger batch cost more? If not, fuller batches are nearly free
    boards = []
    while len(boards) < 96:
        boards += [chess.Board(f) for f in fens]
    REP = 40

    def bench(bs):
        bb = boards[:bs]
        enc = [board_to_ids(b) for b in bb]
        ce = [clock_and_ep(b, e[2]) for b, e in zip(bb, enc)]
        ids = torch.tensor(np.array([e[0] for e in enc]), dtype=torch.long, device=dev)
        cast = torch.tensor(np.array([e[1] for e in enc], np.float32), device=dev)
        hm = torch.tensor([c[0] for c in ce], dtype=torch.long, device=dev)
        ep = torch.tensor([c[1] for c in ce], dtype=torch.long, device=dev)

        def go():
            with torch.autocast("cuda", torch.bfloat16, enabled=(dev != "cpu")):
                return model(ids, cast, hm, ep)
        with torch.no_grad():
            go()
            torch.cuda.synchronize()
            t0 = clock()
            for _ in range(REP):
                go()
            torch.cuda.synchronize()
        return (clock() - t0) / REP * 1000.0

    print("")
    print("  forward pass against batch size, on the contended card")
    print(f"    {'batch':>6} {'ms':>8} {'ms a position':>15}")
    base = None
    for bs in (8, 16, 27, 32, 48, 64, 96):
        ms = bench(bs)
        if bs == 27:
            base = ms
        print(f"    {bs:>6} {ms:>8.2f} {ms/bs:>15.3f}")
    if base:
        at48 = bench(48)
        print("")
        print(f"    a batch of 27 costs {base:.2f} ms and a batch of 48 costs {at48:.2f} ms, "
              f"{at48/base:.2f}x the time for {48/27:.2f}x the work")
