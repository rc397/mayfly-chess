"""Classify the search's collisions and count transpositions in the tree.

A collision is a descent that reaches a node already pending in the current batch. Two kinds are
possible in principle, two descents picking the same child (which virtual loss should prevent) and the
same position reached by a different move order. Every node has one parent, so a transposition builds
a different node with the same key, which the eval cache serves, and never shows as a collision. This
reports how concentrated the collisions are, and how many tree nodes duplicate a position elsewhere in
the tree (their network cost is saved by the cache, but their statistics are split).

    python chess/classify_collisions.py [POSITIONS] [VISITS]
"""
import os, sys, glob, random, collections
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

E = os.environ.get
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
VISITS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
PGN_DIR = E("PGN", os.path.join(HERE, "twic"))


def sample(n, seed=5):
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
                for m in mv[:rng.randrange(14, 45)]:
                    b.push(m)
                if not b.is_game_over() and b.legal_moves.count() > 2:
                    out.append(b.fen())
        if len(out) >= n:
            break
    return out[:n]


if __name__ == "__main__":
    import torch
    from mcts import MCTS
    from play_web import load_version

    class Watched(MCTS):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.hit_node = collections.Counter()   # which node object absorbed the collision
            self.hit_key = collections.Counter()    # which POSITION it was
            self.hit_depth = collections.Counter()

        def _unwind(self, path):
            # _unwind is called at the moment of the collision with the real path, so the pending
            # node is the child the last step chose
            if path:
                # a path step is (node, edge, provisional value)

                parent, act = path[-1][0], path[-1][1]
                hit = parent.kids[act] if parent.kids else None
            else:
                hit = None
            if hit is not None:
                self.hit_node[id(hit)] += 1
                self.hit_key[hit.key] += 1
                self.hit_depth[len(path)] += 1
            return super()._unwind(path)

    def walk(root):
        """Every node in the tree, so duplicates by position can be counted."""
        seen, stack = [], [root]
        while stack:
            n = stack.pop()
            if n is None:
                continue
            seen.append(n)
            if n.kids:
                stack.extend(n.kids)
        return seen

    fens = sample(N_POS)
    model, label, _ = load_version(E("VERSION", "mayfly_v1.pt"))
    dev = str(next(model.parameters()).device)
    print(f"{len(fens)} positions, {VISITS} visits, {label}", flush=True)

    tot = collections.Counter()
    node_hits, key_hits, depth_hits = collections.Counter(), collections.Counter(), collections.Counter()
    dup_nodes = dup_keys = tree_nodes = 0
    biggest_dup = 0
    for fen in fens:
        t = Watched(model, dev, batch=48, cache_size=20000)
        t.search(chess.Board(fen), visits=VISITS, reuse=False)
        tot["collisions"] += t.stats["collisions"]
        tot["evals"] += t.stats["evals"]
        tot["cache_hits"] += t.stats["cache_hits"]
        node_hits.update(t.hit_node)
        key_hits.update(t.hit_key)
        depth_hits.update(t.hit_depth)
        nodes = walk(t.root)
        tree_nodes += len(nodes)
        by_key = collections.Counter(n.key for n in nodes)
        dup_keys += sum(1 for k, c in by_key.items() if c > 1)
        dup_nodes += sum(c - 1 for c in by_key.values() if c > 1)
        biggest_dup = max(biggest_dup, max(by_key.values()))

    print("")
    print(f"  {tot['collisions']:,} collisions, {tot['evals']:,} evals, "
          f"{tot['cache_hits']:,} eval-cache hits")
    print(f"  they landed on {len(node_hits):,} distinct NODES and "
          f"{len(key_hits):,} distinct POSITIONS")
    if node_hits:
        top = node_hits.most_common(5)
        share = 100.0 * sum(c for _, c in top) / sum(node_hits.values())
        print(f"  the five most collided nodes absorbed {share:.1f}% of all collisions")
        print(f"  most collisions on one node: {top[0][1]:,}")
    if depth_hits:
        print("  depth at which the descent hit a pending node:")
        s = sum(depth_hits.values())
        for d in sorted(depth_hits)[:8]:
            print(f"    ply {d:>2}  {depth_hits[d]:>7,}  {100*depth_hits[d]/s:>5.1f}%")

    print("")
    print(f"  TRANSPOSITIONS IN THE TREE, which the collision counter cannot see:")
    print(f"    {tree_nodes:,} nodes built, {dup_nodes:,} of them "
          f"({100*dup_nodes/max(tree_nodes,1):.1f}%) duplicate a position already in the tree")
    print(f"    {dup_keys:,} positions appear more than once, the worst {biggest_dup} times")
    print(f"    that {100*dup_nodes/max(tree_nodes,1):.1f}% is the share of the tree whose "
          f"STATISTICS are split; their network cost is already saved by the eval cache")
