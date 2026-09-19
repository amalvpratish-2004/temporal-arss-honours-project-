"""
smoke_no_torch.py — dependency-free reproduction of the B1/B2/B4 fix logic,
so the fix can be verified right here without installing torch/torch_geometric.

This mirrors arss/sampler.py exactly (same code, just with the torch Data
object swapped for a plain dict) -- it is NOT a substitute for
tests/test_invariants.py, which exercises the real package. Run this one
here-and-now as a sanity check; run the pytest suite in your actual training
environment (where torch + torch_geometric are installed) before you start
Phase 1.

Run: python3 smoke_no_torch.py
"""

import math


# ---------------------------------------------------------------------------
# B1 — negative sampler must never emit an id outside [item_lo, item_hi]
# ---------------------------------------------------------------------------
def negative_sample(rng, src, adj, item_lo, item_hi, max_retries=5):
    out = []
    for s in src:
        d = rng_int(rng, item_lo, item_hi)
        tries = 0
        while d in adj.get(s, ()) and tries < max_retries:
            d = rng_int(rng, item_lo, item_hi)
            tries += 1
        out.append(d)
    return out


def rng_int(rng, lo, hi):
    return lo + int(rng() * (hi - lo + 1))


def test_b1_no_phantom_nodes():
    adj = {
        0: {100, 101}, 1: {100, 102, 103}, 2: {101},
        100: {0, 1}, 101: {0, 2}, 102: {1}, 103: {1},
    }
    item_lo, item_hi = 100, 103
    import random
    rnd = random.Random(0)
    negs = negative_sample(rnd.random, [0, 1, 2] * 50, adj, item_lo, item_hi)
    assert min(negs) >= item_lo, negs
    assert max(negs) <= item_hi, negs
    print(f"  [B1] negatives stay within [{item_lo},{item_hi}]  (range={min(negs)}-{max(negs)})  OK")


# ---------------------------------------------------------------------------
# B2 — roots must land at index 0/1 even when the target edge is a real
# training edge (i.e. v is already in u's own neighbour list)
# ---------------------------------------------------------------------------
def build_all_nodes_BROKEN(u, v, u_neighbors, v_neighbors):
    """The ORIGINAL buggy construction, reproduced verbatim for comparison."""
    u_set, v_set = set(u_neighbors), set(v_neighbors)
    all_nodes = [u, v] + list(u_set | v_set)
    n2l = {n: i for i, n in enumerate(all_nodes)}
    return n2l


def build_all_nodes_FIXED(u, v, u_neighbors, v_neighbors):
    """The FIXED construction from arss/sampler.py."""
    u_set, v_set = set(u_neighbors), set(v_neighbors)
    others = [n for n in (u_set | v_set) if n != u and n != v]
    all_nodes = [u, v] + others
    n2l = {n: i for i, n in enumerate(all_nodes)}
    assert n2l[u] == 0 and n2l[v] == 1
    return n2l


def test_b2_root_indexing():
    # Positive edge: v is in u's own neighbour list (the exact bug scenario)
    u, v = 0, 100
    u_nb = [100, 101]   # v=100 is present -- this is what breaks the original code
    v_nb = [0, 2]        # u=0 is present

    n2l_broken = build_all_nodes_BROKEN(u, v, u_nb, v_nb)
    broken_u_idx, broken_v_idx = n2l_broken[u], n2l_broken[v]
    print(f"  [B2] BROKEN: n2l[u]={broken_u_idx} n2l[v]={broken_v_idx}  "
          f"(should be 0,1 -- {'BUG REPRODUCED' if (broken_u_idx, broken_v_idx) != (0, 1) else 'no bug here?'})")
    assert (broken_u_idx, broken_v_idx) != (0, 1), "expected to reproduce the original bug"

    n2l_fixed = build_all_nodes_FIXED(u, v, u_nb, v_nb)
    fixed_u_idx, fixed_v_idx = n2l_fixed[u], n2l_fixed[v]
    assert (fixed_u_idx, fixed_v_idx) == (0, 1)
    print(f"  [B2] FIXED : n2l[u]={fixed_u_idx} n2l[v]={fixed_v_idx}  OK")


# ---------------------------------------------------------------------------
# B4 — induced subgraph must contain edges between neighbours, not just
# root<->neighbour ("double star") edges
# ---------------------------------------------------------------------------
def build_induced_edges(all_nodes, n2l, adj):
    S = set(all_nodes)
    edges = set()
    for a in all_nodes:
        for b in adj.get(a, ()) & S:
            pair = tuple(sorted((n2l[a], n2l[b])))
            edges.add(pair)
    return edges


def test_b4_induced_edges():
    # 10 and 11 are both neighbours of the query pair, AND connected to
    # each other -- a homogeneous-graph situation the original "double
    # star" builder could never represent.
    adj = {0: {10, 11}, 10: {0, 11}, 11: {0, 10}}
    u, v = 0, 10
    others = [11]
    all_nodes = [u, v] + others
    n2l = {n: i for i, n in enumerate(all_nodes)}

    edges = build_induced_edges(all_nodes, n2l, adj)
    idx_10, idx_11 = n2l[10], n2l[11]
    assert tuple(sorted((idx_10, idx_11))) in edges, edges
    print(f"  [B4] induced edge between non-root neighbours present: {edges}  OK")


if __name__ == "__main__":
    print("Running dependency-free smoke tests for B1 / B2 / B4 ...\n")
    test_b1_no_phantom_nodes()
    test_b2_root_indexing()
    test_b4_induced_edges()
    print("\nAll smoke tests passed. These mirror arss/sampler.py's actual logic;\n"
          "run `pytest tests/test_invariants.py -v` against the real package once\n"
          "torch + torch_geometric are installed in your training environment.")
