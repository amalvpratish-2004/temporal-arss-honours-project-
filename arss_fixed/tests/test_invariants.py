"""
tests/test_invariants.py

These are the Phase 0 deliverable the handoff doc calls for: run them
BEFORE and AFTER each fix so you have a red->green moment per bug, and
keep them in CI (or just `pytest tests/` before every run) so none of
these five bugs can silently come back.

Run: pytest tests/ -v
(needs torch + torch_geometric installed; the pure-adjacency tests --
test_no_phantom_nodes, test_negative_sampler_range -- will still run
without them if you comment out the torch-dependent imports.)
"""

import math
import pytest
import numpy as np

torch = pytest.importorskip("torch")

from arss.sampler import (
    build_anonymous_subgraph,
    make_sampler,
    make_negative_sampler,
    make_temporal_score_fn,
)


# ---------------------------------------------------------------------------
# A small synthetic bipartite graph, standing in for the real Wikipedia
# adjacency, so these tests run in milliseconds and don't need the dataset.
#
#   users: 0, 1, 2      pages: 100, 101, 102, 103
#   0 -- 100, 101
#   1 -- 100, 102, 103
#   2 -- 101
# ---------------------------------------------------------------------------
def make_toy_graph():
    adj = {
        0: {100, 101}, 1: {100, 102, 103}, 2: {101},
        100: {0, 1}, 101: {0, 2}, 102: {1}, 103: {1},
    }
    item_lo, item_hi = 100, 103
    n_users = 3
    return adj, item_lo, item_hi, n_users


# ---------------------------------- B1 ------------------------------------
def test_no_phantom_nodes():
    """Every id the negative sampler can produce must exist in adjacency
    (have at least the possibility of being a real node). This is the
    direct regression test for the double-offset bug: before the fix, the
    negative range extended far past the real item ids."""
    adj, item_lo, item_hi, n_users = make_toy_graph()
    rng = np.random.default_rng(0)
    neg_sampler = make_negative_sampler(rng, adj, item_lo, item_hi)

    src = np.array([0, 1, 2] * 20)
    negs = neg_sampler(src)

    assert negs.min() >= item_lo
    assert negs.max() <= item_hi
    # every possible value in range must be a real page id present in the
    # toy graph's key set (mirrors "set(all_dst) <= set(adj.keys())")
    real_pages = {k for k in adj if item_lo <= k <= item_hi}
    assert real_pages == set(range(item_lo, item_hi + 1))


# ---------------------------------- B2 ------------------------------------
def test_roots_not_isolated():
    adj, item_lo, item_hi, n_users = make_toy_graph()
    u, v = 0, 100  # a real positive edge in the toy graph
    u_nb = [n for n in adj[u] if n != v]        # exclusion applied by caller normally
    v_nb = [n for n in adj[v] if n != u]
    sg = build_anonymous_subgraph(u, v, u_nb, v_nb, adj)

    deg = {}
    src, dst = sg.edge_index
    for s in src.tolist():
        deg[s] = deg.get(s, 0) + 1

    assert deg.get(0, 0) > 0, "root u (index 0) must not be isolated"
    assert deg.get(1, 0) > 0, "root v (index 1) must not be isolated"


def test_root_indices_fixed():
    """n2l[u] == 0 and n2l[v] == 1 must hold even when v is in u's own
    neighbour list (the exact scenario that broke the original code)."""
    adj, item_lo, item_hi, n_users = make_toy_graph()
    u, v = 0, 100
    # deliberately DON'T exclude v from u_nb here -- this is the failure
    # mode the assertion inside build_anonymous_subgraph must catch/handle
    u_nb = list(adj[u])   # includes v = 100
    v_nb = list(adj[v])   # includes u = 0
    sg = build_anonymous_subgraph(u, v, u_nb, v_nb, adj)
    # the assert inside build_anonymous_subgraph already enforces this;
    # reaching this line without an AssertionError is the pass condition
    assert sg.x[0].argmax().item() == 0   # role 0 = source
    assert sg.x[1].argmax().item() == 1   # role 1 = target


def test_target_edge_excluded():
    """The sampler must never return the query's own destination as a
    'neighbour' of the source (or vice versa) when exclude= is passed."""
    adj, item_lo, item_hi, n_users = make_toy_graph()
    t_last = {n: {} for n in adj}
    score_fn = make_temporal_score_fn({}, lambda a, b: 1.0, t_last, t_min=0.0, t_scale=1.0)
    sample_fn = make_sampler(adj, score_fn)

    u, v = 0, 100
    u_nb = sample_fn(u, K=10, t_query=5.0, lam=1.0, exclude={v}, max_hops=2)
    assert v not in u_nb


# ---------------------------------- B4 ------------------------------------
def test_subgraph_has_internal_edges_on_homogeneous_graph():
    """On a graph where neighbours of u and v are themselves connected, the
    induced subgraph must contain edges beyond the root-neighbour double
    star. Toy graph: user 1's neighbours 102, 103 don't connect to each
    other, so extend the toy graph slightly for this one test."""
    adj = {
        0: {10, 11}, 10: {0, 11}, 11: {0, 10},  # 10-11 edge exists (homogeneous augment)
    }
    u, v = 0, 10
    u_nb = [11]
    v_nb = [11]
    sg = build_anonymous_subgraph(u, v, u_nb, v_nb, adj)

    # all_nodes = [0, 10, 11]; edge 10-11 should appear even though neither
    # endpoint is a root -- that's the induced edge B4 adds.
    src, dst = sg.edge_index.tolist()
    pairs = set(zip(src, dst))
    idx_10, idx_11 = 1, 2  # u=0->idx0, v=10->idx1, 11->idx2
    assert (idx_10, idx_11) in pairs or (idx_11, idx_10) in pairs


# ---------------------------------- S3 ------------------------------------
def test_eval_negatives_are_fixed():
    from arss.eval import get_fixed_negatives
    src_arr = np.arange(50)
    neg_a = get_fixed_negatives("val", src_arr, item_lo=100, item_hi=200, seed=999)
    neg_b = get_fixed_negatives("val", src_arr, item_lo=100, item_hi=200, seed=999)
    assert np.array_equal(neg_a, neg_b), "same split+seed must return identical negatives"
