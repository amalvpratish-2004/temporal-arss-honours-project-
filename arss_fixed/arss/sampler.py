"""
sampler.py — neighbour sampling, temporal reweighting, anonymous subgraph
construction, and negative sampling.

FIXES APPLIED vs the original notebook (cells 3, 4, 6, 10):

  B2  Isolated roots / label leakage.
      Original `build_anonymous_subgraph` did:
          all_nodes = [u, v] + list(u_set | v_set)
          n2l = {n: i for i, n in enumerate(all_nodes)}
      For a POSITIVE edge (u, v), v is in u's own neighbour list (it's a
      real training edge), so v appears twice in all_nodes — once at index
      1, once inside the neighbour block — and the dict comprehension lets
      the LATER occurrence win, so n2l[u] and n2l[v] silently point at the
      duplicate positions, not indices 0/1. forward() always reads indices
      0 and 1, which for positives are therefore isolated placeholder nodes
      with a constant, edge-free embedding — while for negatives (where
      u and v never appear in each other's neighbour lists) indices 0/1
      are properly connected. The model was separating classes by
      detecting this degenerate representation, not by learning structure.

      Fix, two parts:
        1. Build all_nodes as [u, v] + <everyone else, u/v excluded>, so
           n2l[u] == 0 and n2l[v] == 1 always hold (asserted).
        2. Exclude v from u's candidate pool (and u from v's) at sampling
           time, so the target edge itself is never visible to the model
           that is trying to predict it (SEAL/NCN-style target exclusion).

  B4  No induced edges between neighbours (rooted-subtree topology).
      Original builder only ever wired root <-> neighbour edges — a double
      star. This defeats the entire premise of ARSS (distinguishing nodes
      whose subtrees are identical but whose subgraphs differ), since a
      double star has no structure beyond a subtree in the first place.

      Fix: after selecting the node set S, add every training edge with
      both endpoints in S ("induce" the subgraph), not just root-neighbour
      edges.

  S1  Hop-2 candidates always scored +inf (t_last never exists for them,
      since by construction they never directly interacted with v_i), so
      w_ik was silently discarded and hop-2 ranking degenerated to
      arbitrary set-iteration order.
      Fix: give unseen pairs a finite worst-case age (T_MIN, i.e. "as old
      as the start of training") instead of None, so w_ik still breaks
      ties via the temporal_score formula.

  S2  exp(lam * dt / T_SCALE) overflows on wider time spans / larger lam.
      Fix: sort in log space (strictly monotone transform of the same
      ranking, numerically safe): score = log(w_base) + lam * dt / T_SCALE.

  S5  Negative destinations could coincide with a real historical edge.
      Fix: negative_sample takes an `adj` lookup and rejects+resamples
      any draw that is already a neighbour of its source (bounded retries,
      falls back to accepting on the rare case retries are exhausted).

B1's range fix (data.py) feeds directly into negative_sample here: it now
takes item_lo/item_hi explicitly instead of a hardcoded (offset, n_items)
pair, so it cannot regress back to sampling phantom node ids.
"""

import math
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

NUM_ROLES = 5


def make_temporal_score_fn(w_cache, compute_w_ik, t_last, t_min, t_scale):
    """Returns temporal_score(v_i, k, t_query, lam) -> float (log-space, S2)."""

    def temporal_score(v_i, k, t_query, lam):
        w_base = w_cache.get((v_i, k))
        if w_base is None:
            w_base = compute_w_ik(v_i, k)
        # S1 fix: unseen pair gets a finite worst-case age (t_min), not inf.
        t_last_ = t_last[v_i].get(k, t_min)
        dt = max(t_query - t_last_, 0.0)
        # S2 fix: log-space, strictly monotone with the original w_base * exp(...).
        return math.log(max(w_base, 1e-12)) + lam * dt / t_scale

    return temporal_score


def make_sampler(adj, temporal_score):
    """Returns sample_subgraph_nodes_temporal(v_i, K, t_query, lam, exclude=None, max_hops=2)."""

    def sample_subgraph_nodes_temporal(v_i, K, t_query, lam, exclude=None, max_hops=2):
        exclude = exclude if exclude is not None else set()
        selected, selected_set = [], set()
        visited = {v_i} | exclude
        frontier = (adj.get(v_i, set()) - exclude).copy()

        for _ in range(1, max_hops + 1):
            candidates = frontier - selected_set - {v_i} - exclude
            if not candidates:
                break
            remaining = K - len(selected)
            if remaining <= 0:
                break

            ranked = sorted(candidates, key=lambda k: temporal_score(v_i, k, t_query, lam))
            batch = ranked if len(candidates) <= remaining else ranked[:remaining]

            selected.extend(batch)
            selected_set.update(batch)
            visited.update(batch)

            if len(selected) >= K:
                break

            frontier = set()
            for nd in selected_set:
                frontier |= adj.get(nd, set())
            frontier -= visited

        return selected[:K]

    return sample_subgraph_nodes_temporal


def build_anonymous_subgraph(u, v, u_neighbors, v_neighbors, adj):
    """B2 + B4 fixed subgraph builder.

    B2: roots u, v are guaranteed indices 0, 1.
    B4: edges are induced over the full sampled node set S, not just
        root<->neighbour ("double star") edges.
    """
    u_set = set(u_neighbors)
    v_set = set(v_neighbors)
    common = u_set & v_set

    # B2 fix: roots first, exclude them from the "everyone else" block so a
    # duplicate occurrence can never overwrite index 0 / 1.
    others = [n for n in (u_set | v_set) if n != u and n != v]
    all_nodes = [u, v] + others
    n2l = {n: i for i, n in enumerate(all_nodes)}
    assert n2l[u] == 0 and n2l[v] == 1, "B2 invariant violated: roots must be indices 0,1"
    N = len(all_nodes)

    roles = torch.zeros(N, dtype=torch.long)
    roles[0] = 0
    roles[1] = 1
    for nd in others:
        roles[n2l[nd]] = 2 if nd in common else (3 if nd in u_set else 4)

    x = F.one_hot(roles, num_classes=NUM_ROLES).float()

    # B4 fix: induce all training edges with both endpoints in S, not just
    # root-neighbour edges. |S| <= 2K+2 so this is cheap.
    S = set(all_nodes)
    esrc, edst = [], []
    seen_pairs = set()
    for a in all_nodes:
        a_idx = n2l[a]
        for b in (adj.get(a, ()) & S):
            b_idx = n2l[b]
            pair = (a_idx, b_idx) if a_idx < b_idx else (b_idx, a_idx)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            esrc += [a_idx, b_idx]
            edst += [b_idx, a_idx]

    if esrc:
        edge_index = torch.tensor([esrc, edst], dtype=torch.long)
    else:
        # Only possible if u and v have zero sampled neighbours at all — keep
        # a harmless self-loop-free placeholder so SAGEConv doesn't choke on
        # an edgeless graph. (This should be rare; on Wikipedia u, v are
        # queried with K=10 so this fires only for near-isolated nodes.)
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, common_mask=(roles == 2))


def make_negative_sampler(rng, adj, item_lo, item_hi, max_retries=5):
    """B1 fix (correct range) + S5 fix (reject destinations that are real
    historical neighbours of the source)."""

    def negative_sample(src):
        out = rng.integers(item_lo, item_hi + 1, size=len(src))
        for i, (s, d) in enumerate(zip(src, out)):
            tries = 0
            while d in adj.get(int(s), ()) and tries < max_retries:
                d = rng.integers(item_lo, item_hi + 1)
                tries += 1
            out[i] = d
        return out

    return negative_sample


def make_batch_builder(sample_fn, adj):
    def build_batch_subgraphs_temporal(src_batch, dst_batch, t_batch, K, lam, max_hops=2):
        subgraphs = []
        for u, v, t_q in zip(src_batch.tolist(), dst_batch.tolist(), t_batch.tolist()):
            # B2 fix, part 2: exclude the target edge from each other's
            # sampling pool so it can't leak into its own subgraph.
            u_nb = sample_fn(u, K, t_q, lam, exclude={v}, max_hops=max_hops)
            v_nb = sample_fn(v, K, t_q, lam, exclude={u}, max_hops=max_hops)
            subgraphs.append(build_anonymous_subgraph(u, v, u_nb, v_nb, adj))
        return subgraphs

    return build_batch_subgraphs_temporal
