"""
data.py — Wikipedia JODIE loading, chronological split, adjacency + w_ik cache.

FIXES APPLIED vs the original notebook (cell 2 / cell 4):

  B1  Double node-ID offset.
      PyG's JODIEDataset.process() ALREADY offsets destination ids so that
      src and dst occupy disjoint integer ranges:
          dst += int(src.max()) + 1
      The original notebook computed N_ITEMS from the *already offset* dst
      (giving 9227 instead of 1000) and then added N_USERS to dst a SECOND
      time, and sampled negatives uniformly over the resulting 9227-wide
      range — of which only 1000 ids were real pages. ~89% of negatives
      were phantom nodes with no adjacency entry.

      Fix: read the offset PyG already applied, do not re-offset, and size
      N_ITEMS from the true (min, max) span of dst.

  S2  T_SCALE label was wrong ("median inter-event gap" in a comment, but
      the code computed full training span). Comment fixed to match code
      (see sampler.py for the actual log-space fix that matters).

Everything else (split ratios, feature normalization) is unchanged from
the original — those parts were correct.
"""

import math
import time
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.datasets import JODIEDataset
from torch_geometric.data import TemporalData


def load_and_split(root="./data", name="Wikipedia", train_frac=0.70, val_frac=0.85):
    dataset = JODIEDataset(root=root, name=name)
    data = dataset[0]

    edge_feat_dim = data.msg.shape[1]
    print(f"Edges      : {data.num_events:,}")
    print(f"Feat dim   : {edge_feat_dim}")
    print(f"Time range : [{data.t.min().item():.0f}, {data.t.max().item():.0f}]")

    # ---- Chronological split -------------------------------------------------
    n = data.num_events
    val_t = data.t[int(n * train_frac)].item()
    test_t = data.t[int(n * val_frac)].item()

    def split(mask):
        return TemporalData(
            src=data.src[mask], dst=data.dst[mask],
            t=data.t[mask], msg=data.msg[mask],
        )

    train_data = split(data.t < val_t)
    val_data = split((data.t >= val_t) & (data.t < test_t))
    test_data = split(data.t >= test_t)

    # ---- Normalize edge features (fit on train only) --------------------------
    mean = train_data.msg.mean(0, keepdim=True)
    std = train_data.msg.std(0, keepdim=True).clamp(min=1e-6)
    for d in [train_data, val_data, test_data]:
        d.msg = (d.msg - mean) / std

    # ---- B1 FIX: node counts, no re-offsetting --------------------------------
    # PyG already made dst globally unique via `dst += src.max() + 1` in
    # JODIEDataset.process(). Do NOT offset again.
    n_users = int(data.src.max()) + 1                      # 8227
    item_lo = int(data.dst.min())                           # 8227 (PyG's offset)
    item_hi = int(data.dst.max())                            # 9226
    n_items = item_hi - item_lo + 1                          # 1000, not 9227

    assert item_lo == n_users, (
        f"Expected PyG's dst offset ({item_lo}) to equal n_users ({n_users}); "
        f"if this fails, PyG's internal offset convention changed — inspect "
        f"torch_geometric.datasets.jodie before proceeding."
    )

    num_nodes = n_users + n_items
    print(f"\nID ranges  : src 0-{data.src.max().item()},  "
          f"dst {item_lo}-{item_hi}  (n_items={n_items}, NOT {item_hi - n_users + n_users})")
    print(f"Nodes      : {num_nodes:,}  ({n_users} users + {n_items} pages)")

    # ---- Flat numpy arrays for the sampler ------------------------------------
    arrs = {
        "train_src": train_data.src.numpy(), "train_dst": train_data.dst.numpy(),
        "train_t": train_data.t.numpy(),
        "val_src": val_data.src.numpy(), "val_dst": val_data.dst.numpy(),
        "val_t": val_data.t.numpy(),
        "test_src": test_data.src.numpy(), "test_dst": test_data.dst.numpy(),
        "test_t": test_data.t.numpy(),
    }
    print(f"Train      : {len(arrs['train_src']):,}")
    print(f"Val        : {len(arrs['val_src']):,}")
    print(f"Test       : {len(arrs['test_src']):,}")

    t_min = float(train_data.t.min())
    t_max_train = float(train_data.t.max())
    t_scale = max(t_max_train - t_min, 1.0)   # full training span, see sampler.py S2 fix
    print(f"T_SCALE    : {t_scale:.1f}s (training time span)")

    meta = dict(
        n_users=n_users, item_lo=item_lo, item_hi=item_hi, n_items=n_items,
        num_nodes=num_nodes, t_min=t_min, t_max_train=t_max_train, t_scale=t_scale,
        edge_feat_dim=edge_feat_dim,
    )
    return arrs, meta


def build_adjacency_and_weights(train_src, train_dst):
    """Static adjacency + w_ik cache (Eq. 2-4). Unchanged from original — this
    part had no bug, it just inherited wrong node ranges from B1."""
    adj = defaultdict(set)
    for u, v in zip(train_src.tolist(), train_dst.tolist()):
        adj[u].add(v)
        adj[v].add(u)

    degree = {node: len(nb) for node, nb in adj.items()}
    d_max = max(degree.values()) if degree else 1

    def psi(x):
        return math.log(1.0 + x)

    psi_dmax = psi(d_max)

    def compute_w_ik(i, k):
        d_i = degree.get(i, 0)
        d_k = degree.get(k, 0)
        pi_ik = (psi(d_i) + psi(d_k)) / (2.0 * psi_dmax)
        common = adj.get(i, set()) & adj.get(k, set())
        if common:
            prod = math.prod(psi(degree.get(u, 1)) / psi_dmax for u in common)
            eta_ik = 1.0 - prod
        else:
            eta_ik = 0.0
        return (2.0 * pi_ik) / (1.0 + eta_ik)

    t0 = time.time()
    w_cache = {}
    for node, neighbours in adj.items():
        for nb in neighbours:
            if (node, nb) not in w_cache:
                w = compute_w_ik(node, nb)
                w_cache[(node, nb)] = w
                w_cache[(nb, node)] = w
    print(f"w_ik cache : {len(w_cache):,} pairs in {time.time() - t0:.1f}s")

    return adj, degree, d_max, w_cache, compute_w_ik


def build_t_last(train_src, train_dst, train_t):
    t_last = defaultdict(dict)
    for u, v, t in zip(train_src.tolist(), train_dst.tolist(), train_t.tolist()):
        t_last[u][v] = max(t_last[u].get(v, -1), t)
        t_last[v][u] = max(t_last[v].get(u, -1), t)
    print(f"t_last     : {sum(len(v) for v in t_last.values()):,} node-pair timestamps")
    return t_last
