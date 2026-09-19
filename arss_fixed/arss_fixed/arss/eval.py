"""
eval.py — evaluation with FIXED negative sets (S3).

FIX APPLIED vs the original notebook (cell 6 / cell 12):

  S3  evaluate() called the module-level negative_sample(), which drew from
      the shared global `rng` and advanced its state every call. So:
        - validation negatives differed between epochs,
        - the λ=1.0 vs λ=10.0 comparison (a 0.0004 AUC gap!) was measured
          against DIFFERENT negative sets,
        - no run was reproducible even with the same seed.

      Fix: generate the val/test negative destination arrays ONCE, from a
      dedicated seeded generator that is never touched again, cache them
      (in memory here; swap in np.save/np.load if you want them on disk
      across process restarts), and reuse the identical array for every
      config/epoch/seed you evaluate.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

_NEG_CACHE = {}


def get_fixed_negatives(split_name, src_arr, item_lo, item_hi, seed=999):
    """Returns a cached, seed-fixed array of negative destinations, one per
    edge in src_arr, for the given split ('val' or 'test'). Same array is
    returned on every call for a given split_name+seed — this is the whole
    point of the fix."""
    key = (split_name, seed, len(src_arr))
    if key not in _NEG_CACHE:
        gen = np.random.default_rng(seed)  # dedicated generator, never reused elsewhere
        _NEG_CACHE[key] = gen.integers(item_lo, item_hi + 1, size=len(src_arr))
    return _NEG_CACHE[key]


def get_batches(src_arr, dst_arr, t_arr, neg_arr, batch_size):
    n = len(src_arr)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        yield src_arr[s:e], dst_arr[s:e], t_arr[s:e], neg_arr[s:e]


@torch.no_grad()
def evaluate(model, build_batch_subgraphs_fn, src_arr, dst_arr, t_arr,
             K, item_lo, item_hi, lam, max_hops, split_name, batch_size=512,
             neg_seed=999):
    model.eval()
    neg_arr = get_fixed_negatives(split_name, src_arr, item_lo, item_hi, seed=neg_seed)

    all_scores, all_labels = [], []
    for src_b, dst_b, t_b, neg_b in get_batches(src_arr, dst_arr, t_arr, neg_arr, batch_size):
        pos_sgs = build_batch_subgraphs_fn(src_b, dst_b, t_b, K, lam, max_hops)
        neg_sgs = build_batch_subgraphs_fn(src_b, neg_b, t_b, K, lam, max_hops)

        pos_logits = model(pos_sgs)
        neg_logits = model(neg_sgs)

        scores = torch.cat([pos_logits, neg_logits]).sigmoid().cpu().numpy()
        labels = np.array([1] * len(pos_logits) + [0] * len(neg_logits))
        all_scores.append(scores)
        all_labels.append(labels)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    return (
        roc_auc_score(all_labels, all_scores),
        average_precision_score(all_labels, all_scores),
    )
