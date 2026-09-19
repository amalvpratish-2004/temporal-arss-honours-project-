"""
train.py — ONE training loop, called with different (seed, lam, max_hops),
replacing the five near-duplicate loops in the original notebook (cells
14, 16, 17, 19).

That drift is exactly what caused:
  - S7: cell 14 referencing `train_t_arr`, which only cell-2's sibling
    `train_t` ever defined -> NameError as shipped.
  - The un-reproducible 0.9142: whatever produced that number lived in a
    branch that no longer matches any cell in the file.

Run this the same way for every (seed, lam, max_hops) combination in your
sweep -- see run_sweep() at the bottom -- so every number in your results
table came from the identical code path.
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import load_and_split, build_adjacency_and_weights, build_t_last
from .sampler import make_temporal_score_fn, make_sampler, make_negative_sampler, make_batch_builder
from .model import TemporalARSSModel
from .eval import evaluate


def bce_loss(pos_logits, neg_logits):
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    return F.binary_cross_entropy_with_logits(logits, labels)


def run_one(seed, lam, max_hops, ctx, K=10, batch_size=512, epochs=15, lr=3e-4,
            readout="root", checkpoint_dir="./checkpoints", log_every=1):
    """ctx: dict from build_context() below (shared across seeds -- the graph
    and w_ik cache don't change, only model init / batch rng do)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_rng = np.random.default_rng(seed)  # ONLY for negative sampling during training

    device = ctx["device"]
    model = TemporalARSSModel(hidden=128, emb_dim=64, dropout=0.3, readout=readout).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    negative_sample = make_negative_sampler(train_rng, ctx["adj"], ctx["meta"]["item_lo"], ctx["meta"]["item_hi"])
    build_batch = ctx["build_batch"]

    arrs = ctx["arrs"]
    best_val_auc, best_epoch, best_state = -1.0, -1, None
    history = {"train_loss": [], "val_auc": [], "val_ap": []}

    n = len(arrs["train_src"])
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        epoch_losses = []
        perm = train_rng.permutation(n)
        src_arr, dst_arr, t_arr = arrs["train_src"][perm], arrs["train_dst"][perm], arrs["train_t"][perm]

        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            src_b, dst_b, t_b = src_arr[s:e], dst_arr[s:e], t_arr[s:e]
            neg_b = negative_sample(src_b)

            pos_sgs = build_batch(src_b, dst_b, t_b, K, lam, max_hops)
            neg_sgs = build_batch(src_b, neg_b, t_b, K, lam, max_hops)

            optimizer.zero_grad()
            pos_logits = model(pos_sgs)
            neg_logits = model(neg_sgs)
            loss = bce_loss(pos_logits, neg_logits)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(mean_loss)

        val_auc, val_ap = evaluate(
            model, build_batch, arrs["val_src"], arrs["val_dst"], arrs["val_t"],
            K, ctx["meta"]["item_lo"], ctx["meta"]["item_hi"], lam, max_hops, split_name="val",
        )
        history["val_auc"].append(val_auc)
        history["val_ap"].append(val_ap)

        if val_auc > best_val_auc:
            best_val_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % log_every == 0:
            print(f"[seed={seed} lam={lam} hops={max_hops}] epoch {epoch:02d} "
                  f"loss={mean_loss:.4f} val_auc={val_auc:.4f} val_ap={val_ap:.4f} "
                  f"({time.time() - t0:.1f}s)")

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"seed{seed}_lam{lam}_hops{max_hops}.pt")
    torch.save(best_state, ckpt_path)

    model.load_state_dict(best_state)
    test_auc, test_ap = evaluate(
        model, build_batch, arrs["test_src"], arrs["test_dst"], arrs["test_t"],
        K, ctx["meta"]["item_lo"], ctx["meta"]["item_hi"], lam, max_hops, split_name="test",
    )

    return dict(
        seed=seed, lam=lam, max_hops=max_hops, readout=readout,
        best_epoch=best_epoch, best_val_auc=best_val_auc,
        test_auc=test_auc, test_ap=test_ap, checkpoint=ckpt_path, history=history,
    )


def build_context(device=None, root="./data"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    arrs, meta = load_and_split(root=root)
    adj, degree, d_max, w_cache, compute_w_ik = build_adjacency_and_weights(
        arrs["train_src"], arrs["train_dst"]
    )
    t_last = build_t_last(arrs["train_src"], arrs["train_dst"], arrs["train_t"])
    temporal_score = make_temporal_score_fn(w_cache, compute_w_ik, t_last, meta["t_min"], meta["t_scale"])
    sample_fn = make_sampler(adj, temporal_score)
    build_batch = make_batch_builder(sample_fn, adj)

    return dict(device=device, arrs=arrs, meta=meta, adj=adj, w_cache=w_cache,
                compute_w_ik=compute_w_ik, t_last=t_last, temporal_score=temporal_score,
                sample_fn=sample_fn, build_batch=build_batch)


def run_sweep(seeds, lambdas, max_hops_list, ctx=None, **kwargs):
    """Replaces cells 14/16/17/19. Every run goes through run_one() with the
    identical code path -- no more drifted duplicate loops."""
    ctx = ctx or build_context()
    results = []
    for max_hops in max_hops_list:
        for lam in lambdas:
            for seed in seeds:
                results.append(run_one(seed, lam, max_hops, ctx, **kwargs))
    return results


if __name__ == "__main__":
    # Example: the Phase 1 ablation -- lam=0 IS Base ARSS through the same
    # code path (cleanest possible base-vs-temporal comparison, no confounds).
    results = run_sweep(
        seeds=[42, 123, 2024],
        lambdas=[0, 1.0, 10.0],
        max_hops_list=[1, 2],
        epochs=15,
    )
    for r in results:
        print(r["seed"], r["lam"], r["max_hops"], r["test_auc"], r["test_ap"])
