# Paper-faithful ARSS (Zhang et al., Chaos, Solitons & Fractals 203 (2026) 117693)

Written 20 Oct 2026. Source of truth: the paper PDF itself (all 14 pages read, including Eq. 2-9, Fig. 2-3, Sec. 3.1-3.2).
Everything below marked "paper" was read from the PDF; everything marked "ours" is a choice the paper does not make.

## 1. Paper -> code mapping

| Paper | Where | Code |
|---|---|---|
| One rooted anonymous subgraph **per node** (Fig. 2) | `arss/fast/paper.py::node_subgraph`, cache of ALL nodes per snapshot | yes |
| w_ik = 2 pi / (1 + eta), pi, eta with psi(x)=log(1+x), d_max = max degree (Eq. 2-4) | `graph.py::log_w` (tested against a from-the-paper reference) | yes |
| Keep K neighbours, **smallest w first**, root not counted in K (Eq. 5, Fig. 3) | `paper.py::select_nodes` | yes, hop 1 only (see 2) |
| Subgraph = root + K nodes + **all** edges among them (Fig. 3) | `paper.py::rooted_subgraph` | yes |
| Anonymous: no labels, h^0 = X_p (Eq. 6) | no role features, no root marker, constant X_p | yes (X_p = 1, see 2) |
| Sum-aggregation message passing, L = 2 (Eq. 6, Fig. 8) | `paper_model.py` SAGEConv(aggr="sum") x2 | yes |
| h_i = mean over all subgraph nodes (Eq. 7) | `PaperARSS.embed` | yes |
| z_ij = (h_i * h_j) \|\| sum of common-neighbour h_u, first-order (Eq. 8) | `score_pairs`, `StaticAdj.common` | yes |
| sigmoid(MLP(z)) (Eq. 9) | `PaperARSS.predict` (logits; sigmoid is monotone) | yes |
| Embedding dim 256, lr 0.0043 (GNN) / 0.0028 (predictor), 800 epochs | `paper_run.py` defaults | dim and lrs yes; epochs no (see 2) |

## 2. What the paper does NOT specify, and what I chose (tell me if you disagree)

1. **Node features.** The paper uses 1433-d (Cora) / 3703-d (Citeseer) features. tgbl-review has none, so X_p = constant 1. Consequence: the network sees only the *unlabeled structure* of each K-node subgraph.
2. **Hops.** Paper: breadth first, then depth, but no maximum h is stated. Implemented: hop 1 only; a node with < K neighbours gets fewer than K nodes (the paper would continue to hop 2). On review this affects the low-degree nodes (roughly 20-40% of nodes by the diagnose output). 2-hop is NOT implemented.
3. **Base GNN layer** (paper: "NGNN's message passing", "default parameters of the open-source code" - code not seen): SAGEConv with sum aggregation, ReLU, dropout 0.3, no normalisation. MLP 2d -> 256 -> 128 -> 1.
4. **Static paper, temporal benchmark.** Subgraphs are computed once per graph snapshot using edges strictly before the cutoff. Val queries use the train graph, test queries the train+val graph (the old "frozen" history). `--lam 0` is Base ARSS. `--lam > 0` is the project's recency term, measured against the snapshot time.
5. **Training protocol (my addition).** The paper reuses one subgraph per node for every pair, so a training positive (u, v) that is an edge of the graph appears inside u's and v's own subgraphs. NCN-style code removes the batch's target edges; that is impossible with cached per-node subgraphs. Replacement: `--sup-frac 0.3` = the **last 30% (by time) of train edges are positives only**; the graph used for training holds the earlier edges. `--sup-frac 0` reproduces the naive protocol (leaks the target edge).
6. **Training negatives.** One uniform negative per positive, **fresh every epoch**: nodes that exist before the query's own time, excluding the source, the true destination and current neighbours in the training graph. No "historical neighbour" negatives (the anonymous model cannot see a root-root edge, so they would add noise).
7. **Common neighbours** capped at 64 per pair (cap not in paper; on review only ~0.2% of pairs have any).
8. **Epochs.** Paper 800 (tiny graphs). Default here 10. Checkpoint = best validation MRR (paper: validation split, metric not stated).
9. **Evaluation** is TGB's MRR against the 100 official negatives, on the same seeded 2,000 val / 10,000 test queries as before, so numbers are comparable with the earlier runs and with the CN / AA / PA / degree baselines (val 0.042 / 0.044 / 0.22 / 0.24; test-frozen 0.049 / 0.051 / 0.26 / 0.27). The paper's AUC/AP on Cora/Citeseer cannot be compared with these.

## 3. Tests (sandbox, synthetic graphs, real torch; 31 tests pass)

* Selected nodes have exactly the K smallest w values computed from Eq. 2-4 written independently in the test; subgraph edges equal ALL edges among root + selected nodes; cutoff is strict; root not counted in K.
* Layer = sum aggregation and pooling = mean (manual 1-layer check on a star graph); embedding is invariant to relabelling nodes, root included (anonymity).
* Common neighbours equal the set intersection at the cutoff; cap works. Supervision positives are not in the training graph except genuine repeats; naive mode shows the leak by construction. Negatives obey their rules.
* Parallel node cache == serial; training resumes from a checkpoint; `score_pairs` equals a manual Eq. 8-9 computation.
* Deliberate bugs (largest-w first; root counted in K; sum pooling; mean aggregation; missing eta term) each made a test fail. (One mutation initially slipped through because my test did not look at the selection the code actually uses; fixed.)

Not tested / unknown: behaviour on the real review data, runtime on the HPC (sandbox: ~100 steps + one small validation pass in 26 s on 2 threads, 60k-node synthetic graph; node caches take seconds there).

## 4. Run on the HPC

```bash
cd ~/arss_fixed/arss_fixed && source .venv/bin/activate        # after unzipping arss_fixed_v4.zip over the repo
pytest tests/test_paper.py -q                                  # expect 9 passed
# 1) timing/sanity run (a few minutes): does the loss move, how long is a step?
python scripts/paper_run.py --prep prep_tgbl-review.npz --lam 0 --seed 42 --epochs 1 --max-steps 200 \
    --workers 64 --threads 16 --out runs_paper_dbg 2>&1 | tee log_paper_dbg.txt
# 2) real run (Base ARSS, paper-faithful)
tmux new -s paper
python scripts/paper_run.py --prep prep_tgbl-review.npz --lam 0 --seed 42 --epochs 10 \
    --workers 64 --threads 16 --out runs_paper 2>&1 | tee -a log_paper.txt
```
If the loss blows up (NaN, or stuck at 0.69 from the first step) rerun with `--lr-gnn 0.001 --lr-pred 0.001`; the paper's rates were tuned for 5k-edge graphs.

## 5. What I expect (all [Guessing])

On review the model has no node features, common neighbours are almost absent, and a K=10 subgraph cannot show degree above 10. The paper itself says ARSS relies on common neighbours (Sec. 4.3). I expect a result far below the degree heuristic (0.27-0.29) and possibly close to chance (0.0515). This is a legitimate finding about the paper's design on this benchmark, not necessarily a bug; the diagnose script and the baselines stay the yardstick.
