# Results summary (post-fix) — Temporal ARSS

Supersedes the numbers in `RESULTS_PHASE1.md` and `RESULTS_PHASE2_ARSS.md`
(both were produced while a label leak was active; see section 2).
`RESULTS_PHASE2_EDGEBANK.md` is unaffected.

## 1. Status at a glance

| Claim | Status | Evidence |
|---|---|---|
| Temporal recency helps ARSS on Wikipedia | **Established (1 dataset)** | +0.038 test AUC, 3/3 seeds (paired std 0.003); +0.055 to +0.064 TGB-protocol MRR (1 seed, n=2000) |
| The originally reported 0.90/0.91 AUC was inflated by pipeline bugs | **Established** | 0.9026 -> 0.7065 for Base after three fixes (section 2) |
| Base ARSS beats EdgeBank, CN and AA on UCI | **Established for this protocol, 1 seed** | +0.06 test MRR over EdgeBank (n=2000, matched history) |
| Temporal ARSS is *worse* than Base on UCI | **Preliminary** | -0.067 test MRR, but one training run per setting |
| Temporal helps on some datasets and not others | **Preliminary** | follows from the two rows above |
| EdgeBank far ahead of ARSS on Wikipedia (TGB protocol) | **Established** | test MRR 0.70 (unlimited) / 0.51 (tw) vs <= 0.35 for ARSS, n=2000, under both history settings |
| Cause of that gap | **Hypothesis, untested** | ARSS never sees earlier interactions of the target pair (section 4) |
| Anything on tgbl-coin / larger datasets | **Not done** | needs CSR adjacency and a working GPU |

## 2. Why the first result was too good (methodology findings)

Original report: Base 0.9026, Temporal 0.9142 test AUC. Each fix changed what
the model could exploit; the first two came from the bug audit, the rest were
found while running experiments.

1. **B1, phantom node ids.** PyG's JODIE loader already offsets destination ids;
   the notebook offset them again, so ~89% of random negatives were node ids
   that do not exist. Fixed in `arss/data.py`.
2. **B2, isolated roots.** A dict-key collision sent the two target nodes of
   every positive edge to an isolated duplicate. Fixed in `arss/sampler.py`.
3. **B4, no induced edges.** Subgraphs were root-neighbour stars only. Induced
   edges added.
4. **Target-edge leak, introduced by the B4 fix and missed at first.** The
   induced-edge step re-added the root-root edge for every training edge, so
   positives carried a "roots are directly linked" signal that random negatives
   lacked. Training loss collapsing to ~0.0003 was the visible symptom. Fixed
   (`if pair == (0, 1): continue`); regression test
   `test_no_root_root_edge_in_final_subgraph`. An earlier "honest" Phase 1
   number (0.8549 Base) was still inflated by this.
5. **TGB's UCI / Enron ids are stored bipartite-style.** Destination id =
   node id + (max source id + 1), so a person has two ids and no two nodes can
   share a neighbour. As loaded, common-neighbour methods scored exactly
   chance (MRR 1/11) and ARSS trained at chance. Folded back by name in
   `arss/tgb_data.py` (`FOLD_DATASETS`); not auto-detected, because Wikipedia has
   the same offset pattern and really is bipartite. Whether TGB intends the two
   ids to be the same entity is not documented; the evidence is that positives
   share neighbours far more often than random pairs after folding (46% vs 30%).
6. **Self-pairs (u, u).** On a folded graph a node can be its own candidate
   destination. It crashed the subgraph builder, and under CN/AA it scored the
   *maximum*. All methods now give it the lowest score.
7. **History asymmetry at test time.** EdgeBank / CN / AA use train+val as
   history for test; ARSS used train only. Corrected with
   `extend_history_to_val` (section 4).

## 3. Results

### 3.1 Wikipedia, PyG loader, project's own protocol (random negatives; 3 seeds, 15 epochs)

| | Test AUC | Test AP |
|---|---|---|
| Base (lambda=0) | 0.7065 +/- 0.0025 | 0.6667 +/- 0.0085 |
| Temporal (lambda=10) | **0.7446 +/- 0.0023** | 0.6902 +/- 0.0082 |
| Paired difference | **+0.0381** (+0.035, +0.042, +0.037) | +0.0235 |

Training loss now ends at 0.47-0.49 (it collapsed to ~0.0003 with the leak).
For scale: the same pipeline with the leak gave 0.8549 / 0.8669 (3 seeds), and the
original report gave 0.9026 / 0.9142 (1 run).
Limits: bipartite graph (common-neighbour term is zero by construction),
hops=1 only, lambda in {0, 10} only.

### 3.2 UCI, TGB protocol (ids folded; 20 negatives per positive; n=2000 positives, seed 0)

| Method | Val MRR | Test MRR |
|---|---|---|
| **ARSS Base (lambda=0), test on train+val graph** | **0.291** | **0.333** |
| EdgeBank (unlimited) | 0.233 | 0.271 |
| ARSS Temporal (lambda=10), test on train+val graph | 0.275 | 0.266 |
| EdgeBank (time-window) | 0.233 | 0.264 |
| Adamic-Adar | 0.221 | 0.260 |
| Common Neighbours | 0.206 | 0.237 |
| Random guess (21 candidates) | | ~0.174 |

ARSS test with a train-only graph (the handicapped setting): Base 0.247,
Temporal 0.237 — shown only to document the asymmetry in section 4.

Rough standard error: ~0.007 per MRR, ~0.01 per difference. This assumes a
per-edge standard deviation of 0.3, which was **not measured**; per-edge MRRs
are not saved by the evaluator (see section 5).

Reading: with matched information, Base ARSS is ahead of every baseline on both
splits by roughly 0.06-0.10. The checkpoint was chosen on val, so test is the
clean comparison for ARSS. Temporal ARSS is above the baselines on val and about
level with them on test.

### 3.3 Wikipedia, TGB protocol (1 seed, 3 epochs; n=2000 positives, 20 negatives, seed 0)

| Method | Val MRR | Test MRR, train-only graph | Test MRR, train+val graph |
|---|---|---|---|
| EdgeBank (unlimited) | 0.663 | | 0.702 |
| EdgeBank (time-window) | 0.553 | | 0.515 |
| ARSS Temporal (lambda=10) | 0.366 | 0.346 | 0.331 |
| ARSS Base (lambda=0) | 0.311 | 0.290 | 0.267 |
| Random guess | | ~0.174 | |

(EdgeBank's memory always includes train+val for test, by definition.)

- **Temporal beats Base** by +0.055 (val) and +0.056 / +0.064 (test, train-only /
  train+val graph). Same direction as the 3-seed AUC result in 3.1. The MRR result
  is one training seed.
- **EdgeBank is far ahead of ARSS**, by roughly 0.35-0.40 MRR. Even the
  time-window variant (0.515) beats the best ARSS number (0.346).
- **Extra validation history did not help ARSS here** (Base -0.023, Temporal
  -0.015), unlike UCI, where it raised Base by +0.086. Cause unknown.
- **Likely cause of the EdgeBank gap (hypothesis, not tested):** ARSS as implemented
  never sees earlier interactions of the target pair (section 4), which is exactly
  the signal EdgeBank uses, and ~88% of Wikipedia test edges are repeats.

## 4. Protocol notes (what the numbers do and do not mean)

- **Capped negatives.** 20 negatives per positive, not TGB's full set (~9,227 on
  Wikipedia, ~1,000 on UCI). A full pass is days of CPU time. These MRRs are
  comparable to each other, **not** to leaderboard entries.
- **Matched baselines.** EdgeBank, CN, AA and ARSS use the same positives, the
  same sampled negatives (same seed, same subsampling code) and TGB's Evaluator.
- **Tie handling.** TGB's Evaluator ranks ties at the average of best and worst
  rank. All-tied scores give MRR 1/(1+20/2) = 0.0909; a random continuous score
  gives ~0.174. Binary scorers (EdgeBank) are penalised by ties relative to
  continuous ones.
- **History at test time.** EdgeBank memory and the CN/AA graph include train+val
  for test queries. ARSS's graph is rebuilt the same way with
  `extend_history_to_val` (`--test-history` in `scripts/reeval_tgb.py`); the model
  is not retrained and the training time-scale is kept. Val edges all precede test
  edges, so no future information is used.
- **Self-pair candidates** are scored lowest by every method; EdgeBank sees them
  as "never seen" (a tie with other unseen candidates), a difference I estimate to
  be far below the noise level but did not measure.
- **The target pair's earlier interactions are invisible to ARSS.** The sampler
  excludes the target destination from the source's neighbours (and vice versa),
  and the root-root edge is never added to the subgraph. That was needed against
  the leak in section 2, but in a *temporal* setting an earlier interaction of the
  same pair is legitimate history, and it is the main signal on repeat-heavy data.
- **Training-time graph is not time-respecting.** Adjacency, degrees and `t_last`
  are built from all training edges. For a training query at time t, neighbours
  whose last interaction is *after* t are still sampled, and their age is clamped
  to 0 (`dt = max(t_query - t_last, 0)`), so they look maximally recent. At test
  time every training edge is in the past, so recency means what it should. The
  recency weighting that distinguishes Temporal from Base is therefore learned
  under a distortion it does not see at evaluation.

## 5. Not established / next

0. **Time-respecting subgraphs.** Build neighbour sets from edges strictly before
   the query time, keep the root-root edge iff the pair interacted before t. This
   removes both design properties in section 4 and is arguably what "temporal
   awareness at the sampling stage" requires. Needs a time-indexed adjacency and a
   retrain; untested whether it closes the EdgeBank gap.
1. **Seed variance on UCI and on any MRR result.** One training seed per setting.
   Needed before the Temporal-vs-Base conclusion on UCI is more than a lead.
2. **Per-edge MRRs are not stored**, so no paired bootstrap or measured standard
   error exists. Storing them in `evaluate_mrr_tgb` is a small change.
3. **Wikipedia under TGB with matched history** — re-score the existing
   checkpoints:
   `python scripts/reeval_tgb.py tgbl-wiki tgb_arss_results.jsonl 2000 --test-history --only-test`
   plus `python scripts/edgebank_tgb_wiki_matched.py 2000`. (Val is also needed at
   n=2000 for a complete table: run reeval without `--only-test`.)
4. **Enron** is blocked on purpose: 13,702 train self-loops after folding. Needs
   an explicit self-loop policy, stated in the write-up.
5. **Larger datasets** (tgbl-coin etc.) need the Phase 3 engineering (CSR
   adjacency, bounded frontier, lazy w_ik) and a working GPU.
6. **hops=2 and a lambda sweep** were never run on the fixed pipeline.

## 6. Reproduce

```bash
pytest tests/ -v                                   # includes the leak and self-pair regression tests
python -m arss.train                               # 3.1 (Wikipedia, PyG loader), ~29 h on CPU
python scripts/edgebank_tgb_uci.py                 # EdgeBank, full negatives
python scripts/train_eval_tgb_uci.py               # 3.2 training (10 epochs x 2 lambdas)
python scripts/reeval_tgb.py tgbl-uci tgb_arss_uci_results.jsonl 2000
python scripts/reeval_tgb.py tgbl-uci tgb_arss_uci_results.jsonl 2000 --test-history --only-test
python scripts/edgebank_tgb_uci_matched.py 2000
python scripts/heuristics_tgb.py tgbl-uci 2000
python scripts/check_dataset.py tgbl-uci --fold    # run before training on any new dataset
```
