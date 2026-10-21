# Final results — Temporal ARSS (post-report work, for the next review)

This document consolidates everything measured or corrected **after** the
submitted report (`honoursreportfinal_signed.pdf`, April 2026): the bug-audit
fixes, the honest re-measurements, the TGB-protocol baseline work, the UCI
expansion, and the first `tgbl-review` run. It supersedes nothing in content —
it organises `findings1.md`, `findings2.md`, `findings3.md`,
`RESULTS_SUMMARY.md` and `RESULTS_REVIEW_RUN1.md` into one place.

Dates: findings gathered 5–19 October 2026.

---

## 1. Executive summary — claims and their current status

| # | Claim | Status | Key evidence |
|---|---|---|---|
| 1 | The report's 0.9026 / 0.9142 test AUC was inflated by pipeline bugs | **Established** | After all fixes: Base 0.7065, Temporal 0.7446 (§3.1). Deflation was predicted in advance (0.75–0.88 band) and confirmed by measurement (§2) |
| 2 | Temporal recency improves Base ARSS on Wikipedia | **Established (1 dataset, 2 protocols)** | +0.038 test AUC paired across 3 seeds (paired std 0.003); +0.055–0.064 TGB-protocol MRR (1 seed) (§3.1, §3.3) |
| 3 | ARSS loses to a parameter-free EdgeBank on repeat-heavy Wikipedia under TGB protocol | **Established** | Test MRR 0.70 (unlimited) / 0.51 (tw) vs ≤ 0.35 for ARSS, n=2000 (§3.3). Cause diagnosed: the sampler makes the target pair's own history invisible — exactly the signal EdgeBank uses (§5.1) |
| 4 | Base ARSS beats EdgeBank, CN and AA on UCI | **Established for this protocol, 1 seed** | Test MRR 0.333 vs 0.271 (EdgeBank unlimited), n=2000, matched history (§3.2) |
| 5 | Temporal ARSS is *worse* than Base on UCI | **Preliminary (1 seed)** | −0.067 test MRR (§3.2) |
| 6 | Recency helps on some datasets and not others | **Preliminary** | Follows from claims 2 and 5 |
| 7 | On tgbl-review ARSS trained at/below chance in run 1 | **Negative result, cause identified, fix implemented, rerun pending** | Val MRR 0.0509 ≈ chance 0.0515; train/eval negative mismatch is the main cause (§3.4) |
| 8 | Anything on tgbl-coin / larger datasets | **Not done** | Needs CSR adjacency, bounded frontier, lazy w_ik, and a working GPU |

---

## 2. Why the report's numbers were too good — the complete bug chronology

The original report quoted Base 0.9026 / Temporal 0.9142 test AUC from one
notebook run. Each fix below removed something the model could exploit; the
first three came from the review bug audit, the rest were discovered while
running the cleaned pipeline.

| Bug | Effect | Fix location |
|---|---|---|
| **B1 — double node-ID offset.** PyG's JODIE loader already offsets destination IDs; the notebook applied it twice, so ~89% of random negatives pointed at node IDs that don't exist. A trivial "does this destination exist" heuristic solved most of the task (89.16% phantom negatives). | `arss/data.py` |
| **B2 — isolated roots on positives.** A dict-key collision sent both target nodes of every positive edge to an isolated duplicate; the positive embedding was constant, so the model separated classes by detecting the degenerate representation. | `arss/sampler.py` (`build_anonymous_subgraph`) |
| **B4 — no induced subgraph edges.** Subgraphs contained only root↔neighbour "double star" edges, never edges between two neighbours — the exact structure ARSS was meant to be more expressive than. | `arss/sampler.py` (`build_anonymous_subgraph`) |
| **Target-edge leak** (introduced *by* the B4 fix, caught later). The induced-edge step re-added the root–root edge for every positive, giving positives a "roots are directly linked" signal random negatives lacked. Symptom: training loss collapsed to ~0.0003. An intermediate "honest" Phase 1 number (0.8549) was still inflated by this. | excluded pairs; regression test `test_no_root_root_edge_in_final_subgraph` |
| **TGB's UCI/Enron IDs are stored bipartite-style** (destination id = node id + max source id + 1). Two IDs per entity, no shared neighbours possible — CN/AA scored exactly chance and ARSS trained at chance. Folded back by name in `arss/tgb_data.py` (`FOLD_DATASETS`); not auto-detected because Wikipedia has the same offset pattern and really is bipartite. Supporting evidence: after folding, positives share neighbours far more often than random pairs (46% vs 30%). |
| **Self-pairs (u, u)** became legal candidates after folding; they crashed the subgraph builder and scored *maximum* under CN/AA. All methods now score them lowest. |
| **History asymmetry at test time.** EdgeBank/CN/AA used train+val as history for test queries; ARSS used train only. Corrected with `extend_history_to_val` (§4, protocol note). |
| **EdgeBank memory bug** (findings2, in the baseline script, not the model). First EdgeBank build used a train-only memory frozen for both splits; tw test collapsed to 0.062. Fixed: val scored against train-only memory, test against train+val. |
| **Train/eval negative mismatch** (tgbl-review run 1, §3.4). Training negatives were easy; TGB's official negatives include 8.8% previous neighbours of the source, carrying a root-root edge signal the model had only ever seen on positives. Fix (`--neg-mode v2 --hist-neg-frac 0.1`) implemented; rerun pending. |

Not fixed (decisions still needed, per the review):
- **R4** — readout is root-embedding by default, not the report's Eq. 3.6 mean
  pooling (pass `readout="mean"` for the paper version; run both as ablation).
- **S4** — LIWC edge features (`msg`) still loaded but unused; state explicitly
  whether that's intentional.
- **S4/B3 follow-on, Phase 2/3 scale-up** (`tgbl-coin`) — separate work.

---

## 3. Results

All MRR numbers below use **capped negatives** (20 per positive unless stated)
and are comparable *only within this project's tables*, never to TGB
leaderboard entries. All methods in a table share the same positives, the same
sampled negatives (same seed, same subsampling code) and TGB's Evaluator.

### 3.1 Wikipedia, PyG loader, project's own protocol (random negatives; 3 seeds, 15 epochs)

The fully honest number — B1/B2/B4 fixed **and** the target-edge leak fixed:

| | Test AUC | Test AP |
|---|---|---|
| Base (λ=0) | 0.7065 ± 0.0025 | 0.6667 ± 0.0085 |
| Temporal (λ=10) | **0.7446 ± 0.0023** | 0.6902 ± 0.0082 |
| Paired difference | **+0.0381** (+0.035, +0.042, +0.037) | +0.0235 |

Training loss now ends at 0.47–0.49 (it collapsed to ~0.0003 under the leak).

Intermediate deflation steps, for the record:

| Stage | Base | Temporal |
|---|---|---|
| Report (3 blocking bugs active, 1 run) | 0.9026 | 0.9142 |
| B1/B2/B4 fixed (findings1, 3 seeds) | 0.8549 ± 0.0199 | 0.8669 ± 0.0209 |
| + leak fixed, + history corrections (final) | 0.7065 ± 0.0025 | 0.7446 ± 0.0023 |

The temporal-vs-base gap survives at every stage: +0.0116 (report, 1 run) →
+0.0120 (findings1, 3 seeds, 4.4σ) → **+0.0381** (final, 3 seeds, paired std
0.003). Limits: bipartite graph (common-neighbour term is zero by
construction), hops=1 only, λ ∈ {0, 10} only in the final run.

### 3.2 UCI, TGB protocol (IDs folded; 20 negatives per positive; n=2000 positives, seed 0)

| Method | Val MRR | Test MRR |
|---|---|---|
| **ARSS Base (λ=0), test on train+val graph** | **0.291** | **0.333** |
| EdgeBank (unlimited) | 0.233 | 0.271 |
| ARSS Temporal (λ=10), test on train+val graph | 0.275 | 0.266 |
| EdgeBank (time-window) | 0.233 | 0.264 |
| Adamic-Adar | 0.221 | 0.260 |
| Common Neighbours | 0.206 | 0.237 |
| Random guess (21 candidates) | | ~0.174 |

ARSS test with a train-only graph (the handicapped setting, documented for the
asymmetry note): Base 0.247, Temporal 0.237 — extra validation history raises
Base MRR by +0.086 here.

Reading: with matched information, Base ARSS is ahead of every baseline on
both splits by roughly 0.06–0.10. Checkpoints were chosen on val, so test is
the clean comparison. Temporal ARSS is above the baselines on val and about
level with them on test. Rough standard error ~0.007 per MRR / ~0.01 per
difference — **assumed** (per-edge MRRs were not saved), not measured.

### 3.3 Wikipedia, TGB protocol (1 seed, 3 epochs; n=2000 positives, 20 negatives, seed 0)

| Method | Val MRR | Test MRR, train-only graph | Test MRR, train+val graph |
|---|---|---|---|
| EdgeBank (unlimited) | 0.663 | | **0.702** |
| EdgeBank (time-window) | 0.553 | | 0.515 |
| ARSS Temporal (λ=10) | 0.366 | 0.346 | 0.331 |
| ARSS Base (λ=0) | 0.311 | 0.290 | 0.267 |
| Random guess | | ~0.174 | |

- **Temporal beats Base**: +0.055 (val), +0.056/+0.064 (test). Same direction
  as the 3-seed AUC result in 3.1; the MRR result is one training seed.
- **EdgeBank is far ahead of ARSS**: ~0.35–0.40 MRR. Even the time-window
  variant (0.515) beats the best ARSS number (0.346).
- Extra validation history did **not** help ARSS here (Base −0.023, Temporal
  −0.015), unlike UCI. Cause unknown.
- Earlier smaller-n matched runs (findings3, 200 positives, 20 negs): EdgeBank
  unlimited 0.7242, Temporal 0.6600, Base 0.6489, EdgeBank tw 0.5255 — same
  ordering, consistent magnitudes.
- EdgeBank under TGB's **full** negative set (findings2, n=23,621): unlimited
  val 0.4138 / test 0.4302; tw val 0.4014 / test 0.3766. These are the only
  full-negative-set numbers; they differ from leaderboard values (tw 0.571 /
  unlimited 0.495 for tgbl-wiki-v2) — protocol-definition differences (tw
  window convention, bundled negative-set version), judged not worth chasing.

### 3.4 tgbl-review, run 1 (causal, λ=10, seed 42, negatives v1) — negative result

Data: 3,413,814 train positives; val 2,000 and test 10,000 queries ×
(1 true + 100 official negatives). Chance MRR = H(101)/101 = 0.0515.

| | val | test frozen | test stream |
|---|---|---|---|
| ARSS causal λ=10 (best epoch 9) | 0.0509 ± 0.0023 | 0.0481 ± 0.0010 | 0.0443 ± 0.0010 |
| Common neighbours | 0.0417 | 0.0486 | 0.0552 |
| Adamic-Adar | 0.0440 | 0.0510 | 0.0587 |
| Preferential attachment | 0.2245 | 0.2585 | 0.2841 |
| Candidate degree | 0.2398 | 0.2672 | **0.2888** |

Training loss 0.6955 → 0.575 flat; val MRR 0.046–0.051 in every epoch.

Model-free diagnosis (`scripts/diagnose_cache.py`), fraction of each set:

| | dest-side nodes = 0 | dest-side saturated (≥10) | ≥1 common nbr | root–root edge |
|---|---|---|---|---|
| train positives | 3.4% | 77.2% | 0.09% | 0.21% |
| train negatives (v1) | 23.8% | 23.6% | 0.10% | **0.00%** |
| val positives | 0.1% | 92.1% | 0.15% | 0.20% |
| val negatives (TGB official) | 1.1% | 60.0% | 0.24% | **8.81%** |

1. **Train/eval negative mismatch — the main cause [Likely].** Training
   negatives were easy (24% had no history; 0% previous neighbours of the
   source), TGB's official negatives mostly active nodes with 8.8% previous
   neighbours carrying a root-root edge — a feature the model saw only on
   positives, so those negatives rank above the true destination. With ~9 such
   negatives per query, expected MRR ~0.026–0.05, matching the observed
   0.044–0.048. Positives have a root edge in only 0.2% of cases — review is a
   "new pair" task.
2. **Common neighbours carry no signal on review** [Certain for this subgraph
   definition]: 0.15% of positives vs 0.24% of negatives; CN/AA at chance.
   ARSS's structural premise has little to work with here.
3. **Degree is invisible above K=10** [Likely]: positives 92% saturated vs
   negatives 60%; the model can only separate the unsaturated 40%, while the
   degree heuristic (exact degree) scores 0.24–0.29.
4. No conclusion about recency (λ) or causal-vs-static can be drawn from this
   run.

**Implemented fix, rerun pending:** `--neg-mode v2 --hist-neg-frac 0.1` —
random negatives uniform over nodes existing at query time (rejecting
source/true destination/current neighbours), plus 10% previous neighbours of
the source (matching the 8.8% measured on val, a protocol statistic, not
labels). Expectation: above chance but still far below the degree heuristic
unless degree information is added — a deviation from the original 5-feature
ARSS that needs an explicit decision.

---

## 4. Protocol notes (what the numbers do and do not mean)

- **Capped negatives.** 20 negatives per positive, not TGB's full set
  (~9,227 on Wikipedia, ~1,000 on UCI). A full pass is days of CPU time.
- **Matched baselines.** EdgeBank, CN, AA and ARSS use the same positives,
  negatives and evaluator within each table.
- **Tie handling.** TGB's Evaluator ranks ties at the average best/worst
  rank. All-tied scores give 1/(1+20/2) = 0.0909; a random continuous score
  gives ~0.174. Binary scorers (EdgeBank) are penalised by ties relative to
  continuous ones.
- **History at test time.** EdgeBank memory and the CN/AA graph include
  train+val for test queries; ARSS's graph is rebuilt the same way via
  `extend_history_to_val` (`--test-history` in `scripts/reeval_tgb.py`). Val
  edges all precede test edges — no future information is used.
- **Self-pair candidates** are scored lowest by every method.
- **Both research questions survive the fixes, in sharper form:** the report's
  question "does temporal recency help?" became answerable only after the
  pipeline bugs were removed; its discussion of structural dominance became
  concrete once quantified as "below a parameter-free memorisation baseline."

---

## 5. Diagnosed causes and open hypotheses

1. **Why EdgeBank beats ARSS on Wikipedia [Hypothesis, untested] (claim 3's
   cause).** The sampler excludes the target destination from the source's
   neighbours and never adds the root-root edge to the subgraph — needed
   against the leak, but in a temporal setting an *earlier interaction of the
   same pair* is legitimate history, and it is the main signal on repeat-heavy
   data (~88% of Wikipedia test edges are repeats). The proposed remedy is
   time-respecting subgraphs (see §6.0), which keeps the anti-leak guarantee
   only for pairs that have not interacted before time t.
2. **Time-respecting training is the deeper fix for the remaining design
   distortion.** The training-time graph is currently not time-respecting:
   neighbours whose last interaction is *after* the query time are still
   sampled, with age clamped to 0 (`dt = max(t_query - t_last, 0)`), so they
   look maximally recent. At test time every training edge is in the past, so
   recency means what it should — the recency weighting that distinguishes
   Temporal from Base is therefore learned under a distortion it does not see
   at evaluation (RESULTS_SUMMARY §4, "Training-time graph is not
   time-respecting"). Fixing this requires the item 0 time-respecting
   subgraphs and a retrain.

---

## 6. Not established / next (ordered)

0. **Time-respecting subgraphs** — build neighbour sets strictly before query
   time; keep the root-root edge iff the pair interacted before t. Removes the
   design distortion (§5.2) and may close the EdgeBank gap. Needs a
   time-indexed adjacency and a retrain.
1. **Negatives v2 rerun on tgbl-review** — fix is implemented; needed before
   any λ or causal-vs-static conclusion (§3.4).
2. **Seed variance on UCI and on all MRR results** — one training seed per
   setting currently.
3. **Store per-edge MRRs** in `evaluate_mrr_tgb` (small change) — enables
   paired bootstrap / measured standard errors.
4. **Wikipedia under TGB with matched history, n=2000 at val too** —
   re-score existing checkpoints: `python scripts/reeval_tgb.py tgbl-wiki
   tgb_arss_results.jsonl 2000 --test-history` (drop `--only-test` for val).
5. **Enron** — blocked on purpose: 13,702 train self-loops after folding;
   needs an explicit self-loop policy stated in the write-up.
6. **Scale-up** (tgbl-coin etc., Phase 3): CSR adjacency, bounded frontier,
   lazy w_ik, and a working GPU (currently CPU-only ~1,100 s/epoch).
7. **hops=2 and a λ sweep** were never run on the fixed pipeline.
8. **Open design decisions carried from the review:** R4 readout ablation
   (root-embedding vs Eq. 3.6 mean pooling); S4 LIWC feature disposition.

---

## 7. Limitations to state in the review

- UCI Temporal-vs-Base rests on one training seed (claim 5's caveat).
- MRR standard errors on UCI were assumed (per-edge std ~0.3), not measured.
- Capped-negative MRRs are internally comparable only — never
  leaderboard-comparable.
- The EdgeBank-vs-ARSS gap on Wikipedia was measured at 3 training epochs, 1
  seed in the AUC-finding runs; undertraining remains the most likely single
  factor to shift it (findings3's own caveat).
- Heavy-run artifacts (`scripts/`, `results.jsonl`, `tgb_*.json`) live on the
  training machine and are **not in this repository** — archive them before
  the review.

## 8. Reproduce

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
python scripts/edgebank_tgb_wiki.py                # findings2 EdgeBank, full negatives
python scripts/train_eval_tgb_wiki.py              # 3.3 ARSS under TGB
python scripts/edgebank_tgb_wiki_matched.py 2000   # matched-protocol EdgeBank
python scripts/diagnose_cache.py                   # 3.4 tgbl-review model-free diagnosis
```
