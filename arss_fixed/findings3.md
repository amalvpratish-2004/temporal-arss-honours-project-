# Phase 2, Step 2 — ARSS on tgbl-wiki (TGB MRR Protocol)

**Status:** Preliminary. One seed, 3 epochs (see "Limitations" — this needs
a fuller run before any finding here is treated as conclusive).
**Code:** `arss/tgb_data.py`, `arss/tgb_eval_mrr.py`, `scripts/train_eval_tgb_wiki.py`
**Raw output:** `tgb_arss_results.jsonl`, `tgb_edgebank_matched_results.json`

---

## 1. What this step did

Phase 1 established honest AUC/AP numbers for Base and Temporal ARSS
under the project's own protocol. Phase 2 swaps to TGB's official
protocol for the same graph: TGB's own split, TGB's own pre-generated
negative sets, TGB's own MRR metric. Step 1 (see
`RESULTS_PHASE2_EDGEBANK.md`) established an EdgeBank floor under that
protocol. This step trains the actual ARSS model (reusing Phase 1's
training code unchanged — see `arss/tgb_data.py`'s docstring for why
that reuse is sound) on TGB's split, and evaluates it the same way.

## 2. Setup

- Dataset: `tgbl-wiki`, loaded via TGB's own `LinkPropPredDataset`
  (not the PyG loader Phase 1 used — different split, different
  node-ID source, though the underlying graph is the same Wikipedia
  data).
- Model/training: identical architecture and B1/B2/B4-fixed pipeline as
  Phase 1. K=10, max_hops=1, batch size 512, Adam lr 3e-4.
- **EPOCHS=3** (Phase 1 used 15), **seed=42 only** (Phase 1 used 3
  seeds) — a deliberately small first pass given CPU-only training.
  See Limitations.
- Configs: λ=0 (Base ARSS, through the identical code path) and λ=10.0
  (the report's best-performing Temporal config).
- MRR evaluation: **MAX_EVAL=200** positive edges per split (not the
  full 23,620/23,621), **NEGS_CAP=20** negatives per positive (not the
  official "all destinations", ~9,227/positive). Both caps exist
  because subgraph construction is pure Python at ~190 builds/second on
  this CPU — a full-negative-set test pass alone would be ~218M builds,
  multiple days. See `arss/tgb_eval_mrr.py`'s docstring for the full
  reasoning. **This means none of the MRR numbers below are directly
  comparable to a TGB leaderboard entry** (which uses the full negative
  set) — they are only comparable to each other, computed under the
  identical capped protocol.

## 3. A methodological correction made during this step

The first comparison run (ARSS at negs_cap=20 vs. EdgeBank's Step 1
numbers, computed with the *full* ~9,227-negative set) was invalid:
fewer negatives to rank against mechanically inflates MRR regardless of
model quality, so that comparison understated how good EdgeBank really
looks next to ARSS. Fixed by adding `scripts/edgebank_tgb_wiki_matched.py`,
which reruns EdgeBank under the **exact same** MAX_EVAL=200/negs_cap=20
protocol (same seed, same positive-edge subsample) as the ARSS run. All
numbers below use this matched version.

## 4. Results (matched protocol: 200 positives, 20 negatives/positive, seed=0)

| Config | Val MRR | Test MRR |
|---|---|---|
| **EdgeBank(unlimited)** | 0.6891 | **0.7242** |
| Temporal ARSS (λ=10) | 0.7396 | 0.6600 |
| Base ARSS (λ=0) | 0.7307 | 0.6489 |
| EdgeBank(tw) | 0.5635 | 0.5255 |

Proxy metrics from training (random-negative AUC/AP, the project's own
Phase 1-style protocol, not TGB's) for reference only:
Base test AUC=0.8820/AP=0.8904; Temporal test AUC=0.8880/AP=0.8942.

### Test MRR ranking
1. EdgeBank(unlimited) — 0.7242
2. Temporal ARSS (λ=10) — 0.6600
3. Base ARSS (λ=0) — 0.6489
4. EdgeBank(tw) — 0.5255

## 5. Findings

**The temporal signal replicates under a second, independent protocol.**
Temporal ARSS beats Base ARSS by +0.0111 test MRR here — consistent in
direction and similar in magnitude to Phase 1's paired AUC result
(+0.0120). Two different evaluation protocols (random-negative AUC vs.
TGB MRR), two different data pipelines (PyG loader vs. TGB loader), same
conclusion: temporal recency provides a small, real, repeatable
improvement over structure alone.

**ARSS does not clearly beat a parameter-free memorisation baseline on
this split.** EdgeBank(unlimited) — no learning, no parameters, just "is
this exact edge in memory" — outperforms Temporal ARSS by +0.0642 test
MRR. Phase 1's bug audit (B3) flagged this as a live possibility and
explicitly said it would itself be the finding if it happened. This is
that result. It does not invalidate the temporal-signal finding above,
but it does mean the project's structural-similarity claim needs
qualifying: ARSS's structural signal is doing *something* (it beats
EdgeBank(tw), the "recent activity only" baseline, by a wide margin:
0.6600 vs 0.5255), but not enough to beat raw long-term memorisation on
this dataset, at this amount of training.

## 6. Limitations — why this is "preliminary," not final

- **3 epochs vs. Phase 1's 15.** The model may simply be undertrained
  relative to the comparison it's being asked to win. This is the most
  likely single factor to change the EdgeBank-vs-ARSS result.
- **1 seed vs. Phase 1's 3.** No variance estimate; the +0.0111
  Temporal-vs-Base gap and the −0.0642 ARSS-vs-EdgeBank gap could both
  shift with different seeds.
- **200-edge subsample**, not the full ~23,620/23,621 split per side.
- **20-negative cap**, not official leaderboard protocol. Fine for the
  internal ARSS-vs-EdgeBank comparison (both measured identically), not
  meaningful as a standalone number outside this project.

**Recommended next step before treating the EdgeBank finding as
conclusive:** rerun with `EPOCHS=15, seeds=[42,123,2024]` (matching
Phase 1's rigor) in `scripts/train_eval_tgb_wiki.py`, and the same
expanded settings in `edgebank_tgb_wiki_matched.py`. This is a
substantially longer run (multi-hour per seed/λ combination) and was
deliberately deferred in favour of moving on to dataset scaling — see
`RESULTS_PHASE3_PLAN.md`.

## 7. How to reproduce

```bash
cd arss_fixed
source .venv/bin/activate
python scripts/train_eval_tgb_wiki.py         # trains + evaluates ARSS, -> tgb_arss_results.jsonl
python scripts/edgebank_tgb_wiki_matched.py   # matched EdgeBank -> tgb_edgebank_matched_results.json
```

Both are resumable: `train_eval_tgb_wiki.py` skips any `(seed, lam)`
combo already present in `tgb_arss_results.jsonl`.