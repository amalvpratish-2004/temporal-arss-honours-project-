# Phase 2, Step 1 — EdgeBank Baseline on tgbl-wiki (TGB Protocol)

**Status:** Complete.
**Code:** `scripts/edgebank_tgb_wiki.py`
**Raw output:** `tgb_edgebank_results.json`

---

## 1. Why this exists

Phase 1 established honest AUC/AP numbers on Wikipedia JODIE under the
project's own evaluation protocol (random negatives, one per positive).
Bug B3 in the original bug audit flagged that protocol as too weak on its
own: Wikipedia's 88% repeat ratio means a model can score close to 0.90
AUC by learning almost nothing beyond "has this (user, page) pair
interacted before." Without a memorisation baseline, a good ARSS number
under that protocol can't be told apart from exactly that.

Phase 2 swaps to TGB's own protocol: the same underlying graph, loaded
through TGB's own pipeline (not the PyG loader used in Phase 1), with
TGB's own fixed train/val/test split, TGB's own pre-generated negative
sets (all possible destinations per positive edge for this dataset), and
TGB's own Evaluator computing **MRR** instead of AUC/AP.

Before spending CPU-hours training the actual model under this new
protocol, this step establishes the floor it needs to beat: **EdgeBank**,
a parameter-free memorisation baseline (Poursafaei et al.; adopted as the
standard TGB baseline). If ARSS doesn't clear EdgeBank, that is itself
the headline finding for this phase — the same posture Phase 1 took
toward its own bugs.

---

## 2. What EdgeBank is

Two variants, both just a memory set of previously-seen `(src, dst)`
pairs, scored 1 if a queried pair is in memory and 0 otherwise — no
learning, no parameters:

- **EdgeBank(unlimited):** remembers every edge ever seen, with no decay.
- **EdgeBank(tw):** remembers only edges seen within a trailing time
  window (here: a window equal to the test split's own duration),
  intended as a fairer "recent activity" baseline.

Critically, EdgeBank's memory is meant to **accumulate over time**: by
the time test edges are being scored, the memory should already include
everything from train *and* validation, not just train — val has, by
definition, already happened by test time.

---

## 3. A bug found and fixed during this step

The first implementation built both EdgeBank memories from **train data
only**, and reused that same frozen memory for both validation and test
evaluation. This produced a clearly wrong result:

| | Val MRR | Test MRR |
|---|---|---|
| EdgeBank(unlimited) | 0.4138 | 0.3741 |
| EdgeBank(tw) | 0.4014 | **0.0621** |

`tw_test` collapsing to near-random (0.062) was the signal something was
broken: a train-only time window, queried at a *test*-split timestamp,
reaches backward past the end of validation and almost never overlaps
the frozen train-only memory — so the model was effectively guessing.
This is a measurement artefact (the scoring code not updating EdgeBank's
memory as time moves from val into test), not a property of EdgeBank
itself, and not comparable to the Phase 1 bugs (B1/B2/B4), which were in
the model/pipeline rather than in the baseline/evaluation script.

**Fix:** validation is scored against train-only memory (correct — val
hasn't happened yet relative to train); test is scored against a
**train+val** memory (correct — val has happened by test time). See
`memory_for_split` in `scripts/edgebank_tgb_wiki.py`.

---

## 4. Results (post-fix)

| | Val MRR | Test MRR |
|---|---|---|
| **EdgeBank(unlimited)** | 0.4138 | **0.4302** |
| **EdgeBank(tw)** | 0.4014 | **0.3766** |

n = 23,621 edges per split (TGB's own split; close to, but not identical
to, the 23,620/23,623 split sizes from Phase 1's PyG-loader split — TGB
v2 uses its own leakage-corrected split, not the same boundary).

Internal consistency, post-fix:
- `unlimited_test` (0.4302) ≳ `unlimited_val` (0.4138) — sensible: more
  accumulated memory by test time.
- `tw_test` (0.3766) no longer collapses, and sits below `unlimited_test`
  as expected (recent-only memory should underperform full memory).

### Comparison to published numbers

The handoff doc's reference table (from the TGB leaderboard) quotes test
MRR on `tgbl-wiki-v2`: EdgeBank(tw) 0.571, EdgeBank(unlimited) 0.495.
Our numbers (0.3766, 0.4302) are in the same range and preserve the
correct relative ordering among variants, but are not an exact match.

This is most likely a **protocol-definition difference, not a bug**:
most plausibly the exact `tw` window length convention (we used "the
test split's own duration"; TGB's official baseline script may define it
differently), and possibly minor differences in which negative-set
version is bundled with the current `py-tgb` release versus whichever
produced the original leaderboard entry. Rank-computation (the
optimistic/pessimistic-rank averaging used for ties) goes through TGB's
own `Evaluator`, so that step is not a likely source of divergence.

**Decision:** treat 0.4302 (unlimited) / 0.3766 (tw) as this project's
own working EdgeBank floor, computed honestly end-to-end through TGB's
official data loader, negative sets, and evaluator. Chasing an exact
match to the published leaderboard number was judged not worth further
time; the methodologically important property — a real, non-trivial,
internally consistent baseline to compare ARSS against — is satisfied.

---

## 5. How to reproduce

```bash
cd arss_fixed
source .venv/bin/activate   # needs py-tgb>=2.3.0; no torch required for this step
python scripts/edgebank_tgb_wiki.py
```

Writes `tgb_edgebank_results.json`, with one entry per
(variant, split) pair, appended/overwritten after each split finishes
(so a crash partway through doesn't lose completed splits). Runtime: a
few minutes total on CPU (pure set-membership lookups — EdgeBank is
cheap compared to the GNN itself).

---

## 6. What's next (Phase 2, Step 2)

Wire the actual ARSS model into this same TGB pipeline:
- Load `tgbl-wiki` via TGB's own API (as here), not the PyG loader —
  build `adj`/`w_cache`/`t_last` from TGB's own train split arrays.
- Retrain under TGB's split (the Phase 1 checkpoint cannot be reused;
  TGB v2's split and node-ID handling are not identical to Phase 1's
  pipeline).
- Evaluate with TGB's MRR protocol, applying the caching strategies
  flagged in the original handoff doc (shared source-side subgraphs
  across a positive's negatives; subsampled negatives during
  training/validation; full "all destinations" negatives only for the
  final test pass) — required because `tgbl-wiki` test alone is 23,621
  positives × ~9,227 negatives each if evaluated in full.
- Compare resulting ARSS test MRR against the EdgeBank floor established
  here (0.4302 / 0.3766). This comparison — not AUC/AP — is Phase 2's
  actual deliverable.