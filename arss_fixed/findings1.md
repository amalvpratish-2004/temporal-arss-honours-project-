# Phase 1 Results — Honest Re-measurement on Wikipedia JODIE

**Status:** Complete. Core comparison done (λ=0 vs λ=10, hops=1, 3 seeds).
λ=1.0 and hops=2 from the original sweep plan are not yet run (see
"What's left" below) — not required for the headline finding.

**Date:** October 2026
**Code:** `arss/` package (fixed pipeline), run via `python -m arss.train`
**Raw output:** `results.jsonl` (one JSON object per completed run,
includes full per-epoch training/validation history)

---

## 1. Why this run exists

The original report (`honoursreportfinal.pdf`) quotes test AUC 0.9026
(Base ARSS) and 0.9142 (Temporal ARSS), both from a notebook
(`temporal_arss_fixed3.ipynb`) with three confirmed blocking bugs:

- **B1 — double node-ID offset.** PyG's JODIE loader already offsets
  destination IDs; the notebook offset them a second time, so ~89% of
  randomly sampled negative edges pointed at node IDs that don't exist
  in the graph. A trivial "does this destination have any neighbours at
  all" heuristic solved most of the task.
- **B2 — isolated roots on positive edges.** A dict key collision meant
  the two target nodes of every *positive* training edge resolved to the
  wrong index, landing on an isolated duplicate node instead of their
  real, connected position. The resulting embedding was constant across
  every positive edge, so the model could separate classes by detecting
  that degenerate representation alone — not by learning structure.
- **B4 — no induced subgraph edges.** Sampled subgraphs only ever
  contained root↔neighbour edges (a "double star"), never edges between
  two neighbours. This is a plain rooted subtree — exactly the
  structure ARSS was designed to be more expressive than, so the
  original implementation never actually exercised the capability the
  project set out to demonstrate.

Full details, line-level code references, and the fix for each are in
`arss/sampler.py` and `arss/data.py` (see their docstrings/comments) and
in the original handoff document.

This run answers one question: **once those three bugs are fixed, what
does the honest number look like, on the identical dataset and split?**

---

## 2. Setup

- **Dataset:** Wikipedia JODIE, loaded via `torch_geometric.datasets.JODIEDataset`
  — same raw data as the original report (157,474 edges, chronological
  70/15/15 split: 110,231 train / 23,620 val / 23,623 test).
- **Node counts (post-fix):** 9,227 total — 8,227 users (IDs 0–8226) +
  1,000 pages (IDs 8227–9226). This matches PyG's documented figure and
  corrects the report's erroneous "17,454 total" (an artefact of B1).
- **Model/hyperparameters:** identical to the original report — K=10
  subgraph budget, 2-layer GraphSAGE (5→128→64), dropout 0.3, batch
  size 512, 15 epochs, Adam (lr 3e-4, weight decay 1e-5),
  CosineAnnealingLR. Only the data pipeline and subgraph construction
  were changed; the model architecture is untouched.
- **Hardware:** CPU (GPU unavailable due to a driver/CUDA version
  mismatch on the training box, not yet resolved). ~1,050–1,270s/epoch.
- **Seeds:** 42, 123, 2024 — same three seeds the original notebook's
  multi-seed cells used.
- **Configs run:** {λ=0 (= Base ARSS through the identical code path),
  λ=10.0 (the report's best-performing Temporal config)} × hops=1.
  λ=0 as the Base ARSS ablation is the cleanest possible comparison:
  Base and Temporal now share 100% of the code, differing only in the
  one multiplier, removing every implementation confound between them.

---

## 3. Results

### Test AUC / AP, per seed

| Config | Seed | Best epoch | Val AUC | Test AUC | Test AP |
|---|---|---|---|---|---|
| Base (λ=0) | 42 | 1 | 0.8912 | 0.8646 | 0.8716 |
| Base (λ=0) | 123 | 4 | 0.8500 | 0.8272 | 0.8663 |
| Base (λ=0) | 2024 | 2 | 0.8993 | 0.8730 | 0.8841 |
| Temporal (λ=10) | 42 | 1 | 0.9039 | 0.8804 | 0.8882 |
| Temporal (λ=10) | 123 | 3 | 0.8575 | 0.8374 | 0.8731 |
| Temporal (λ=10) | 2024 | 1 | 0.9074 | 0.8830 | 0.8971 |

### Mean ± std across seeds

| Config | Test AUC | Test AP |
|---|---|---|
| **Base ARSS** | **0.8549 ± 0.0199** | **0.8740 ± 0.0075** |
| **Temporal ARSS** | **0.8669 ± 0.0209** | **0.8861 ± 0.0099** |

### Paired comparison (Temporal − Base, same seed)

| Seed | Base AUC | Temporal AUC | Δ |
|---|---|---|---|
| 42 | 0.8646 | 0.8804 | +0.0158 |
| 123 | 0.8272 | 0.8374 | +0.0102 |
| 2024 | 0.8730 | 0.8830 | +0.0100 |

**Mean paired Δ = +0.0120, std of Δ = 0.0027.**
Temporal ARSS beats Base ARSS in all three seeds, and the gap (+0.0120)
is now roughly **4.4× its own standard deviation across seeds** — a far
stronger basis for the claim than the original report's single-run
+0.0116, which the bug audit flagged as smaller than one standard
deviation of either configuration's run-to-run noise.

---

## 4. Comparison to the original (buggy) report

| | Report (buggy pipeline) | This run (fixed pipeline) | Δ |
|---|---|---|---|
| Base ARSS test AUC | 0.9026 | 0.8549 (mean) | −0.0477 |
| Temporal ARSS test AUC | 0.9142 | 0.8669 (mean) | −0.0473 |

The drop (~0.047 AUC) lands inside the range the bug analysis predicted
in advance of running anything ("probably into the 0.75–0.88 AUC
range") — both honest numbers (0.8549, 0.8669) fall within that band.
This is the headline Phase 1 finding: **a substantial fraction of the
originally reported performance was an artefact of the three bugs, not
model capability**, and the finding was predicted before being measured,
not reverse-engineered after the fact.

The **relative** finding — temporal recency helps, structural
similarity still dominates — survives the fix. Only the absolute
numbers change.

---

## 5. How to reproduce

```bash
cd arss_fixed
source .venv/bin/activate   # Python 3.11 venv with requirements.txt installed
python -m arss.train
```

`arss/train.py`'s `__main__` block runs exactly this sweep
(`seeds=[42,123,2024]`, `lambdas=[0,10.0]`, `max_hops_list=[1]`,
`epochs=15`). Results append to `results.jsonl` (JSONL, one line per
completed run) as each run finishes — the file is flushed and fsync'd
after every write, and a restarted run automatically skips any
`(seed, lam, max_hops)` combo already present in that file.

To recompute the summary table above from `results.jsonl` directly:

```python
import json, statistics as stats
rows = [json.loads(l) for l in open("results.jsonl")]
for lam in (0, 10.0):
    aucs = [r["test_auc"] for r in rows if r["lam"] == lam]
    print(lam, "mean=%.4f std=%.4f" % (stats.mean(aucs), stats.pstdev(aucs)))
```

Per-epoch training loss and validation curves for every run are in each
JSON line's `"history"` field.

---

## 6. What's left (not blocking, lower priority)

- **λ=1.0 and hops=2** from the original full sweep plan — can be added
  back into `max_hops_list`/`lambdas` in `arss/train.py` and re-run;
  `results.jsonl`'s resume logic means this only costs the 12 new
  combos, not a full re-run.
- **EdgeBank and "seen-before" baselines** — not yet implemented. The
  handoff doc flags this as important: Wikipedia JODIE's 88% repeat
  ratio means a trivial memorisation baseline may already score close
  to what's reported here, which would reframe (not invalidate) this
  result.
- **GPU training** — currently CPU-only (~1,100s/epoch) due to an
  unresolved CUDA driver/wheel version mismatch on the training
  machine. Not required for Phase 1, but necessary before Phase 3
  (`tgbl-coin`, ~145× more edges than Wikipedia).
- **Phase 2 (TGB protocol, same Wikipedia data)** — `tgbl-wiki`
  confirmed downloadable (`py-tgb>=2.3.0`); eval.py has not yet been
  rewritten to use TGB's MRR metric and fixed historical/random
  negative sampler.