# Temporal ARSS — fixed package (Phase 0 → Phase 1)

This replaces `temporal_arss_fixed3.ipynb`'s five drifted training loops
with one package. It applies the B1, B2, B4, S1, S2, S3, S5 fixes from the
review. It does **not** yet do Phase 2 (TGB protocol) or Phase 3 (scale to
`tgbl-coin`) — this is Phase 0 + Phase 1 only.

## What's fixed, and where

| Bug | File | What changed |
|---|---|---|
| B1 double node-ID offset | `arss/data.py` | Read PyG's existing dst offset once; removed the second offset; `N_ITEMS` now 1000, not 9227 |
| B2 isolated roots / label leakage | `arss/sampler.py` (`build_anonymous_subgraph`) | Roots forced to indices 0/1; target edge excluded from each root's own candidate pool via `exclude=` |
| B4 no induced subgraph edges | `arss/sampler.py` (`build_anonymous_subgraph`) | Edges induced over the whole sampled node set, not just root-neighbour "double star" edges |
| S1 hop-2 candidates always `inf` | `arss/sampler.py` (`make_temporal_score_fn`) | Unseen pairs get a finite worst-case age (`t_min`), not `None`/`inf` |
| S2 `exp()` overflow | `arss/sampler.py` (`make_temporal_score_fn`) | Scoring done in log-space (monotone-equivalent, numerically safe) |
| S3 non-reproducible eval negatives | `arss/eval.py` | Val/test negatives generated once from a dedicated seeded RNG and cached, reused across every config |
| S5 negative can equal a real edge | `arss/sampler.py` (`make_negative_sampler`) | Rejects and resamples if the draw is already a neighbour of the source |
| S7 `train_t_arr` NameError, drifted loops | `arss/train.py` | One `run_one()` function, called per (seed, λ, max_hops) — see `run_sweep()` |

**Not changed** (per the review, these need a decision, not a silent fix):
- R4 — model uses root-embedding readout by default; pass `readout="mean"`
  to `TemporalARSSModel` to get the report's Eq. 3.6 mean-pooling instead.
  Consider running both as an ablation.
- S4 — `msg` (LIWC features) is still loaded/normalized but unused. Decide
  and state explicitly in the report whether that's intentional (ARSS
  anonymises nodes by design) or something to wire in later.
- B3 (weak eval protocol) and Phase 2/3 (TGB protocol, `tgbl-coin` scale-up)
  are separate follow-on work, not part of this package.

## IMPORTANT — what I could and couldn't verify here

I don't have `torch`/`torch_geometric` or network access in this sandbox,
so I could **not** run the actual model or the full test suite end-to-end.
What I did verify:
- All files pass `python -m py_compile` (no syntax errors).
- The B1 arithmetic reproduces the review's reported 89.16% phantom-negative
  figure exactly, and the fixed logic eliminates it (see the numbers below —
  reran with the actual N_USERS/dst-range constants from your report).
- The logic in `build_anonymous_subgraph`, the sampler, and `eval.py` was
  written and re-read carefully against the bug descriptions, but **you
  should run `pytest tests/ -v` yourself** in an environment with torch +
  torch_geometric (Colab, your local GPU box) before trusting any numbers
  from it. Treat this as a strong first draft you review, not a merged PR.

```
OLD (buggy): negatives drawn from (8227, 17453) width 9227
OLD (buggy): real page ids in that range: 1000 (10.84% real -> 89.16% phantom)

NEW (fixed): negatives drawn from (8227, 9226) width 1000 -- all real pages, 0% phantom
```

## How to run

```bash
pip install torch torch_geometric scikit-learn numpy pytest

# 1. Run the invariant tests first (should mostly pass immediately since
#    the fixes are already applied; they're here so you can re-run them
#    if you modify anything, or intentionally revert a fix to see them fail)
pytest tests/ -v

# 2. Run the Phase 1 sweep (downloads Wikipedia JODIE on first call)
python -m arss.train
```

Or from a notebook:
```python
from arss.train import build_context, run_sweep

ctx = build_context()   # loads data once, builds adjacency + w_ik cache once
results = run_sweep(
    seeds=[42, 123, 2024, 7, 999],
    lambdas=[0, 0.1, 0.5, 1, 2, 5, 10, 20],   # lam=0 IS Base ARSS, same code path
    max_hops_list=[1, 2],
    ctx=ctx,
    epochs=15,
)
import pandas as pd
df = pd.DataFrame(results)
df.groupby(["lam", "max_hops"])[["test_auc", "test_ap"]].agg(["mean", "std"])
```

## Expected outcome

Per the review: expect test AUC to drop substantially from the original
0.9026/0.9142 — probably into the 0.75–0.88 range — once B1 and B2 are
fixed. **That drop is the point.** Document it as the finding, add
EdgeBank / "seen-before" baselines next (not yet in this package), and
move to Phase 2 (TGB protocol) once these numbers are stable across seeds.
