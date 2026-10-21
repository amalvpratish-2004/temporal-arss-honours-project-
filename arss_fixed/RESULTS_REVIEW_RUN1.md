# tgbl-review, run 1 (causal, lam=10, seed 42, negatives v1) -- NEGATIVE RESULT, cause identified

Command: `fast_run.py --variant causal --lam 10 --seed 42 --epochs 10 --workers 64 --threads 32 --baselines`
Data: 3,413,814 train positives; val 2,000 and test 10,000 queries x (1 true + 100 official negatives). Chance MRR = H(101)/101 = 0.0515.

## Numbers (all from the user's terminal screenshots)
| | val | test frozen | test stream |
|---|---|---|---|
| ARSS causal lam=10 (best epoch 9) | 0.0509 +/- 0.0023 | 0.0481 +/- 0.0010 | 0.0443 +/- 0.0010 |
| Common neighbours | 0.0417 | 0.0486 | 0.0552 |
| Adamic-Adar | 0.0440 | 0.0510 | 0.0587 |
| Preferential attachment | 0.2245 | 0.2585 | 0.2841 |
| Candidate degree | 0.2398 | 0.2672 | 0.2888 |
Training loss 0.6955 -> 0.575 in epoch 1, then flat to 0.573 (epoch 10). Val MRR 0.046-0.051 in every epoch.
Timings: graph 4 s; train cache 888 s; val eval cache 14 s; test eval caches ~77 s each; training 3,700-13,400 s per epoch (varies 3.5x, ~0.55-2 s/batch).

## Diagnosis (scripts/diagnose_cache.py, model-free)
| | dest-side nodes = 0 | dest-side saturated (>=10) | >=1 common nbr | root-root edge |
|---|---|---|---|---|
| train positives | 3.4% | 77.2% | 0.09% | 0.21% |
| train negatives (v1) | 23.8% | 23.6% | 0.10% | **0.00%** |
| val positives | 0.1% | 92.1% | 0.15% | 0.20% |
| val negatives (TGB official) | 1.1% | 60.0% | 0.24% | **8.81%** |

1. **Train/eval negative mismatch (the main cause, [Likely]).** My training negatives were easy (24% had no history; 0% were previous neighbours of the source). TGB's official negatives are mostly active nodes and 8.8% are *previous neighbours of the source*, which carry a root-root edge. The model saw a root-root edge only on positives, so it ranks those negatives high. With ~9 such negatives per query ahead of the true destination the expected MRR falls to ~0.026-0.05, i.e. at or below chance, matching 0.044-0.048. (Positives have a root edge in only 0.2% of cases: review is a "new pair" task.)
2. **Common neighbours carry no signal on review** (0.15% of positives vs 0.24% of negatives); CN/AA are at chance. ARSS's structural premise has little to work with here. [Certain for this subgraph definition]
3. **Degree is invisible above K=10.** Positives: 92% saturated, negatives 60%. The model can only separate the unsaturated 40%; the degree heuristic uses exact degree and scores 0.24-0.29. [Likely]
4. No conclusion about recency (lam) or causal-vs-static can be drawn from this run.

## Next run (implemented): negatives v2
`--neg-mode v2 --hist-neg-frac 0.1` (default): random negatives are uniform over nodes that exist at the query time, reject the source/true destination/current neighbours; 10% of negatives are previous neighbours of the source. 10% was chosen to match the 8.8% measured on val queries (a protocol statistic, not labels). Expectation: above chance but still far below the degree heuristic unless degree information is added (a deviation from the original 5-feature ARSS; not done, needs the user's decision). [Guessing]
