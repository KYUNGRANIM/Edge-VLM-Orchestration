# Profiling pipeline

Scores every response collected during profiling, builds the reference termination labels, and trains the two routers
that `deployment/` runs. Run the steps from the repository root, in order.

| Step | Script | Output (`profiling/results/pipeline_outputs/` unless noted) |
|---|---|---|
| 00 | `00_gpt_judge.py` | `00_gpt_judge_scores.jsonl`: GPT-4o text scores (paid API calls; needs `OPENAI_API_KEY` and the images). The scores are included, so this step can be skipped. |
| 01 | `01_quality_components.py` | `01_quality_components.jsonl`: category, severity, and text scores per tier |
| 02 | `02_cost_components.py` | `02_cost_components.jsonl`: latency, communication, and energy per tier, normalized |
| 03 | `03_oracle_labels.py` | `03_oracle_labels.jsonl`: reference termination tier per image |
| 04 | `04_train_eval_router.py` | `04_routing_main.jsonl`: five-fold out-of-fold evaluation of the two-stage router |
| 05 | `05_sensitivity.py` | `05_sensitivity.jsonl`: the same under ten other reference-label settings |
| 06 | `06_export_routers.py` | `deployment/routers/`: the trained routers (joblib and readable tree text) |

`run_all.sh` runs steps 01–06 and skips the paid step 00 unless `RUN_GPT_JUDGE=1` is set.

## Reference labels

`configs/primary_balanced_d10.json` defines the operating point:

```text
Q[t]     = 0.3 Category[t] + 0.3 Severity[t] + 0.4 Text[t]
feasible = tiers with max(Q) - Q[t] <= 0.10
reference tier = the feasible tier with the lowest cost
Cost[t]  = 0.1 Latency[t] + 0.1 Communication[t] + 0.8 Energy[t]   (each normalized)
```

## Results on the 5,526 profiling images

```text
Reference tiers: M5 2,384 | Pi 1,755 | Jetson 1,387
Reference: mean quality regret 0.0154, latency 39.5 s, energy 246.3 J

Two-stage router (decision trees of depth 4, out-of-fold):
  agreement with the reference tier       55.0% (balanced 53.1%)
  images sent to a feasible tier          65.9%
  mean quality regret                     0.110
  mean latency / energy                   39.4 s / 245.0 J (50.3% less energy than always using Jetson)
```

Across the ten other reference-label settings of step 05 (five quality weightings × tolerances 0.05 and 0.10), the
router's agreement ranged from 42.3% to 61.6%, the share sent to a feasible tier from 47.8% to 70.6%, and the energy
saving against always using Jetson from 50.3% to 59.4%.

## Held-out set

The same steps 01–03 score the 600 held-out images; `README.md` at the repository root gives the commands.
