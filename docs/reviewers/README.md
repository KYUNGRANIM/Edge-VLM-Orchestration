# Guide for reviewers

This page maps the results in the article and in the response to reviewers to the scripts that compute them. Every
script reads only files inside this repository and prints its results. Run them from the repository root after
`pip install -r requirements.txt`; `docs/reproducibility.md` records that each one was re-run and matched.

Related pages: [`prompts.md`](prompts.md) (every prompt, verbatim) and
[`human_evaluation.md`](human_evaluation.md) (the nine-rater check of the GPT-4o judge).

## Results in the article

| Article item | Script |
|---|---|
| Table I: fixed and direct baselines, Confidence-Only, our policy, reference | `analysis/main_results/results_table.py` |
| Table I: M5–Pi Cascade and Primary-Only; Section IV-C comparisons, reference gap, permutation test; training-set sizes | `analysis/main_results/paper_numbers.py` |
| Table I: FrugalGPT-inspired and M5–Jetson cascades; the quality-constrained cascade | `analysis/main_results/learned_cascade.py` |
| Confidence intervals (Newcombe; paired bootstrap) | `analysis/main_results/paired_statistics.py` |
| Human–GPT agreement and the Jetson-offset check | `analysis/judge_validation/judge_sensitivity_nonauthor.py` |
| Stage 2 retrained on out-of-fold forwarded samples | `analysis/router_analysis/stage2_mismatch.py` |
| Earlier-response-set sensitivity check | `analysis/robustness/robustness_supplement.py` (section C) |
| Auxiliary-path latency check | `analysis/system_costs/critical_path.py` |
| Held-out images with near-duplicates in the profiling set | `analysis/sensitivity/exclusion_sensitivity.py` |
| Power measurements and their repeatability | `analysis/system_costs/power_summary.py` |
| Profiling cost (device-hours, energy) | `analysis/router_analysis/profiling_requirements.py`, `analysis/router_analysis/profiling_cost.py` |
| EIDSeg results | `revision/eidseg/runtime/pipeline/paper_numbers_eidseg.py` |
| Low-bandwidth latency scenario | `analysis/system_costs/bandwidth_sensitivity.py` |
| Fig. 4 and Fig. 5 | `analysis/main_results/figures.py` (writes `analysis/results/figures/`) |

The image-distribution traffic for the two auxiliary M5 devices (about 460 KiB per request) is two copies of the
230 KiB camera upload in Table I; no script computes it.

## Additional analyses in the response to reviewers

The analysis scripts are grouped by topic (see `analysis/README.md`).

| Topic | Where |
|---|---|
| GPT-4o judge: human agreement, tier bias, re-scoring, repeatability | `analysis/judge_validation/` |
| Paired intervals and the non-inferiority margin | `analysis/main_results/paired_statistics.py` |
| Multiple comparisons, event-level resampling, leave-one-event-out evaluation | `analysis/robustness/` |
| Cost weights, quality weights, the reference tolerance, cost-matching sources, near-duplicate images | `analysis/sensitivity/` |
| Learned-cascade baselines and full operating curves | `analysis/main_results/learned_cascade.py`, `analysis/main_results/operating_points.py` |
| Router behavior: agreement among agents, the Primary-Only gap, Stage 2 training versus deployment inputs, how much profiling the routers need | `analysis/router_analysis/` |
| Latency accounting, network conditions, energy measurement, a shared Jetson | `analysis/system_costs/`, `revision/load_test/` |
| Output processing: parsing rules, retries, prompt order, severity mapping | `analysis/output_checks/` |
| Generalization to another dataset | `revision/eidseg/` |
| Comparison with the first submission | `revision/first_submission/`, `analysis/router_analysis/primary_only_gap.py` |

Each script's docstring states what it computes.

## Rebuilding labels, routers, and device code

```bash
# labels and routers from the collected responses (step 00, the paid GPT-4o judge, is skipped)
bash profiling/code/pipeline/run_all.sh

# held-out labels
H=heldout/results/labels
python profiling/code/pipeline/01_quality_components.py --collection heldout/results/collection.jsonl \
  --judge heldout/results/gpt_judge_scores.jsonl --medic heldout/results/medic_manifest_test.csv --out $H/01_quality_components.jsonl
python profiling/code/pipeline/02_cost_components.py --collection heldout/results/collection.jsonl --out $H/02_cost_components.jsonl
python profiling/code/pipeline/03_oracle_labels.py --quality $H/01_quality_components.jsonl --cost $H/02_cost_components.jsonl --out $H/03_oracle_labels.jsonl

# device code from the routers, checked against the routers and against the live run
python deployment/export_device_router.py --collection profiling/results/collection.jsonl \
  --runtime deployment/live_run/online_routing_final.jsonl
```

These commands rewrite files that are already in the repository; the rewritten files match the committed ones except
for the absolute paths recorded in their summary rows.

## What is not included

- The MEDIC and EIDSeg images. Scripts that need them (the GPT-4o judge, the rating-page builders, the EIDSeg
  preparation, and the load test) cannot be re-run without the datasets.
- New runs on the devices, which need the hardware.
