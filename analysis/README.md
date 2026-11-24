# Analysis scripts

Each script prints its results; run it from the repository root, for example
`python analysis/main_results/results_table.py`. Its docstring states what it computes.

| Folder | Contents |
|---|---|
| `main_results/` | Table I (`results_table.py`), the comparisons quoted in the article (`paper_numbers.py`), learned-cascade baselines, paired statistics, operating curves, and Figs. 4–5 (`figures.py`) |
| `judge_validation/` | The GPT-4o judge against the human raters, its tier bias, re-scoring, and repeatability |
| `sensitivity/` | Cost weights, quality weights, the reference tolerance, cost-matching sources, and near-duplicate images |
| `router_analysis/` | Agreement among agents, the Primary-Only gap, Stage 2 training versus deployment inputs, and how much profiling the routers need |
| `robustness/` | Multiple comparisons, event-level resampling, and leave-one-event-out evaluation |
| `system_costs/` | Latency accounting, network conditions, power measurements, and a shared Jetson |
| `output_checks/` | Parsing rules, retries, prompt order, and severity mapping, checked against the stored responses |
| `results/` | Figures and tables written by the scripts |

Most scripts reuse `main_results/results_table.py` for the data and the cost accounting.
