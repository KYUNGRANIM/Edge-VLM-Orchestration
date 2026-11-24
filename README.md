# Edge-VLM-Orchestration

Quality–cost-aware orchestration of vision-language models (VLMs) across a hierarchy of edge devices.

This repository contains the code, prompts, and data for the article *Hierarchical Edge VLM Inference in Practice:
Quality–Cost-Aware Runtime Orchestration* (submitted to IEEE Communications Magazine).

## Overview

Small edge devices can run VLMs next to a camera, but their responses are weaker; larger devices give better responses
at a higher cost in time and energy. In a hierarchy, each image can stop at the first tier whose response is good
enough or be sent on to the next tier. This work makes that decision per image.

1. **Offline profiling.** Every tier answers every image of a profiling set. Each response is scored on disaster
   category, damage severity, and text quality, and the latency, communication, and energy of reaching each tier are
   measured. Each image is labeled with its reference termination tier: the cheapest tier whose quality is within 0.10
   of the best tier's.
2. **Runtime routing.** Two decision trees learn the reference decision from signals available at run time: the
   lower tier's predicted category and severity, its confidence, and the agreement among agents on the same tier.
   After Tier 1 the first tree decides whether to stop or escalate; after Tier 2 the second tree decides again.
3. **Physical evaluation.** The routers run on a three-tier prototype and are evaluated on held-out images.

## System

| Tier | Devices | Model and runtime |
|---|---|---|
| 1 | 3 × M5Stack CoreS3 + Module LLM (one main agent, two auxiliary agents) | InternVL2.5-1B, AX630C package |
| 2 | 2 × Raspberry Pi 5 (one main agent, one auxiliary agent) | SmolVLM2-2.2B-Instruct, Q8_0, Ollama |
| 3 | 1 × Jetson Orin Nano | Qwen2.5-VL-3B-Instruct, 4-bit NF4, vLLM |

Each tier's response comes from its main agent; the auxiliary agents provide the agreement signals used for routing.
`docs/runtime_specification.md` gives every setting.

## Results

600 held-out MEDIC images. Latency, communication, and energy are reconstructed from measured device traces.

| Method | Q-Sat (%) | Latency (s) | Communication (KiB) | Energy (J) |
|---|---|---|---|---|
| Stop at Tier 1 (M5) | 52.3 | 9.2 | 0 | 11.2 |
| Stop at Tier 2 (Pi) | 64.3 | 66.2 | 230 | 428.7 |
| Stop at Tier 3 (Jetson), through all tiers | 76.0 | 77.9 | 460 | 559.6 |
| Jetson only (direct upload) | 76.0 | 13.4 | 230 | 130.9 |
| **Learned routing (this work)** | **66.2** | **39.5** | **291** | **376.2** |

Q-Sat is the share of responses with a quality score of at least 0.65. Compared with stopping at the Pi, learned routing
had 40.3% lower latency and 12.2% lower energy but 26.5% more communication; the Q-Sat difference (+1.8 points) was not
statistically significant. In this configuration, sending every image directly to the Jetson gave a better quality–cost
point. The article discusses these results and further baselines.

## Repository layout

```text
profiling/     offline profiling on 5,526 MEDIC images: collection firmware and scripts, the scoring and training
               pipeline (code/pipeline), and the collected responses, scores, labels, and power logs (results)
heldout/       the all-tier collection on the 600 held-out images: firmware, servers, responses, and labels
deployment/    the deployed system: M5 firmware, Pi and Jetson servers, trained routers, and the live run on the
               600 held-out images
analysis/      scripts that compute the article's tables, figures, and statistics
revision/      material added during the review: human evaluation of the judge, EIDSeg experiments, a load test,
               label checks, and the deployment of the first submission
docs/          runtime specification, reproducibility record, and documentation for reviewers
```

## Getting started

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python analysis/results_table.py                 # Table I
python analysis/paper_numbers.py                 # the comparisons quoted in the article
python analysis/figures.py                       # Fig. 4 and Fig. 5 -> analysis/results/figures/
bash profiling/code/pipeline/run_all.sh          # labels and routers from the collected responses
```

Run commands from the repository root. Two checks also need a C++ compiler. `requirements-judge.txt` is needed only to
re-run the paid GPT-4o judge, and `requirements-devices.txt` only to run the Pi and Jetson servers.

## Data

Included: the collected VLM responses, GPT-4o scores, quality and reference labels, device timing traces, power logs,
and the human ratings. Not included: the MEDIC and EIDSeg images; the manifests in `profiling/results/`,
`heldout/results/`, and `revision/eidseg/dataset/` list every image used. MEDIC is distributed under CC BY-NC-SA 4.0.

## License

The code is released under the MIT License (`LICENSE`). Files derived from the MEDIC and EIDSeg datasets (manifests,
labels, and model responses to their images) remain subject to the terms of those datasets.

## For reviewers

- [`docs/reviewers/README.md`](docs/reviewers/README.md): where each result in the article comes from and how to
  reproduce it, including the additional analyses in the response to reviewers.
- [`docs/reviewers/prompts.md`](docs/reviewers/prompts.md): every prompt, verbatim.
- [`docs/reviewers/human_evaluation.md`](docs/reviewers/human_evaluation.md): the nine-rater check of the GPT-4o judge.
- [`docs/reproducibility.md`](docs/reproducibility.md): what was re-run for this release and what matched.
