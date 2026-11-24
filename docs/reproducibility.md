# Reproducibility record

What was re-run for this release, what matched, and what was not re-run. Checked in November 2026 on macOS with
Python 3.12.7.

## How the checks were run

Every script listed in `docs/reviewers/README.md` was run from the repository root, both in the analysis environment
used for the article and in a fresh virtual environment built only from `requirements.txt`. The outputs were compared
with those of the original working repository from which this release was assembled.

During each run, a Python audit hook recorded every file the script opened. Apart from output folders given on the
command line, no script opened a file outside the repository.

## Results

**Labels and routers.** The profiling pipeline (`profiling/code/pipeline/`, steps 01–05) was run on the profiling set
and on the held-out set, and step 01 on the live run's responses. Every output row matched the committed files. Only
the summary rows differ, because they record the paths and step names of the run. Step 06 regenerated the three
routers in `deployment/routers/` byte for byte (identical MD5).

**Device code and the live run.** `deployment/export_device_router.py` regenerated `router_policy.cpp` (M5) and
`stage2_policy.py` (Pi) byte for byte. Its checks:

- the generated code matched the trained routers on all 5,526 training records;
- it matched the devices' logged decisions in the live run (Stage 1: 600/600, Stage 2: 297/297);
- the Stage 2 features logged by Pi_A matched the recomputed features (297/297).

**Article numbers and figures.** All twelve rows of Table I and every result quoted in Sections IV and V were
reproduced by the scripts in `docs/reviewers/README.md`. The one exception is the 460 KiB image-distribution estimate,
which is arithmetic rather than a script output. `analysis/figures.py` redrew Fig. 4 and Fig. 5; the PNG images and
their data files were byte-identical to the committed ones, and the PDFs differ only in their embedded creation date.

**Parsing rules.** `analysis/output_checks/parsing_rules_reimpl.py` re-implements the category and severity rules in
`analysis/output_checks/parsing_rules.tex`, the listing given in the response to reviewers. It reproduced every stored
label: 20,178 M5 responses and 12,838 Pi responses, with 0 mismatches. `parsing_rules_difftest.py` compared the rules
with the deployed code, and all outputs agreed:

| Parser | Stored texts | Generated texts |
|---|---|---|
| M5 firmware function (compiled on the host) | 36,584 | 60,000 |
| Pi deployed Python | 36,626 | 60,000 |
| Jetson deployed parser | — | 30,000 label blocks |

**Power.** `analysis/power_summary.py` reproduced the device powers used in the cost model from the raw logs: M5
1.2154 W, Pi 7.5564 W, and Jetson 11.199 W. The two Pi recordings differed by 0.56% and the two Jetson recordings by
0.70%. The USB meter's accumulated energy agreed with mean power × time within 0.07%.

**Prompts.** The prompt constants are identical in the profiling, held-out, deployed, and first-submission code.
`docs/reviewers/prompts.md` was generated from them.

## How this release differs from the working repository

The folders were reorganized and renamed, and the paths inside the scripts were updated to match. Every script was
run before and after the reorganization. The outputs were identical, except for printed file paths, one printed list
whose items appear in a different order, and the column widths of one table.

| Change | Files |
|---|---|
| Added the MEDIC test-split manifest; scripts read it from the repository instead of an external drive. | `heldout/results/medic_manifest_test.csv` |
| Saved the pre-correction label files instead of reading them from the version-control history; the files are byte-identical to those versions. | `revision/first_submission/labels_before_severity_fix/` |
| The parsing check reads the listing stored next to it. | `analysis/output_checks/parsing_rules.tex` |
| Added the Jetson power logs, copied from the device with matching MD5, and a script that computes the device powers. | `profiling/results/power/`, `analysis/power_summary.py` |
| Renamed the human-rating files to `rater00_author` and `rater01`–`rater09`, keeping their order. The rating site's user and page identifiers were removed; the scores are unchanged. | `revision/human_evaluation/ratings/` |
| Updated the generator's path in the first comment line of the generated router code. | `deployment/m5_firmware/router_policy.cpp`, `revision/eidseg/runtime/firmware/router_policy.cpp` |
| Removed the router BSSID and network name; the optional setting is commented out. | `platformio.ini` |
| Removed: internal drafts and notes, the first submission's notebook and figures (the revised article no longer uses their numbers), an exploratory script whose results are not reported, a superseded collection file, a corrupted SD-card log, and the dataset images. | — |

## Not re-run

- New executions on the M5, Pi, or Jetson devices, a PlatformIO firmware build, and the load test in
  `revision/load_test/`, all of which need the hardware.
- The GPT-4o judge (`profiling/code/pipeline/00_gpt_judge.py`), which makes paid API calls; its outputs are included.
- Scripts that need the dataset images: the GPT-4o judge, `revision/eidseg/dataset/prepare_eidseg.py`, the page
  builders in `revision/human_evaluation/rating_page/` and `revision/label_checks/`,
  `heldout/code/sample_test_images.py`, and the `eidseg` mode of `analysis/exclusion_sensitivity.py`.
- Collecting human ratings again.
