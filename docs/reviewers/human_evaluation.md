# Human evaluation of the GPT-4o judge

Textual usefulness, 40% of the response-quality score, is scored by a GPT-4o judge. This study checks the judge against
human raters who scored the same responses with the same criteria.

## Design

| Item | Setting |
|---|---|
| Images | 60 held-out MEDIC images, 10 per disaster category (fire, flood, hurricane, landslide, earthquake, other). They are the first 60 of a category-stratified sample drawn with seed 2026 by `revision/human_evaluation/rating_page/build_rating_page.py`. |
| Responses | For each image, the three tier responses (M5_A, Pi_A, Jetson) from the all-tier held-out collection (`heldout/results/collection.jsonl`): 180 responses. |
| Blinding | Per image, the three responses were shown in shuffled order under the labels A, B, and C. The mapping to tiers (`rating_page/human_eval_key.json`) was not shown. |
| Criteria | The five criteria of the GPT-4o judge, each 0–5: damage grounding, damage overclaim, response appropriateness, response operationality, and target alignment. The rating page reads the criteria text from the judge script, so humans and GPT-4o saw the same wording (see `prompts.md`). |
| Reading aid | A Korean translation was shown under each response (`rating_page/human_eval_ko.json`). |
| Raters | Nine raters who are not authors (`ratings/rater01.json` … `rater09.json`) independently scored all 180 responses: 1,620 ratings. The first author's ratings (`ratings/rater00_author.json`) are excluded from the primary analysis because of authorship and are used only in a sensitivity analysis. One rater also scored a 61st image; the analyses use the 60 images every rater completed. |

Each rating is converted to a textual-usefulness score with the judge's own formula. The judge's score and each human
score then enter the article's quality score (0.3 category + 0.3 severity + 0.4 textual usefulness), and Q-Sat means a
quality score of at least 0.65.

## Results (nine non-author raters)

| Measure | GPT-4o vs each rater (mean) | Rater vs rater (mean, 36 pairs) |
|---|---|---|
| Q-Sat agreement | 84.7% | 83.9% |
| Cohen's κ on Q-Sat | 0.69 (range 0.56–0.78) | 0.68 (range 0.56–0.86) |
| Tie-aware Spearman correlation of textual usefulness | 0.46 | 0.47 |

Against the median of the nine raters, GPT-4o agreed on Q-Sat for 88.3% of the responses (κ = 0.77).

**Tier-related difference.** GPT-4o minus the human median, textual usefulness on a 0–1 scale, with 95% bootstrap
confidence intervals:

| Tier | Difference |
|---|---|
| M5 | −0.008 [−0.041, +0.025] |
| Pi | +0.017 [−0.040, +0.070] |
| Jetson | +0.096 [+0.049, +0.147] |

GPT-4o scored Jetson responses higher than every one of the nine raters did. Mean response lengths were similar across
tiers (34–36 words), and the correlation between length and the GPT–human difference was −0.08.

**Effect on the article's results.** Removing the Jetson offset from the judge's scores of the 600-image live run
changes Q-Sat as follows: our policy 66.2% → 64.5%, Fixed-Termination@Pi 64.3% (unchanged), Fixed-Termination@Jetson
76.0% → 69.8%, Confidence-Only 68.5% → 65.3%.

**Re-scoring with the final snapshot.** The 180 responses were originally scored with a GPT-4o snapshot that was not
recorded, so they were re-scored with `gpt-4o-2024-08-06`, the snapshot of the final evaluation. Mean GPT–human κ
became 0.70, agreement with the human median stayed at 88.3% (κ = 0.77), and the Jetson difference was +0.107
[+0.062, +0.152].

**Including the first author** (ten raters): GPT–human Q-Sat agreement 84.7%, rater–rater 84.1%.

## Files

| Path | Contents |
|---|---|
| `revision/human_evaluation/ratings/` | The ten rating files: `rater01`–`rater09` (non-authors), `rater00_author`. |
| `revision/human_evaluation/rating_page/` | Rating page template, the script that builds it, the A/B/C key, and the Korean reading aid. The built page embeds dataset images and is not included. |
| `revision/human_evaluation/analyze_human_eval.py` | Agreement statistics for a set of rating files. |
| `revision/human_evaluation/rescored_gpt4o/` | The 180 responses re-scored with `gpt-4o-2024-08-06`. |

## Reproducing the numbers

```bash
python analysis/judge_sensitivity_nonauthor.py   # nine raters (primary) and all ten; agreement, tier difference, Jetson offset
python analysis/judge_sensitivity_rejudge.py     # the same with the re-scored judge outputs
```

Both scripts print the values above.
