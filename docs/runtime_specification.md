# Runtime specification of the deployed hierarchy (MEDIC held-out rerun)

Everything below is transcribed from the code that ran the held-out evaluation (firmware `deployment/m5_firmware/`,
device servers `deployment/servers/`, feature code `profiling/code/pipeline/common.py`).

## 1. Models, runtimes and decoding

| Agent | Device | Model and runtime | Decoding | Image given to the model | Retries |
|---|---|---|---|---|---|
| M5_A, M5_B, M5_C | M5Stack CoreS3 + Module LLM (AX630C NPU) | InternVL2.5-1B, module build `internvl2.5-1b-364-ax630c` | `max_token_len` 300. The module API exposes no temperature, top-p, top-k or seed, so the module's defaults apply | JPEG downscaled on the device to a 640-pixel long side (short side at least 300 pixels) | One retry with the role's backup prompt when the answer fails the text check (Section 5.3) |
| Pi_A, Pi_B | Raspberry Pi 5 | SmolVLM2-2.2B-Instruct through Ollama (`ahmadwaqar/smolvlm2-2.2b-instruct`) | temperature 0.2, seed 42, top-k and top-p not set (Ollama defaults), `num_predict` 80 (Pi_A) and 40 (Pi_B), token log-probabilities returned | Original image file | Pi_A: one format retry. Pi_B: up to two (Section 5.1, 5.2) |
| Jetson | Jetson Orin Nano | Qwen2.5-VL-3B-Instruct on vLLM 0.7.4 (container `dustynv/vllm:0.7.4-r36.4.0-cu128-24.04`), bitsandbytes 4-bit weights, `--max-model-len 4096 --max-num-seqs 1 --gpu-memory-utilization 0.65` | temperature 0.3, seed 42, top-k and top-p not set, `max_tokens` 220 (main call) and 80 (repair call) | Resized to a 768-pixel long side, JPEG quality 85; on a call failure retried at 640, 512 and 384 | One repair call for the countermeasure line (Section 5.4) |

## 2. Prompts (verbatim)

The M5 and Pi prompts are sent with the image as a single user turn; the Jetson uses a system and a user message.

**M5_A** (main / backup)
```
You are a disaster expert.
Describe the damage and the most urgent action to take. Two sentences.
Do not start with Assessing.
```
```
You are a disaster expert.
Describe what happened and what should be done immediately. Two sentences.
```

**M5_B** (main / backup)
```
You are a disaster expert.
Describe the physical destruction or collapse visible in this image. One sentence.
```
```
You are a disaster expert.
Describe the broken or blocked structures here. One sentence.
```

**M5_C** (main / backup)
```
You are a disaster expert.
Describe how people could be hurt or killed here. One sentence.
```
```
You are a disaster expert.
Describe the danger to people near this scene. One sentence.
```

**Pi_A** (main)
```
You are a disaster response specialist. Look at this disaster photograph.
Write exactly two labeled lines:
Situation: one sentence describing the visible damage, blockage, or hazard.
Action: one immediate field command to address that damage.

Each line is 12-18 words. Name specific structures, locations, or materials. Action starts with an imperative verb. No news summaries, no cause guessing, no organization names.
```
**Pi_A** (retry; `{prev_raw}` is the first answer)
```
Same image. Write exactly two labeled lines (12-18 words each):
Situation: visible damage, blockage, or hazard.
Action: immediate field command starting with a verb.

Previous output:
{prev_raw}
```

**Pi_B** (main)
```
Disaster photo. Output exactly one line:
Description: <one sentence describing what is damaged, blocked, or collapsed>

12-18 words. Name specific structures, materials, and location. Mention what is blocked, fallen, submerged, or at risk. No action plan, no quotes, no bullets, no news headlines.
```
**Pi_B** (retry)
```
Rewrite for the same image as one line (12-18 words):
Description: <what is damaged, blocked, or collapsed — name structures and location>

Previous output:
{prev_raw}
```
**Pi_B** (second retry, only when an answer is a short "U.S."-style landmark fragment)
```
The previous answer was too short: {prev_raw}
Ignore landmark names, country names, and organization names. Describe only the visible physical hazard in the image.
If water is visible, mention standing water or flooding. Return exactly one line:
Description: <8-18 words about what is flooded, blocked, damaged, or at risk>
```
If that still returns such a fragment, Pi_B outputs the fixed sentence "Standing water covers the paved area near vehicles
and buildings, creating a possible flooding hazard."

**Jetson** (system message)
```
You are a disaster response specialist writing a concise field report
from visual evidence only. Look carefully at one disaster photograph.
Output exactly four labeled lines.

damage: one natural English sentence describing the visible disaster
  situation. Include concrete visual details such as material, color,
  count, structure, position, water, smoke, flames, debris, or people.
  Describe only what is plainly visible. Do not infer hidden injuries,
  deaths, trapped people, causes, or damage to vehicles, boats,
  aircraft, or equipment unless clearly visible. Do not name a building's
  function unless signage confirms it. Match the full visible scene, not
  just a foreground fragment.

countermeasure: one natural English sentence prescribing one
  concrete first-response action. Begin with a specific action verb
  (e.g., Deploy, Clear, Pump, Cordon, Search, Extinguish, Shore,
  Evacuate, Remove, Stabilize, Cover, Seal, Drain, Reroute). Name
  the specific target visible in the damage line (the collapsed
  structure, the flooded street, the burning building, the fallen
  tree). Do not start with 'Immediate countermeasure', 'Immediate
  action', 'Immediate response', or any label prefix. Do not use
  generic safety phrases such as 'ensure safety', 'prevent further
  damage', 'evacuate to a safer place', or 'secure the area' as the
  whole sentence — bind the action to a specific visible target.
  Avoid 'should', 'must', and past-tense narration.

cat: choose one based on the dominant visible evidence:
  fire — visible flames, active smoke, or charred surfaces.
  flood — pooled or flowing water, submersion, or waterlines on
          structures.
  hurricane — wind-torn roofs, snapped or uprooted trees, laterally
              blown debris. Storm-surge flooding may be present;
              classify hurricane when wind damage is the dominant
              visible signal.
  landslide — hillside earth, mud, or rocks displaced downslope;
              hillside debris burying a road or structure at a
              slope base.
  earthquake — cracked, leaning, tilted, pancaked, or collapsed
               masonry; structural collapse without active fire or
               dominant water.
  other_disaster — when the image mainly shows people, shelters,
                   responders, equipment, vehicles, or aftermath
                   without a clear visible hazard from the categories
                   above.

sev: 0 = minor or superficial damage; 1 = one section seriously damaged;
  2 = widespread or total destruction.

No markdown. No blank lines. Output:
damage: ...
countermeasure: ...
cat: ...
sev: ...
```
**Jetson** (user message): `Write the 4-line disaster field report for this photograph.`

**Jetson** repair (system message)
```
You are a disaster response specialist looking at a disaster photograph.
Write one concrete first-response action as an imperative sentence.

Begin with a specific action verb (Deploy, Clear, Pump, Cordon,
Search, Extinguish, Shore, Evacuate, Remove, Stabilize, Cover, Seal,
Drain, Reroute). Name the specific visible target from the damage
description.

Do not start with 'Immediate countermeasure', 'Immediate action',
'Immediate response', or any label prefix. Do not use generic safety
phrases ('ensure safety', 'prevent further damage', 'evacuate to a
safer place', 'secure the area') as the whole sentence. Avoid
'should', 'must', and past-tense narration.

Output one sentence only. No labels. No markdown.
```
Repair user message: `Damage: "<the damage sentence>"` followed by a blank line and
`Write one countermeasure sentence for this photograph.`

## 3. Category and severity

Categories: earthquake, fire, flood, hurricane, landslide, other_disaster. Severity: 0 little or none, 1 mild, 2 severe.
All matching is on lower-cased text; "contains" means substring.

### 3.1 M5 (labels derived from the free text; the M5 prompts do not ask for labels)

Category, step 1: the first rule that matches.
1. contains earthquake or quake: earthquake
2. hurricane, cyclone, typhoon or tornado: hurricane
3. landslide, mudslide, rockslide or rockfall: landslide
4. flood or inundat: flood
5. fire, wildfire, blaze or flame: fire

Step 1.5: if step 1 gave landslide, the text has no slope evidence (slope, hillside, mud, soil, debris flow, mudflow,
rockfall, erosion, cliff face, embankment) and has structural evidence (concrete, structural, rubble, crumbled,
earthquake, aftershock, collapsed building, cracked wall, collapsed structure), the category becomes earthquake.

Step 2, only when step 1 matched nothing: if the text contains burn together with burning, burned, burnt or smoke: fire.
Otherwise points are added and the highest wins (ties: earthquake, then flood, then hurricane, then landslide);
if the highest is below 2: other_disaster.
- earthquake: +4 seismic, tremor or aftershock; +3 rubble or crumbled; +4 fissure or sinkhole; +3 collapsed building,
  collapsed structure or building collapse; +2 structural with collapse or damage; +2 cracked with wall, road or
  foundation; +2 concrete with collapse, damage or crumble; +1 collapsed or collapse, and +1 debris, each only if
  landslide, flood and hurricane are all below 3
- hurricane: +4 storm surge, storm damage or tropical storm; +2 wind or gust; +3 uprooted; +3 fallen tree, tree fell
  or toppled tree; +2 roof with blown or torn; +2 power line, powerline or downed; +3 coastal with damage or flood
- flood: +4 submerge or waterlog; +3 overflow or water level; +4 inundat; +2 water with rising or rose
- landslide: +3 slope or hillside; +3 cliff or boulder; +3 erosion or eroded; +4 debris flow or mudflow

Severity: severe (not "not severe", "no severe", "not a severe"): 2; else moderate: 1; else mild, low severity, minor,
low damage, minimal damage, little or no, little to no: 0. Otherwise points: +2 massive, devastating, catastroph or
widespread; +2 casualt, killed, fatal, dead or death; +2 significant, major, extensive or destruct; +1 destroy,
collaps or rubble; +1 injur, trapped, evacuat or rescue; +1 block, scatter or displace. Severity is 2 at 3 points or
more, 1 at 2 points, else 0. (In the held-out rerun none of the 1,800 M5 answers contained "mild" or "moderate".)

### 3.2 Pi_A and Pi_B (identical rules, applied to the final answer text)

Category: points per pattern (`\b` word boundaries; `*` any word ending); the highest total wins.
- fire: +10 fire, wildfire, blaze; +6 flame(s), burning, burned, burnt, charr*, soot, scorch*, ablaze, inferno,
  smoldering; +3 smoke and +4 engulfed, each only if the text has no flood, water or wave
- flood: +10 flood, flooded, flooding, inundat*; +6 floodwater(s), submerged, waterlogged, waterline; +4 water with
  rising, level, deep, knee, waist, washed, standing, covering or filled; +6 river or stream followed by overflow*,
  burst or breached
- hurricane: +15 hurricane; +12 cyclone, typhoon, tornado; +6 storm with roof, tree, boat, pole, wind, torn, uprooted,
  ripped or surge; +4 uprooted, torn off/apart/away, ripped off/apart/away; +5 snapped, fallen, toppled, downed or
  knocked down followed by tree(s), branches, pole(s), power line(s), wire(s), fence(s); +5 wind or storm followed by
  damage or debris; +4 roof, shingle(s) or siding with torn, ripped, blown, peeled or twisted; +6 storm surge, coastal
  wave, tidal wave; +6 coast, coastal, shore, shoreline, beach, seafront or waterfront with damaged, destroyed, wreck,
  debris, scattered, battered, ravaged, flooded, submerged, broken, smashed or wreckage; +3 twisted or blown
  off/away/down with roof, wall, tree, pole, fence, sign or structure; +4 fallen, toppled or uprooted tree with road,
  street or path; +3 boat, ship, vessel or yacht with beach, shore, coast, damaged, destroyed or wreck
- landslide: +10 landslide, mudslide, rockslide; +8 rockfall, debris slide, slope failure, slope collapse; +5 boulder,
  hillside, embankment or slope with block, road, path, collaps, down or onto; +4 road, highway, path, route, railway or
  track(s) together with block*, obstruct* or cover* and debris, rock(s), mud, soil, earth, dirt or boulder
- earthquake: +10 earthquake, seismic, quake, tremor, aftershock; +5 collaps* with building, structure, wall, floor,
  story, storey, house, apartment, concrete or tower; +4 rubble with building, structure, concrete or brick(s), else +2
  rubble; +3 cracked, tilted or leaning with building, wall, tower, structure or column; +3 ground crack(s) or
  fissure(s)
- No points at all: hurricane if a coast word appears with damaged, destroyed, broken, wrecked, debris or wreckage;
  earthquake if damaged, destroyed, broken or wrecked appears with building, house, wall, structure or apartment;
  landslide if debris, rock(s) or mud appears with road, street, path, highway or track(s); otherwise other_disaster.

Severity: 2 if any of (a) widespread, extensive, massive, total destruction, total collapse, devastat*, engulf*,
demolished, ravaged, reduced to rubble, levelled, leveled, flattened, catastrophic, inferno, ablaze, deep water,
submerged, washed out, blocked, blocking, cut off, impassable, completely destroyed/collapsed/demolished/buried/
covered/submerged, entirely destroyed/collapsed/demolished/buried, totally destroyed/collapsed/demolished;
(b) buildings, structures, houses or homes together with collaps*, destroyed, damaged or in ruins; (c) building,
structure, house, home, apartment, bridge or tower together with collaps*, in ruins, crumbled, caved in, completely
destroyed, entirely destroyed or reduced to rubble. Else 0 if minor, small, slight, superficial, limited, little, no
visible, no major, intact, undamaged, mostly intact, barely, minimal or negligible. Else 1.

### 3.3 Jetson

Read from the `cat:` and `sev:` lines of its report. An unknown category becomes other_disaster; a missing severity
becomes 1.

## 4. Confidence (integer 1-5)

### 4.1 M5 (no token probabilities are available on the module)

score = 0.30 x (category keywords) + 0.25 x (specific terms) + 0.25 x (location anchors) - 0.30 x (hedges)
- 0.50 x (contradiction) + 0.02 x min(word count, 40), where each count is the number of listed words present:
- category keywords, for the extracted category only: landslide: landslide, mudslide, rockslide, rockfall, slope,
  cliff, erosion, boulder, mudflow; earthquake: earthquake, quake, seismic, fissure, sinkhole, rubble, crumbled, tremor;
  flood: flood, submerge, inundat, waterlog, overflow, water level; hurricane: hurricane, cyclone, typhoon, storm, wind,
  gust, surge; fire: fire, wildfire, burn, blaze, flame, smoke, ash, scorch
- specific terms: blocked, buried, trapped, injured, displaced, evacuate, shatter, toppled, scattered, piled, damaged,
  severed
- location anchors: road, bridge, building, highway, vehicle, house, wall, roof, river, forest, town, village, street
- hedges: " may ", " might ", " could ", " possibly ", " appears ", "appear ", " seem", "seems ", "seemed ",
  " perhaps ", " likely ", " unlikely "
- contradiction = 1 if: flood with burning, fire, blaze, smoke or flame; fire with flood, submerge, overflow or
  waterlog; earthquake with flood and without quake

The score is mapped to 1-5 by role-specific quintile boundaries fixed on a 914-answer pilot run:

| Role | 1 if score <= | 2 if <= | 3 if <= | 4 if <= | else 5 |
|---|---|---|---|---|---|
| M5_A | 0.97 | 1.20 | 1.39 | 1.62 | 5 |
| M5_B | 0.55 | 0.76 | 0.93 | 1.13 | 5 |
| M5_C | 0.40 | 0.57 | 0.75 | 0.98 | 5 |

### 4.2 Pi_A and Pi_B

From the token log-probabilities of the chosen answer: base = 5 if the mean log-probability is above -0.62, 4 above
-0.67, 3 above -0.72, 2 above -0.76, else 1 (3 if no log-probabilities were returned). Subtract 1 if the minimum token
log-probability is below -3.0, then subtract the semantic penalty = min(2, [answer failed the format gate of Section
5] + [the first retry's answer was chosen]). Clip to 1-5.

### 4.3 Jetson

Same thresholds over the tokens of the main and repair calls, with no semantic penalty. The Jetson confidence is
recorded but not used by any routing decision.

## 5. Format gates, retries and text clean-up

### 5.1 Pi_A
The answer is parsed into a Situation line and an Action line (aliases Disaster, Damage, Scene, Description and
Observation count as Situation; cause clauses such as "due to ..." are cut; one sentence each). Penalty points:
- +3 Situation missing or under 5 words; +3 Action missing or under 3 words
- +1 Situation over 25 words; +1 Action over 25 words
- +1 "due to", "caused by" or "because of" anywhere
- +2 descriptive action: rescue worker(s), emergency responder(s), emergency worker(s), firefighter(s), responder(s),
  relief worker(s) or response team(s) followed within 80 characters by must, need(s/ed), arrive(d), mobilize(s),
  swiftly, quickly, urgently, are, is, rush(es), working, seen, trying, begin(s), started, clearing, searching,
  responding, inspecting, conducting, assessing or coordinating
- +1 news-style Situation: has struck, has left, has devastated, has caused, has hit, in its wake, its wake, leaving
  destruction, leaving behind, leaving <word> in ruins, swept through, ravaged the
- +1 Situation contains none of: road, highway, railway, tracks, beach, shore, coast, street, bridge, building, house,
  forest, slope, cliff, hillside, canyon, route, river, village, town, city, area
- +1 Situation contains none of: rocks, boulders, debris, slope, cliff, beach, road, water, smoke, flames, trees, mud,
  soil, building, bridge, tracks, highway, hillside, route, shore, roof, wall, car, boat, rubble, structure, house

The gate passes at 5 points or fewer. A retry is made when a line is missing, the gate fails, the Action is
descriptive, or the Situation is news-style with none of the visual words. Of the two answers the one with the higher
selection score is kept: +5 both labels present, +3 Situation of 5 words or more, +3 Action of 3 words or more, minus
the penalty, plus the number of visual words (ties keep the first answer). Pi_A's answer is "Situation Action".

### 5.2 Pi_B
One Description sentence is parsed; a sentence containing non-ASCII characters is treated as empty. Penalty points:
+3 under 5 words; +1 over 40 words; +2 for each of "response action", "first-response", recommend(ed), and
should / must / will need to / needs to; +1 due to, caused by, because of, indicating or suggesting; +4 any non-ASCII
character; +1 an opener "a scene of destruction/devastation/extensive/severe"; +1 no location word; +1 no visual word
(the Pi_A lists). The gate passes at 5 or fewer; a retry is made when the text is empty, under 5 words or fails the
gate. The kept answer maximises +5 label present, +3 at least 5 words, minus the penalty, plus the visual words.

### 5.3 M5
First the role's sentences are cleaned (5.3.2). The cleaned answer is then accepted when all of 5.3.1 hold; otherwise
the backup prompt is tried once.

#### 5.3.1 Acceptance
- 12-500 characters, ASCII only, at least 5 words; at most 50 words for M5_A and 35 for M5_B and M5_C
- M5_A: no label field (cat:, category:, sev:, severity:, conf:, confidence:)
- none of: ch11, chapter, introduction, machine learning, data set, dataset, ###, http://, https://, 2020-, 2021-,
  2022-, 2023-, chia -, lassig, regular expression, regex, python, match a string, chef, renowned, chengcheng,
  amaciyla, amacyla, amac, reevaluation, evaluate results, implement changes; none of: sentence 1, sentence 2, first
  sentence, second sentence, output only, analysis:; "chia" fewer than 3 times; no "injuries or injuries" or
  "risk of risk"
- does not end with a dangling word: or, and, with, to, of, by, in, for, into, onto, from, at, on, near, under, over,
  around, through, the, a, an, is, are, was, were, indicating, suggesting, this, that, potential (also before a
  final period, comma or colon)
- starts with a letter; first sentence at least 4 words (M5_A: the second sentence too, when present)
- also not flagged by these markers: ###, chapter, introduction, machine learning, data set, dataset, chakra, venus,
  python, regex, http://, https://, 2020-, chia -, lassig, evaluate results, implement changes, chef, renowned,
  chengcheng, amac, regular expression, match a string, in this chapter, we will discuss
- M5_A only: contains a disaster-related word: damage, destroy, collaps, block, crack, break, broke, rock, boulder,
  debris, rubble, landslide, mudslide, flood, erosion, eroded, erode, fire, burn, blaze, earthquake, quake,
  hurricane, storm, cyclone, typhoon, harm, injur, danger, risk, hazard, threat, fatal, rescue, evacuate, emergency,
  respond, road, bridge, building, structure, infrastructure, fallen, scatter, swept, buried, trapped, unstable,
  unsafe, severe, massive, cliff, slope, hillside, trench, sinkhole, vehicle, car, truck

#### 5.3.2 Sentence selection
- M5_A keeps its first two clean sentences, each cut to 25 words. If only one sentence is returned, the firmware
  appends "Responders should secure the area and perform a rapid field safety assessment."
- M5_B and M5_C keep the highest-scoring sentence among the first four that starts with a letter and has 4-25 words
  (cut to 25 words, dangling words removed).
  - M5_B: +6 if it contains a damage word (landslide, rockfall, debris, damag, collapse, collapsed, failure, failed,
    erosion, eroded, crack, cracked, blocked, obstruct, buried, fallen, mud, disruption, displaced); +4 an anchor
    (road, highway, lane, track, rail, slope, cliff, hillside, bridge, shoreline, coast, beach, debris, mud, rock,
    boulder, barrier, vehicle, building, house); +3 road, bridge, path, lane, barrier or shore; -3 speculation
    (" may ", " could ", " might ", " likely ", " appears ", " seem", " suggest", possibly, probably, maybe);
    -4 scenic wording (clear sky, cloudy sky, beautiful, scenic, picturesque, lush, volcanic landscape, landscape
    with, weather-beaten, green hills); -5 boiler-plate (affected terrain, route section, visible damaged zone, nearby
    routes, route continuity, route corridor, damaged terrain, affected route, physical disruption of the visible
    route, "(a large) section of the cliff has collapsed, exposing the rocky surface").
  - M5_C: +8 a human-risk word (risk, danger, hazard, unsafe, injury, injured, hurt, fatal, evacuat, rescue,
    trapped, bury, blocked access, access, people, pedestrian, nearby residents, public safety, life-threatening);
    +4 evacuat or rescue; +4 injury, danger or hazard; +3 collapse or collapsed; +2 rockfall, boulder or debris;
    +2 cliff, slope or hillside; +2 blocked, buried or trapped; -2 shoreline or partially submerged; -2 vegetation
    or background.

### 5.4 Jetson
The `damage` and `countermeasure` lines keep their first sentence. Leading meta phrases are stripped from the
countermeasure ("The immediate countermeasure should be to ...", "Immediate action is required to ...", "The first
step is to ...", "To prevent further damage, ...", "Action:", "Immediately, ...", "First, ..." and similar).
One repair call is made when the countermeasure:
- is empty or has fewer than 6 words;
- leaks the prompt (starts with "unstable structure -", "piled debris or rubble -", "trapped person -",
  "exposed line -", "fire or smoke -", "water - pump/drain/divert", or lists three of shore/brace/stabilize or
  clear/remove/extract);
- is a fragment (first word ends in -ing, other than "being");
- contains "should";
- contains a past-tense response verb (began, organized, arrived, evacuated, started, conducted, rushed, mobilized,
  cleared, worked, dispatched, surveyed, searched, boarded, responded, coordinated, initiated, completed, managed,
  deployed, secured, removed, assessed, established, cordoned);
- is descriptive ("is/are + -ing" within the first 8 words);
- narrates in the simple present (one of is, are, was, were, uses, takes, operates, holds, applies, performs,
  places, moves, pulls, pushes, lifts, carries, searches, sifts, climbs, walks, stands, sits, drives, digs, clears,
  checks, examines, assesses within the first 10 words, unless should, must, need or needs appears).
The less broken of the original and the repaired sentence is kept, ranked empty < too short < fragment < prompt leak
< descriptive < simple-present narrative < past tense < should; the repair wins ties. The Jetson answer is
"damage countermeasure".

## 6. Routing features

Stage 1 (computed on M5_A from the three M5 answers; 7 features):

| Feature | Definition |
|---|---|
| m5a_cat | M5_A category (categorical, one-hot) |
| m5a_sev | M5_A severity (0-2) |
| m5_conf_mean | mean of the M5_A, M5_B, M5_C confidences |
| m5_conf_min | minimum of the three confidences |
| m5_sev_range | maximum minus minimum of the three severities |
| m5_cat_agree | share of the three categories equal to the most common one (1, 2/3 or 1/3) |
| m5_sev_agree | the same for the three severities |

Stage 2 (computed on Pi_A; the 7 above plus 9):

| Feature | Definition |
|---|---|
| pia_cat, pia_sev | Pi_A category and severity |
| pi_conf_mean, pi_conf_min | mean and minimum of the Pi_A and Pi_B confidences |
| pi_cat_match, pi_sev_match | 1 if Pi_A and Pi_B give the same category (severity), else 0 |
| pia_min_logprob | minimum token log-probability of Pi_A's answer (-5 if unavailable) |
| m5a_pia_cat_match | 1 if M5_A and Pi_A give the same category |
| m5a_pia_sev_diff | absolute difference of the M5_A and Pi_A severities |

Numerical features are standardised and the category features one-hot encoded inside the scikit-learn pipeline;
both trees are depth 4, minimum 5 samples per leaf, balanced class weights, random_state 42.

## 7. The deployed decision rules

Stage 1 (true = terminate at M5):
- If M5_A severity is 0 or 1:
  - M5_A category other_disaster or hurricane: terminate if m5_conf_mean <= 1.5, otherwise escalate.
  - any other category: escalate.
- If M5_A severity is 2:
  - category not other_disaster: terminate.
  - other_disaster: terminate only if m5_conf_min = 1 and all three severities differ (m5_sev_agree = 1/3).

Stage 2 (true = terminate at Pi):
- Pi_A and Pi_B categories differ:
  - Pi_A category not hurricane: escalate if m5_conf_min = 1; otherwise terminate only if Pi_A says earthquake.
  - Pi_A says hurricane: if M5_A category is not other_disaster, terminate only if Pi_A severity is 2; otherwise
    terminate only if pia_min_logprob > -2.2132.
- Pi_A and Pi_B categories agree:
  - Pi_A severity 0 or 1: category other_disaster: terminate only if pia_min_logprob <= -2.2607; landslide:
    terminate; any other: escalate.
  - Pi_A severity 2: category not other_disaster: terminate; other_disaster: terminate only if m5_conf_min >= 2.

## 8. Why agents running the same model answer differently

Agents of one tier receive different role prompts (Section 2): M5_A damage and action, M5_B structural destruction,
M5_C harm to people; Pi_A situation and action, Pi_B a damage description. Pi agents also sample at temperature 0.2.

| | Profiling (5,526) | Held-out rerun (600) |
|---|---|---|
| M5 categories: all three agree | 41.0% | 40.2% |
| two of three agree | 49.6% | 51.0% |
| all differ | 9.4% | 8.8% |
| Pi_A and Pi_B categories differ | 29.6% (of 3,142 escalated by the reference) | 33.3% (of the 297 escalated) |

M5_A's category is correct in 82.3% / 81.7% of images when all three M5 agents agree, 56.4% / 49.7% when two agree and
34.1% / 22.6% when all differ (profiling / rerun). Pi_A's category is correct in 81.9% / 85.4% when the two Pi agents
agree and 41.7% / 49.5% when they differ.
