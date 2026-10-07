# Experiment log

Running record of decisions and results. Append, never rewrite. Convert relative dates to
absolute. Every figure copied from the literature must be verified against its primary source
before it is relied on.

## 2026-10-06 — Project stood up

- Scaffolded the full package (`src/caesd/`): backbone + blocks A/B/C/D, metrics, calibration,
  data, train, eval, baselines. Environment: `uv` + Python 3.12, CUDA PyTorch on an RTX 3090 Ti.
- Metric suite built and unit-tested against known-answer identities (Cllr=1 for zero-info LLRs,
  EER=0 for separable scores, minCllr≤Cllr). Full numeric cross-check against BOSARIS / the
  ASVspoof5 eval package is still **to do** before any headline number is published.

## 2026-10-06 — Data pipeline + baselines

- **AASIST** cloned to `F:/datasets/audio/aasist`; pretrained weights load and score on the
  RTX 3090 Ti under torch 2.6. Bona-fide score = `logit[:,1]`; clips tiled to 64600 samples.
- **Sources verified** (web search, this date): In-the-Wild = HF `mueller91/In-The-Wild`
  (one 8.16 GB zip, English, bona/spoof); IndicSynth = HF `vdivyasharma/IndicSynth` (ACL 2025,
  845 GB total, 12 Indian languages, synthetic only — we pull a lean few-shard subset).
- **Downloading** (staged lean): In-the-Wild full + IndicSynth Telugu/Hindi/Tamil shards.
- Phase 1 measurement runner added (`eval/measure.py`, `scripts/phase1_measure.py`).
- Pipeline verified on synthetic scores end-to-end; 26 unit tests pass; ruff clean.

## Phase 0 — go / no-go (pending real data)

Pipeline is ready. Once In-the-Wild + the IndicSynth subset finish downloading:
1. `scripts/make_manifests.py` (or prepare step) → clean manifest, disjoint by speaker.
2. `scripts/build_condition_grid.py` → codec grid + condition labels.
3. `scripts/score_grid.py` → AASIST scores parquet.
4. `scripts/phase0_falsify.py` → the verdict below.

### First run — 2026-10-06 (In-the-Wild base, pretrained AASIST, 500 clips × 6 codec cells)

Per-condition AASIST discrimination (higher score = bona fide):

| condition | EER | minDCF |
| --- | --- | --- |
| clean | 0.424 | 0.932 |
| mp3_16k | 0.468 | 0.960 |
| opus_16k | 0.456 | 0.992 |
| amr_16k | 0.488 | 0.972 |
| speex8k_16k | 0.472 | 0.984 |
| aac_16k | 0.484 | 0.976 |

Scores vary (500 unique) and separate in the correct direction (bona mean −0.96 > spoof −1.78);
sign-flip makes EER worse (0.576). So the pipeline is wired correctly.

**Decision: NO-GO for this data configuration — but this is a corpus-choice result, not a
falsification of the premise.** Clean EER 0.424 faithfully reproduces the known collapse
(proposal P1: AASIST 1.89% in-domain → 41.85% on ML-ITW). On In-the-Wild, AASIST's
**discrimination is already destroyed** (minDCF ≈ 0.93 even clean), so there is no
"good discrimination + bad calibration" regime on top of which to measure a calibration gap
(held-out codec gap actDCF−minDCF = 0.008). The falsification harness correctly reports NO-GO.

**Conclusion.** The calibration-gap premise must be tested on a base corpus where AASIST
discriminates well — ASVspoof 2019 LA (AASIST's in-domain set, ~1% EER) or ASVspoof 5 — with the
condition grid applied there. In-the-Wild is a held-out *collapse* benchmark, not a valid base
for isolating calibration from discrimination. Next data item: **ASVspoof 2019 LA**.

Two results already banked from this run:
1. Reproduced the AASIST in-the-wild collapse (clean EER 0.42 ≈ literature 0.4185).
2. The end-to-end Phase-0 pipeline (prepare → grid → score → falsify) runs on real data.

### 2026-10-06 — ASVspoof 5 base attempt

- ASVspoof 5 protocols confound codec with label (bona fide clean, spoof always AC-coded) in
  BOTH train and dev; there are no clean spoof clips.
- Pretrained AASIST on ASVspoof 5 dev (800-clip balanced subset): **EER 0.3575, minDCF 0.985** —
  another collapse (out-of-domain; AASIST was trained on ASVspoof 2019 LA).
- Conclusion: the cheap "pretrained AASIST, no training" Phase 0 is not viable on ASVspoof 5.
  To stay on ASVspoof 5 we must TRAIN a countermeasure (the redesign), with our codec grid
  applied uniformly to both classes so the model cannot exploit the codec↔label shortcut.

### 2026-10-06 — Redesigned Phase 0 on ASVspoof 5: **GO**

Trained a quick countermeasure (XLS-R-300M frozen + LoRA, 3.35M params, RawBoost on both
classes) on 4,591 ASVspoof 5 train clips, 6 epochs (loss 2.19 → 0.036). Evaluated on a
speaker-disjoint dev subset (800 clips) passed through our frozen codec grid.

Per-condition, global calibration fit on clean dev, **balanced prior p=0.5** (test split):

| condition | minDCF | actDCF | calib gap | far@fixed-thresh |
| --- | --- | --- | --- | --- |
| clean | 0.323 | 0.349 | 0.025 | 0.073 |
| opus_16k | 0.274 | 0.357 | 0.083 | 0.046 |
| mp3_16k | 0.397 | 0.465 | 0.068 | 0.174 |
| amr_16k | 0.291 | 0.502 | 0.210 | 0.009 |
| **speex8k_16k (held out)** | 0.288 | 0.634 | **0.346** | 0.000 |
| aac_16k | 0.311 | 0.661 | 0.349 | 0.014 |

**Decision: GO.** Held-out (speex8k) calibration gap 0.346 (≥ 0.10); discrimination (minDCF)
stays ~0.3 across codecs while actDCF blows up — calibration, not discrimination, breaks under
codec shift. The per-condition lookup table closes **70% of the gap on seen codecs** (a
competitive baseline, the reviewer's objection) but **0% on the held-out codec** — so the
condition-dependent hypernetwork (block C) is justified exactly where the table cannot reach.

Caveats (honest):
- At ASVspoof 5's strict prior p=0.05 the quick CM saturates (minDCF=1.0, EER ≈ 0.17): too weak
  to have an operating point there. A stronger detector (full Phase 2: more data/epochs) is
  needed to operate at 0.05, where the calibration gap will matter even more.
- The RawBoost-on-both-classes training mitigates but may not fully remove the ASVspoof 5
  codec↔label confound; minDCF staying ~stable across codecs is evidence the gap is calibration,
  not a codec shortcut.

**Next:** Phase 2 — build blocks A + C (condition encoder + calibration hypernetwork) and show
condition-dependent calibration closes the held-out gap the global/lookup baselines cannot.

### 2026-10-06 — Phase 2: condition-dependent calibration closes the held-out gap

Trained the condition-aware CM on a codec grid of ASVspoof 5 train (clean/mp3/opus/amr/aac;
**speex8k held out**), with block A's channel head supervised on the codec (codec loss
1.49 → 1.04 over 5 codecs vs random 1.61) so the condition vector is meaningful and block C can
generate per-condition calibration. Evaluated on the ASVspoof 5 dev grid (all 6 codecs).

Calibration gap (actDCF − minDCF, p=0.5): global logistic vs block-C hypernetwork.

| condition | minDCF | gap global | gap block C |
| --- | --- | --- | --- |
| clean | 0.309 | 0.005 | 0.075 |
| mp3_16k | 0.315 | 0.005 | 0.038 |
| opus_16k | 0.318 | 0.014 | 0.058 |
| amr_16k | 0.350 | 0.072 | 0.015 |
| aac_16k | 0.321 | 0.157 | 0.005 |
| **speex8k_16k (held out)** | 0.351 | **0.102** | **0.004** |

**Result.** On the held-out codec the condition-dependent hypernetwork cuts the calibration gap
**0.102 → 0.004 (97%)**, and most on the heavily-degraded codecs (aac 0.157 → 0.005). Because the
Phase-0 lookup table covers 0% of the held-out cell, global calibration *is* the best fixed
baseline there — and block C beats it by interpolating in condition space. This is the method
paper's core contrast (condition-dependent vs global calibration).

Honest caveats:
- **Cllr regression (important):** by Cllr — overall LLR quality, the forensic metric — block C
  is WORSE than global everywhere (pooled Cllr 0.941 vs 0.670; e.g. clean 1.087 vs 0.645). It
  shrinks the actDCF gap at the p=0.5 operating point on shifted codecs but degrades the LLRs in
  general. So the method is promising on the specific operating-point claim but is NOT yet a
  strict improvement. Fixing the Cllr regression (clean-anchor regulariser, more data/epochs,
  possibly training the hypernetwork to minimise Cllr across conditions rather than pooled) is
  the top Phase-2 follow-up.
- Trade-off: block C is worse than global on clean/mild codecs at the operating point too
  (clean gap 0.005 → 0.075). It redistributes calibration toward the conditions where shift hurts.
- Discrimination is modest: EER ≈ 0.22, minDCF ≈ 0.31–0.35 at p=0.5.
- Discrimination is still modest (minDCF ≈ 0.31–0.35 at p=0.5); the calibration claim is
  orthogonal to and holds despite that. A stronger detector (more data/epochs, all train shards)
  is the Phase-2 scaling step.
- Lean scale (2k train base, 800 dev); confirm with bootstrap CIs and more data before publishing.

**Still to do in Phase 2:** full ablation ladder (temperature scaling, +FiLM-only, +SAE, +gate),
bootstrap CIs, and the five generalization mechanisms measured on the held-out eval corpora.

### 2026-10-07 — Published baselines under our protocol (same dev grid)

Two published detectors run faithfully (plug-and-play, no head-guessing) on the exact ASVspoof 5
dev grid, EER by condition (test split):

| condition | AASIST (2019 LA) | HF wav2vec2 deepfake | our CM (in-domain) |
| --- | --- | --- | --- |
| clean | 0.329 | 0.438 | 0.259 |
| mp3_16k | 0.366 | 0.443 | 0.223 |
| opus_16k | 0.351 | 0.453 | 0.237 |
| amr_16k | 0.417 | 0.397 | 0.220 |
| aac_16k | 0.397 | 0.484 | 0.220 |
| speex8k_16k (held out) | 0.402 | 0.383 | 0.220 |
| **POOLED** | **0.383** | **0.425** | **0.225** |

Both published baselines were trained out-of-domain (AASIST on ASVspoof 2019 LA; the HF wav2vec2
model on other deepfake data) and collapse on ASVspoof 5 — the P1 generalization problem. Our
in-domain CM beats both but is still far from reported ASVspoof 5 SOTA (EER ~3.4%, Cllr ~0.19).

Literature SOTA (proposal §9, unverified 2026 preprints, different protocols — NOT reproduced
here): ASVspoof 2021 LA 0.87% EER; ASVspoof 5 best fused EER 3.42% / Cllr 0.19. The authoritative
SSL SOTA checkpoint (nii-yamagishilab/xls-r-2b-anti-deepfake) is a backbone-only release
(Wav2Vec2ForPreTraining, 8.65 GB) needing its official downstream head; running it faithfully is
a scoped follow-up, not hacked here.

### 2026-10-07 — Scale-up (a): EER plateaus, and the calibration win does NOT reproduce

Trained a 2.5x-larger condition-aware CM (22,955 train clips, 8 epochs, batch 32; same grid with
speex8k held out) and scored the same dev grid.

- **EER did not improve:** pooled 0.225 (2k) → 0.226 (5k); per-condition flat ~0.22. Discrimination
  is plateaued at this configuration (frozen XLS-R + small LoRA, this recipe). Reaching SOTA
  (~3-5% EER) needs much more than a lean scale-up — full train set, more adapter capacity, and
  tuning.
- **Block-C calibration REGRESSED** — the earlier 97% held-out improvement did NOT reproduce:

  | held-out speex8k gap | small CM (2k) | scaled CM (5k) |
  | --- | --- | --- |
  | global calibration | 0.102 | 0.079 |
  | block C (hypernetwork) | **0.004** | **0.152 (worse)** |

  On the scaled run block C is worse than global on every condition (e.g. mp3 0.000 → 0.280,
  clean 0.043 → 0.231). The hypernetwork overfit its calibration on the train score distribution
  (train cllr loss fell to 0.084) and is miscalibrated on dev.

**Conclusion (important, cautionary):** the Phase-2 calibration result is **training-unstable** —
the earlier small-model win was not robust and reversed under scaling. Before any calibration
claim is made, block C needs stabilising: a clean-anchor regulariser, a Cllr-across-conditions
objective (not pooled), early stopping on a held-out calibration criterion, and bootstrap CIs
with several seeds to measure variance. The actDCF/Cllr caveats compound this. **This supersedes
the optimistic read of the first Phase-2 run.**

### 2026-10-07 — Root-cause diagnosis of the 22% EER plateau (supersedes parts of the above)

Diagnostics on the `phase2_condaware_big` checkpoint and on frozen XLS-R-300M features. Dev numbers
are on the 800-clip balanced ASVspoof 5 dev subset (4 s crop), so each per-attack cell has 42-63
spoof clips: treat them as indicative, not final. Single seed.

1. **The model memorises its training attacks.** EER on seen train clips 0.000; on held-out
   speakers with seen attacks A01-A08 (train-partition dev split, all 5 codecs) 0.000-0.005; on
   ASVspoof 5 dev (unseen attacks A09-A16) 0.217.
2. **Two attacks cause the plateau.** Per-attack dev EER (clean): A09 0.046, A10 0.081, A11 0.018,
   **A12 0.777, A13 0.509**, A14 0.000, A15 0.091, A16 0.128. Without A12/A13 the EER is 0.065.
3. **Our training makes the backbone worse.** A logistic-regression probe on mean-pooled *frozen*
   XLS-R features (no LoRA, trained on the same 4,591 clean clips) gives dev EER 0.010 (layer 20),
   0.010-0.018 for layers 10-22, 0.022 for the last layer; A12 <= 0.02 and A13 <= 0.04 for layers
   19-22. Which component of our recipe causes the loss (LoRA, FiLM, joint Cllr loss, codec
   supervision, epochs) is NOT yet isolated.
4. **The "codec<->label confound" was a misreading.** In `ASVspoof5.train.tsv` and
   `ASVspoof5.dev.track_1.tsv` the three codec columns are `-` for every row. Column 7 (AC1/AC2/AC3)
   is a separate tag on spoof rows, not a codec. Train and dev have no codec confound.
5. **speex8k is not a held-out condition.** Block A's channel head labels 679/800 speex clips as
   AMR (both 8 kHz narrowband). It also labels 723/800 Opus clips as clean.
6. **The Phase-0 GO does not hold for a strong detector.** With the layer-20 probe (clean-trained;
   EER 0.010 clean, 0.022 speex8k) and global calibration fit on clean dev, the p=0.5 calibration
   gap is 0.005-0.018 on every codec except AAC (0.115); held-out speex8k gap 0.018 (Phase 0
   measured 0.346 with the weak CM). Cllr - minCllr: clean 0.023, speex8k 0.027, AAC 0.196. At
   p=0.05 the gaps are larger (up to 0.43) but n=413 per cell is too small to trust them.
7. **Scale.** We train on 4,591 unique clips = 2.5% of the 182,357-clip train partition (the
   "22,955" figure is the same clips x 5 codecs).

Consequence: replace the trained CM with a frozen-feature baseline first, re-run Phase 0 on a
large dev sample at p=0.05, and decide go/no-go on the calibration premise from that.

### 2026-10-07 — Rebuilt pipeline, interim results on partial data (branch `sota-baseline`)

Setup. Frozen backbone (XLS-R-300M, WavLM-Large; all 25 hidden layers time-pooled, cached in fp16),
learned layer weights, MLP head (0.26M params). Train: ASVspoof 5 train tars aa+ab (73,000 of 182,357
clips, attacks A01-A08), clean view + one simulated codec per clip. Validation and calibration: dev
tar aa (47,400 of 140,950 clips, A09-A16), split by speaker into `select` (23,347) and `calib`
(24,053). Clips up to 16 s. Single seed (1337). The eval set is not used yet.
Claim served: a frozen SSL backbone generalises to unseen attacks; block C is judged against
pooled-global and lookup calibration.

**Detection (dev `select`, unseen attacks A09-A16; EER).**

| run | clean | simulated codec | A12 / A13 (clean) |
| --- | --- | --- | --- |
| old trained CM (2026-10-06, 800-clip dev subset, not comparable in size) | 0.217 | – | 0.777 / 0.509 |
| frozen XLS-R head | 0.0082 | 0.0269 | 0.003 / 0.017 |
| frozen WavLM head | 0.0159 | 0.0315 | 0.011 / 0.039 |
| frozen XLS-R, clean training only | 0.0030 | 0.0246 | 0.001 / 0.005 |
| frozen XLS-R + block A / FiLM | 0.0074 | 0.0323 | 0.013 / 0.010 |
| fusion XLS-R + WavLM (weights fit on `calib`; v1 / v2 heads) | 0.0054 / 0.0060 | 0.0203 / 0.0199 (mean over the 8 codecs) | – |

- The root cause is confirmed: a frozen backbone with a small head removes the A12/A13 failure.
- FiLM / block A adds nothing. Codec augmentation in training did not help here (clean-only
  training is no worse on the codec view). Both are single-seed, single-run differences of
  <1 point on ~23k clips: not established.
- The head overfits: train BCE 0.0003 at epoch 29, best dev epoch 3, dev EER then fluctuates
  0.9-2.1%. A rerun with the full 30-epoch one-cycle schedule (v2) gave identical best-epoch
  numbers for XLS-R, so the schedule was not the problem. (I had suspected it; that was wrong.)
  Regularisation or more data is the lever, not the schedule.
- Not comparable with published results: this is dev, not eval, and 34% of dev.

**Calibration (fused scores, Cllr; lower is better; clip-weighted mean over conditions).**

| | global_clean | global_pooled | lookup_oracle | lookup_pred | block C |
| --- | --- | --- | --- | --- | --- |
| dev, seen codecs | 0.074 | 0.063 | 0.053 | 0.066 | **0.041** |
| dev, leave-one-codec-out | 0.124 | 0.091 | 0.091 | 0.095 | **0.209** (v2: 0.517) |

- Block C wins on seen codecs and loses badly on unseen ones (speex_wb Cllr 0.69 vs 0.11 for
  pooled), which is exactly the case the method claims. Its leave-one-out result varies from
  0.209 to 0.517 between two runs of the same study (different codec-classifier training).
- **Block C is not a pure calibrator.** Its LLR changes the EER versus the raw score (clean
  0.5% -> 0.3%, mean -0.4 points on seen codecs, +0.5 on unseen). A monotone score map cannot do
  that. The codec posterior it reads comes from a classifier trained on the same features, so it
  carries class information (the codec classifier is also weak on clean: 43% accuracy, 37% on
  m4a). The seen-codec "win" is therefore not evidence of better calibration.
- Pooled-global calibration is within 0.01-0.02 Cllr of the PAV floor (minCllr) on most codecs.
  The remaining gap is at the strict prior: actDCF(p=0.05), speex_nb 0.270 (pooled) vs 0.192
  (oracle lookup) vs minDCF 0.176; this is closed by per-condition calibration with the TRUE
  codec, and only partly (0.192 vs 0.176).

**Decision on the calibration premise: weakly supported, not established.** Pooled calibration
removes most of the loss; per-condition calibration helps at p=0.05 on seen codecs when the codec
is known. Block C as built does not beat pooled calibration on unseen codecs. Do not claim it.

Open, in order: (1) eval-set scoring (download: 3 of 10 tars; eval extraction ~7 h GPU);
(2) a leak-free condition input for block C, or drop it; (3) more train data and regularisation
for the head; (4) LoRA / full fine-tune on full data; (5) multiple seeds and CIs on EER.
