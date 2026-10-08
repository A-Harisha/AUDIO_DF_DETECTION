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

### 2026-10-08 — First ASVspoof 5 eval results; minCllr bug fixed

**Metric bug (affects every minCllr above this entry).** `min_cllr` took the logit of the PAV
posterior without removing the empirical prior log-odds. That is only correct for balanced classes.
Check: equal-variance Gaussians, true LLR = 4*score, 5% bona fide: Cllr(true LLR) 0.078 but
min_cllr 0.182 (a floor above an achievable value). Fixed in `metrics/cllr.py`; test added. All sets
here are ~20% bona fide, so every earlier minCllr is too high (eval pooled 0.267 -> 0.213). Read the
earlier statement "pooled calibration is within 0.01-0.02 of the floor" as: dev seen codecs, pooled
0.062 vs corrected floor 0.046. EER, minDCF, actDCF and Cllr are not affected.

**Setup.** As the previous entry: frozen backbones, heads trained on 40% of train (A01-A08), selected
on dev `select` (34% of dev). Eval: all 680,774 track-1 clips (A17-A32, codecs C01-C11), up to 16 s,
scored once. No eval number was used for any decision. Single seed. Run dirs: `frozen_xlsr_final`,
`frozen_wavlm_final`, `cal_eval` under `experiments/runs_partial/`.

**Detection on eval.**

| system | EER | minDCF (p=0.05) | minCllr | EER, clean clips only |
| --- | --- | --- | --- | --- |
| frozen XLS-R head | 0.0767 | 0.453 | 0.279 | 0.0431 |
| frozen WavLM head | **0.0440** | **0.255** | **0.162** | 0.0147 |
| fusion (weights fit on dev calib: 0.63 XLS-R, 0.15 WavLM) | 0.0579 | 0.311 | 0.213 | 0.0230 |

- Dev and eval disagree on the better backbone: dev prefers XLS-R (0.8% vs 1.6% clean), eval
  prefers WavLM (4.4% vs 7.7%). The dev-fit fusion therefore down-weights the better system and
  is worse than WavLM alone. Dev (A09-A16, no codecs) is a poor proxy for eval.
- Reference only, UNVERIFIED and from a different training setup: proposal section 9 lists a best
  fused ASVspoof 5 result of EER 3.42% / Cllr 0.19 [CITATION NEEDED]. We are not at that level.
- Per codec (fused, EER): clean 2.3%, C05 2.5%, C06 2.8%, C11 2.8%, C02 3.3%, C01 3.7%, C03 4.4%,
  C09 4.6%, C10 5.9%, C08 6.2%, **C04 11.6%, C07 14.0%**. C04/C07 are the EnCodec conditions, never
  simulated in training. C11 (devices) is also unsimulated and is not hard.

**Calibration on eval (fused scores, all fits on dev calib; Cllr).**

| | global_clean | global_pooled | lookup_oracle | lookup_pred | block C | floor (minCllr) |
| --- | --- | --- | --- | --- | --- | --- |
| eval pooled | 0.310 | 0.250 | 0.243 | 0.245 | **0.862** | 0.213 |
| C04 (unseen) | 0.976 | 0.645 | 0.645 | 0.581 | **2.965** | 0.370 |
| C07 (unseen) | 1.475 | 0.962 | 0.962 | 0.878 | **6.263** | 0.426 |
| C11 (unseen) | 0.146 | 0.190 | 0.190 | 0.242 | 0.248 | 0.105 |
| clean | 0.111 | 0.147 | 0.111 | 0.144 | 0.179 | 0.086 |

actDCF (p=0.05), eval pooled: global_clean 0.334, global_pooled 0.368, lookup_oracle 0.374,
lookup_pred 0.399, block C 0.539; minDCF 0.311.

- **The premise holds on the unseen neural codecs:** on C04/C07 every calibration is far from
  the floor (0.645 vs 0.370; 0.962 vs 0.426). Calibration does break under an unseen codec.
- **No condition-aware method fixes it.** The lookup table has no cell for C04/C07/C11. The
  predicted-codec lookup gains 0.06-0.08 Cllr on C04/C07 and loses on C11 and clean.
- **Block C fails on eval:** worst on every condition, catastrophic on the unseen codecs
  (paired 95% CI of Cllr difference to global_pooled, eval pooled: [+0.591, +0.631]). On dev
  seen codecs its Cllr (0.039) is below the corrected floor (0.046), which confirms that it adds
  discrimination through the codec posterior instead of calibrating. **Block C as built is
  rejected.** Do not use the earlier "97%" or the dev seen-codec numbers as support.

Open, in order: (1) select and fuse on data that looks like eval (codec-coded dev), not clean
dev; (2) simulate EnCodec; (3) regularise the head and use all of train; (4) LoRA / fine-tune
with unseen-attack validation; (5) seeds and CIs; (6) a calibration method that degrades
gracefully on unseen conditions (shrink toward LLR = 0), judged on C04/C07.

### 2026-10-08 — Block C rebuilt: leak-free condition input, bounded extrapolation, held-out stopping

Implementation only. No claim is made or retracted here: the redesign changes what block C *can*
do, and the numbers that decide its fate still need the real descriptor pass and a rerun of the
calibration study. Verified end to end on a synthetic score/descriptor set, which exercises the
wiring and the two guarantees but is not an experiment.

The three recorded defects (2026-10-07) and what now prevents each:

1. **Leakage.** v1's condition vector was the codec log-posterior of a `Head` trained on the
   detector's own frozen SSL features, so (a, b) carried class evidence and the LLR moved the EER
   (clean 0.5% -> 0.3%). Block C now reads `channel_descriptor` (`data/condition_grid.py`): the
   long-term average spectrum in 24 log-spaced bands plus the 99% rolloff edge, from the waveform
   only. It is exactly loudness-invariant (relative noise floor, interpolated bands) and it tracks
   bandwidth (clean vs 8 kHz round trip: L2 distance 10.6; clean vs mild quantisation noise: 0.01).
   `models.label_probe_eer` is now a reported **precondition**: a logistic probe on the descriptor
   alone must give ~0.5 EER. Anything lower means the descriptor leaks and no calibration claim
   here is valid. This is the test that would have caught v1.
2. **Extrapolation.** v1's unconstrained MLP mapped an unseen condition anywhere, so its
   leave-one-codec-out Cllr swung 0.209 -> 0.517 between two runs of the same study. The residual
   is now a kernel average over K=8 learned anchors against a **null anchor that carries no
   residual**: as z leaves the anchors' span every kernel weight vanishes and the map becomes
   exactly `softplus(a0+0.5413)*s + b0`, one global calibration. On the synthetic leave-one-codec-out
   the null weight is 1.00 on all 8 held-out codecs, i.e. the guarantee holds by construction.
   `null_weight` is reported per condition: it says when block C acted at all.
   - The first version of this fix was **not enough, and the synthetic run caught it**: the null
     anchor guarantees *a* global calibration, not a good one. With (a0, b0) optimised only inside
     the full map, an unseen codec reached the fallback at null weight 1.00 and still scored Cllr
     1.39 against 0.56 for pooled. The fix: the global map gets its own term in the loss
     (`anchor_weight`) **and** is detached from the residual path, so only that term moves it and
     the two objectives cannot fight. Unseen-codec Cllr went 1.39 -> 1.02 (own term) -> 0.87
     (detached). A unit test asserts the fallback stays within 1.15x of pooled calibration.
   - Consequence worth stating plainly: the fallback minimises Cllr **averaged over conditions**,
     which is the right uniform prior for a condition never seen, while `global_pooled` optimises
     a pooled class-balanced logistic fit. The two objectives differ, so block C can lose to
     pooled on an individual condition even at null weight 1.00. On the synthetic set it beat
     pooled on 6 of 8 held-out codecs and lost on 2 (opus_wb 0.87 vs 0.56, amr_wb 0.68 vs 0.55).
     This is reported, not tuned away: the synthetic shifts are +-4 and far larger than real ones.
3. **Overfitting.** v1 ran a fixed 2000 full-batch steps on all of the calibration data. The calib
   split is now halved by speaker (`assign_split_by`, 0.75 / 0.25) into `fit` / `stop`, and the step
   is selected on the held-out `stop` half; the best state is restored. `+hyper_steps` is a knob.

Also: block C is run over `+hyper_seeds` (default 3) and the per-seed Cllr / actDCF mean and
standard deviation are reported per condition, so the run-to-run variance that reversed the
Phase-2 result is measured rather than assumed.

**What the synthetic end-to-end run verifies** (wiring and guarantees only, not performance):
the precondition passes at 0.505; `eer/hypernet - eer` is **+0.0000 on every condition** in both
studies, i.e. block C is now a genuine per-condition calibrator and no longer moves discrimination
the way v1 did; null weight is 1.00 on all 8 held-out codecs; and the per-seed Cllr standard
deviation on leave-one-codec-out is **0.000**, because an unseen condition returns the
seed-independent fallback. v1's headline instability cannot recur in this architecture.

**Net effect on the code.** `calibrate.py` lost `fit_condition_classifier` and `condition_vector`,
and no longer loads SSL features at all: the leave-one-codec-out loop no longer retrains a codec
classifier per codec, which deletes the exact variance source behind the 0.209 / 0.517 swing and
cuts the study's GPU time to zero. The `lookup_pred` baseline keeps its predicted codec, now from
a logistic classifier on the 25-dim descriptor. Descriptors are joined on `utt`, so nothing assumes
a shared row order between the score files and the feature cache.

`extract_features.py` writes `<features>/descriptor/<partition>_<view>.npy` [N, 25] float32 in the
same pass over the audio, so the pending eval extraction gets it for free. `+descriptor_only=true`
is the CPU-only pass for the sets already extracted: it never loads a backbone. Measured at
1500 dev clips in 27 s with 4 workers, so the full train+dev descriptor pass is minutes, not hours.

**Note on ffmpeg.** `+view=codec` fails with `FileNotFoundError` unless ffmpeg is on PATH. It is
installed but not on the default PATH; `scripts/chain_eval.sh` exports the WinGet location, and any
descriptor or extraction run must do the same.

**Relation to the eval entry above.** That entry rejects block C as built, with a far stronger case
than the dev-only picture: eval pooled Cllr 0.862 against 0.250 for `global_pooled`, and 2.965 /
6.263 on the unseen neural codecs C04 / C07. Its diagnosis is the same leakage this rebuild removes
-- "its Cllr (0.039) is below the corrected floor (0.046), which confirms that it adds
discrimination through the codec posterior instead of calibrating". The rebuild is therefore the
answer to that entry's open item (6), "a calibration method that degrades gracefully on unseen
conditions", with one difference: it shrinks toward the best **global calibration** rather than
toward LLR = 0, which is the stronger baseline of the two. C04 / C07 are now the test that decides
it, and the null weight on those conditions says whether the shrinkage actually engaged.

Open, in order (this supersedes nothing above; it is the block C thread only): (1) descriptor pass
for train/dev clean+codec and eval clean, then rerun `calibrate.py +eval=true` and read
`descriptor_label_probe_eer` before any other number; (2) judge block C on C04 / C07 against
`global_pooled`, with the paired CI and the per-seed spread, and reject it again if it does not
degrade gracefully there. The eval entry's items (1)-(5) are unchanged and independent of this.

### 2026-10-08 — The block C precondition FAILS on dev and passes across attacks

First real-data run of the leakage gate added with the block C rebuild. Descriptors from
`chain_descriptor.sh` (clean views; 73,000 train and 47,400 dev clips). The gate fired, which is
what it was for.

| probe (logistic, class-balanced, on the 25-dim descriptor alone) | EER | verdict |
| --- | --- | --- |
| fit dev `calib` (A09-A16), test dev `select` (A09-A16) | **0.2553** | FAILS |
| fit train (A01-A08), test dev `select` (A09-A16) | **0.5527** | passes (chance) |
| fit train speakers, test held-out train speakers (A01-A08) | 0.2798 | FAILS |
| rolloff edge alone, fit calib test select | 0.4808 | passes |
| the 24 LTAS bands alone, fit calib test select | 0.2555 | FAILS |

**Reading.** The long-term average spectrum carries an **attack-specific** signature, not a general
"spoof has a different spectrum" effect. A probe recovers the class whenever it is fit and tested on
the SAME attacks (0.255, 0.280) and gets nothing at all across attacks (0.553, i.e. worse than
chance -- the train-attack signature does not transfer to A09-A16). The rolloff edge, which is the
part that actually measures bandwidth, is label-free on its own (0.481); the leak is in the spectral
shape. Per-generator rolloff means are flat (0.550-0.697 across A01-A08 and bona fide 0.586), so
this is not a bandwidth confound.

**Consequence for the study protocol, which is the useful part.** Block C is fit on dev `calib` and
tested on dev `select` -- the same attacks, i.e. exactly the 0.255 case. So the dev seen-codec and
leave-one-codec-out numbers are **structurally unusable** for a calibration claim, with this
descriptor or with v1's codec posterior: the condition input carries class evidence either way, and
that is how v1 produced a dev Cllr (0.039) below the corrected floor (0.046). The eval protocol is
different: calibration is fit on dev `calib` (A09-A16) and applied to eval (A17-A32), which is the
cross-attack case where the descriptor is label-free. **Therefore eval is the only legitimate test
of block C, and the dev numbers must not be reported as support.** This retrospectively explains
why v1's dev win did not survive to eval.

**Not patched, deliberately.** Tuning the descriptor until the gate passes would be fitting to the
gate. The design's own falsification criterion (stated with the rebuild) is met for dev.

Still open on this thread: the codec views are extracting, so whether the descriptor separates the
simulated codecs on real audio is not yet measured -- that is its actual job and it is necessary for
block C to be worth running at all. Then `calibrate.py +eval=true`, reading ONLY the eval section
and C04 / C07, with `null_weight` to say whether the shrinkage engaged.
