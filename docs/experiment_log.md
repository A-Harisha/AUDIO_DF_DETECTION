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

### 2026-10-08 — Block C v2 on eval: no longer catastrophic, still worse than pooled. Rejected again.

Study `experiments/runs_partial/cal_eval_v2` (fused `frozen_xlsr_final` + `frozen_wavlm_final`, as
`cal_eval`; block C v2 over seeds 0-2; all fits on dev `calib`). Per the previous entry only the
eval section is a legitimate test; the dev sections are not read as support.
Claim tested: block C gives better LLRs than one pooled calibration and degrades gracefully on
unseen codecs (C04, C07).

**Preconditions.**
- Label probe on the descriptor alone, dev `select`, clean + codec views: EER **0.319** (must be
  ~0.5). Fails, as on clean views (0.255).
- The descriptor does not identify the simulated codecs. Codec-classifier accuracy on dev: clean
  0.84, mp3_wb 0.93, opus_nb 0.91, amr_nb 0.66, speex_nb 0.66, amr_wb 0.48, m4a_wb 0.20, speex_wb
  0.19, **opus_wb 0.007**. It separates narrowband from wideband and little else. This was the
  open question of the previous entry; the answer is no.

**Eval (Cllr; paired 95% CI of block C minus global_pooled).**

| | global_clean | global_pooled | lookup_oracle | block C v2 (seed mean +- std) | CI of difference | floor | null weight |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eval pooled | 0.310 | **0.250** | 0.243 | 0.295 (0.289 +- 0.006) | [+0.044, +0.047] | 0.213 | 0.66 |
| C04 (unseen) | 0.976 | **0.645** | 0.645 | 0.735 (0.708 +- 0.019) | [+0.077, +0.100] | 0.370 | 0.53 |
| C07 (unseen) | 1.475 | **0.962** | 0.962 | 1.216 (1.169 +- 0.034) | [+0.238, +0.269] | 0.426 | 0.49 |
| C11 (unseen) | 0.146 | 0.190 | 0.190 | 0.215 (0.221 +- 0.008) | [+0.024, +0.025] | 0.105 | 0.99 |
| clean | 0.111 | 0.147 | 0.111 | 0.183 (0.181 +- 0.006) | [+0.035, +0.038] | 0.086 | 0.59 |
| C10 (speex_nb) | 0.312 | 0.220 | 0.229 | **0.212** (0.212 +- 0.001) | [-0.010, -0.008] | 0.204 | 0.99 |

actDCF (p=0.05), eval pooled: global_clean 0.334, global_pooled 0.368, block C 0.476
(CI of difference [+0.105, +0.113]); minDCF 0.311.

- v2 removed the v1 collapse (eval pooled Cllr 0.862 -> 0.295; C07 6.26 -> 1.22) and its seed
  spread is small (std <= 0.034). The rebuild did what it was built to do on stability.
- It is still **worse than global_pooled on 11 of 12 conditions**, with CIs that exclude zero. The
  single win (C10) is 0.009 Cllr.
- **It does not degrade gracefully on the unseen neural codecs.** The null-anchor weight (1 = plain
  global calibration) is 0.53 on C04 and 0.49 on C07, the same as on clean (0.59): the descriptor
  does not see EnCodec audio as unfamiliar, so block C acts there with confidence and makes Cllr
  worse. It does retreat on narrowband (0.99 on C08-C10) and on C11.
- Simplest strong baselines on eval: `global_pooled` for Cllr (0.250), `global_clean` for actDCF
  (0.334). No condition-aware method beats both.

**Decision.** By the criterion written before this run (must beat `global_pooled` on C04 / C07 and
degrade gracefully there), **block C v2 is rejected.** Not tuned further against eval: that would
be fitting to the test set. The calibration premise itself stands (C04 / C07 pooled Cllr 0.645 /
0.962 vs floor 0.370 / 0.426) and is unsolved. What a third attempt would need, if made: a
condition signal that recognises an unseen codec as unseen (the long-term spectrum does not),
developed on a held-out codec family in dev, never on eval.

### 2026-10-09 — DCF cost model was wrong; corrected. Single-system comparison with the literature.

**Metric bug (affects every minDCF and actDCF above this entry).** `metrics/dcf.py` used a 0.05
BONA FIDE prior with C_miss = C_fa = 1. The ASVspoof 5 track-1 cost model is spoof prior 0.05,
C_miss = 1, C_fa = 10 (stated in arXiv:2501.05545, Appendix A; still to be confirmed against the
evaluation plan). Defaults corrected, known-answer test added. Earlier minDCF / actDCF values are on
a different scale and must not be compared with published ones. EER, Cllr, minCllr are unaffected.
The calibration conclusions (block C rejected) rest on Cllr with paired CIs and stand; the actDCF
columns of those entries are at the wrong operating point and have NOT been recomputed.

**Eval, official cost model (scores unchanged; runs `frozen_*_final`, heads on 40% of train, one seed).**

| system | EER | minDCF |
| --- | --- | --- |
| frozen WavLM-Large + head | 4.40% | **0.1227** (was reported as 0.255) |
| frozen XLS-R-300M + head | 7.67% | 0.2161 (was 0.453) |

WavLM per codec (EER / minDCF): clean 1.47 / 0.039, C05 1.85 / 0.047, C06 2.24 / 0.058,
C11 2.37 / 0.061, C02 2.77 / 0.070, C01 2.93 / 0.075, C03 3.81 / 0.096, C09 4.44 / 0.111,
C08 4.80 / 0.123, C10 5.71 / 0.138, **C04 8.20 / 0.220, C07 9.78 / 0.261**.

**Published results, read from the papers (not from search summaries).**

| system | type | EER | minDCF | source |
| --- | --- | --- | --- | --- |
| T45, challenge winner, open | 6-model ensemble | 2.59% | 0.0750 | arXiv:2408.08739 |
| 4-model fusion | ensemble | 2.58% | – | arXiv:2501.05545, Table 12 |
| XLS-R 128 + ResNet34, MaskedSpec + RawBoost | single, fine-tuned | **4.17%** | 0.119 | arXiv:2501.05545, Table A.13 |
| own SSL pretraining on ASVspoof 5 + ResNet34 | single, closed | 5.44% | **0.117** | same table |
| WavLM + ResNet34, MaskedSpec + codecs | single, fine-tuned | 4.77% | 0.132 | same table |
| SZU-AFS D4 | 2-model fusion | 4.04% | 0.115 | arXiv:2408.09933 |
| BUT, open primary | 3-model fusion | 9.28% | 0.2573 | BUT ASVspoof 5 paper |

- Our frozen WavLM single model (4.40% / 0.123) is level with the best single systems I could
  verify (4.17% / 0.117), without fine-tuning and with 40% of train. It is not SOTA overall: the
  ensembles reach 2.6% / 0.075.
- A search summary attributed "Wav2vec2 + MHFA, minDCF 0.0848, EER 3.30%" to the eval set. It is a
  DEV result (BUT paper, Table 2). Not used.
- Not checked: arXiv:2602.13596 (BreathNet) result tables; a better single system may exist.
- Single-model target (the user will not use ensembles): EER < 4.17% and minDCF < 0.117 on eval.

**Adversarial attacks in eval.** Which attacks are adversarial, and their base attacks, are from
arXiv:2408.08739 Table 2: A18, A20, A23, A27, A30 (Malafide / Malacopula on A17, A12, A09, A26,
A18), A31, A32 (Malacopula on A22, A25). **The EER numbers below are OURS**, computed from
`scores_eval_clean.parquet`; an earlier version of this entry presented them under the paper's
citation, which was wrong. A20 (base A12) and A23 (base A09) have dev-set bases, so they have no
in-eval pair.

Paired EER, base -> adversarial, our two single frozen systems:

| pair | XLS-R | WavLM |
| --- | --- | --- |
| A17 -> A18 | 1.62 -> 7.15 (4.4x) | 0.68 -> 1.92 (2.8x) |
| A18 -> A30 | 7.15 -> 14.76 (2.1x) | 1.92 -> 3.00 (1.6x) |
| A22 -> A31 | 2.68 -> 17.11 (6.4x) | 2.84 -> 6.21 (2.2x) |
| A25 -> A32 | 1.61 -> 4.41 (2.7x) | 0.92 -> 1.80 (2.0x) |
| A26 -> A27 | 1.68 -> 6.29 (3.7x) | 1.08 -> 2.16 (2.0x) |

Mean per-attack EER, benign (9 attacks) vs adversarial (7):

| system | benign | adversarial | ratio |
| --- | --- | --- | --- |
| frozen XLS-R-300M | 3.35% | 8.22% | **2.45x** |
| frozen WavLM-Large | 3.00% | 2.75% | **0.92x** |

- **WavLM is adversarially robust here and XLS-R is not.** On the mean over attacks WavLM does not
  degrade at all (0.92x); the adversarial filter multiplies XLS-R error 2.1-6.4x per pair and 2.45x
  on the mean. The per-pair ratio is still ~2x for WavLM, so the filter does hurt it; the mean is
  flat because WavLM's benign mean is dragged up by A28 (below).
- **For WavLM the binding constraint is NOT adversarial.** Its worst attacks are **A28 12.13%** and
  **A24 5.38%**, both benign; the worst adversarial one is A31 at 6.21%. For XLS-R the two worst are
  adversarial (A31 17.11%, A30 14.76%) with A28 third at 13.25%. A28 is the single largest error
  source for the backbone we would actually ship. What A28 is: [CITATION NEEDED].
- Every eval clip carries a codec label, so these per-attack numbers already pool over C01-C11.
  Adversarial and codec effects do not compound on WavLM (coded clips: benign 5.14%, adversarial
  3.35%, 0.65x) and do on XLS-R (5.72% vs 9.55%, 1.67x).
- Bearing on the goal: picking WavLM as the single backbone already buys most of the adversarial
  robustness, so "adversarially safe" is close to free and A28 is the open problem. Single seed,
  40% of train, no fine-tuning. No CIs yet.

**Head sweep (dev select only; clean / codec EER).** WavLM: seeds 1337, 1, 2 = 1.59 / 3.15,
1.50 / 2.99, 1.63 / 3.30; dropout 0.5 = **1.13 / 2.76**; wd 0.1 + lr 3e-4 = 1.61 / 3.22.
XLS-R: seeds = 0.82 / 2.69, 0.78 / 2.37, 0.78 / 2.45; dropout 0.5 = **0.44 / 2.36**; wd = 0.98 / 2.74.
Dropout 0.5 is outside the three-seed range on clean for both backbones. Not yet scored on eval.

**Infrastructure.** The EnCodec extraction (chain_encodec.sh) crashed on 2026-10-08 after 3 min
(DataLoader worker exited). On 2026-10-09 the Windows WMI service hung: `platform.machine()` blocks,
so `import pandas` blocks, so every script blocks. Unrelated to our code; needs a service restart.

## 2026-10-09 — Dropout 0.5 on eval; EnCodec moved to the GPU

**Claim tested:** the dev-only head sweep said dropout 0.5 beats the three-seed range. Does it hold on eval?
Runs `frozen_{wavlm,xlsr}_drop50_final` (views clean + codec, 40% of train, seed 1337, one seed each);
controls `frozen_{wavlm,xlsr}_final` (dropout 0.2, otherwise identical). Official cost model.

| Eval (680,774 clips) | EER % | minDCF | C04 EER | C07 EER | clean EER |
|---|---|---|---|---|---|
| WavLM, dropout 0.2 | 4.40 | 0.1227 | 8.20 | 9.78 | 1.47 |
| WavLM, dropout 0.5 | 4.95 | 0.1399 | 9.29 | 11.28 | 1.48 |
| XLS-R, dropout 0.2 | 7.67 | 0.2161 | 14.19 | 16.83 | 4.31 |
| XLS-R, dropout 0.5 | 6.80 | 0.1970 | 14.35 | 17.30 | 2.47 |

Adversarial attacks, EER % (attack vs all bona fide): A18 / A20 / A23 / A27 / A30 / A31 / A32
- WavLM 0.2: 1.92 / 1.93 / 2.25 / 2.16 / 3.00 / 6.21 / 1.80
- WavLM 0.5: 2.21 / 1.79 / 1.97 / 2.59 / 3.66 / 6.59 / 2.35
- XLS-R 0.2: 7.15 / 2.59 / 5.27 / 6.29 / 14.76 / 17.11 / 4.41
- XLS-R 0.5: 5.50 / 3.29 / 4.04 / 6.03 / 9.77 / 11.94 / 5.02

**Result:** the dev gain does not transfer for WavLM (eval gets worse, mostly on the codecs); it does for
XLS-R (clean and most codecs better, C04 / C07 not). One seed per cell and no eval seed range, so the
size of either effect is not established. This is one more case where dev (clean, A09-A16) misleads.
Dropout is NOT chosen from this table (that would be selection on eval); it needs the stress-dev set.
Neither head reaches the single-model target (EER < 4.17%, minDCF < 0.117).

**EnCodec view:** on CPU in DataLoader workers, EnCodec took 7 min per batch of 24 and failed with
"bad allocation" (cause of the 8 Oct crash is presumed the same; not confirmed). It now runs on the GPU
in the main process; workers do only the MP3 step for the C07 simulation. Dev view: 47,400 clips in 2 h.

## 2026-10-09 — EnCodec training view on eval

**Claim tested:** eval C04 / C07 are hardest because the detector never saw a neural codec; simulating
EnCodec in training lowers their EER. Runs `frozen_{wavlm,xlsr}_enc_drop50` (views clean + codec +
encodec, selected on dev clean + codec + encodec, dropout 0.5, 40% of train, seed 1337). Controls:
`frozen_{wavlm,xlsr}_drop50_final` (identical without the encodec view). Official cost model.

| Eval (680,774 clips) | EER % | minDCF | C04 EER | C07 EER | clean EER |
|---|---|---|---|---|---|
| WavLM, control | 4.95 | 0.1399 | 9.29 | 11.28 | 1.48 |
| WavLM, + encodec | 4.17 | 0.1144 | 6.60 | 7.77 | 1.60 |
| XLS-R, control | 6.80 | 0.1970 | 14.35 | 17.30 | 2.47 |
| XLS-R, + encodec | 5.97 | 0.1717 | 11.09 | 13.27 | 2.48 |

Adversarial attacks, EER % (A18 / A20 / A23 / A27 / A30 / A31 / A32):
- WavLM + encodec: 1.08 / 1.49 / 1.33 / 2.27 / 2.21 / 6.18 / 1.74
- XLS-R + encodec: 3.70 / 3.83 / 3.01 / 6.63 / 7.66 / 10.05 / 5.03

**Result:** claim supported for both backbones: C04 and C07 EER drop by 2.7-4.0 points, pooled EER by
0.8 points. The other codecs and clean move by a few tenths in either direction (C02, C09 slightly worse).
C04 / C07 are still the worst conditions, so the simulation closes only part of the gap (that ASVspoof 5
used the public 24 kHz EnCodec model is still unverified).

**Against the literature (verified single systems, arXiv:2501.05545 Table A.13: EER 4.17%, best
minDCF 0.117):** WavLM + encodec has minDCF 0.1144 and EER equal to 4.17% at two decimals. This is a
frozen backbone with a 0.26M-parameter head on 40% of train. Limits: one seed, no confidence interval,
and the comparison with the dropout 0.2 control (4.40%) is confounded by the dropout change.

## 2026-10-09 — EnCodec view at dropout 0.2 (removes the dropout confound)

Run `frozen_wavlm_enc`: identical to `frozen_wavlm_enc_drop50` except dropout 0.2. Its control is
`frozen_wavlm_final` (dropout 0.2, views clean + codec). 40% of train, seed 1337, official cost model.

| Eval | EER % | minDCF | C04 EER | C07 EER | clean EER |
|---|---|---|---|---|---|
| WavLM, dropout 0.2, control | 4.40 | 0.1227 | 8.20 | 9.78 | 1.47 |
| WavLM, dropout 0.2, + encodec | 3.96 | 0.1083 | 6.06 | 7.15 | 1.69 |
| WavLM, dropout 0.5, + encodec | 4.17 | 0.1144 | 6.60 | 7.77 | 1.60 |

Adversarial EER %, dropout 0.2 + encodec (A18 / A20 / A23 / A27 / A30 / A31 / A32):
1.20 / 1.70 / 1.85 / 2.16 / 2.48 / 6.76 / 1.77

**Result:** at fixed dropout the EnCodec view lowers pooled EER by 0.44 (dropout 0.2) and 0.78 (dropout
0.5) points; C04 / C07 by 2.1-3.5 points. The price is visible at dropout 0.2: clean and the classic
codecs get 0.1-0.7 points worse (clean 1.47 -> 1.69, C08 4.80 -> 5.50), i.e. the 0.26M head trades
capacity between views. Both EnCodec heads are at or below the verified single-system reference
(EER 4.17%, minDCF 0.117); dropout 0.2 gives 3.96% / 0.1083.
**Caution:** two dropout values were both scored on eval, so quoting the better one is a selection on
eval of size two. One seed each; the 0.21-point gap between them is not shown to exceed seed noise.

## 2026-10-09 — Seed range of the WavLM + EnCodec head: the 3.96% does not hold

Runs `frozen_wavlm_enc_seed{1,2}`: identical to `frozen_wavlm_enc` (dropout 0.2, views clean + codec +
encodec, 40% of train) except the seed.

| Seed | best epoch | select EER % | eval EER % | eval minDCF | C04 | C07 | clean | A31 |
|---|---|---|---|---|---|---|---|---|
| 1337 | 19 | 4.61 | 3.96 | 0.1083 | 6.06 | 7.15 | 1.69 | 6.76 |
| 1 | 21 | 4.85 | 4.09 | 0.1132 | 6.21 | 7.53 | 1.88 | 8.05 |
| 2 | 1 | 4.92 | 5.13 | 0.1358 | 7.43 | 8.48 | 2.88 | 11.09 |
| mean | | | 4.39 | 0.1191 | | | | |

**Result:** the seed range (3.96-5.13%) is wider than every effect measured today. The three-seed mean
(4.39% / 0.119) does not beat the single-system reference (4.17% / 0.117), and the earlier entries'
"at or below the reference" holds for two seeds of three only. What does hold on all three seeds: C04 and
C07 are below the no-EnCodec control (8.20 / 9.78); the control itself is one seed.
**Cause seen:** the select EER moves by +-0.5 points from epoch to epoch (e.g. 5.38 -> 6.90 -> 4.88), so
"best epoch" is the minimum of a noisy curve. Seed 2 stopped at epoch 1, an undertrained head. Its last
ten epochs are flat (5.2-5.6), like the other seeds.
**Also seen:** with the encodec view, dev clean EER is ~3.0-3.4% against ~1.5% without it. On dev the
view costs clean accuracy; on eval the cost is 0.2 points. One more dev / eval disagreement.
**Not done (would be selection on eval):** rescoring eval with the last-epoch weights. The selection
rule must be fixed on dev (last epoch or an average of the last epochs), then all seeds rerun.

## 2026-10-09 — Weight averaging (EMA) against the noisy selection curve: rejected on dev

**Claim tested:** validating and saving an EMA (decay 0.999 per step) of the head weights removes the
selection noise and gives a better and more stable model. Runs `frozen_wavlm_enc_ema_seed{1337,1,2}`;
controls `frozen_wavlm_enc`, `frozen_wavlm_enc_seed{1,2}` (identical without EMA). Dev only; eval was
not scored. Held-out = dev `calib` split (not used for selection), EER % on clean / codec / encodec.

| Seed | raw: best epoch, held-out mean | EMA: best epoch, held-out mean |
|---|---|---|
| 1337 | 19, 5.14 | 23, 5.57 |
| 1 | 21, 5.38 | 25, 5.56 |
| 2 | 1, 5.65 | 23, 5.88 |

Mean epoch-to-epoch change of select EER, last 20 epochs: raw 0.15-0.33 points, EMA 0.04-0.06.

**Result:** EMA does what it should mechanically (smooth curve, best epoch 23-25 on every seed) but the
held-out dev EER is 0.2-0.4 points worse on all three seeds. The averaged model sits at the late-training
weights, and those are worse on the unseen dev attacks than the raw model's best epochs: the head
overfits the train attacks, and the "noise" partly is that drift. EMA is therefore not adopted and the
code was not kept (patch not committed; `fit(..., ema=)` on `torch.Tensor.lerp_`, 20 lines).
**Open:** the seed spread on eval (3.96-5.13%) stays unexplained by selection alone. The frozen head
is at its limit for a stable result; the next lever is the backbone (LoRA fine-tune, full train set).

## 2026-10-10 — Smoothing the epoch-selection statistic: rejected on held-out dev

**Claim tested:** the kept epoch is the argmin of a noisy per-epoch curve, so seed 2 of
`frozen_wavlm_enc` selected epoch 1, an undertrained head. Choosing the epoch by the lowest
3-epoch mean of the select EER, and keeping that middle epoch's own weights (not an average, which
EMA already showed to be worse), should remove the selection noise. Runs `sel3_seed{1337,1,2}`:
identical to `frozen_wavlm_enc_seed*` (WavLM, views clean + codec + encodec, dropout 0.2, 40% of
train) except `experiment.select_window=3`. Held-out = dev `calib` split, never used for selection;
EER % is the mean over clean / codec / encodec.

| rule | seed 1337 | seed 1 | seed 2 | mean | spread |
|---|---|---|---|---|---|
| argmin (`select_window=1`) | 5.14 (ep 19) | 5.38 (ep 21) | 5.65 (ep 1) | 5.39 | 0.51 |
| 3-epoch mean (`=3`) | 5.57 (ep 20) | 5.60 (ep 25) | 5.88 (ep 21) | 5.68 | 0.31 |
| change | +0.43 | +0.22 | +0.23 | **+0.29** | -0.20 |

**Result: rejected.** Worse on all three seeds, same sign each time. The window moves selection later
(epochs 20 / 25 / 21 against 19 / 21 / 1) and late weights overfit the train attacks, which is the
same mechanism that rejected EMA on 2026-10-09. The variance narrowed by 0.20 and the mean paid 0.29.
Default returned to `select_window=1`; the code and its tests are kept and configurable.

**What the experiment did establish, and it matters more.** Within seed 1337, consecutive epochs 19
and 20 give held-out 5.14 and 5.57: **0.43 points one epoch apart, on a split selection never saw.**
The epoch-to-epoch movement is therefore real model variance, not measurement noise on the select
split. No selection rule can fix it, and the 1.17-point seed spread on eval (3.96-5.13) is not a
selection artefact. The lever must reduce the variance itself: the full train set instead of 40%, a
lower max learning rate so the end of the one-cycle schedule is quieter, or stronger regularisation.

**Side result (weak, n = 3).** Under the argmin rule the held-out `calib` ranking 5.14 < 5.38 < 5.65
matches the eval ranking 3.96 < 4.09 < 5.13 exactly. If that holds, `calib` is a model-selection
signal that costs no eval scoring. Worth testing on the runs already scored. Not established.

## 2026-10-10 — Lower max learning rate against the seed spread: rejected on held-out dev

**Claim tested:** the one-cycle schedule still takes real steps late in training, so a lower max
learning rate should quieten the end of the run, shrink the per-epoch movement and shrink the seed
spread. Runs `lr3e4_seed{1337,1,2}`: identical to `frozen_wavlm_enc_seed*` (WavLM, views clean +
codec + encodec, dropout 0.2, 40% of train, `select_window=1`) except `experiment.lr=3.0e-4`.
Held-out = dev `calib` split; EER % is the mean over clean / codec / encodec.

| lr | seed 1337 | seed 1 | seed 2 | mean | spread |
|---|---|---|---|---|---|
| 1e-3 (baseline) | 5.14 (ep 19) | 5.38 (ep 21) | 5.65 (ep 1) | 5.39 | 0.51 |
| 3e-4 | 5.46 (ep 10) | 5.45 (ep 8) | 5.80 (ep 4) | 5.57 | 0.35 |
| change | +0.32 | +0.07 | +0.15 | **+0.18** | -0.16 |

**Result: rejected.** Worse on all three seeds. The selected epoch moves much earlier (10 / 8 / 4
against 19 / 21 / 1): the lower rate reaches its minimum sooner and then drifts.

**The pattern, now on three independent attempts.** EMA (2026-10-09), the 3-epoch selection window
(this morning) and the lower learning rate all buy variance with error, at about the same rate:

| intervention | held-out mean | seed spread |
|---|---|---|
| EMA weight averaging | +0.2 to +0.4 | curve much smoother |
| 3-epoch selection window | +0.29 | -0.20 |
| lr 1e-3 -> 3e-4 | +0.18 | -0.16 |

Three levers, same trade, same sign. The 0.26M head on 40% of train sits on a bias-variance frontier
and these knobs only slide along it. The variance is therefore not a tuning problem, and the
remaining levers are data quantity and head capacity. Full-train extraction (clean + codec, WavLM,
182,357 clips) started 13:26; dev stays at the partial 47,400 clips, hardlinked, so train is the
only variable. The encodec view of full train (+6.8 h) is deliberately not extracted until the
2-view result says whether more data shrinks the spread at all.

## 2026-10-13 — The ensemble premise, measured first: members are not diverse, so fusion gains ~0.1 points

**Why this entry exists.** The goal changed to "beat SOTA with an ensemble that also survives adversarial
attack", which reverses the single-model rule of 2026-10-09. Before building a fusion stage, the
premise was measured on the nine runs already scored on eval. Fusion was fit on the dev `calib`
split only (clean + codec, the views every run has) and tested on the full eval set.

**Current best single model (full train + encodec view, not previously logged):**

| run | eval EER | eval minDCF |
|---|---|---|
| fullenc_seed1337 | **3.89%** | **0.1061** |
| fullenc_seed1 | 4.07% | 0.1097 |
| fullenc_seed2 | 4.21% | 0.1138 |
| mean (3 seeds) | 4.06% | 0.1099 |

Full train narrowed the seed spread from 1.17 points (40% of train: 3.96-5.13) to **0.32** (3.89-4.21),
and the mean now sits below the verified single-system reference (4.17% / 0.119, arXiv:2501.05545
Table A.13). This is the first intervention that moved the variance, and it confirms the 2026-10-10
conclusion that the lever was data quantity, not tuning.

**Member correlation on eval (Pearson, scores z-scored on dev calib).** Six WavLM runs: **0.974-0.994**.
Three XLS-R runs: 0.946-0.983. WavLM against XLS-R: **0.879-0.928**. Different seeds, different dropout,
different views and even a different backbone all give near-duplicate scores.

**Fusion gain (best subset of each size; the subset was chosen ON EVAL, so these are upper bounds).**

| rule | best k | eval EER | eval minDCF |
|---|---|---|---|
| best single | 1 | 3.89% | 0.1061 |
| mean of z-scores | 4 | **3.74%** | **0.1027** |
| logistic, fit on dev calib | 2 | 3.88% | 0.1061 |

**Result: the fusion stage is not the bottleneck.** The plain mean buys 0.15 EER points at its
eval-selected optimum, which is inside the 0.32-point seed spread, and adding members past four makes
it worse (k=9: 4.15%). Logistic fusion fit on dev is worse than the unweighted mean at every size and
collapses as members are added (k=9: 5.74%), the same dev/eval disagreement seen throughout this log.
The ensemble SOTA target (T45, 2.59% / 0.0750, arXiv:2408.08739) is 1.3 EER points away. No fusion rule
closes that over members correlated at 0.98.

**Cause found, and it is mechanical.** The learned layer weighting of every unbanded WavLM head is the
same and nearly uniform (top five layers 24, 10, 9, 8-11; maximum weight 0.067 against 0.040 for
uniform). All six WavLM members therefore read almost the same vector. XLS-R picks a different band
(17-19, plus 5) and that single difference is the whole 0.99 -> 0.89 drop in correlation.

**Consequence for the adversarial goal.** At correlation 0.98 a perturbation that moves one member's
score moves every member's score, so this ensemble would also carry no robustness benefit. Accuracy and
adversarial robustness have the same binding constraint here: member diversity.

**Next, in this order.** (1) Layer-band members: restrict each head to one slice of the backbone, so the
feature paths differ by construction (`Head(band=...)`, `scripts/chain_bands.sh`, four bands over
WavLM's 25 outputs, seed 1337, everything else at default; control `fullenc_seed1337`). Report per-member
EER, the correlation matrix and the mean-fusion gain. (2) Build the condition gate only if correlation
actually drops. (3) Then the adversarial harness (white-box PGD on the waveform, and black-box: transfer
plus the Malafide / Malacopula attacks already in eval), reported separately.

## 2026-10-13 — Layer-band members decorrelate and the ensemble wins; the condition gate is rejected on eval

**Claim tested.** Member diversity, not the fusion rule, is what an ensemble needs here. Restricting
each head to one band of backbone layers should decorrelate the members, and a combiner over them
should then gain much more than the 0.15 points the near-duplicate runs gave this morning.

Runs `band{0_7,7_13,13_19,19_25}`: WavLM, full train, views clean + codec + encodec, seed 1337,
`experiment.band` the only change from `fullenc_seed1337` (3.89% / 0.1061). Study `ens_bands`:
combiners fit on dev `calib`, compared on dev `select`, eval scored once.

| member | dev select EER | eval EER | eval minDCF | layer mass |
|---|---|---|---|---|
| band0_7 | 0.1200 | 10.60% | 0.2987 | layers 0-6 |
| band7_13 | **0.0549** | 4.06% | 0.1136 | layers 7-12 |
| band13_19 | 0.0668 | 4.83% | 0.1302 | layers 13-18 |
| band19_25 | 0.0770 | 6.71% | 0.1689 | layers 19-24 |

**Diversity: achieved.** Pairwise eval correlation 0.59-0.94, mean off-diagonal **0.813**, against
0.974-0.994 for the six unbanded WavLM runs. Layers 7-12 alone reach 4.06%, within 0.2 points of the
whole 25-layer stack, so banding costs little where the evidence lives.

**Eval, pooled over all conditions (n = 680,774). Paired bootstrap CI of the EER difference against
best_single (= band7_13); negative means the combiner wins.**

| combiner | EER | minDCF | d_EER vs best_single, CI95 |
|---|---|---|---|
| best_single (band7_13) | 4.06% | 0.1136 | — |
| mean | **3.67%** | 0.1021 | [-0.0043, -0.0035] |
| global_mix | 3.70% | **0.1011** | [-0.0040, -0.0033] |
| logistic | 3.83% | 0.1064 | [-0.0027, -0.0018] |
| gated | 3.97% | 0.1071 | [-0.0013, -0.0005] |

**Result 1: the diversity claim holds.** Four banded members, three of which are individually worse
than the control, combine to 3.67% / 0.1011 and beat the best single model in the project
(`fullenc_seed1337`, 3.89% / 0.1061). The gain over the best member is 0.39 points with a CI that
excludes zero, against 0.15 points (eval-selected, so an upper bound) for the near-duplicate members.
**The weakest member is the most useful one**: band0_7 is 10.60% alone and correlates at 0.59-0.86,
and dropping it makes the mean of the other three worse (4.10% against 3.66%).

**Result 2: the condition gate is REJECTED on eval.** It is the best combiner on dev `select`
(3.84% against 4.35% for logistic) and the worst ensemble on eval (3.97%, behind the plain mean's
3.67%), worse than `mean` on 9 of 12 codecs. This was predictable and was predicted: the descriptor
label probe fails within attacks (dev calib -> dev select EER **0.328**, chance 0.50) and passes
across attacks (0.553), so the dev gate fit on A09-A16 and tested on A09-A16 is fitting an
attack-specific spectral signature that does not exist in A17-A32 (2026-10-08 entry). The dev gated
number is therefore not evidence for anything, and the eval number says the mechanism does not
transfer. This is the third condition-dependent mechanism rejected in this project after block C's
two attempts. **The combiner is a plain unweighted mean, or the global mixture if minDCF is the
target.** The gate code and its tests are kept and are not used for headline results.

**Still far from the target.** Ensemble SOTA is T45 at 2.59% / 0.0750 (arXiv:2408.08739, six models).
3.67% / 0.1011 beats every verified single system but leaves 1.1 EER points. Every member here is a
0.26M head on a FROZEN backbone, while the 4.17% single system and T45's members are fine-tuned, so
the next lever is member strength (LoRA on the full train set), not another combiner.

**Side note on protocol.** `report` now prints `null_weight`, the mean weight the gate puts on its
null anchor. On eval it sat at 0.31, i.e. the gate acted on most clips rather than falling back --
so the rejection is a rejection of what it did, not of a gate that never fired.

## 2026-10-13 — Fusion of frozen-backbone members saturates at 3.6%; the lever is member strength

Three member sets, same protocol throughout (combiners fit on dev `calib`, compared on dev `select`,
eval scored once). Eval EER % / minDCF, full 680,774-clip set, official cost model.

| study | members | mean off-diag corr | best_single | mean | global_mix | logistic | gated |
|---|---|---|---|---|---|---|---|
| ens_bands | 4 bands | 0.813 | 4.06 / .1136 | **3.67 / .1021** | 3.70 / **.1011** | 3.83 / .1064 | 3.97 / .1071 |
| ens_mixed | + full-stack head | 0.851 | 3.89 / .1061 | **3.61 / .1001** | 3.69 / .1017 | 3.75 / .1034 | 3.75 / .1037 |
| ens_cross | + XLS-R member | 0.854 | 5.97 / .1717 | **3.60 / .1004** | 4.26 / .1190 | 4.39 / .1232 | 4.54 / .1279 |

**Result 1: the plain mean wins every time, and the ensemble saturates.** Four bands give 3.67%.
Adding the unbanded full-stack head gives 3.61%. Adding a whole second backbone (XLS-R, a member
that cost its own 7-hour extraction) gives 3.60% -- within noise of doing nothing. Every learned
combiner (logistic, global_mix, gated) is worse than the unweighted mean on every set, and the gap
widens as members are added: on ens_cross the mean is 3.60% and the next best combiner is 4.26%.
Fitting combiner weights on dev does not transfer to eval, which is this project's oldest pattern.

**Result 2: band diversity is as valuable as a second backbone, and far cheaper.** Four banded WavLM
heads (5 minutes each on cached features) reach 3.67%; adding XLS-R on top of the bands plus the
full-stack head moves nothing. Cross-backbone correlation is 0.80-0.90, inside the band-to-band range
of 0.59-0.94, so the second backbone brings no diversity the bands did not already have.

**Result 3, and it is the strongest argument for the ensemble here: the mean removes the
model-selection risk.** On ens_cross the member with the best dev `select` EER (0.0428,
`frozen_xlsr_enc_drop50`) is the WORST on eval (5.97%). Picking a single model on dev would have cost
2.37 points against the mean of the same six members (3.60%). Given how often dev and eval disagree in
this log, that insurance is worth more than the 0.29-point accuracy gain.

**Result 4: the gate is rejected on all three sets**, not only on the bands: 3.97, 3.75, 4.54 against
the mean's 3.67, 3.61, 3.60. The cross-attack precondition itself passes (descriptor label probe fit
on train A01-A08, tested on dev A09-A16: EER **0.456**, chance 0.50), so the rejection is not a
leakage artefact -- the mechanism simply does not transfer from A09-A16 to A17-A32. Closed.

**Where this leaves the SOTA target.** Ensemble SOTA is 2.59% / 0.0750 (T45, six models,
arXiv:2408.08739). We are at 3.60% / 0.1001, which beats every verified single system (best 4.17% /
0.119) and loses to the ensembles by 1.0 EER points. The arithmetic says why: T45's six fine-tuned
members fuse from roughly 4% each down to 2.59%, a 35% relative gain, while our six frozen members
fuse from 3.89% to 3.60%, a 7% gain. Their members are far more decorrelated because they differ in
front end and are each fine-tuned; ours all share one frozen backbone and one time-mean pooling, and
banding the layer weighting is the only decorrelation available inside that design.

**Therefore the next lever is member STRENGTH and feature-path diversity, not another combiner.**
In cost order:
1. LoRA fine-tuned members on the full train set. Fine-tuning both strengthens a member and moves its
   features, so it buys accuracy and decorrelation together. Cost: ~5 h training plus ~8-10 h to score
   eval over raw audio, per member.
2. A member with a genuinely different front end (raw-waveform or spectrogram, e.g. an AASIST-style
   graph model). It shares no backbone with anything here, so it is the largest decorrelation
   available; it is also small and trains from scratch in hours.
3. Narrower bands (eight 3-layer bands, `scripts/chain_bands8.sh`, ~40 min). Cheapest, but the
   saturation above makes a large gain unlikely; it is worth running as the control that says whether
   more members of the same kind are exhausted.

## 2026-10-14 — Adversarial robustness, measured: white-box is a total break, black-box holds

**Claim served.** The project goal names adversarial safety, and the 2026-10-09 rule says robustness
must be reported as its own measured result, never assumed. Both halves of the threat model are
reported apart, as decided on 2026-10-13.

**Protocol.** PGD cannot run on 680,774 clips, so both studies use the SAME stratified subsample:
40 clips per (codec, class) cell and 10 per (adversarial attack, codec) cell, 1,799 clips in all,
480 bona fide, 12 codecs, 17 attacks, cropped to 4 s. 20 PGD steps, L2 budget set by target SNR,
perturbations projected over the valid samples only so zero padding is never attacked. The clean
metrics of that same subsample are the baseline; they are NOT the full-eval numbers, because the
subsample deliberately over-weights C04, C07 and the adversarial attacks.

**Path agreement (the precondition for reading any of this as a statement about our system).** The
attack path runs the backbone on a 4 s crop; the eval table comes from features pooled over up to
16 s. On the same 1,799 clips the two rank the clips the same way: Pearson **0.9618**, Spearman
**0.9340**. `scripts/attack.py` now enforces a 0.9 Spearman floor and raises below it. The residual
0.07 of rank variance is the crop, and it bounds how finely these numbers can be read.

**Single model `fullenc_seed1337` (3.89% / 0.1061 on full eval). Study `adv_single`.**

| condition | EER | minDCF | success@0.1% FAR | success@1% FAR | median SNR |
|---|---|---|---|---|---|
| clean | 0.0478 | 0.1287 | 0.0015 | 0.0106 | — |
| PGD white-box 30 dB | **1.0000** | 1.0000 | **1.0000** | 1.0000 | 32.5 dB |
| PGD white-box 20 dB | **1.0000** | 1.0000 | **1.0000** | 1.0000 | 23.1 dB |
| SPSA black-box 20 dB, 320 queries/clip | 0.2668 | 0.7395 | **0.0106** | 0.0326 | 23.5 dB |

Benchmark Malafide / Malacopula attacks already in eval, scored at no extra cost: EER 0.0186 (A23),
0.0252 (A32), 0.0283 (A18), 0.0330 (A20), 0.0398 (A27), 0.0451 (A30), **0.0830 (A31)**; success at
0.1% FAR 0.0061-0.0069 throughout.

**Result 1: white-box robustness is zero.** EER 1.0000 with minDCF 1.0000 is complete rank
inversion -- the attack pushes spoofs up and bona fide down until the ordering fully reverses -- and
every spoof clip crosses the threshold the clean scores put at 0.1% false-accept. A perturbation
32 dB below the signal suffices. No budget tested was survivable. The detector is undefended and
this is what undefended means, quantified.

**Result 2: black-box robustness is real.** A gradient-free attacker with 320 queries per clip
reaches 1.06% success at the same operating point where the white-box attacker reaches 100%. That
is a 94-point gap, and it is the number a deployment faces. The benchmark's own adversarial attacks,
crafted against other detectors, cost at most 3.5 EER points (A31: 8.30% against 4.78% clean), so
transfer from an unknown detector is weak too.

**Result 3 (study `adv_bands`, in progress): the ensemble changes nothing white-box.** The four-band
ensemble, clean 0.0478 / 0.1123 on the same subsample, gives PGD white-box 30 dB EER **1.0000**,
success 1.0000, median SNR 32.3 dB -- identical to the single model. The cause is structural: the
attacker holds all four members' gradients and the mean combiner is differentiable, so PGD attacks
the average. Member correlation of 0.81 is irrelevant when every member is visible at once.
**An ensemble is not an adversarial defence.** It bought 0.28 EER points of clean accuracy
(3.89% -> 3.61%) and zero white-box robustness.

**Consequence for the plan.** Only adversarial training can move the white-box number. Before
committing ~30 h to waveform PGD training, run the cheap probe: feature-space PGD on the existing
band heads costs minutes on cached features. It is honestly a weak defence -- the attacker goes
through the waveform while this hardens only the head -- but it says whether the real thing is worth
the compute. Transfer and SPSA against the ensemble are still running.

## 2026-10-14 — The ensemble gives ZERO adversarial robustness benefit (study `adv_bands`, complete)

**Claim tested.** Member diversity decorrelates the members' clean scores (0.98 -> 0.81) and gains
0.28 EER points. Does it also raise the cost of an adversarial attack? Answer: no, on every attack
tested. Same 1,799-clip subsample, same budgets and step count as `adv_single`, so the two compare
directly. Clean baselines: single 0.0478 / 0.1287, ensemble 0.0478 / 0.1123.

| condition | single `fullenc_seed1337` | four-band ensemble |
|---|---|---|
| clean | EER 0.0478, success@0.1%FAR 0.0015 | EER 0.0478, 0.0015 |
| PGD white-box 30 dB | **1.0000**, 1.0000 | **1.0000**, 1.0000 |
| PGD white-box 20 dB | **1.0000**, 1.0000 | **1.0000**, 1.0000 |
| PGD transfer 30 dB (crafted on the single model) | — | **1.0000**, 1.0000 |
| PGD transfer 20 dB | — | **1.0000**, 1.0000 |
| SPSA black-box 20 dB, 320 queries/clip | 0.2668, 0.0106 | 0.2624, 0.0144 |

**Result 1: every gradient attack is a total break, on both systems.** EER 1.0000 is complete rank
inversion and 100% of spoof clips cross the clean 0.1%-FAR threshold. Neither budget nor member
count changed anything.

**Result 2: TRANSFER is the finding that matters, and it is worse news than white-box.** A
perturbation crafted on `fullenc_seed1337` alone defeats all four band members it never saw, at both
budgets, completely. White-box assumes the attacker holds our exact system; transfer only assumes
they hold SOME WavLM-based detector, and WavLM is a public checkpoint. Correlation with the surrogate
was 0.78-0.96, far too high to break the gradient. The operative question is therefore not white-box
versus black-box but whether the attacker has any gradient from a model sharing our backbone -- which
is why SPSA, with no gradient at all, is the only attack that struggles.

**Result 3: the black-box difference is NOT real.** 0.0106 against 0.0144 is 14 against 19 of 1,319
spoof clips, a difference of 0.9 pooled standard deviations. The ensemble is neither better nor worse
than the single model here, and reporting 1.06% against 1.44% as a degradation would be reading noise.

**Result 4: benchmark Malafide / Malacopula attacks move a little, both ways.** On the ensemble:
A23 0.0175, A18 0.0272, A30 0.0278, A32 0.0377, A20 0.0458, A27 0.0538, **A31 0.0775**. Against the
single model the same attacks gave 0.0186, 0.0283, 0.0451, 0.0252, 0.0330, 0.0398, 0.0830. The
ensemble is better on A30 (0.0278 vs 0.0451) and A31, worse on A20, A27 and A32. No pattern worth a
claim at these sample sizes (about 630 clips per attack).

**Conclusion, and it constrains the project goal.** Adversarial safety is not reachable by
architecture. The ensemble bought 0.28 EER points of clean accuracy (3.89% -> 3.61% on full eval) and
nothing at all in robustness. The cause is structural, not a tuning failure: the attacker holds every
member's gradient and the mean combiner is differentiable, so PGD attacks the average. Only
adversarial training can move these numbers.

**Next, in cost order.** (1) Feature-space PGD training on the existing band heads -- minutes on
cached features, honestly a weak defence because the attacker goes through the waveform while this
hardens only the head, but it says whether waveform adversarial training (~30 h on the raw member) is
worth the compute. (2) `scripts/chain_raw_transfer.sh`, staged: does a WavLM gradient reach a member
that shares no backbone with it? If not, backbone heterogeneity rather than score-level diversity is
where robustness has to come from.

## 2026-10-15 — The raw member fails as a detector; it breaks the mean and proves the mean's limit

**Run `raw_seed1337`** (learnable sinc filterbank + 5 residual stages, 0.5M parameters, from scratch,
RawBoost on, codec augmentation at p=0.5, 12 epochs over the full train set, seed 1337).

| | best dev select EER | eval EER | eval minDCF |
|---|---|---|---|
| without RawBoost (abandoned at epoch 3) | 0.2790 | not scored | — |
| with RawBoost (12 epochs, kept epoch 8) | 0.2825 | **36.61%** | 0.8478 |

**Result 1: RawBoost made no difference, and the prediction that it would was wrong.** The 2026-10-13
entry called augmentation "a precondition, not a variable to isolate" for a from-scratch raw model,
on the evidence that the un-augmented run was memorising its train attacks. With RawBoost the model
plateaus at the same place: dev select EER 0.370 -> 0.308 -> 0.296 -> 0.293 -> 0.292, up to 0.326,
best 0.2825 at epoch 8, then flat at 0.286-0.290. Train BCE still falls to 0.066 while dev stalls,
so it still memorises train attacks; augmentation slowed that without preventing it. At 36.61% eval
EER this is not a usable detector. The cause is not augmentation but capacity and data: 0.5M
parameters from scratch on 182,357 clips against a 300M-parameter backbone pretrained on far more.

**Result 2: heterogeneous front ends DO decorrelate, hard.** Eval correlation of `raw_seed1337` with
the SSL members: band19_25 0.233, band13_19 0.343, fullenc_seed1337 0.365, band7_13 0.450,
band0_7 0.592 -- against 0.59-0.94 among the bands themselves and 0.78-0.96 with the WavLM surrogate.
The architecture does what it was built to do; it just cannot detect. Including it drops the
ensemble's mean off-diagonal correlation from 0.851 to 0.699.

**Result 3 (study `ens_raw`): the first study where the plain mean LOSES.** Six members, eval pooled:

| combiner | EER | minDCF | d_EER vs best_single, CI95 |
|---|---|---|---|
| best_single (fullenc_seed1337) | 3.89% | 0.1061 | — |
| mean | **4.37%** | 0.1206 | **[+0.0044, +0.0052]** (worse) |
| logistic | 3.71% | 0.1035 | [-0.0022, -0.0014] |
| global_mix | **3.67%** | **0.1011** | [-0.0025, -0.0020] |
| gated | 3.76% | 0.1031 | [-0.0016, -0.0011] |

The unweighted mean has no way to ignore a member, so a 36.61% member drags it to 4.37%, below the
best single model. The learned combiners ignore it correctly: `global_mix` gives it weight **0.0035**
and `logistic` gives it a NEGATIVE weight (-0.674), i.e. uses it as a correction rather than as a
detector. **This corrects the 2026-10-13 conclusion that "the plain mean beats every learned
combiner": that holds only while every member is competent.** The rule is that the mean is best when
members are comparable and a learned combiner is required when one is not.

**Result 4: weak-and-decorrelated has a floor.** band0_7 at 10.60% helps (removing it makes the rest
worse); raw_seed1337 at 36.61% hurts. Decorrelation does not redeem a member near chance.

**Headline unchanged.** `ens_mixed` (four bands + the full-stack head, unweighted mean) stays the best
result at **3.61% EER / 0.1001 minDCF**. `global_mix` on the six-member set reaches 3.67% while
putting 0.70 on fullenc_seed1337 and 0.28 on band7_13 -- effectively a two-member system -- so
nothing here beats it. The raw member's value was diagnostic, not operational.

**Note on the fusion code.** `ens_raw` first failed outright: `member_scores` required every member to
have scored the same clips, and a feature-trained head scores the 47,400 extracted dev clips while an
audio-trained member scores the whole 140,950-clip dev manifest. Members are now aligned by `utt`
with the intersection taken, a 90% floor against fusing over an unintended corner, and tests for
shuffled order, a superset member, a near-disjoint pair and a disjoint pair.

## 2026-10-15 — Three-seed variance of the ensemble: the headline is 3.80%, not 3.61%

**Claim tested.** The 3.61% EER / 0.1001 minDCF of 2026-10-13 rested on one seed. Three interventions
in this log (EMA, the 3-epoch selection window, a lower max learning rate) were each rejected only
after three seeds, so a single-seed ensemble number could not be reported as a result.

Runs `band{0_7,7_13,13_19,19_25}_s{1,2}` (seed 1337 already existed), studies `ens_mixed_s1` and
`ens_mixed_s2`: four bands plus that seed's full-stack head, identical in every other respect.

**Member spread on eval, per band:**

| band | 1337 | 1 | 2 | mean | spread |
|---|---|---|---|---|---|
| band0_7 | 10.60 | 11.78 | 10.62 | 11.00 | **1.18** |
| band7_13 | 4.06 | 4.18 | 4.38 | 4.21 | 0.32 |
| band13_19 | 4.83 | 5.01 | 4.92 | 4.92 | 0.18 |
| band19_25 | 6.71 | 6.78 | 7.03 | 6.84 | 0.32 |

The weakest band is by far the most seed-sensitive, and seed 1337 is the best or tied-best seed for
three of the four bands -- so the single-seed headline was built on a favourable draw.

**Ensemble, eval pooled, EER % per seed:**

| combiner | 1337 | 1 | 2 | **mean** | spread |
|---|---|---|---|---|---|
| best_single | 3.89 | 4.07 | 4.21 | 4.06 | 0.33 |
| mean | 3.61 | 3.88 | 3.89 | **3.80** | 0.28 |
| logistic | 3.75 | 3.88 | 4.04 | 3.89 | 0.29 |
| global_mix | 3.69 | 3.85 | 3.86 | **3.80** | **0.17** |
| gated | 3.75 | 4.25 | 3.93 | 3.98 | 0.50 |

minDCF 3-seed means: global_mix **0.1046** (spread 0.0051), mean 0.1056 (0.0085), logistic 0.1079
(0.0080), best_single 0.1099 (0.0077), gated 0.1090 (0.0118).

**Result 1: the headline is 3.80% / 0.1056, and 3.61% / 0.1001 was the best of three seeds.** Every
earlier entry quoting 3.61% as the result must be read as a best-seed figure. The three-seed mean
still beats the verified best single system (4.17% / 0.119, arXiv:2501.05545 Table A.13) on both
metrics; the gap to ensemble SOTA (T45, 2.59% / 0.0750, arXiv:2408.08739) widens to 1.21 EER points.

**Result 2: the ensemble gain is real and reproducible.** best_single 4.06% -> 3.80% is a 0.26-point
gain on three-seed means, against 0.28 measured on seed 1337 alone. The gain survives; only its
absolute level moves.

**Result 3: averaging does NOT reduce seed variance here.** Ensemble spread 0.28 against 0.33 for the
single models. The reason was anticipated in `chain_seeds.sh`: the four bands of one seed are trained
on the same data order, so their errors move together across seeds, and averaging over correlated
errors cannot shrink the spread. Reducing the ensemble's variance therefore needs seed diversity
INSIDE the member set, not more bands.

**Result 4: `global_mix` is the better combiner, reversing the single-seed verdict.** It ties the mean
on EER (3.80), beats it on minDCF (0.1046 against 0.1056) and has 40% less spread (0.17 against 0.28).
On one seed the unweighted mean looked best and this log said so; across three seeds it is the less
stable choice. Together with `ens_raw` -- where the mean broke on a junk member and global_mix did not
-- the mean's apparent superiority was an artefact of one seed and uniformly competent members.
`gated` has the worst spread of all five (0.50), a fourth independent reason it stays rejected.

**Method note.** All eight heads were trained in ONE invocation via `+sweep_bands` / `+sweep_seeds`,
sharing a single feature load: 50 minutes against 8+ hours for separate runs, which spend an hour
each reading 70 GB at about 16 MB/s to do three minutes of training. Per-head run directories,
configs, selection and scores are unchanged.

## 2026-10-16 — Backbone heterogeneity is NOT a transfer defence; the architectural route is closed

**Claim tested.** Transfer from `fullenc_seed1337` defeated all four WavLM layer-band members
completely (2026-10-14). Every member there read WavLM, so that result could not separate "the
perturbation is general" from "the perturbation rides the shared backbone". Studies
`adv_xlsr_from_wavlm` and `adv_wavlm_from_xlsr` test both directions across two different SSL
backbones, with no training: `frozen_xlsr_enc_drop50` (5.97% EER) and `fullenc_seed1337` (3.89%)
already existed. Same 1,799-clip subsample and budgets as every earlier attack study.

| system under test | surrogate | budget | white-box EER | transfer EER | transfer success@0.1%FAR |
|---|---|---|---|---|---|
| XLS-R member | WavLM | 30 dB | 1.0000 | 0.9333 | 0.8324 |
| XLS-R member | WavLM | 20 dB | 1.0000 | 0.9960 | **0.9841** |
| WavLM member | XLS-R | 30 dB | 1.0000 | 0.8789 | 0.8923 |
| WavLM member | XLS-R | 20 dB | 1.0000 | 0.9938 | **0.9947** |

Clean baselines on the subsample: XLS-R 0.0921 / 0.2635, WavLM 0.0478 / 0.1287. Path agreement for
the XLS-R member, checked inline for the first time: pearson 0.9465, spearman 0.9307.

**Result: a perturbation crafted on one SSL backbone defeats a different one 98-99% of the time at
20 dB.** The result is symmetric, so it is not an artefact of which model was the surrogate. Transfer
does cost the attacker something at the tighter budget -- 83-89% at 30 dB against 100% white-box --
but a slightly larger perturbation erases that, and 20 dB is still 20 dB below the signal. Backbone
heterogeneity raises the required budget marginally and defends nothing.

**This corrects the 2026-10-14 reading.** That entry attributed the 100% transfer to the shared
WavLM backbone and concluded that "backbone heterogeneity rather than score-level diversity is where
robustness has to come from". That hypothesis is now falsified: the perturbation exploits something
common to SSL speech models, not something about WavLM.

**The architectural route to robustness is closed.** Three designs measured, all fail:

| design | white-box | transfer |
|---|---|---|
| single model | 100% | — |
| four layer bands, one backbone | 100% | 100% |
| across backbones (WavLM <-> XLS-R) | 100% | 98-99% |

**Consequence.** Under the threat model scoped on 2026-10-15 (white-box excluded, transfer kept),
the only honest robustness claim is against a QUERY-ONLY attacker: SPSA at 320 queries per clip
reaches 1.06% success on the single model and 1.44% on the band ensemble, a difference that is noise.
Anything holding a gradient from any SSL speech detector succeeds. No ensemble design changes this,
so the only remaining lever is adversarial training, and the cheap feature-space probe should decide
whether the ~30 h waveform version is worth it.

**Side note for the accuracy work.** This also means the two goals do not want the same member set:
XLS-R added nothing to accuracy (3.61% -> 3.60% on seed 1337) and adds nothing to robustness either,
so there is no reason to keep a second backbone in the ensemble.

## 2026-10-16 — Feature-space adversarial training: rejected, but it was the wrong test

**Claim tested.** Every architectural route to robustness is closed (2026-10-14, 2026-10-16), leaving
adversarial training. The waveform version costs about 30 h, so this was meant as the cheap gate on
it: PGD in the CACHED FEATURE space, L2 budget 5% of each clip's feature norm, 3 steps, which runs in
minutes because the head is 0.26M parameters. Run `advtrain_seed1337`, otherwise identical to
`fullenc_seed1337`; study `adv_advtrain` uses the same attack set and budgets as every earlier one.

| | dev select EER | eval EER | eval minDCF |
|---|---|---|---|
| control `fullenc_seed1337` | 0.0435 | 3.89% | 0.1061 |
| `advtrain_seed1337` | 0.1169 | **14.36%** | 0.3294 |

Attack success at the clean 0.1%-FAR threshold, both on the 1,799-clip subsample:

| attack | control | hardened head |
|---|---|---|
| white-box 30 dB | 1.0000 | 0.9932 |
| white-box 20 dB | 1.0000 | **1.0000** |
| transfer 30 dB | 1.0000 | 0.7763 |
| transfer 20 dB | 1.0000 | **0.9773** |

EER is 1.0000 under every attack in both systems: the ranking stays fully inverted and only the
fixed-threshold crossing rate moves. Path agreement for the hardened head: pearson 0.9344,
spearman 0.9279.

**Result: rejected.** The hardening costs 10.5 EER points of clean accuracy and buys nothing durable.
Its only visible effect is at the tightest budget -- transfer success 0.7763 at 30 dB -- and that
collapses to 0.9773 when the attacker spends a 20 dB perturbation instead. White-box returns to
1.0000 at 20 dB. The asymmetry at 30 dB is mechanistically sensible (a white-box attacker
re-optimises against the hardened head, a transfer attacker carries an un-hardened surrogate's
gradient) but its magnitude makes it useless.

**A side result worth keeping.** That a 5%-of-norm perturbation in the pooled-layer space can take a
head from 3.89% to 14.36% EER says the spoofing evidence in those features is concentrated in
directions that small a perturbation can largely erase. That is a statement about the features, not
about the defence.

**What this does NOT settle, and the probe design is at fault.** It hardened the HEAD against
perturbations in FEATURE space, while the attacker perturbs the WAVEFORM and arrives through the
backbone. The failure is exactly what that mismatch predicts, so the probe gives no positive evidence
for waveform adversarial training and only weak negative evidence -- the mechanism explaining its
failure does not apply to the real thing. The docstring of `feature_pgd` flagged this before the run;
the gate was still set as though a proxy in the wrong space could answer the question, which it
cannot.

**What would decide it.** A small-scale WAVEFORM adversarial-training run: backbone in the loop,
PGD-3 on waveforms, a 20k-clip subset, 3 epochs -- roughly 4-6 h, and it hardens the path that is
actually attacked. Not started; it needs a decision, because the honest prior after three closed
architectural routes is that robustness here is expensive and may not be reachable at a usable
accuracy.

## 2026-10-16 -- LoRA passes the gate, and the 2026-10-06 failure was never LoRA's fault

**Claim served.** Fine-tuning the backbone is the only credible route to the remaining 1.21 EER
points to T45 (2.59% / 0.0750), because our ensemble members are frozen backbones carrying 0.26M
heads while T45's six are fine-tuned and gain 35% relative from fusion against our 6%. LoRA had
failed here once, so the question was whether it fails at full data scale.

**Run.** `lora_probe`: WavLM-Large, LoRA rank 16 on q/k/v/out_proj (3.15M trainable adapters, which
matches 2*16*1024*4*24 exactly), plain BCE, all 182,357 train clips, batch 32, 8 epochs,
backbone_lr 1e-5, head lr 1e-3, codec_p 0.5. Validated every epoch on the dev `select` split
(A09-A16, unseen). `score_sets=[]`: eval deliberately NOT scored, because 9 h of scoring is
worthless until dev says fine-tuning helps at all. 11:32 to 17:34, exit 0.

**GATE: beat the control's dev select EER of 0.0308 on the same two views.** Note the gate value:
`fullenc_seed1337` publishes 0.0435, but that is its THREE-view mean and this run validates on TWO.
The comparable number is 0.0308. The error was caught before the result was read.

| epoch | train BCE | dev clean | dev codec | A12 | A13 | select |
|---|---|---|---|---|---|---|
| 0 | 0.0677 | 0.0145 | 0.0313 | 0.0101 | 0.0382 | 0.0229 |
| 1 | 0.0216 | 0.0132 | 0.0328 | 0.0117 | 0.0350 | 0.0230 |
| 2 | 0.0178 | 0.0117 | 0.0299 | 0.0133 | 0.0215 | 0.0208 |
| 3 | 0.0135 | 0.0066 | 0.0278 | 0.0051 | 0.0159 | **0.0172** |
| 4 | 0.0092 | 0.0073 | 0.0305 | 0.0066 | 0.0179 | 0.0189 |
| 5 | 0.0084 | 0.0067 | 0.0313 | 0.0066 | 0.0120 | 0.0190 |
| 6 | 0.0060 | 0.0073 | 0.0305 | 0.0082 | 0.0107 | 0.0189 |
| 7 | 0.0059 | 0.0074 | 0.0308 | 0.0082 | 0.0123 | 0.0189 |

**Result: 0.0172 at epoch 3, 44% below the gate.** The probe passes.

**The 2026-10-06 failure is now diagnosed, and it was not LoRA.** That run reached dev EER 0.217 on
unseen attacks while a logistic probe on the frozen features reached 0.010, and the log recorded
train-attack memorisation as the suspicion without isolating a cause. It used 4,591 clips and
confounded LoRA with FiLM, a joint Cllr loss and codec supervision. Here, LoRA alone on 182,357
clips never memorises: A12 0.0051 and A13 0.0159 at the kept epoch, against 0.777 and 0.509 then.
The cause was data scale or one of the confounded components, not low-rank adaptation. A prediction
I made earlier -- that the raw member's memorisation and the LoRA failure shared one mechanism --
survives for the raw member and is withdrawn for LoRA.

**Epochs 4-7 are saturation, not overfitting, and the per-view split is the evidence.** Train BCE
halves from 0.0135 to 0.0059 while select sits inside 0.0001 of 0.0189 for four straight epochs.
Clean holds at its best (EER 0.0067-0.0074, min_dcf 0.0166 at epoch 5, the lowest of the run) and
A13 keeps IMPROVING to 0.0107. Only the codec view loses ground, 0.0278 -> 0.0313. So the late-epoch
cost is channel specialisation, not attack memorisation; if it is worth fixing, the lever is more
codec augmentation during fine-tuning, not fewer epochs. The epoch-4 rise of +0.0017 was reported as
possible-noise at the time and the plateau confirms that reading.

**An unasked-for finding.** The learned layer weighting moves. A frozen WavLM probe peaks near layer
20 (2026-10-07); this head peaks at layer 8 (0.072) with a nearly flat profile over layers 4-10 and
a maximum only 1.8x uniform. LoRA redistributes the useful evidence down the stack and spreads it,
which is a plausible reason fine-tuned members should fuse better than our band members, whose
diversity is imposed by a mask rather than learned.

**Hope, stated as arithmetic and not more.** The control gives ONE measured dev-to-eval mapping on
this setup: dev select 0.0308 -> eval 3.89% EER, a ratio of 1.26. Applying it to 0.0172 gives about
2.2% EER, below T45's 2.59%. That is an extrapolation from a single point, over a dev split that
lacks eval's unseen codecs and its A18-A32 adversarial attacks. It is a reason to spend 9 h scoring
eval, not a result.

**A blocker found while verifying the implementation.** `ScoreEnsemble.from_runs`
(src/caesd/models/ensemble.py:95) raises NotImplementedError when `lora_rank` is non-zero, because
it caches one shared backbone per tag -- correct for frozen backbones, wrong for LoRA, where every
member carries its own adapters. So a LoRA member can be trained and scored alone today but cannot
enter the ensemble or the attack harness. The fix is to key the cache by run and rebuild with the
run's own rank, the shape the raw front end already uses in the branch below. It is a precondition
for a fine-tuned ensemble, not for scoring this member's eval.

**Robustness is unchanged by all of this.** LoRA is an accuracy lever only. Every gradient-bearing
attacker still succeeds at 98-100% across single models, four layer bands, two backbones and
feature-space hardening.

## 2026-10-17 -- LoRA on eval: the gain is real, and dev stops being a valid selector

**Claim served.** Whether fine-tuning closes the remaining gap to T45 (2.59% EER / 0.0750 minDCF)
from our 3.80%. `lora_probe`'s kept epoch-3 checkpoint was re-scored with the new
`experiment.score_only=true`: dev clean and dev codec in full (140,950 clips each, 65 min), then
eval (680,774 clips, 3 h 22 min). Nothing was retrained.

**First, two errors in the dev verdict of 2026-10-16, both mechanical.** The control was wrong:
`fullenc_seed1337`'s 0.0308 is a run that trains on a THIRD (encodec) view, and the matched
two-view control is `full_seed1337` at select 0.0210. And the margin carried no error bar, while
the 6,000-clip validation subsample has a standard error of about 0.0035 on an EER of 0.0172. Both
faults are now mechanised away in `scripts/score_report.py`, which reports a PAIRED bootstrap
difference against each named control and prints "not established" when the interval straddles
zero. Validated control-against-control before use: it resolves the 0.0026 seed gap between
`full_seed2` and `full_seed1337` as significant on 23,347 clips.

**Dev, on the full select split (69,469 clips per view), is better than the subsample said.**

| view | LoRA | control (23,347 paired) | d | CI 95% | verdict |
|---|---|---|---|---|---|
| clean | 0.0060 | 0.0119 full_seed1337 | -0.0063 | [-0.0077, -0.0049] | LoRA better |
| clean |        | 0.0142 full_seed2    | -0.0087 | [-0.0103, -0.0071] | LoRA better |
| codec | 0.0262 | 0.0302 full_seed1337 | -0.0041 | [-0.0064, -0.0018] | LoRA better |
| codec |        | 0.0276 full_seed2    | -0.0015 | [-0.0040, +0.0009] | not established |

Select mean 0.0161 against the controls' 0.0210 and 0.0209. LoRA halves the clean error and ties
the best seed on codec -- exactly what the per-epoch curve predicted.

**Eval: 3.99% EER / 0.1143 minDCF / 0.1478 minCllr.** I extrapolated 2.2% from dev using the
control's single measured dev-to-eval ratio of 1.26. That was wrong by 1.8 points: LoRA's own ratio
is 2.48. The extrapolation was flagged as single-point every time it was quoted, and it still failed.

| comparison (paired, 680,774 clips) | LoRA | control | d | CI 95% | verdict |
|---|---|---|---|---|---|
| full_seed1337_eval (2 views) | 0.0399 | 0.0512 | -0.0113 | [-0.0119, -0.0109] | LoRA better |
| full_seed2_eval (2 views)    | 0.0399 | 0.0446 | -0.0047 | [-0.0052, -0.0043] | LoRA better |
| fullenc_seed1337 (3 views)   | 0.0399 | 0.0389 | +0.0010 | [+0.0005, +0.0015] | control better |

**So fine-tuning DOES transfer to eval, by 0.5 to 1.1 points over the matched frozen controls.** The
only frozen run that beats it is the one with more channel augmentation. I first reported LoRA as
"worse than the frozen control" by comparing it to that three-view run; that was wrong and is
corrected here.

**The deficit is channel, not attack, and the per-codec table isolates it.** Eval clean is 0.0058,
against dev clean 0.0060 -- the clean gain transferred EXACTLY, losing nothing. The real codecs
never simulated in training are what break it: C07 0.0968, C04 0.0771, against C11 0.0138 (also
never simulated) and clean 0.0058. This is the channel specialisation named from the dev curve
before eval ran, and that prediction held.

**Fusion: the best eval number the project has produced, and we cannot claim it.** Score-level
fusion needs no backbone, so the `ScoreEnsemble.from_runs` LoRA refusal (white-box path only) does
not apply. `ens_2view_base` and `ens_2view_lora` differ ONLY by the added member; both drop the
encodec view, because `lora_probe` has no dev_encodec scores, so neither is comparable to the
published three-view headline.

| eval, 680,774 | best_single | mean | logistic | global_mix | gated |
|---|---|---|---|---|---|
| base, 5 members | 3.89 / .1061 | 3.62 / .1001 | 3.81 / .1051 | 3.66 / .1010 | 3.84 / .1074 |
| + lora_probe    | 3.99 / .1143 | **3.45 / .0961** | 4.02 / .1152 | 3.99 / .1142 | 4.02 / .1155 |

**But dev select ranks the combiners almost in reverse: logistic 0.0179 (best) to mean 0.0272
(worst).** Selecting on dev, which the discipline requires, picks logistic and lands on 4.02% --
worse than the base ensemble. The mechanism is explicit in the log: the learned combiners put
nearly all weight on LoRA (logistic weight 7.95 against 1.06 for the next member; global mixture
0.994 on lora_probe) because dev select, whose codecs are SIMULATED, ranks it best by a wide
margin. Eval's real codecs punish that. Only the unweighted mean keeps the other five members.

**The finding worth keeping is methodological: the dev select split is not a valid selector for a
combiner when one member's channel-robustness profile differs from the others'.** The rank
inversion is the evidence, and it is the same channel-shift story as the per-codec table and the
loss to the three-view control -- three symptoms, one cause.

**This also puts a question on the existing headline,** whose combiner is the global mixture, not
the mean (corrected 2026-10-19; this entry first said "the mean"). Whether it was dev-selected needs
checking against the three-seed studies' dev rows. Flagged, not asserted: those rows have not been
read. [Checked 2026-10-19, below: it was not.]

**Next step this argues for, and it is one run.** LoRA with `views=[clean,codec,encodec]`, about 6 h.
It attacks the single cause behind all three symptoms. The three-member fine-tuned ensemble (~34 h)
should wait for it. Robustness is untouched by any of this: LoRA is an accuracy lever, and every
gradient-bearing attacker still succeeds at 98-100%.


## 2026-10-19 -- the headline combiner was not chosen by the protocol the README states

**Claim served.** The integrity of the project's single headline number, 3.80% EER / 0.1046 minDCF.
README.md line 61 states the protocol: "combiners: fit on dev calib, chosen on dev select, eval
scored once". This checks the headline against it. No compute: the rows were already in the logs.

**The headline combiner is `global_mix`, not the mean.** README line 16 is the global mixture at
3.80% / 0.1046; the mean is line 17 at 3.80% / 0.1056. My 2026-10-17 entry said "3.80% uses the
mean" and was wrong; corrected above.

**`global_mix` is not the dev select winner on ANY of the three seeds.** Combiner order is
best_single, mean, logistic, global_mix, gated.

| seed | dev select EER (5 combiners) | dev winner | eval EER of global_mix |
|---|---|---|---|
| 1337 | .0456 .0455 .0425 .0447 .0424 | gated .0424 | .0369 |
| 1    | .0522 .0470 .0458 .0475 .0428 | gated .0428 | .0385 |
| 2    | .0509 .0472 .0464 .0470 .0428 | gated .0428 | .0386 |

**Taken literally, the protocol picks `gated` unanimously, which scores eval 3.98% / 0.1090** -- 0.18
EER points worse than the reported headline.

**But excluding `gated` from dev selection is principled and already documented**, so that is not the
honest correction. Every ensemble log prints the precondition: the within-attack descriptor label
probe FAILS on dev (EER 0.3194 today), and the line says so in words -- "the dev gated numbers are
unusable". The gate's channel descriptor carries class evidence within an attack on dev, so gated's
dev number is invalid by a test that runs before the result is read. Rejecting it is right.

**With `gated` excluded, the protocol picks `logistic` -- again unanimously, on all three seeds --
and that scores eval 3.89% / 0.1079.** Not `global_mix`.

| combiner | dev select rank | eval EER (3-seed) | eval minDCF (3-seed) |
|---|---|---|---|
| logistic   | 1st on all 3 seeds | 3.89 | 0.1079 |
| global_mix | 3rd, 4th, 3rd      | **3.80** | **0.1046** |
| gated      | (rejected by its own precondition) | 3.98 | 0.1090 |

**So the headline is 0.09 EER points and 0.0033 minDCF better than the stated protocol yields.** The
reason given for `global_mix` in the 2026-10-16 entry -- "ties the mean on EER (3.80), beats it on
minDCF (0.1046 against 0.1056)" -- quotes EVAL numbers. The seed-spread argument (0.17 against 0.28)
and the `ens_raw` argument (the mean broke on a junk member, global_mix did not) are legitimate and
do not touch eval, but they were not the stated rule, and spread was measured on eval too.

**This is the same class of error as the LoRA control and the empty attack breakdown: a decision rule
stated in one place and a different one applied in practice.** It is smaller than either -- 0.09
points -- and the project's standing beats the verified single-system baseline (4.17% / 0.119)
either way. It is not smaller than it needs to be reported.

**Two honest ways to fix it, and the choice is a judgement, not a measurement.**
  1. Report `logistic` at 3.89% / 0.1079 as the headline, which is what "chosen on dev select"
     yields once gated is excluded for failing its precondition. Costs 0.09 points, needs no compute.
  2. Keep `global_mix` and declare it A PRIORI, with a rationale that cites no eval number: its
     condition-independent shape, and that it is the only combiner that survived a junk member in
     `ens_raw`. Then say plainly in the README that the combiner is fixed by design, not selected on
     dev, and delete the "chosen on dev select" claim.
Option 2 is defensible and is probably the better system; option 1 is the one that needs no argument.
Either way README line 61 is currently false and must change.

**Not touched by this.** Every member-level number, every paired CI, the LoRA results, and the
robustness measurements. This concerns only which of five combiners the headline reports.

## 2026-10-19 -- the neural codec was the whole deficit: 3.31% EER, and fusion stops helping

**Claim served.** That fine-tuning closes the gap to T45 (2.59% / 0.0750), and that `lora_probe`'s
eval weakness was channel specialisation with one identifiable cause.

**The cause, and why `views` was not the lever.** `lora_probe` beat both matched two-view frozen
controls on eval but lost to `fullenc_seed1337`, the one frozen run trained with the cached
`encodec` view. Its per-codec table said the same thing from the other side: eval clean 0.0058, its
BEST condition and exactly its dev clean of 0.0060, against C07 0.0968 and C04 0.0771 -- the two
REAL NEURAL codecs. A features=false run had never seen a neural codec, because `CodecAugment` runs
in the DataLoader workers and EnCodec cannot (GPU only; on CPU it took minutes per batch and died,
2026-10-09). I first proposed fixing this with `views=[clean,codec,encodec]`. That would have done
NOTHING: `views` selects cached feature sets and a features=false run ignores it entirely.

**So `experiment.encodec_p` was added:** that fraction of every training batch goes through EnCodec
on the GPU, relabelled the way `extract_features` does -- MP3-then-EnCodec becomes C07, plain
EnCodec becomes C04. `lora_enc` is `lora_probe` with `encodec_p=0.3` and NOTHING else changed:
same seed, batch 32, 8 epochs, same rates, same two validation views. Cost 9.2% per step (0.535 s
against 0.490), about 4.3 min an epoch, not the 11 min I predicted.

**Measured before shipping: EnCodec's output depends on batch SIZE,** by 22-37 dB SNR, because a
hair of numerical difference at the encoder flips a residual-quantiser code. Checked for leakage
before accepting it: identical rows give identical output and a clip is unaffected by loud
neighbours, so there is no cross-talk, and the difference sits 24 dB or more BELOW the degradation
the codec itself applies. Both are valid realisations of the same codec; a batched crop is not
bit-identical to the cached view. One bandwidth per call, so the selected clips cost ONE forward --
per-clip rates cost up to five and about 30 min an epoch. A test asserts that cost property.

**Dev: select 0.0150 at the kept epoch 5, against `lora_probe`'s 0.0172.** Paired on the full split,
the trade is exactly what harder augmentation should buy: clean 0.0070 against 0.0060 (control
better, CI [+0.0005, +0.0017]) and codec 0.0224 against 0.0262 (run better, CI [-0.0048, -0.0030]).
`lora_probe` saturated at epoch 3 and its codec view then drifted WORSE, 0.0278 to 0.0313; here
codec keeps improving to epoch 5, 0.0365 to 0.0218.

**Eval: 3.31% EER / 0.0942 minDCF / 0.1256 minCllr. The best single system this project has.** Paired
over all 680,774 clips: -0.0068 against `lora_probe` [-0.0071, -0.0064] and -0.0058 against
`fullenc_seed1337` [-0.0062, -0.0053]. It also beats the five-member ensemble headline (3.80% /
0.1046) by 0.49 points, as ONE model.

**The mechanism is confirmed, not merely consistent.** The targeted conditions moved most and clean
did not move at all.

| condition | lora_probe | lora_enc | change |
|---|---|---|---|
| C07 (neural) | 0.0968 | 0.0770 | -20.5% |
| C04 (neural) | 0.0610 was 0.0771 | 0.0610 | -20.9% |
| C01 | 0.0235 | 0.0182 | -22.6% |
| C03 | 0.0312 | 0.0255 | -18.3% |
| C08 | 0.0582 | 0.0513 | -11.9% |
| C09 | 0.0327 | 0.0291 | -11.0% |
| C10 | 0.0516 | 0.0464 | -10.1% |
| C11 | 0.0138 | 0.0125 | -9.4% |
| C02 | 0.0168 | 0.0157 | -6.5% |
| C05 | 0.0075 | 0.0074 | -1.3% |
| C06 | 0.0118 | 0.0121 | +2.5% |
| clean | 0.0058 | 0.0058 | 0.0% |

Ten of eleven codecs improved; the two neural ones improved most; clean is unchanged to four
decimals. The gap to T45 is now 0.72 EER points, down from 1.21.

**And fusion has stopped helping, which REVERSES what I said an hour earlier.**

| eval, 680,774 | best_single | mean | logistic | global_mix | gated |
|---|---|---|---|---|---|
| 5 members + lora_probe | 3.99 / .1143 | 3.45 / .0961 | 4.02 / .1152 | 3.99 / .1142 | 4.02 / .1155 |
| 5 members + lora_enc   | 3.31 / .0942 | 3.35 / .0933 | 3.72 / .1065 | 3.31 / .0941 | 3.31 / .0941 |
| 5 members + both LoRAs | 3.31 / .0942 | 3.31 / .0922 | 3.84 / .1102 | 3.31 / .0941 | 3.33 / .0946 |

The learned combiners collapse ONTO `lora_enc` -- global mixture 0.995, logistic weight 9.99 -- and
simply reproduce the single model. The mean buys 0.0020 of minDCF and no EER. The reason is in the
member list: the band members sit at dev select 0.0433 to 0.1071 against `lora_enc`'s 0.0165, so
averaging a strong member with five much weaker ones cannot help, and the learned combiners are
right to discard them. **The band members are now obsolete as ensemble material.**

**What this does and does not say about the 34 h three-member plan.** It does NOT refute it: that
plan is three `lora_enc`-class members, mutually diverse, which is untested. It DOES refute the
cheap version -- bolting one strong member onto the existing five. I said an hour ago that this
result "justifies" the 34 h run; the sharper statement is that an ensemble is only worth building
from several members of this strength, and nothing here measures whether three such members
disagree enough to fuse.

**The dev/eval combiner inversion repeats.** Dev select prefers logistic (0.0158, 0.0161) and eval
punishes it (3.84%, 3.72%), for the third study running. Dev's codecs are simulated; this is the
same cause as everything else in this entry.

**Robustness unchanged.** Every gradient-bearing attacker still succeeds at 98-100%.
