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
