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
