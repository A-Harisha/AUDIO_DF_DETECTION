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

## Phase 0 — go / no-go (pending)

Record here, after running `scripts/phase0_falsify.py`:

- [ ] Global-calibration gap (actDCF − minDCF) per condition and per language.
- [ ] Lookup-table closure on seen cells vs held-out codecs/languages.
- [ ] Threshold-transfer drift from clean dev to each held-out condition.
- [ ] **Decision:** GO / NO-GO, with the numbers that justify it.
