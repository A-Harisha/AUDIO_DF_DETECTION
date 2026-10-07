# CAESD — Condition-Aware Evidence for Speech Deepfake Detection

A speech deepfake detector built on a frozen self-supervised backbone, with a post-hoc calibration
stage that depends on the recording condition (codec). The open research question is whether
condition-dependent calibration gives better likelihood ratios under codec shift than one global
calibration. `docs/experiment_log.md` is the record of what is and is not supported so far.

See [Condition-Aware Evidence for Speech Deepfake Detection.md](Condition-Aware%20Evidence%20for%20Speech%20Deepfake%20Detection.md)
for the research proposal and [Claude.md](Claude.md) for the working rules.

## Setup

```bash
uv sync                 # creates .venv (Python 3.12) and installs everything
uv run python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

ffmpeg must be on PATH with libopus, libspeex, libmp3lame, aac, libopencore_amrnb and libvo_amrwbenc.

## Protocol (ASVspoof 5, track 1)

| Partition | Attacks | Use |
| --- | --- | --- |
| train | A01-A08 | train the detector and the codec classifier |
| dev, `select` half | A09-A16 | model selection, early stopping |
| dev, `calib` half | A09-A16 | fit score fusion and every calibration method |
| eval | A17-A32, codecs C01-C11 | reporting only |

The two dev halves share no speaker. Train and dev clips are uncoded; codecs are simulated with
ffmpeg from `ASVspoof5.codec.config.csv`. Eval codecs C04, C07 and C11 are never simulated, so
they are unseen conditions.

## Pipeline

```bash
uv run python scripts/prepare_asvspoof5.py                                   # index the flac tars
uv run python scripts/extract_features.py +partition=train +view=clean "+backbones=[xlsr,wavlm]"
uv run python scripts/extract_features.py +partition=train +view=codec "+backbones=[xlsr,wavlm]"
#   ... the same for dev (clean, codec) and eval (clean)
uv run python scripts/train.py experiment.name=frozen_xlsr                    # head on cached features
uv run python scripts/train.py experiment.name=frozen_wavlm backbone=wavlm
uv run python scripts/train.py experiment.name=lora_xlsr experiment.features=false experiment.lora_rank=16
uv run python scripts/calibrate.py +study=cal_fused "+runs=[frozen_xlsr,frozen_wavlm]" +eval=true
```

Every run writes its own directory under `experiments/runs/` (config, per-epoch metrics, best
checkpoint, scores) and refuses to overwrite an existing one.

## Code map

| Module | Role |
| --- | --- |
| `caesd.data.sources` | ASVspoof 5 index with random access into the tars (no extraction). |
| `caesd.data.condition_grid` | Simulated codecs, named as in the ASVspoof 5 codec table. |
| `caesd.models.backbone` | Frozen XLS-R / WavLM (optional LoRA); returns every layer, time-pooled. |
| `caesd.models.detector` | Head: learned layer weights + MLP. Optional block A + FiLM. |
| `caesd.models.block_c_hypernet` | Block C: post-hoc calibration hypernetwork. |
| `caesd.train.loop` | Training with validation on unseen attacks and early stopping. |
| `caesd.metrics`, `caesd.calibration` | EER, minDCF, actDCF, Cllr, minCllr; logistic and lookup calibration. |
| `caesd.models.block_b_sae`, `block_d_conformal` | Phase 3 (evidence, competence gate); not yet used. |
