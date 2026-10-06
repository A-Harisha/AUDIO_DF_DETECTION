# CAESD — Condition-Aware Evidence for Speech Deepfake Detection

One model that estimates each recording's **condition** (codec, bandwidth, SNR, language), scales the forensic
evidence (LLR) to that condition through a calibration hypernetwork, and **refuses to judge** outside a validated
region. The headline claim is an accuracy claim: *at a fixed operating point, the system makes fewer errors under
channel and language shift.*

See [Condition-Aware Evidence for Speech Deepfake Detection.md](Condition-Aware%20Evidence%20for%20Speech%20Deepfake%20Detection.md)
for the full research proposal and [Claude.md](Claude.md) for the working rules.

## Setup

```bash
uv sync                 # creates .venv (Python 3.12) and installs everything
uv run python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

ffmpeg must be on PATH (codecs: mp3, opus, amr, aac) for the condition grid.

## Architecture

| Block | Module | Role |
| --- | --- | --- |
| Backbone | `caesd.models.backbone` | Frozen XLS-R / WavLM + LoRA; emits SSL features. |
| A | `caesd.models.block_a_condition` | Condition encoder → continuous condition vector. |
| B | `caesd.models.block_b_sae` | Sparse autoencoder → named evidence directions. |
| C | `caesd.models.block_c_hypernet` | Hypernetwork → per-condition calibration params (**core novelty**). |
| D | `caesd.models.block_d_conformal` | Mondrian conformal competence gate. |

## Pipeline

| Phase | Goal |
| --- | --- |
| 0 | Falsify the premise with pretrained AASIST → go/no-go. |
| 1 | Establish the problem; build baselines (measurement paper). |
| 2 | Blocks A + C + generalization mechanisms (method paper). |
| 3 | Blocks B + D; faithfulness + conformal coverage (evidence paper). |

Run the go/no-go check:

```bash
uv run python scripts/phase0_falsify.py experiment=phase0
```
