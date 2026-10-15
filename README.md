# CAESD — speech deepfake detection on ASVspoof 5

A speech deepfake detector on frozen self-supervised backbones. The goal is in-domain state of the
art on ASVspoof 5 track 1 with a system that is also robust under codec shift and under adversarial
attack. `docs/experiment_log.md` is the record of what is and is not supported, including what was
rejected; nothing in this README is a claim that the log does not carry.

Where it stands (2026-10-13, full eval set, official cost model: spoof prior 0.05, C_miss 1, C_fa 10):

All figures are THREE-SEED MEANS (1337, 1, 2) with the spread, not a best seed: on one seed the
ensemble reads 3.61% / 0.1001, which is the best draw of the three.

| system | EER | minDCF |
| --- | --- | --- |
| best single model (frozen WavLM + 0.26M head) | 4.06% (spread 0.33) | 0.1099 |
| ensemble, four layer bands + the full-stack head, global mixture | **3.80%** (spread **0.17**) | **0.1046** |
| ensemble, same members, unweighted mean | 3.80% (spread 0.28) | 0.1056 |
| verified best single system in the literature (arXiv:2501.05545 Table A.13) | 4.17% | 0.119 |
| ensemble SOTA, T45 challenge winner, 6 models (arXiv:2408.08739) | 2.59% | 0.0750 |

Adversarial robustness is measured, never assumed: see `scripts/attack.py` and the log. An
undefended detector has essentially none under white-box PGD.

The earlier framing of this repository — condition-dependent calibration, "block C" — was tested and
rejected twice, and the condition gate built for the ensemble was rejected a third time on eval. That
code is kept and tested but is not used for any headline result.

## Setup

```bash
uv sync                 # creates .venv (Python 3.12) and installs everything
uv run python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

ffmpeg must be on PATH with libopus, libspeex, libmp3lame, aac, libopencore_amrnb and libvo_amrwbenc.

## Protocol (ASVspoof 5, track 1)

| Partition | Attacks | Use |
| --- | --- | --- |
| train | A01-A08 | train the members |
| dev, `select` half | A09-A16 | model selection, early stopping, choosing the combiner |
| dev, `calib` half | A09-A16 | fit the combiner; a speaker-disjoint quarter selects the gate's step |
| eval | A17-A32, codecs C01-C11 | reporting only, scored once |

The two dev halves share no speaker. Train and dev clips are uncoded; codecs are simulated with
ffmpeg from `ASVspoof5.codec.config.csv`. Eval codecs C04, C07 and C11 are never simulated, so they
are unseen conditions. A18, A20, A23, A27, A30, A31 and A32 are the adversarial attacks
(Malafide / Malacopula, arXiv:2408.08739 Table 2).

## Pipeline

```bash
uv run python scripts/prepare_asvspoof5.py                                   # index the flac tars
uv run python scripts/extract_features.py +partition=train +view=clean "+backbones=[xlsr,wavlm]"
#   ... the same for views codec and encodec, and for dev and eval

# members: one head per backbone-layer band, so the members do not read the same vector
bash scripts/chain_bands.sh

# combiners: fit on dev calib, chosen on dev select, eval scored once
uv run python scripts/ensemble.py +study=ens_mixed "+views=[clean,codec,encodec]" \
    "+runs=[band0_7,band7_13,band13_19,band19_25,fullenc_seed1337]"

# adversarial robustness: white-box and black-box, reported separately
bash scripts/chain_attack.sh
```

Every run writes its own directory under `experiments/runs*/` (config, per-epoch metrics, best
checkpoint, scores) and refuses to overwrite an existing one.

## What makes the ensemble work

Members built the obvious way do not differ. Six WavLM runs across seeds, dropout and training views
all converge on the same near-uniform weighting of layers 8-11 + 24, correlate at 0.974-0.994 on
eval, and fusing them gains 0.15 EER points. `Head(band=...)` restricts a head to one slice of the
backbone, which drops pairwise correlation to 0.59-0.94 and raises the fusion gain to 0.39 points.
The weakest member helps most: `band0_7` is 10.60% EER alone and removing it makes the rest worse.

The combiner is the condition-independent global mixture: across three seeds it ties the unweighted
mean on EER, beats it on minDCF and has 40% less spread. The unweighted mean looked better on a
single seed, and it also breaks outright when one member is near chance, because it cannot
down-weight anything. A condition-dependent gate was built and rejected four times — see the log.

## Code map

| Module | Role |
| --- | --- |
| `caesd.data.sources` | ASVspoof 5 index with random access into the tars (no extraction). |
| `caesd.data.condition_grid` | Simulated codecs, named as in the ASVspoof 5 codec table. |
| `caesd.models.backbone` | Frozen XLS-R / WavLM (optional LoRA); returns every layer, time-pooled. |
| `caesd.models.detector` | Head: learned layer weights, optionally banded, plus an MLP. |
| `caesd.models.ensemble` | The waveform-to-score path of a whole ensemble, as one graph. |
| `caesd.attacks` | PGD (L-inf, SNR-set L2, EOT) and SPSA, with budget and realised SNR reported. |
| `caesd.eval.fusion` | Aligned member scores, the calib scale, the channel descriptor. |
| `caesd.train.loop` | Training with validation on unseen attacks and early stopping. |
| `caesd.metrics` | EER, minDCF, actDCF, Cllr, minCllr, fixed-operating-point metrics. |
| `caesd.models.gate`, `block_c_hypernet` | The rejected condition gate and block C. Kept, not used. |
| `caesd.models.block_b_sae`, `block_d_conformal` | Serve no live claim; candidates for removal. |
