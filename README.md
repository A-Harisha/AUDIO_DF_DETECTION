# CAESD — speech deepfake detection on ASVspoof 5

A speech deepfake detector on self-supervised backbones. The goal is in-domain state of the art on
ASVspoof 5 track 1 with a system that is also robust under codec shift and under adversarial attack.
`docs/experiment_log.md` is the record of what is and is not supported, including what was rejected
and which of our own predictions failed; nothing in this README is a claim the log does not carry.

Where it stands (2026-10-24, full eval set, official cost model: spoof prior 0.05, C_miss 1, C_fa 10):

| system | EER | minDCF |
| --- | --- | --- |
| frozen WavLM + 0.26M head, three-seed mean | 4.06% (spread 0.33) | 0.1099 |
| four frozen layer bands + full-stack head, global mixture, three-seed mean | 3.80% (spread 0.17) | 0.1046 |
| LoRA + neural-codec augmentation (`lora_enc`) | 3.31% | 0.0942 |
| the same features with attentive pooling (`lora_mhfa`) | 3.44% | 0.0957 |
| **mean of `lora_enc` and `lora_mhfa`** | **3.21%** | **0.0905** |
| verified best single system in the literature (arXiv:2501.05545 Table A.13) | 4.17% | 0.119 |
| challenge winner T45 = USTC-KXDIGIT, 12 systems (arXiv:2409.01695) | 2.59% | 0.0750 |

**Read the seeds before comparing rows.** The first two rows are three-seed means (1337, 1, 2) with
the spread. Every row below them is **one seed, 1337**, so its variance is unmeasured; the frozen
single models varied by 0.33 EER points across seeds, which is larger than the 0.10-point gain the
two-member mean shows over `lora_enc` alone. The headline is the best result we have, not a result
with error bars over seeds.

**The comparison to T45 is not like for like, in both directions.** T45 fuses 12 systems over three
back-end families and trains on four corpora (ASVspoof 5, ASVspoof 2021 LA+DF, their own business
data, and VITS-synthesised fakes); we fuse two and train on one. Their fusion weights were also
chosen manually on the evaluation set progress subset, while every number here is selected on dev
with eval scored once.

Adversarial robustness is measured, never assumed. Any gradient-bearing attacker succeeds on 98-100%
of clips; a query-budgeted black-box attacker succeeds on about 1%. Four defences were tested and
all failed: a single model, four layer bands, two SSL backbones, and feature-space adversarial
training. See `scripts/attack.py` and the log. Waveform adversarial training is the one route not
yet tried.

The earlier framing of this repository — condition-dependent calibration, "block C" — was tested and
rejected twice, and the condition gate built for the ensemble was rejected a third time on eval.
That code is kept and tested but is not used for any headline result.

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
| dev, `calib` half | A09-A16 | fit the combiner; a speaker-disjoint quarter selects the gate step |
| eval | A17-A32, codecs C01-C11 | reporting only, scored once |

The two dev halves share no speaker. Train and dev clips are uncoded; codecs are simulated with
ffmpeg from `ASVspoof5.codec.config.csv`, and EnCodec is applied on the GPU during training
(`experiment.encodec_p`). Ten of the eleven eval conditions are simulated in training; only C11
(devices) is not. A18, A20, A23, A27, A30, A31 and A32 are the adversarial attacks (Malafide /
Malacopula, arXiv:2408.08739 Table 2).

The headline mean is chosen by this protocol: on dev `select` the unweighted mean ranks first
(0.0156) ahead of the best single member (0.0165), and it also ranks first on eval. That had not
been true before — in five earlier studies dev preferred a learned combiner that eval then punished,
and the retired 3.80% headline used a combiner that was never the dev winner (log, 2026-10-19).

## Pipeline

```bash
uv run python scripts/prepare_asvspoof5.py                                   # index the flac tars
uv run python scripts/extract_features.py +partition=train +view=clean "+backbones=[xlsr,wavlm]"
#   ... the same for views codec and encodec, and for dev and eval. Cached features are the
#   TIME-MEAN of every layer; a back-end that models time cannot use them (see below).

bash scripts/chain_lora_enc.sh     # member 1: LoRA + EnCodec augmentation
bash scripts/chain_mhfa.sh         # member 2: attentive pooling over the frames, then the fusion

# score a finished run on more sets without retraining
uv run python scripts/train.py experiment.name=lora_enc experiment.score_only=true \
    experiment.features=false experiment.lora_rank=16 "experiment.score_sets=[eval_clean]"

# the verdict, with paired bootstrap CIs against named controls
uv run python scripts/score_report.py +run=lora_enc "+sets=[eval_clean]" +group=codec

bash scripts/chain_attack.sh       # adversarial robustness, white-box and black-box separately
```

Every run writes its own directory under `experiments/runs*/` (config, per-epoch metrics, best
checkpoint, scores) and refuses to overwrite an existing one.

## What makes the ensemble work: the view, not the members

Fusion pays only for members that fail on different CLIPS, and almost nothing we tried produced
them. Six WavLM runs across seeds, dropout and training views converge on the same near-uniform
weighting of layers 8-11 + 24 and correlate at 0.974-0.994. `Head(band=...)` restricts a head to one
slice of the backbone, which drops correlation to 0.851 and was worth 0.39 points — but those
members are weak (4.06% at best) and are now obsolete: adding `lora_enc` to them gains nothing,
because the combiners discard them.

Three fine-tuned members did not fuse either. The diagnosis is in the correlations:

| members | eval correlation | best member | fusion gain |
| --- | --- | --- | --- |
| `lora_enc` + `lora_enc_p70` (more EnCodec augmentation) | 0.994 | 3.37% | 0.03 pts |
| `lora_enc` + `lora_enc_xlsr` (XLS-R backbone) | 0.855 | 8.67% — broken | none |
| `lora_enc` + `lora_mhfa` (attentive pooling) | **0.917** | 3.44% | **0.10 pts** |

Diversity and competence were anti-correlated until the view changed. `lora_enc_p70` has a
complementary per-codec profile and still correlates at 0.994: differing on which CONDITIONS you
fail is not differing on which CLIPS you fail. The cause was that every member consumed the
time-MEAN of the backbone layers, so 199 frames of a 4 s clip became one point and nothing saw time.
`Backbone(pooled=False)` returns the frames, and `MHFA` pools them by learned attention — the first
member with a different view of the same features, and the first fusion gain worth having.

XLS-R under LoRA memorises its training attacks where WavLM does not: train BCE falls ninefold while
dev EER rises, with A12 and A13 carrying the whole failure. It reproduces a failure first seen on
2026-10-06, which was also XLS-R.

## Code map

| Module | Role |
| --- | --- |
| `caesd.data.sources` | ASVspoof 5 index with random access into the tars (no extraction). |
| `caesd.data.condition_grid` | Simulated codecs, EnCodec on the GPU, the channel descriptor. |
| `caesd.models.backbone` | XLS-R / WavLM, optional LoRA; every layer, time-pooled or as frames. |
| `caesd.models.detector` | `Head` (layer weights + MLP) and `MHFA` (attentive pooling over frames). |
| `caesd.models.ensemble` | The waveform-to-score path of a whole ensemble, as one graph. |
| `caesd.attacks` | PGD (L-inf, SNR-set L2, EOT) and SPSA, with budget and realised SNR reported. |
| `caesd.eval.fusion` | Aligned member scores, the calib scale, the channel descriptor. |
| `caesd.train.loop` | Training with validation on unseen attacks and early stopping. |
| `caesd.metrics` | EER, minDCF, actDCF, Cllr, minCllr, paired bootstrap CIs, operating points. |
| `scripts/score_report.py` | Score parquets to a verdict, with paired CIs and a "not established" call. |
| `caesd.models.gate`, `block_c_hypernet` | The rejected condition gate and block C. Kept, not used. |
| `caesd.models.block_b_sae`, `block_d_conformal` | Serve no live claim; candidates for removal. |
