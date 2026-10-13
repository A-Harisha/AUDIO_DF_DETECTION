#!/usr/bin/bash
# A raw-waveform member, trained from scratch, then fused with the SSL members.
# Claim served: fusion of frozen-SSL members saturated at 3.60% EER / 0.1001 minDCF on 2026-10-13 --
# four layer bands, the full-stack head and a whole second SSL backbone all landed within 0.07
# points, because cross-backbone correlation (0.80-0.90) sat INSIDE the band-to-band range
# (0.59-0.94). A member that shares no weights, no pretraining and no pooling grid with any of them
# should be the first real decorrelation, and the test is whether the fused EER drops below 3.60%.
#
# What this member is, named honestly: a learnable band-pass filterbank (SincNet / RawNet2 style,
# implemented from the filter definition) plus six residual 1-D convolution stages, 0.6M parameters,
# one pooled vector per stage so the existing Head takes it unchanged. It is NOT AASIST -- there is
# no graph-attention stack -- and no result from it may be reported under that name.
#
# Sizing. The filterbank runs with a 10-sample hop (0.625 ms), not at the sample rate: without it
# one epoch took over an hour. Raw audio, so batch 64 not 512, and the whole model trains (backbone_lr is its real rate,
# 1e-3, not the 1e-5 meant for fine-tuning a pretrained backbone). 12 epochs over 182,357 clips.
# RawBoost is ON, and it is not optional here. The first attempt ran without it and overfitted its
# training attacks instead of learning a general detector: train BCE fell 0.34 -> 0.10 over four
# epochs while dev select EER went 0.341 -> 0.327 -> 0.288 -> 0.354, and A12 sat at 76-80% EER, i.e.
# the model scored A12 spoofs as MORE bona fide than real speech. A frozen SSL member inherits
# attack-independent features from its pretraining; a from-scratch raw model has to be forced to
# find them, and signal-level augmentation is that force. The verified 4.17% single system also
# trains with RawBoost (arXiv:2501.05545 Table A.13). The abandoned run is kept for the record as
# raw_norawboost_abandoned.
#
# Known limit of this pass: the audio path's CodecAugment does not do the neural codecs, so this
# member never sees an EnCodec view, which is exactly what helped the SSL members most on C04 / C07.
# If the member is worth keeping, that view is the next thing to add.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run raw_seed1337 scripts/train.py $F backbone=raw experiment.name=raw_seed1337 seed=1337 \
  experiment.features=false experiment.batch_size=64 experiment.epochs=12 \
  experiment.rawboost=true \
  experiment.backbone_lr=1.0e-3 experiment.lr=1.0e-3 \
  "experiment.score_sets=[dev_clean,dev_codec,eval_clean]"

# Fuse it with the set that won on 2026-10-13 (mean of four bands + the full-stack head, 3.61%).
# Only dev clean + codec: the raw member has no encodec view to score.
run ens_raw scripts/ensemble.py $F "+views=[clean,codec]" +study=ens_raw \
  "+runs=[band0_7,band7_13,band13_19,band19_25,fullenc_seed1337,raw_seed1337]"
echo "== CHAIN DONE"
