#!/usr/bin/bash
# Interim experiments on the partial download (train tars aa+ab = 40%, dev tar aa = 34%).
# Frozen-feature runs only: they take minutes. LoRA and full fine-tuning (about 1 h per epoch on
# this data, and they compete with feature extraction for the GPU) wait for the full data.
# Sequential: one GPU job at a time. Starts when the last cached feature file is complete.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
P="paths.manifests=experiments/data_partial paths.features=data/features_partial paths.runs=experiments/runs_partial"
L=experiments/logs_partial; mkdir -p $L
until [ -f data/features_partial/wavlm_large/train_codec.parquet ]; do sleep 60; done
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python "$@" > $L/$name.log 2>&1; echo "== $(date +%H:%M) end $name (exit $?)"; }
S="experiment.score_sets=[dev_clean,dev_codec]"
run frozen_xlsr        scripts/train.py $P experiment.name=frozen_xlsr $S
run frozen_wavlm       scripts/train.py $P experiment.name=frozen_wavlm backbone=wavlm $S
run frozen_xlsr_clean  scripts/train.py $P experiment.name=frozen_xlsr_clean "experiment.views=[clean]" $S
run film_xlsr          scripts/train.py $P experiment.name=film_xlsr experiment.film=true $S
run cal_frozen_xlsr    scripts/calibrate.py $P +study=cal_frozen_xlsr "+runs=[frozen_xlsr]"
run cal_frozen_wavlm   scripts/calibrate.py $P +study=cal_frozen_wavlm backbone=wavlm "+runs=[frozen_wavlm]"
run cal_fused          scripts/calibrate.py $P +study=cal_fused "+runs=[frozen_xlsr,frozen_wavlm]"
echo "== CHAIN DONE"
