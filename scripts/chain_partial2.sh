#!/usr/bin/bash
# Rerun of the frozen runs with the full one-cycle schedule (patience >= epochs), then the fused
# calibration study. v1 stopped the codec-augmented runs at epoch 3-8 of 30, at peak learning rate.
cd /d/AUDIO_DEEPFAKE
P="paths.manifests=experiments/data_partial paths.features=data/features_partial paths.runs=experiments/runs_partial"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python "$@" > $L/$name.log 2>&1; echo "== $(date +%H:%M) end $name (exit $?)"; }
S="experiment.score_sets=[dev_clean,dev_codec]"
run frozen_xlsr_v2       scripts/train.py $P experiment.name=frozen_xlsr_v2 $S
run frozen_wavlm_v2      scripts/train.py $P experiment.name=frozen_wavlm_v2 backbone=wavlm $S
run frozen_xlsr_clean_v2 scripts/train.py $P experiment.name=frozen_xlsr_clean_v2 "experiment.views=[clean]" $S
run cal_fused_v2         scripts/calibrate.py $P +study=cal_fused_v2 "+runs=[frozen_xlsr_v2,frozen_wavlm_v2]"
echo "== CHAIN DONE"
