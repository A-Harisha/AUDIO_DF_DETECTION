#!/usr/bin/bash
# Eval-set run. Claim served: the benchmark number (ASVspoof 5 eval, A17-A32, codecs C01-C11).
# 1. cache eval features for both backbones (~5 h GPU); 2. retrain the two frozen heads with the
# same seed and settings as *_v2 and score eval (the head trains in ~1 min); 3. fusion + calibration
# study on the eval set. Eval is only scored: no decision uses it.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.features=data/features_partial paths.runs=experiments/runs_partial"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }
run extract_eval     scripts/extract_features.py $F paths.manifests=experiments/data_full +partition=eval +view=clean "+backbones=[xlsr,wavlm]"
S="experiment.score_sets=[dev_clean,dev_codec,eval_clean]"
run frozen_xlsr_final  scripts/train.py $F experiment.name=frozen_xlsr_final $S
run frozen_wavlm_final scripts/train.py $F experiment.name=frozen_wavlm_final backbone=wavlm $S
run cal_eval           scripts/calibrate.py $F +study=cal_eval "+runs=[frozen_xlsr_final,frozen_wavlm_final]" +eval=true
echo "== CHAIN DONE"
