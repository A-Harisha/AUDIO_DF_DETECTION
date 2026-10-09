#!/usr/bin/bash
# EnCodec training views. Claim served: eval C04 / C07 (EnCodec) are the hardest conditions because
# the detector never saw a neural codec; simulating it in training should lower their EER.
# 1. cache the encodec view of train and dev for both backbones; 2. retrain the frozen heads with
# views clean + codec + encodec, selected on dev clean + codec + encodec; 3. score eval once.
# Controls: the *_drop50_final runs (clean + codec, dropout 0.5, same seed, same data).
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
P="paths.manifests=experiments/data_partial paths.features=data/features_partial paths.runs=experiments/runs_partial"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }
run extract_dev_encodec   scripts/extract_features.py $P +partition=dev   +view=encodec "+backbones=[xlsr,wavlm]"
run extract_train_encodec scripts/extract_features.py $P +partition=train +view=encodec "+backbones=[xlsr,wavlm]"
V="experiment.dropout=0.5 experiment.views=[clean,codec,encodec] experiment.val_views=[clean,codec,encodec] experiment.score_sets=[dev_clean,dev_codec,dev_encodec,eval_clean]"
run frozen_wavlm_enc_drop50 scripts/train.py $P experiment.name=frozen_wavlm_enc_drop50 backbone=wavlm $V
run frozen_xlsr_enc_drop50  scripts/train.py $P experiment.name=frozen_xlsr_enc_drop50 $V
echo "== CHAIN DONE"
