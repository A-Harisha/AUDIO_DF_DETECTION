#!/usr/bin/bash
# Block C's channel descriptor for every cached feature set. CPU only: no backbone is loaded.
# Claim served: block C needs a condition input that cannot carry class evidence (the SSL-feature
# codec posterior did -- docs/experiment_log.md, 2026-10-08). Writes
# data/features_partial/descriptor/<partition>_<view>.npy, aligned to the feature parquets by utt.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.features=data/features_partial"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }
D="+descriptor_only=true"
for v in clean codec; do
  run desc_train_$v scripts/extract_features.py $F paths.manifests=experiments/data_partial +partition=train +view=$v $D
  run desc_dev_$v   scripts/extract_features.py $F paths.manifests=experiments/data_partial +partition=dev   +view=$v $D
done
run desc_eval_clean scripts/extract_features.py $F paths.manifests=experiments/data_full +partition=eval +view=clean $D
echo "== CHAIN DONE"
