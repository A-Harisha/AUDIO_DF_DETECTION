#!/usr/bin/bash
# Full-train EnCodec view, three seeds. Claim served: the EnCodec training view and the full train
# set each helped separately; this is the untested cell that combines them, and the only
# configuration with a credible path under the single-system reference (EER 4.17% / minDCF 0.117,
# arXiv:2501.05545 Table A.13) on a three-seed MEAN rather than a best seed.
#
# Established evidence for the view (2026-10-09, three seeds, 40% of train): C04 / C07 EER fall
# 2.1-3.5 points on every seed; pooled mean 4.39% against 4.63% for full-train clean+codec.
#
# 1. cache the encodec view of the FULL train set, WavLM only (182,357 clips, 8.7 GB, ~6-8 h:
#    EnCodec runs on the GPU in the main process, so one backbone does not speed it up).
#    dev_encodec is already cached at full scale, so only train is extracted.
# 2. train three seeds on views clean + codec + encodec, selected on the same three dev views.
#    dropout 0.2 and select_window=1 are the config defaults: dropout 0.5 and select_window=3 were
#    both rejected, and re-picking dropout here would be a selection on eval.
# 3. score dev (for the held-out calib analysis) and eval once per seed.
# Controls: full_seed{1337,1,2} (full train, clean+codec) and frozen_wavlm_enc_seed* (40%, +encodec).
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run extract_fulltrain_encodec scripts/extract_features.py $F +partition=train +view=encodec "+backbones=[wavlm]"

V="experiment.views=[clean,codec,encodec] experiment.val_views=[clean,codec,encodec] experiment.score_sets=[dev_clean,dev_codec,dev_encodec,eval_clean]"
for seed in 1337 1 2; do
  run fullenc_seed${seed} scripts/train.py $F experiment.name=fullenc_seed${seed} seed=$seed backbone=wavlm $V
done
echo "== CHAIN DONE"
