#!/usr/bin/bash
# Eval-set scoring of the full-train WavLM head. Claim served: the benchmark number
# (ASVspoof 5 eval, unseen attacks A17-A32, real codecs C01-C11).
#
# 1. cache eval features for WavLM only (680,774 clips, 32.5 GB, ~3-4 h GPU). The eval manifest
#    already carries the real codec per clip, so the view is `clean`: the clip as released.
# 2. retrain the three full-train heads under *_eval names and score eval. train.py scores only at
#    the end of a run and there is no standalone scorer, so the head is refit (~1 min, identical
#    given the fixed seed) instead of loading model.pt.
# Eval is only scored: no decision uses it.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.features=data/features_fulltrain paths.runs=experiments/runs_partial paths.manifests=experiments/data_full"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run extract_eval_clean scripts/extract_features.py $F +partition=eval +view=clean "+backbones=[wavlm]"

S="experiment.views=[clean,codec] experiment.val_views=[clean,codec] experiment.score_sets=[eval_clean]"
for seed in 1337 1 2; do
  run full_seed${seed}_eval scripts/train.py $F experiment.name=full_seed${seed}_eval seed=$seed backbone=wavlm $S
done
echo "== CHAIN DONE"
