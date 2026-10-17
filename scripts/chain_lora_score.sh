#!/usr/bin/bash
# Re-score the FINISHED lora_probe checkpoint. Nothing is trained: experiment.score_only=true loads
# experiments/runs_partial/lora_probe/model.pt (best epoch 3) and writes score parquets beside it.
#
# WHY. The probe's dev number, 0.0172 select EER, was measured on a 6,000-clip subsample, because a
# features=false run validates on experiment.val_n clips of the dev select split. The frozen
# controls validate on the 23,347 cached select clips. So the comparison that the LoRA decision
# rests on had two faults:
#   1. the control was WRONG. I used fullenc_seed1337's two-view mean, 0.0308, but that run trains
#      on a third (encodec) view and pays for it. The matched control on the same manifest, the same
#      backbone and the same two views is full_seed1337 at select 0.0210 (best seed, full_seed2,
#      0.0209). The real margin is 18%, not 44%.
#   2. the margin, 0.0210 - 0.0172 = 0.0038, is about ONE standard error of an EER measured on
#      1,372 bona fide clips (se ~ 0.0035). It is not established.
#
# STEP 1 scores the whole dev partition, 140,950 clips per view. That is 2 h rather than the 20 min
# of scoring only the 23,347 the controls used, and it buys three things: the exactly comparable
# number on those 23,347 utterances, a 3x tighter estimate on all 69,469 select clips, and the calib
# scores any later ensemble work needs. No utterance filtering is plumbed in, which is the point.
#
# STEP 2 scores eval, 680,774 clips, about 5-9 h. It is the number that decides whether LoRA closes
# the 1.21 EER points to T45 (2.59% / 0.0750) from our 3.80%. It runs second DELIBERATELY: if step 1
# says the dev gain is noise, kill this job before it spends the night.
#
# Claim served: fine-tuning the backbone is the route to the remaining gap to T45. A dev gain that
# survives a correct control and a real error bar is the precondition; eval is the measurement.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
M="backbone=wavlm experiment.name=lora_probe seed=1337 experiment.features=false
   experiment.lora_rank=16 experiment.score_only=true data.num_workers=12"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run lora_score_dev  scripts/train.py $F $M "experiment.score_sets=[dev_clean,dev_codec]"
run lora_score_eval scripts/train.py $F $M "experiment.score_sets=[eval_clean]"
echo "== CHAIN DONE"
