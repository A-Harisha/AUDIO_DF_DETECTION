#!/usr/bin/bash
# Score lora_enc's kept epoch-5 checkpoint. The dev gate passed (select 0.0150 against lora_probe's
# 0.0172), but dev CANNOT see what this run was built to fix: dev's codecs are simulated and the
# fix targets eval's REAL neural codecs, C04 and C07, which were lora_probe's two worst conditions
# (0.0771 and 0.0968) against its best, clean, at 0.0058.
#
# Dev first, on the full partition, so lora_enc and lora_probe are compared on identical clips with
# a paired bootstrap. Then eval, 680,774 clips, about 3 h 22 min.
#
# THE TEST, in order of what it licenses:
#   1. eval C04 and C07 improve. That is the mechanism, and it is the only thing that confirms the
#      diagnosis rather than a coincidence.
#   2. eval overall beats lora_probe's 3.99%.
#   3. eval beats fullenc_seed1337's 3.89%, the only frozen run that beat lora_probe. If so,
#      fine-tuning plus neural augmentation is this project's best single system and the
#      three-member fine-tuned ensemble (~34 h) is finally justified.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
M="backbone=wavlm experiment.name=lora_enc seed=1337 experiment.features=false
   experiment.lora_rank=16 experiment.score_only=true data.num_workers=12"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run encscore_dev  scripts/train.py $F $M "experiment.score_sets=[dev_clean,dev_codec]"
run encscore_eval scripts/train.py $F $M "experiment.score_sets=[eval_clean]"
echo "== CHAIN DONE"
