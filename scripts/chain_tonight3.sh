#!/usr/bin/bash
# The raw-waveform member and its ensemble. adv_bands is already complete and is not repeated.
#
# Claim served: fusion of frozen-SSL members saturated at 3.60% EER (four bands, the full-stack head
# and a second SSL backbone all landed within 0.07 points), and every gradient attack on that
# ensemble succeeded completely, including transfer from a single WavLM model. This member shares no
# weights, no pretraining and no pooling grid with WavLM, so it is the first candidate to move either
# number. RawBoost is on; the run without it overfitted its training attacks (see chain_raw.sh).
#
# num_workers=20, NOT 8. This job is CPU-bound on augmentation -- RawBoost plus ffmpeg codec
# simulation per clip -- with the GPU at 0%, so worker count sets the epoch time: 20 workers gave
# 0.41 s/step (19 min/epoch) and 8 workers gave 1.13 s/step (53 min/epoch). The 8-worker setting was
# a mitigation for a memory crash caused by 63 orphaned workers from killed runs plus two concurrent
# 20-worker jobs. Neither condition holds here: one job, no orphans, 117 GB free. Reap the process
# family after any kill (Get-Process python | Stop-Process -Force) or the next run pays for it.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run raw_seed1337 scripts/train.py $F backbone=raw data.num_workers=20 \
  experiment.name=raw_seed1337 seed=1337 \
  experiment.features=false experiment.batch_size=64 experiment.epochs=12 \
  experiment.rawboost=true experiment.backbone_lr=1.0e-3 experiment.lr=1.0e-3 \
  "experiment.score_sets=[dev_clean,dev_codec,eval_clean]"

# Only dev clean + codec: the raw member has no encodec view, so the calibration scale must come
# from the views every member scored. Report this as its own row, not as a replacement for the
# 3.61% of ens_mixed, which was fit on three views.
run ens_raw scripts/ensemble.py $F "+views=[clean,codec]" +study=ens_raw \
  "+runs=[band0_7,band7_13,band13_19,band19_25,fullenc_seed1337,raw_seed1337]"
echo "== CHAIN DONE"
