#!/usr/bin/bash
# LoRA WITH the neural codec in training. One variable against lora_probe: experiment.encodec_p.
#
# WHY. lora_probe (2026-10-16/17) beats both matched two-view frozen controls on eval -- 3.99% EER
# against 5.12% and 4.46%, paired over all 680,774 clips -- and loses only to fullenc_seed1337
# (3.89%), the one frozen run trained with the cached `encodec` view. Its per-codec table names the
# same cause from the other side: eval clean 0.0058 (its best condition, and exactly its dev clean
# of 0.0060, so the clean gain transferred without loss) against C07 0.0968 and C04 0.0771 -- the
# two REAL NEURAL codecs. A features=false run never saw a neural codec, because CodecAugment runs
# in the DataLoader workers and EnCodec cannot (it needs the GPU; on CPU it took minutes per batch
# and died, 2026-10-09).
#   The same cause explains the third symptom: the ensemble study's dev/eval rank inversion, where
#   dev select prefers logistic (0.0179) and eval punishes it (4.02%) because the learned combiners
#   put 0.994 of the mixture on LoRA -- dev's codecs are SIMULATED and flatter it.
#
# So `experiment.encodec_p=0.3` puts 30% of every training batch through EnCodec on the GPU. Note
# `views` is NOT the lever here: it only selects cached feature sets, and a features=false run
# ignores it entirely.
#
# HOLD EVERYTHING ELSE: same seed, same batch 32, same 8 epochs, same learning rates, same two
# validation views. lora_probe is the control and the comparison must be one variable.
#
# GATES, in order of what they would license:
#   1. dev select beats lora_probe's 0.0161 (full split) -- or at least does not lose on clean,
#      which is where the fine-tuning gain lives. A dev LOSS here with an eval win is possible and
#      is the point: dev's codecs are simulated, so dev cannot see this fix working.
#   2. eval beats 3.99%, and C04 / C07 improve specifically. That is the real test.
#   3. eval beats fullenc_seed1337's 3.89%. If so, fine-tuning plus neural augmentation is the best
#      single system this project has, and the three-member ensemble (~34 h) is finally justified.
# eval is NOT scored here: that is 3 h 22 min and is spent only once dev says the run is sane.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run lora_enc scripts/train.py $F backbone=wavlm experiment.name=lora_enc seed=1337 \
  experiment.features=false experiment.lora_rank=16 experiment.batch_size=32 experiment.epochs=8 \
  experiment.backbone_lr=1.0e-5 experiment.lr=1.0e-3 experiment.encodec_p=0.3 \
  data.num_workers=12 "experiment.score_sets=[]"
echo "== CHAIN DONE"
