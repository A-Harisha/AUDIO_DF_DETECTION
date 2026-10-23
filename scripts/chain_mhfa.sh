#!/usr/bin/bash
# STEP 1 + 2 of the back-end route: the first member that sees TIME.
#
# THE PROBLEM IT ATTACKS. backbone.py time-averages every hidden layer, so a 4 s clip's 199 frames
# become one point and nothing in the model sees time. All six members ever built sit on that one
# view and differ only in WHICH layers they average, which is why they correlate at 0.85-0.99 on
# eval and fusing three fine-tuned ones gained 0.03 points for 22 h (2026-10-23). The bottleneck is
# the shared view, not the members.
#
# lora_mhfa is lora_enc with ONE change: experiment.backend=mhfa. Multi-head factorized attentive
# pooling learns WHERE in the clip to look, with two separate layer weightings -- the layers that
# say where to look need not be the layers that carry the evidence. Everything else is held:
# WavLM-Large, LoRA 16, encodec_p=0.3, batch 32, 8 epochs, seed 1337, same two validation views.
#
# Measured before launch: head 0.53M params (against Head's 0.26M), 3.94 GiB peak at batch 32 on 4 s
# crops, 9.47 GiB at batch 24 on 16 s eval clips, 174 ms of pure compute per training step. Frames
# are NOT cacheable -- un-pooled, the 34.8 GB eval cache would be about 870 GB -- so this runs
# features=false with the backbone in the loop, exactly as the LoRA runs do.
#
# THE GATE IS CORRELATION, NOT EER. ens_pair_mhfa prints the eval member correlation against
# lora_enc, and that is the decision:
#   r < 0.95 and EER near 3.31%  -> the first competent decorrelated member; the fusion route is
#                                   open again and more back-ends (LFCC hybrid, AASIST-style graph
#                                   attention -- reference code is at F:/datasets/audio/aasist) are
#                                   worth their 12 h each.
#   r > 0.98                     -> attentive pooling is not enough architectural distance. Skip
#                                   straight to the back-ends that change the REPRESENTATION, not
#                                   just the pooling.
# For reference: lora_enc + lora_enc_p70 was r=0.994, lora_enc + lora_enc_xlsr r=0.855 (but that
# member was broken at 8.67%), four frozen bands r=0.851.
#
# lora_enc stays the standing best at 3.31% / 0.0942 throughout; nothing here can make it worse.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
M="backbone=wavlm experiment.name=lora_mhfa seed=1337 experiment.features=false
   experiment.backend=mhfa experiment.lora_rank=16"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run lora_mhfa scripts/train.py $F $M experiment.batch_size=32 experiment.epochs=8 \
  experiment.backbone_lr=1.0e-5 experiment.lr=1.0e-3 experiment.encodec_p=0.3 \
  data.num_workers=12 "experiment.score_sets=[]"
run mhfascore_dev  scripts/train.py $F $M experiment.score_only=true data.num_workers=12 \
  "experiment.score_sets=[dev_clean,dev_codec]"
run mhfascore_eval scripts/train.py $F $M experiment.score_only=true data.num_workers=12 \
  "experiment.score_sets=[eval_clean]"
run ens_pair_mhfa scripts/ensemble.py $F "+views=[clean,codec]" +study=ens_pair_mhfa \
  "+runs=[lora_enc,lora_mhfa]"
echo "== CHAIN DONE"
