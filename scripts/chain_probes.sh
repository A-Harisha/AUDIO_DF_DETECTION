#!/usr/bin/bash
# Two probes, each a GATE on a much larger commitment. Sequential: one GPU, one job.
#
# PROBE 1 -- robustness. Every architectural route is now closed: single model, four layer bands on
# one backbone, and across two SSL backbones all break at 98-100% under any gradient-bearing
# attacker (2026-10-14, 2026-10-16). Only adversarial training is left, and the waveform version
# costs about 30 h. This probe is the cheap version: feature-space PGD on the cached features, which
# runs in minutes because the head is 0.26M parameters.
#   It hardens the HEAD ONLY. The attacker perturbs the waveform and arrives through the backbone,
#   so there is no reason to expect full transfer of the hardening -- the question is whether ANY of
#   it survives. If the white-box success rate drops materially below 1.000, waveform adversarial
#   training is worth its 30 h. If it stays at 1.000, the robustness section stays a measurement and
#   we stop spending compute on it.
#   eps_rel = 0.05 is a first guess, documented as arbitrary; sweep it only if the probe shows life.
#   Control: fullenc_seed1337, identical except adv_eps=0 (3.89% EER, white-box success 1.0000).
#
# PROBE 2 -- accuracy. LoRA on the full train set, DEV ONLY. It is the only credible route to the
# remaining 1.21 EER points to T45 (2.59% / 0.0750), because our members are frozen backbones with
# 0.26M heads while T45's six are fine-tuned, and their fusion gains 35% relative against our 6%.
#   LoRA has FAILED here before: on 2026-10-06, XLS-R + LoRA on 4,591 clips reached dev EER 0.217 on
#   unseen attacks while a plain logistic probe on the FROZEN features reached 0.010. The diagnosis
#   was train-attack memorisation, the same failure the raw member hit on 2026-10-14. That run also
#   confounded LoRA with FiLM, a joint Cllr loss and codec supervision, and the log says which
#   component caused the loss was never isolated. This run is LoRA alone, plain BCE, all 182,357
#   clips, validated on unseen attacks every epoch.
#   GATE: beat fullenc_seed1337's dev select EER of 0.0435 on the same three views. The failure
#   signature -- train BCE falling while dev select EER rises -- is visible by epoch 3-4, so this is
#   a 6-8 h question, not a 16 h one. eval is NOT scored: that is 9 h and is worthless until dev
#   says fine-tuning helps at all.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
V3="experiment.views=[clean,codec,encodec] experiment.val_views=[clean,codec,encodec]"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

# 1a. the adversarially trained head
run advtrain_seed1337 scripts/train.py $F backbone=wavlm experiment.name=advtrain_seed1337 seed=1337 \
  $V3 "experiment.score_sets=[dev_clean,dev_codec,dev_encodec,eval_clean]" \
  experiment.adv_eps=0.05 experiment.adv_steps=3

# 1b. the same attack battery as adv_single, so the two are directly comparable. SPSA off: it was
#     measured twice and answers a different question.
run adv_advtrain scripts/attack.py $F data.num_workers=8 "+views=[clean,codec,encodec]" \
  "+n_per_cell=40" "+steps=20" "+snr_db=[30,20]" "+spsa=false" \
  +study=adv_advtrain "+runs=[advtrain_seed1337]" "+surrogate=[fullenc_seed1337]"

# 2. the LoRA probe. batch 32 because WavLM-Large forward+backward on 4 s crops is the cost; 8
#    epochs is enough to see the gate decided either way.
run lora_probe scripts/train.py $F backbone=wavlm experiment.name=lora_probe seed=1337 \
  experiment.features=false experiment.lora_rank=16 experiment.batch_size=32 experiment.epochs=8 \
  experiment.backbone_lr=1.0e-5 experiment.lr=1.0e-3 data.num_workers=20 \
  "experiment.score_sets=[]"
echo "== CHAIN DONE"
