#!/usr/bin/bash
# THE THREE-MEMBER FINE-TUNED ENSEMBLE. The last untested route to T45 (2.59% EER / 0.0750 minDCF).
#
# WHERE THIS STARTS. lora_enc (WavLM + LoRA 16 + encodec_p=0.3) is the project's best single system
# at eval 3.31% / 0.0942, beating the old five-member headline (3.80% / 0.1046) on its own. But
# 2026-10-19 measured that bolting it onto the five frozen band members buys NOTHING: the learned
# combiners collapse onto it (global mixture 0.995) and the mean buys 0.0020 of minDCF and no EER.
# The band members sit at dev select 0.0433-0.1071 against lora_enc's 0.0165, so they are obsolete
# as ensemble material. Fusion only pays when members are BOTH strong and in disagreement, and
# nothing here has ever measured whether several members of this strength disagree.
#   That is exactly the gap to T45: their six fine-tuned systems gain 35% relative from fusion while
#   our frozen members gain 6%. 0.72 EER points remain.
#
# THE TWO DIVERSITY AXES, one member each, chosen because they are the only two with evidence:
#   M2 lora_enc_xlsr -- BACKBONE. XLS-R-300M instead of WavLM-Large, everything else identical.
#      Architectural diversity is untested for FINE-TUNED members. It is worth noting the prior is
#      not good: a second frozen SSL backbone moved the ensemble 3.61% -> 3.60% (2026-10-13), and
#      cross-backbone transfer of adversarial perturbations was 98-99% (2026-10-16). Those say the
#      two backbones learn similar things when FROZEN. Fine-tuning is what makes them diverge, so
#      this is a real test rather than a repeat.
#   M3 lora_enc_p70 -- CHANNEL AUGMENTATION. encodec_p=0.7 instead of 0.3, same WavLM backbone.
#      This axis has measured effect on the ERROR PROFILE, which is what fusion eats: at p=0.0 the
#      model is clean-specialised (eval clean 0.0058, C07 0.0968) and at p=0.3 it is channel-robust
#      (clean 0.0058, C07 0.0770). Pushing to 0.7 should trade more clean for more codec and give a
#      third distinct profile. It may be individually WEAKER than lora_enc; that is acceptable in a
#      member and is the point of measuring rather than assuming.
#   Seed is 1337 throughout: the project's fixed-seed rule, multi-seed only for final claim variance.
#
# GATES. Member-level: each must land near lora_enc's 3.31%, not near the frozen members. Ensemble:
# the fusion must beat 3.31% by more than the 0.0020 minDCF that five weak members bought. If three
# strong members still collapse onto one, the fusion route to T45 is closed and the honest
# conclusion is that this architecture family does not produce fusable diversity.
#
# COST about 22 h serialised: 2 trainings at ~6.5-7 h, 2 scorings at ~4.4 h (dev 1 h + eval 3.4 h),
# fusion studies in minutes. lora_enc is already trained AND scored, which saves 10 h of the 34 h
# originally quoted.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
TRAIN="experiment.features=false experiment.lora_rank=16 experiment.batch_size=32 experiment.epochs=8
       experiment.backbone_lr=1.0e-5 experiment.lr=1.0e-3 data.num_workers=12 experiment.score_sets=[]"
SCORE="experiment.features=false experiment.lora_rank=16 experiment.score_only=true data.num_workers=12"
V="+views=[clean,codec]"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

# M2: the backbone axis
run lora_enc_xlsr scripts/train.py $F backbone=xlsr experiment.name=lora_enc_xlsr seed=1337 \
  $TRAIN experiment.encodec_p=0.3
run xlsrscore_dev  scripts/train.py $F backbone=xlsr experiment.name=lora_enc_xlsr seed=1337 $SCORE \
  "experiment.score_sets=[dev_clean,dev_codec]"
run xlsrscore_eval scripts/train.py $F backbone=xlsr experiment.name=lora_enc_xlsr seed=1337 $SCORE \
  "experiment.score_sets=[eval_clean]"

# M3: the augmentation axis
run lora_enc_p70 scripts/train.py $F backbone=wavlm experiment.name=lora_enc_p70 seed=1337 \
  $TRAIN experiment.encodec_p=0.7
run p70score_dev  scripts/train.py $F backbone=wavlm experiment.name=lora_enc_p70 seed=1337 $SCORE \
  "experiment.score_sets=[dev_clean,dev_codec]"
run p70score_eval scripts/train.py $F backbone=wavlm experiment.name=lora_enc_p70 seed=1337 $SCORE \
  "experiment.score_sets=[eval_clean]"

# The fusion. ens_3member is the claim; the two pairs isolate which axis carries it.
run ens_3member  scripts/ensemble.py $F $V +study=ens_3member  "+runs=[lora_enc,lora_enc_xlsr,lora_enc_p70]"
run ens_pair_bb  scripts/ensemble.py $F $V +study=ens_pair_bb  "+runs=[lora_enc,lora_enc_xlsr]"
run ens_pair_aug scripts/ensemble.py $F $V +study=ens_pair_aug "+runs=[lora_enc,lora_enc_p70]"
echo "== CHAIN DONE"
