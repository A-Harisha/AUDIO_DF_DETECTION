#!/usr/bin/bash
# The variance of the headline ensemble number. Claim served: "3.61% EER / 0.1001 minDCF beats the
# verified single-system reference (4.17% / 0.119)" currently rests on ONE seed, and this log is full
# of effects that turned out to be seed noise -- EMA, the 3-epoch selection window and a lower max
# learning rate were each rejected after measuring three seeds.
#
# The single models span 0.32 points at full train (3.89 / 4.07 / 4.21). The ensemble's spread is
# unmeasured, and it is not safe to assume averaging shrinks it: the four bands of a given seed are
# trained on the same data order, so their errors may move together across seeds.
#
# ens_mixed is 4 bands + the full-stack head, so each seed needs its own bands AND its own
# full-stack head; fullenc_seed1 and fullenc_seed2 already exist. Seed 1337 is already done
# (ens_mixed). Eight band runs at about 5 minutes each on cached features, plus two studies.
# Cheapest necessary work in the queue: about an hour for the error bar the claim needs.
cd /d/AUDIO_DEEPFAKE
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
V="experiment.views=[clean,codec,encodec] experiment.val_views=[clean,codec,encodec] experiment.score_sets=[dev_clean,dev_codec,dev_encodec,eval_clean]"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

for seed in 1 2; do
  NAMES=""
  for band in "0,7" "7,13" "13,19" "19,25"; do
    tag=band$(echo $band | tr ',' '_')_s$seed
    NAMES="$NAMES,$tag"
    run $tag scripts/train.py $F experiment.name=$tag seed=$seed backbone=wavlm $V "experiment.band=[$band]"
  done
  run ens_mixed_s$seed scripts/ensemble.py $F "+views=[clean,codec,encodec]" +study=ens_mixed_s$seed \
    "+runs=[${NAMES#,},fullenc_seed$seed]"
done
echo "== CHAIN DONE"
