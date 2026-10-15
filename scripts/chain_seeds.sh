#!/usr/bin/bash
# The variance of the headline ensemble number. Claim served: "3.61% EER / 0.1001 minDCF beats the
# verified single-system reference (4.17% / 0.119)" rests on ONE seed, and this log is full of
# effects that turned out to be seed noise -- EMA, the 3-epoch selection window and a lower max
# learning rate were each rejected only after three seeds were measured.
#
# The single models span 0.32 points at full train (3.89 / 4.07 / 4.21). The ensemble's spread is
# unmeasured, and averaging cannot be assumed to shrink it: the four bands of one seed share a data
# order, so their errors may move together.
#
# ONE invocation trains all eight heads (4 bands x seeds 1 and 2) from a single feature load.
# Separate runs re-read 28 GB of train views, 7 GB of dev and a 34.8 GB eval cache EACH, at about
# 16 MB/s on this disk -- an hour of I/O for three minutes of training, eight times over. The first
# attempt at eight separate runs took 1h47m for run one alone. Nothing else differs: the per-head
# run directory, config, selection rule and scores are exactly what separate runs produce, which
# the throwaway seed-99 sweep confirmed (band0_7 select EER 0.1244 against band7_13's 0.0072, so
# the per-head band mask is applied).
#
# ens_mixed is 4 bands + the full-stack head, so each seed uses its own bands and its own
# full-stack head; fullenc_seed1 and fullenc_seed2 already exist. Seed 1337 is already done.
cd /d/AUDIO_DEEPFAKE
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
V="experiment.views=[clean,codec,encodec] experiment.val_views=[clean,codec,encodec] experiment.score_sets=[dev_clean,dev_codec,dev_encodec,eval_clean]"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run sweep_seeds scripts/train.py $F backbone=wavlm experiment.name=unused $V \
  "+sweep_bands=[[0,7],[7,13],[13,19],[19,25]]" "+sweep_seeds=[1,2]"

for seed in 1 2; do
  run ens_mixed_s$seed scripts/ensemble.py $F "+views=[clean,codec,encodec]" +study=ens_mixed_s$seed \
    "+runs=[band0_7_s$seed,band7_13_s$seed,band13_19_s$seed,band19_25_s$seed,fullenc_seed$seed]"
done
echo "== CHAIN DONE"
