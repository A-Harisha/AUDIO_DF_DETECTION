#!/usr/bin/bash
# Head sweep on cached features, judged on dev select only (eval is not scored here).
# Claim served: the head overfits its train attacks (best epoch 3 of 30); seeds give the
# run-to-run noise that every single-seed comparison so far lacks, and two regularisation settings
# test whether the overfit can be reduced. Baselines: frozen_{xlsr,wavlm}_final (seed 1337).
cd /d/AUDIO_DEEPFAKE
P="paths.features=data/features_partial paths.runs=experiments/runs_partial"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python scripts/train.py $P experiment.name=$name "experiment.score_sets=[dev_clean,dev_codec]" "$@" > $L/$name.log 2>&1; echo "== $(date +%H:%M) end $name (exit $?)"; }
for bb in wavlm xlsr; do
  run sweep_${bb}_seed1   backbone=$bb seed=1
  run sweep_${bb}_seed2   backbone=$bb seed=2
  run sweep_${bb}_drop50  backbone=$bb experiment.dropout=0.5
  run sweep_${bb}_wd      backbone=$bb experiment.weight_decay=0.1 experiment.lr=3e-4
done
echo "== CHAIN DONE"
