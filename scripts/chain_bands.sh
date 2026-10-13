#!/usr/bin/bash
# Layer-band ensemble members. Claim served: member DIVERSITY, not the fusion rule, is what blocks
# an ensemble from beating the ASVspoof 5 ensemble SOTA (T45, 2.59% EER / 0.0750 minDCF,
# arXiv:2408.08739), and banding the layer weighting is what creates it.
#
# Measured premise (docs/experiment_log.md, 2026-10-13): fusing the nine runs already scored on eval
# gains 0.15 EER points at best (3.89% -> 3.74%, itself selected on eval), because every WavLM head
# converges to the same near-uniform weighting of layers 8-11 + 24 and their eval scores correlate
# at 0.98-0.99. Cross-backbone correlation is still 0.89. A frozen backbone's layers differ strongly
# (a linear probe gets 2.2% EER on the last layer and 1.0% on layer 20), so a head restricted to one
# band must use a different feature path.
#
# Four bands over WavLM-Large's 25 outputs (CNN output + 24 transformer layers), seed 1337, views
# clean + codec + encodec, every other setting at the config default so the band is the only change.
# Control: fullenc_seed1337 (identical, no band): 3.89% EER / 0.1061 minDCF.
#
# Reported next: per-member eval EER, the member score correlation matrix, and the gain of the plain
# mean over the best member. The condition gate is built only if the correlation actually drops.
cd /d/AUDIO_DEEPFAKE
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

V="experiment.views=[clean,codec,encodec] experiment.val_views=[clean,codec,encodec] experiment.score_sets=[dev_clean,dev_codec,dev_encodec,eval_clean]"
for band in "0,7" "7,13" "13,19" "19,25"; do
  tag=band$(echo $band | tr ',' '_')
  run $tag scripts/train.py $F experiment.name=$tag seed=1337 backbone=wavlm $V "experiment.band=[$band]"
done
echo "== CHAIN DONE"
