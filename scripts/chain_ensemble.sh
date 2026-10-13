#!/usr/bin/bash
# Ensemble studies over the layer-band members. Claim served: once members read different slices of
# the backbone their scores decorrelate, and only then does a combiner win anything worth the name.
#
# Measured baseline to beat (docs/experiment_log.md, 2026-10-13): nine near-duplicate runs correlate
# at 0.87-0.99 on eval and their best fusion gains 0.15 EER points over the best single member
# (3.89% -> 3.74%, itself selected on eval). Ensemble SOTA is T45 at 2.59% / 0.0750 (arXiv:2408.08739).
#
# Three member sets, each fitted on dev calib, chosen on dev select, scored once on eval:
#   ens_bands   the four WavLM bands alone -- does banding decorrelate at all?
#   ens_mixed   the bands plus the unbanded full-stack head (fullenc_seed1337, 3.89% / 0.1061)
#   ens_cross   ens_mixed plus the XLS-R member (frozen_xlsr_enc_drop50, 5.97%), the only member
#               with a different backbone, so a second kind of diversity is in the set
cd /d/AUDIO_DEEPFAKE
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
V="+views=[clean,codec,encodec]"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

B="band0_7,band7_13,band13_19,band19_25"
run ens_bands scripts/ensemble.py $F $V +study=ens_bands "+runs=[$B]"
run ens_mixed scripts/ensemble.py $F $V +study=ens_mixed "+runs=[$B,fullenc_seed1337]"
run ens_cross scripts/ensemble.py $F $V +study=ens_cross "+runs=[$B,fullenc_seed1337,frozen_xlsr_enc_drop50]"
echo "== CHAIN DONE"
