#!/usr/bin/bash
# Does the diversity gain keep scaling with narrower bands? Claim served: the 2026-10-13 result that
# four 6-layer bands decorrelate to 0.813 and combine to 3.67% EER / 0.1011 minDCF, against 3.89% /
# 0.1061 for the best single model. If diversity is the lever, eight 3-layer bands should decorrelate
# further and combine better; if the members get too weak, it should get worse. Either answer is
# worth the 40 minutes, because it tells us whether to spend compute on more members or on stronger
# ones (LoRA fine-tuning, the lever the log names next).
#
# Eight bands of three layers over WavLM-Large's 25 outputs, seed 1337, everything else at the
# config default so the band width is the only change. Layer 0 (the CNN output) is left out: it is
# the one output with no transformer context and band0_7 already showed the early layers are the
# weak end (10.60% EER alone).
# Controls: band{0_7,7_13,13_19,19_25} (four 6-layer bands) and fullenc_seed1337 (no band).
cd /d/AUDIO_DEEPFAKE
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
V="experiment.views=[clean,codec,encodec] experiment.val_views=[clean,codec,encodec] experiment.score_sets=[dev_clean,dev_codec,dev_encodec,eval_clean]"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

NAMES=""
for band in "1,4" "4,7" "7,10" "10,13" "13,16" "16,19" "19,22" "22,25"; do
  tag=b3_$(echo $band | tr ',' '_')
  NAMES="$NAMES,$tag"
  run $tag scripts/train.py $F experiment.name=$tag seed=1337 backbone=wavlm $V "experiment.band=[$band]"
done
run ens_bands8 scripts/ensemble.py $F "+views=[clean,codec,encodec]" +study=ens_bands8 "+runs=[${NAMES#,}]"
echo "== CHAIN DONE"
