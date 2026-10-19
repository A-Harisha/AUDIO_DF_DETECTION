#!/usr/bin/bash
# Does the LoRA member HELP THE ENSEMBLE, even though it is worse alone on eval?
#
# Alone on eval it is 3.99% EER against the frozen control's 3.89% and the ens_mixed headline's
# 3.80% (three-seed mean). But its error profile is unlike every existing member: best-in-class on
# clean eval (0.58% EER, against dev clean 0.60% -- the clean gain transferred exactly) and worst on
# the real codecs the training never simulated (C07 9.68%, C04 7.71%). Fusion pays for exactly that
# kind of disagreement, and our existing members disagree barely at all: they share one frozen
# backbone and differ only by a layer-band mask, which is why ens_mixed gains 6% relative while
# T45's six fine-tuned members gain 35%.
#
# This is score-level fusion over parquets, so it needs no GPU and no backbone. The
# NotImplementedError in ScoreEnsemble.from_runs blocks the white-box ATTACK path only.
#
# views=[clean,codec]: lora_probe has no dev_encodec scores, so the encodec view is dropped. The
# baseline is re-run on the SAME two views, so the only difference between the two studies is the
# added member -- the headline's three-view numbers are NOT comparable to these and are not used.
cd /d/AUDIO_DEEPFAKE
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
V="+views=[clean,codec]"
B="band0_7,band7_13,band13_19,band19_25"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run ens_2view_base scripts/ensemble.py $F $V +study=ens_2view_base "+runs=[$B,fullenc_seed1337]"
run ens_2view_lora scripts/ensemble.py $F $V +study=ens_2view_lora "+runs=[$B,fullenc_seed1337,lora_probe]"
echo "== CHAIN DONE"
