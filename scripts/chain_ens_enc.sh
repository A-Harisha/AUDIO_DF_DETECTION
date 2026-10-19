#!/usr/bin/bash
# lora_enc is now the best SINGLE system in the project (eval 3.31% / 0.0942), beating the
# five-member ensemble headline (3.80% / 0.1046). So the fusion must be refitted with it: a member
# that beats the whole ensemble changes which members are worth keeping.
#
# Same two views as ens_2view_base / ens_2view_lora, so all three studies are comparable and the
# only change is which fine-tuned member is present. Score-level fusion: no GPU, no backbone.
cd /d/AUDIO_DEEPFAKE
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
V="+views=[clean,codec]"
B="band0_7,band7_13,band13_19,band19_25"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run ens_2view_enc  scripts/ensemble.py $F $V +study=ens_2view_enc  "+runs=[$B,fullenc_seed1337,lora_enc]"
run ens_2view_both scripts/ensemble.py $F $V +study=ens_2view_both "+runs=[$B,fullenc_seed1337,lora_probe,lora_enc]"
echo "== CHAIN DONE"
