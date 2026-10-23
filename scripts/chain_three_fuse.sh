#!/usr/bin/bash
# The fusion half of chain_three_member.sh, rerun after the descriptor guard was corrected.
# All three members are score_only runs that scored the WHOLE dev partition, so nothing forced the
# clip set down to the 47,400 with a cached channel descriptor the way a frozen member used to.
cd /d/AUDIO_DEEPFAKE
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
V="+views=[clean,codec]"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run ens_3member  scripts/ensemble.py $F $V +study=ens_3member  "+runs=[lora_enc,lora_enc_xlsr,lora_enc_p70]"
run ens_pair_bb  scripts/ensemble.py $F $V +study=ens_pair_bb  "+runs=[lora_enc,lora_enc_xlsr]"
run ens_pair_aug scripts/ensemble.py $F $V +study=ens_pair_aug "+runs=[lora_enc,lora_enc_p70]"
echo "== CHAIN DONE"
