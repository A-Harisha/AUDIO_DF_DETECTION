#!/usr/bin/bash
# 1. ens_raw: does a very weak but very DECORRELATED member help or hurt?
#    raw_seed1337 is 36.61% EER / 0.8478 minDCF on eval -- near chance, not a usable detector -- but
#    its eval correlation with the SSL members is 0.233-0.592 against 0.59-0.94 among the bands. The
#    precedent says weak-and-decorrelated can help: band0_7 is 10.60% alone and removing it makes the
#    rest worse. This asks where that stops. It is also the first case where a LEARNED combiner
#    should beat the plain mean, because global_mix and logistic can down-weight a junk member while
#    an unweighted mean cannot.
# 2. chain_seeds: the error bar for the 3.61% headline, which currently rests on one seed.
cd /d/AUDIO_DEEPFAKE
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run ens_raw scripts/ensemble.py $F "+views=[clean,codec]" +study=ens_raw \
  "+runs=[band0_7,band7_13,band13_19,band19_25,fullenc_seed1337,raw_seed1337]"
bash scripts/chain_seeds.sh
echo "== FOLLOWUP DONE"
