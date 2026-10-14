#!/usr/bin/bash
# The remaining work of 2026-10-13, SEQUENTIALLY. Claim served: both are measurements the project
# goal needs -- adv_bands is the ensemble's half of the adversarial result (adv_single is already
# done: white-box PGD inverts it completely, black-box SPSA succeeds 1.1% of the time at a fixed
# operating point), and raw_seed1337 is the heterogeneous member that tests whether fusion can get
# past the 3.60% plateau the frozen-SSL members sit on.
#
# Sequential ON PURPOSE. Running them together cost about five hours today: 40 dataloader workers
# plus orphaned workers from killed runs exhausted memory (OpenBLAS allocation failure killed the
# raw member), left adv_bands with 7 seconds of CPU in two hours, and at one point two raw runs
# wrote to the same log and the same run directory. One GPU, one job.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
W=data.num_workers=8
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

# 1. the ensemble under attack, white-box and black-box, with the single model as transfer surrogate
run adv_bands scripts/attack.py $F $W "+views=[clean,codec,encodec]" \
  "+n_per_cell=40" "+steps=20" "+snr_db=[30,20]" "+spsa_samples=8" \
  +study=adv_bands "+runs=[band0_7,band7_13,band13_19,band19_25]" "+surrogate=[fullenc_seed1337]"

# 2. the raw-waveform member, then its ensemble. RawBoost is on; see chain_raw.sh for why.
run raw_seed1337 scripts/train.py $F backbone=raw $W experiment.name=raw_seed1337 seed=1337 \
  experiment.features=false experiment.batch_size=64 experiment.epochs=12 \
  experiment.rawboost=true experiment.backbone_lr=1.0e-3 experiment.lr=1.0e-3 \
  "experiment.score_sets=[dev_clean,dev_codec,eval_clean]"

run ens_raw scripts/ensemble.py $F "+views=[clean,codec]" +study=ens_raw \
  "+runs=[band0_7,band7_13,band13_19,band19_25,fullenc_seed1337,raw_seed1337]"
echo "== CHAIN DONE"
