#!/usr/bin/bash
# Adversarial robustness: the single model against the ensemble, same attack set and same budgets.
# Claim served: the ensemble's robustness is a measured result. The 2026-10-13 finding is that the
# nine near-duplicate runs correlate at 0.87-0.99, so a perturbation that moves one member moves
# them all; the layer bands cut that to 0.59-0.94, and this chain measures whether the drop buys
# robustness as well as accuracy.
#
# Both halves of the threat model, reported apart (user decision, 2026-10-13):
#   white-box  PGD on the waveform against the exact system under test, including its combiner
#   black-box  SPSA with a stated query budget, transfer from the single model as surrogate, and
#              the Malafide / Malacopula attacks already in the eval set
#
# The attack set is a stratified eval subsample, because PGD cannot run on 680,774 clips. Its clean
# metrics are reported beside the attacked ones, so no drop is ever read against the full-eval number.
# One member set per study, so adv_single and adv_bands are directly comparable.
cd /d/AUDIO_DEEPFAKE
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
V="+views=[clean,codec,encodec]"
B="band0_7,band7_13,band13_19,band19_25"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; uv run python "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

# First pass sizing: 40 clips per (codec, class) cell and 10 per (adversarial attack, codec) cell,
# 20 PGD steps, two SNR budgets. One white-box PGD pass over the set costs ~20 min on the 3090 Ti and
# SPSA at 320 queries per clip costs ~2 h, so this is one overnight run, not a sweep. Widen
# n_per_cell and steps only once the first pass says the measurement is worth more compute.
S="+n_per_cell=40 +steps=20 +snr_db=[30,20] +spsa_samples=8 data.num_workers=8"

# 1. the control: the strongest single model (3.89% EER / 0.1061 minDCF on full eval)
#run adv_single scripts/attack.py $F $V $S +study=adv_single "+runs=[fullenc_seed1337]"
# 2. the band ensemble, white-box and black-box, with the single model as the transfer surrogate
run adv_bands scripts/attack.py $F $V $S +study=adv_bands "+runs=[$B]" "+surrogate=[fullenc_seed1337]"
echo "== CHAIN DONE"
