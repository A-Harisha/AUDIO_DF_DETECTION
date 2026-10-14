#!/usr/bin/bash
# Does an adversarial gradient need a SHARED BACKBONE to transfer?
#
# Claim served: adversarial robustness as a measured result. On 2026-10-14 every gradient attack on
# the four-band ensemble succeeded completely -- white-box at 30 and 20 dB, and transfer from
# fullenc_seed1337 at both budgets, all EER 1.0000 with 100% success at the clean 0.1%-FAR
# threshold. Transfer mattering more than white-box is the uncomfortable part: the attacker then
# needs only SOME WavLM-based detector, and WavLM is a public checkpoint. Every member of that
# ensemble reads WavLM, and member correlation with the surrogate was 0.78-0.96.
#
# raw_seed1337 is the first member that shares no weights, no pretraining and no pooling grid with
# WavLM. If a WavLM gradient does not reach it, the mechanism behind the 100% transfer is the shared
# backbone, and backbone heterogeneity -- not score-level diversity -- is where robustness has to
# come from. If it DOES reach it, the perturbation is attacking something about speech rather than
# about WavLM, which is a stronger and more interesting negative result.
#
# Two studies, same attack set and budgets as adv_single and adv_bands so all three compare:
#   adv_raw        the raw member ALONE against the WavLM surrogate. This is the clean test of the
#                  mechanism. Its clean EER will be poor, so read it as a statement about gradients,
#                  not as a usable system.
#   adv_mixed_raw  the whole ensemble including the raw member. Expect dilution: the raw member is
#                  one of six in an unweighted mean, so five WavLM paths still carry the gradient.
#                  Reported to size that dilution, not as a defence.
# White-box stays 100% in both by construction -- the attacker holds every member's gradient and the
# mean combiner is differentiable -- so transfer is the number to read.
#
# views=[clean,codec] throughout: the raw member has no encodec view (the audio path's CodecAugment
# has no neural codec), so the calibration scale must come from the views every member scored.
cd /d/AUDIO_DEEPFAKE
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
S="+n_per_cell=40 +steps=20 +snr_db=[30,20] +spsa_samples=8 data.num_workers=8"
B="band0_7,band7_13,band13_19,band19_25"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run adv_raw scripts/attack.py $F $S "+views=[clean,codec]" \
  +study=adv_raw "+runs=[raw_seed1337]" "+surrogate=[fullenc_seed1337]"

run adv_mixed_raw scripts/attack.py $F $S "+views=[clean,codec]" \
  +study=adv_mixed_raw "+runs=[$B,fullenc_seed1337,raw_seed1337]" "+surrogate=[fullenc_seed1337]"
echo "== CHAIN DONE"
