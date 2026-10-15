#!/usr/bin/bash
# Does an adversarial perturbation survive a change of BACKBONE?
#
# Claim served: adversarial robustness under the threat model the user scoped on 2026-10-15, which
# excludes white-box (the attacker holding our weights) and keeps transfer, because transfer needs
# only SOME public detector and scored 100% success at the clean 0.1%-FAR threshold.
#
# What is established (2026-10-14, study adv_bands): a perturbation crafted on fullenc_seed1337
# defeats all four WavLM layer-band members completely, at 30 and 20 dB, EER 1.0000. Every member
# there reads WavLM, so that experiment cannot separate "the attack is general" from "the attack
# rides the shared backbone". Nothing so far has tested transfer ACROSS backbones.
#
# Both directions are run, because an asymmetric result would not support the claim:
#   adv_xlsr_from_wavlm   WavLM surrogate -> XLS-R member
#   adv_wavlm_from_xlsr   XLS-R surrogate -> WavLM member
# Each study also reports its own white-box number as the reference ceiling, so the transfer figure
# is read against what a full-gradient attacker achieves on the SAME model.
#
# No training: frozen_xlsr_enc_drop50 (5.97% EER) and fullenc_seed1337 (3.89%) both already exist
# and are scored on every set. SPSA is off (+spsa=false): it costs about two hours per study, it was
# already measured twice (1.06% and 1.44% success), and it answers a different question.
#
# Reading the result. If the WavLM gradient does NOT reach the XLS-R member, backbone heterogeneity
# is a transfer defence and the ensemble should be built across backbones rather than across layer
# bands -- which also explains why the four-band ensemble bought nothing in robustness. If it DOES
# reach it, the perturbation exploits something common to SSL speech models, heterogeneity is not a
# defence, and only adversarial training remains.
cd /d/AUDIO_DEEPFAKE
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
S="+n_per_cell=40 +steps=20 +snr_db=[30,20] +spsa=false data.num_workers=8"
V="+views=[clean,codec,encodec]"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

run adv_xlsr_from_wavlm scripts/attack.py $F $S $V +study=adv_xlsr_from_wavlm \
  "+runs=[frozen_xlsr_enc_drop50]" "+surrogate=[fullenc_seed1337]"

run adv_wavlm_from_xlsr scripts/attack.py $F $S $V +study=adv_wavlm_from_xlsr \
  "+runs=[fullenc_seed1337]" "+surrogate=[frozen_xlsr_enc_drop50]"
echo "== CHAIN DONE"
