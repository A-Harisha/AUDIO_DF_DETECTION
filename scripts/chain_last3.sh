#!/usr/bin/bash
# THREE JOBS, serial because there is one GPU.
#
# A. OUT-OF-DOMAIN REPORTING. Every number this project has is in-domain ASVspoof 5, where train,
#    dev and eval share a recording pipeline and the attacks are the known families A01-A32. A
#    headline of 3.21% there says nothing about audio found in the world. In-the-Wild is 31,779
#    clips of real and synthesised speech of 54 public figures collected from the internet:
#    different speakers, different channels, no codec labels, unknown spoofing systems. Reporting
#    only -- nothing is selected or calibrated on it. The expectation is that the numbers are far
#    worse, and the point is to MEASURE that instead of leaving it unstated. About 10 min a member.
#
# B. WAVEFORM ADVERSARIAL TRAINING, the last untried robustness route. Four defences have failed:
#    a single model, four layer bands, two SSL backbones and feature-space hardening all leave any
#    gradient-bearing attacker at 98-100% success. The feature-space probe could not answer the
#    question at all -- it hardened the HEAD against perturbations in FEATURE space while the
#    attacker perturbs the WAVEFORM and arrives through the backbone, a mismatch its own docstring
#    flagged before the run (log, 2026-10-16). This perturbs the waveform and backpropagates
#    through the backbone, so it hardens the path that is actually attacked.
#      Half of each batch at 30 dB, PGD-3. Half rather than all because training purely on attacked
#      audio is the version that destroys clean accuracy, and clean accuracy is the result here.
#      Cost is about 2.5x a normal epoch, so roughly 17 h rather than the 30 h first quoted.
#      lora_enc is the control: identical except wav_adv_snr=0, eval 3.31% / 0.0942.
#
# C. THE ROBUSTNESS MEASUREMENT, which needed a blocker fixed first. The white-box attack path
#    refused a LoRA member because it shares one backbone pass between members; it now keys the
#    cache by run for LoRA and rebuilds the adapters, verified by taking a live gradient of the
#    score with respect to the waveform. Without that this whole run would be unmeasurable.
#      Same battery as adv_single and adv_advtrain so the three are directly comparable.
#
# WHAT DECIDES B. White-box PGD success is 1.000 for every model this project has built. If it
# drops materially below 1.000 while eval EER stays near 3.31%, adversarial training is a real
# defence here and the robustness section stops being a measurement of a field-wide failure. If it
# stays at 1.000, that is the fifth closed route and the honest claim is that this threat model is
# not defensible at a usable accuracy -- which is worth stating plainly, with five routes behind it.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }

# ---- A: in-the-wild, every member we have ----------------------------------------------------
itw() {  # name backend
  run "$1_itw" scripts/train.py $F backbone=wavlm experiment.name="$1" seed=1337 \
    experiment.backend="$2" experiment.features=false experiment.lora_rank=16 \
    experiment.score_only=true data.num_workers=12 "experiment.score_sets=[itw_clean]"
}
itw lora_enc head
itw lora_mhfa mhfa
itw lora_lfcc lfcc
itw lora_graph graph
run itw_report scripts/score_report.py paths.runs=experiments/runs_partial +run=lora_enc \
  "+sets=[itw_clean]" "+controls=[lora_mhfa,lora_lfcc,lora_graph]" +n_boot=300

# ---- B: waveform adversarial training --------------------------------------------------------
run advwav scripts/train.py $F backbone=wavlm experiment.name=advwav seed=1337 \
  experiment.features=false experiment.lora_rank=16 experiment.batch_size=32 experiment.epochs=8 \
  experiment.backbone_lr=1.0e-5 experiment.lr=1.0e-3 experiment.encodec_p=0.3 \
  experiment.wav_adv_snr=30.0 experiment.wav_adv_steps=3 experiment.wav_adv_fraction=0.5 \
  data.num_workers=12 "experiment.score_sets=[]"
run advwav_dev scripts/train.py $F backbone=wavlm experiment.name=advwav seed=1337 \
  experiment.features=false experiment.lora_rank=16 experiment.score_only=true \
  data.num_workers=12 "experiment.score_sets=[dev_clean,dev_codec]"
run advwav_eval scripts/train.py $F backbone=wavlm experiment.name=advwav seed=1337 \
  experiment.features=false experiment.lora_rank=16 experiment.score_only=true \
  data.num_workers=12 "experiment.score_sets=[eval_clean]"
run advwav_report scripts/score_report.py paths.runs=experiments/runs_partial +run=advwav \
  "+sets=[eval_clean]" "+controls=[lora_enc]" +group=codec +n_boot=300

# ---- C: the robustness battery, hardened model against its control ---------------------------
run adv_wavtrain scripts/attack.py $F data.num_workers=8 "+views=[clean,codec]" \
  "+n_per_cell=40" "+steps=20" "+snr_db=[30,20]" "+spsa=false" \
  +study=adv_wavtrain "+runs=[advwav]" "+surrogate=[lora_enc]"
run adv_control scripts/attack.py $F data.num_workers=8 "+views=[clean,codec]" \
  "+n_per_cell=40" "+steps=20" "+snr_db=[30,20]" "+spsa=false" \
  +study=adv_control "+runs=[lora_enc]" "+surrogate=[advwav]"
echo "== CHAIN DONE"
