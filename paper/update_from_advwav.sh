#!/usr/bin/bash
# Fill Table 4 (robustness) and Figure 3 (training curves) once advwav and its attack battery land.
# Waits rather than racing: chain_last3.sh still owns the GPU.
#
# Table 4 currently carries four closed defence routes. advwav is the FIFTH -- waveform adversarial
# training, half of each batch replaced by 30 dB PGD examples crafted through the backbone. Unlike
# the feature-space row already in the table, it hardens the path that is actually attacked, so its
# result is the one that decides whether the robustness section is a defence or a measurement.
# Figure 3's AdvWav curve is truncated at the epochs that existed when the paper was drafted.
cd /d/AUDIO_DEEPFAKE
L=experiments/logs_partial
DEADLINE=$(( $(date +%s) + 30*3600 ))

until grep -q "CHAIN DONE" $L/chain_last3.log 2>/dev/null; do
  if grep -qE "end [a-z_0-9]+ \(exit [1-9]" $L/chain_last3.log 2>/dev/null; then
    echo "== $(date +%H:%M) chain_last3 failed; refreshing from whatever finished"; break
  fi
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "== $(date +%H:%M) timed out"; exit 1; }
  sleep 300
done

echo "== $(date +%H:%M) refreshing paper data and figures"
.venv/Scripts/python.exe paper/collect.py > $L/paper_collect.log 2>&1
echo "== collect exit $?"
.venv/Scripts/python.exe paper/make_figures.py > $L/paper_figures.log 2>&1
echo "== figures exit $?"
.venv/Scripts/python.exe paper/robustness_table.py > $L/paper_robust.log 2>&1
echo "== robustness table exit $?"
cat $L/paper_robust.log
echo "== DONE -- rebuild with: cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main"
