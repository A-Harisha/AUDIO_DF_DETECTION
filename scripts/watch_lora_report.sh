#!/usr/bin/bash
# The second half of the automation: chain_lora_score.sh writes the score parquets, this turns each
# one into a verdict the moment it lands, so nothing waits on a human to run the analysis.
#
# It does NOT touch the running chain. Editing a bash script while bash is executing it re-reads the
# file by byte offset and runs the wrong lines, so the watcher is a separate process that waits on
# the chain's own log lines.
#
# Dev is reported against the MATCHED controls -- full_seed1337 and full_seed2, same manifest, same
# backbone, same two views -- with a paired bootstrap over the 23,347 select clips all three scored.
# Paired matters: a control-vs-control run of this script resolved a seed-to-seed gap of 0.0026 as
# significant, so the test is sensitive enough to settle LoRA's 0.0038 margin either way.
# Eval is reported alone, per real codec, against the two published reference points.
cd /d/AUDIO_DEEPFAKE
L=experiments/logs_partial
R=experiments/runs_partial
DEADLINE=$(( $(date +%s) + 16*3600 ))

# Wait for a marker in a log, or give up: a chain that died must not leave this spinning all night.
wait_for() {
  until grep -q "$1" "$2" 2>/dev/null; do
    if grep -qE "end [a-z_0-9]+ \(exit [1-9]" $L/chain_lora_score.log 2>/dev/null; then
      echo "== $(date +%H:%M) scoring chain failed; no report"; exit 1
    fi
    [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "== $(date +%H:%M) timed out waiting for $1"; exit 1; }
    sleep 30
  done
}

wait_for "scored dev_codec for lora_probe" $L/lora_score_dev.log
echo "== $(date +%H:%M) dev scores landed; reporting"
.venv/Scripts/python.exe scripts/score_report.py paths.runs=$R +run=lora_probe \
  "+sets=[dev_clean,dev_codec]" "+controls=[full_seed1337,full_seed2]" +split=select \
  +group=generator > $L/report_lora_dev.log 2>&1
echo "== $(date +%H:%M) dev report exit $?"

wait_for "scored eval_clean for lora_probe" $L/lora_score_eval.log
echo "== $(date +%H:%M) eval scores landed; reporting"
.venv/Scripts/python.exe scripts/score_report.py paths.runs=$R +run=lora_probe \
  "+sets=[eval_clean]" +group=codec > $L/report_lora_eval.log 2>&1
echo "== $(date +%H:%M) eval report exit $?"
echo "== WATCHER DONE"
