#!/usr/bin/bash
# Reports each set the moment it lands. Separate process: editing a running bash script makes bash
# re-read it by byte offset and run the wrong lines.
cd /d/AUDIO_DEEPFAKE
L=experiments/logs_partial
R=experiments/runs_partial
DEADLINE=$(( $(date +%s) + 12*3600 ))
wait_for() {
  until grep -q "$1" "$2" 2>/dev/null; do
    if grep -qE "end [a-z_0-9]+ \(exit [1-9]" $L/chain_lora_enc_score.log 2>/dev/null; then
      echo "== $(date +%H:%M) scoring chain failed; no report"; exit 1
    fi
    [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "== $(date +%H:%M) timed out waiting for $1"; exit 1; }
    sleep 30
  done
}
wait_for "scored dev_codec for lora_enc" $L/encscore_dev.log
echo "== $(date +%H:%M) dev scores landed; reporting"
.venv/Scripts/python.exe scripts/score_report.py paths.runs=$R +run=lora_enc \
  "+sets=[dev_clean,dev_codec]" "+controls=[lora_probe,full_seed1337,full_seed2]" +split=select \
  +group=generator > $L/report_enc_dev.log 2>&1
echo "== $(date +%H:%M) dev report exit $?"
wait_for "scored eval_clean for lora_enc" $L/encscore_eval.log
echo "== $(date +%H:%M) eval scores landed; reporting"
.venv/Scripts/python.exe scripts/score_report.py paths.runs=$R +run=lora_enc \
  "+sets=[eval_clean]" "+controls=[lora_probe,fullenc_seed1337]" +group=codec \
  +n_boot=300 > $L/report_enc_eval.log 2>&1
echo "== $(date +%H:%M) eval report exit $?"
echo "== WATCHER DONE"
