#!/usr/bin/bash
# Fix the SELECTION SET, then re-select. Waits for chain_backends.sh rather than racing it: one GPU.
#
# THE PROBLEM. On 2026-10-24 the dev select ranking of member sets came out inverted against eval.
# scripts/select_members.py prints it:
#   rank 1  dev 0.0146  enc+mhfa        eval 3.21%  <== what the protocol picks
#   rank 4  dev 0.0183  enc+lfcc        eval 2.93%  <== the best system we have
#   rank 7  dev 0.0296  lfcc            eval 3.11%  <== the best SINGLE member, ranked last
# The selection set costs 0.28 EER points. The cause is not subtle: dev holds only clean and
# ffmpeg-simulated codecs, while eval's hardest conditions are the NEURAL codecs C04 (6.1% for
# lora_enc) and C07 (7.7%). A member whose evidence is cepstral generalises to real channels in a
# way dev cannot observe, so dev penalises exactly what we want.
#
# THE FIX. Score every member on a dev `encodec` view -- the same construction as the cached view,
# half the clips through MP3 first (C07) and the rest plain EnCodec (C04), deterministic per clip --
# and select over three views instead of two.
#
# WHAT IT DECIDES.
#   If a three-view dev select ranks enc+lfcc first, then 2.93% / 0.0819 becomes a CLAIMABLE result
#   and the gap to T45 (2.59% / 0.0750) is 0.34 EER points.
#   If it still picks enc+mhfa, the honest conclusion is that this dev split cannot select these
#   systems at all, and the headline stays 3.21% with that stated as a limitation. That is a result
#   too, and a more interesting one than the number.
# Either way the answer is mechanical and comes from the script, not from reading an eval table.
#
# COST about 4.5 h: four members at roughly 1.1 h each, 140,950 dev clips with EnCodec on the GPU.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
DEADLINE=$(( $(date +%s) + 20*3600 ))

# Wait for the back-end chain; a failure there means lora_graph has no checkpoint to score.
until grep -q "CHAIN DONE" $L/chain_backends.log 2>/dev/null; do
  if grep -qE "end [a-z_0-9]+ \(exit [1-9]" $L/chain_backends.log 2>/dev/null; then
    echo "== $(date +%H:%M) chain_backends failed; scoring only the members that finished"
    break
  fi
  [ "$(date +%s)" -lt "$DEADLINE" ] || { echo "== $(date +%H:%M) timed out waiting"; exit 1; }
  sleep 120
done

run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }
score() {  # name backend
  [ -f experiments/runs_partial/$1/model.pt ] || { echo "== skip $1 (no checkpoint)"; return 0; }
  run "$1_encdev" scripts/train.py $F backbone=wavlm experiment.name="$1" seed=1337 \
    experiment.backend="$2" experiment.features=false experiment.lora_rank=16 \
    experiment.score_only=true data.num_workers=12 "experiment.score_sets=[dev_encodec]"
}

score lora_enc head
score lora_mhfa mhfa
score lora_lfcc lfcc
score lora_graph graph

# Two selections, so the effect of the view is isolated rather than asserted.
run sel_2view scripts/select_members.py $F "+views=[clean,codec]" +study=sel_2view_final \
  "+members=[lora_enc,lora_mhfa,lora_lfcc,lora_graph]"
run sel_3view scripts/select_members.py $F "+views=[clean,codec,encodec]" +study=sel_3view \
  "+members=[lora_enc,lora_mhfa,lora_lfcc,lora_graph]"
echo "== CHAIN DONE"
