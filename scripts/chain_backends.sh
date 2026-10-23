#!/usr/bin/bash
# STEP 3 of the back-end route: two members that change the REPRESENTATION, not just the pooling.
#
# WHERE THIS STARTS. MHFA changed only the pooling -- attention over the frames instead of their
# mean -- and that alone dropped the eval correlation with lora_enc from 0.994 to 0.917 and bought
# 0.10 EER points, taking the project to 3.21% / 0.0905 (log, 2026-10-24). It was the first fusion
# gain worth having, and the first headline that dev select actually picks. But MHFA still reads
# nothing except the SSL features, so the two members share every bit of their evidence.
#
# THE TWO MEMBERS, one per axis beyond pooling:
#   lora_lfcc  -- a second REPRESENTATION. 20 linear-frequency cepstral coefficients with their
#                 first and second deltas, computed from the waveform and aligned frame-for-frame
#                 with the backbone, concatenated with the SSL frames before attentive pooling.
#                 Part of its evidence cannot correlate with anything the backbone encodes. This is
#                 T45's third family (arXiv:2409.01695).
#   lora_graph -- a second TOPOLOGY. Graph attention over two node sets, time slots and backbone
#                 layers, using AASIST's GraphAttentionLayer (MIT, NAVER Corp). It is NOT AASIST:
#                 no heterogeneous cross-attention, no master node, and the spectral sub-graph is
#                 replaced by a layer sub-graph because SSL frames have no frequency axis. No result
#                 from it may be reported under the AASIST name.
#
# Both hold everything else at lora_enc's settings: WavLM-Large, LoRA 16, encodec_p=0.3, batch 32,
# 8 epochs, seed 1337, the same two validation views. One variable each.
#
# Measured before launch: head 0.50M and 0.30M params, 3.94 and 3.96 GiB peak at batch 32 on 4 s
# crops, 9.47 GiB at batch 24 on 16 s eval clips, 170 ms per step for both -- the same cost as MHFA.
#
# SERIAL, NOT PARALLEL. One GPU, and each run needs about 20 GB in practice; two at once would
# contend and an OOM mid-run would cost more than the wait. Total is the same either way: about
# 23 h, two trainings at ~7 h and two scorings at ~4.4 h.
#
# WHAT DECIDES IT. The fusion studies at the end, in order of what they answer:
#   ens_pair_lfcc / ens_pair_graph -- each member's eval correlation with lora_enc alone. Below
#       0.917 means changing the representation or the topology buys MORE decorrelation than
#       changing the pooling did.
#   ens_trio  -- lora_enc + lora_mhfa + lora_lfcc, the claim: does a third view beat 3.21%?
#   ens_quad  -- all four. If the gain keeps compounding, the view axis is the answer to the 0.62
#       points still separating us from T45. If it saturates at three, say so.
# A gain is only claimable if dev select also ranks the winning combiner first, as it did for the
# two-member mean and did NOT for the five studies before it.
cd /d/AUDIO_DEEPFAKE
export PATH="$PATH:/c/Users/COE/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin"
F="paths.manifests=experiments/data_full paths.features=data/features_fulltrain paths.runs=experiments/runs_partial"
L=experiments/logs_partial
V="+views=[clean,codec]"
TRAIN="experiment.features=false experiment.lora_rank=16 experiment.batch_size=32
       experiment.epochs=8 experiment.backbone_lr=1.0e-5 experiment.lr=1.0e-3
       experiment.encodec_p=0.3 data.num_workers=12 experiment.score_sets=[]"
run() { name=$1; shift; echo "== $(date +%H:%M) start $name"; .venv/Scripts/python.exe "$@" > $L/$name.log 2>&1; rc=$?; echo "== $(date +%H:%M) end $name (exit $rc)"; [ $rc -eq 0 ] || exit $rc; }
member() {  # name backend
  run "$1" scripts/train.py $F backbone=wavlm experiment.name="$1" seed=1337 \
    experiment.backend="$2" $TRAIN
  run "$1_dev" scripts/train.py $F backbone=wavlm experiment.name="$1" seed=1337 \
    experiment.backend="$2" experiment.features=false experiment.lora_rank=16 \
    experiment.score_only=true data.num_workers=12 "experiment.score_sets=[dev_clean,dev_codec]"
  run "$1_eval" scripts/train.py $F backbone=wavlm experiment.name="$1" seed=1337 \
    experiment.backend="$2" experiment.features=false experiment.lora_rank=16 \
    experiment.score_only=true data.num_workers=12 "experiment.score_sets=[eval_clean]"
}

member lora_lfcc lfcc
member lora_graph graph

run ens_pair_lfcc  scripts/ensemble.py $F $V +study=ens_pair_lfcc  "+runs=[lora_enc,lora_lfcc]"
run ens_pair_graph scripts/ensemble.py $F $V +study=ens_pair_graph "+runs=[lora_enc,lora_graph]"
run ens_trio scripts/ensemble.py $F $V +study=ens_trio "+runs=[lora_enc,lora_mhfa,lora_lfcc]"
run ens_quad scripts/ensemble.py $F $V +study=ens_quad \
  "+runs=[lora_enc,lora_mhfa,lora_lfcc,lora_graph]"
echo "== CHAIN DONE"
