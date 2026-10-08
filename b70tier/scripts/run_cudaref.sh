#!/bin/bash
# N137 step 2: exllamav3 CUDA reference dump on the 3090 (container n137-cudaref, <2 GB VRAM, ~10 min).
# Waits for N129's LATF_DONE (agreed window), then blocks on ~/gpu3090.lock; kguard before and every 60 s.
set -u
R=$HOME/freetoken-exl3/runs/N137-b70tier; KG=$HOME/freetoken-exl3/runs/N116-glm53-nvme/s2/hom272/kguard.sh
LATF=$HOME/freetoken-exl3/runs/N116-glm53-nvme/s2/hom272/latf.log
echo "$(date -Is) waiting for LATF_DONE"
until grep -q LATF_DONE "$LATF" 2>/dev/null; do sleep 30; done
echo "$(date -Is) LATF_DONE seen; waiting for gpu3090.lock"
exec 9>$HOME/gpu3090.lock; flock 9
echo "$(date -Is) lock held"
bash $KG || { echo "PRE GUARD TRIP"; exit 3; }
( while sleep 60; do bash $KG >/dev/null || { echo "$(date -Is) GUARD TRIP -> stop"; timeout 60 docker stop -t 5 n137-cudaref; touch $R/out/TRIPPED; exit; }; docker ps -q -f name=n137-cudaref | grep -q . || exit; done ) &
timeout 1500 docker run --rm --name n137-cudaref --gpus '"device=0"' --cpuset-cpus 2-5 --memory 16g --memory-swap 16g \
  --network none -v $HOME/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw:/models:ro -v $R/b70:/b70:ro -v $R/out:/out \
  -e OUT=/out/cuda_ref.npz --entrypoint python3 4732a063fa9e /b70/cuda_ref.py
rc=$?; echo "$(date -Is) docker rc=$rc"
bash $KG
exit $rc
