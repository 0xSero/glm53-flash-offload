#!/bin/bash
# N134 prerouter phase A: capture per decode token x MoE layer the MoE input row, top-8 ids + weights, and the tier
# each pick was served from, on the shipped nvme3 serving config (prefetch OFF so tiers are pure demand residency).
# usage: run_cap.sh <tag> [ENV=val ...]    env: MEM (55g | 16g), TARGET (decode tokens), ONLY_FIRST (prompt subset)
# Holds ~/nvx_bench.lock for the whole run (caller: flock). Container n134-<tag>, port 30334, GPU0, cpuset 2-39.
set -u
TAG=$1; shift
N=$HOME/freetoken-exl3/runs/N134-prerouter
S=$HOME/freetoken-exl3/runs/N116-glm53-nvme/s2; H=$S/hom272
OUT=/mnt/nvx/n134/$TAG; R=$N/runs/$TAG; mkdir -p $R $OUT $N/build/nv2cap
PORT=${PORT:-30334}; URL=http://127.0.0.1:$PORT; CN=n134-$TAG
IMG=ghcr.io/0xsero/glm53-flash-offload@sha256:bb633b0bcb85573ad40b6c408af5e1036ae062c593e42479f4e4d2ab521e551d
K=$H/kguard.sh
log() { echo "[$(date +%T)] $*" | tee -a $R/arm.log; }
mkdir -p $R/code && cp -a $N/glm53cap/*.py $N/glm53cap/*.cpp $N/glm53cap/*.cu $N/glm53cap/*.h $N/run_cap_v2.sh $N/cap_client.py $R/code/ 2>/dev/null
[ -f $N/STOP ] && { log "STOP present"; exit 1; }
$K > $R/kcheck_before.txt 2>&1 || { log "kcheck TRIPPED before start"; cat $R/kcheck_before.txt; exit 9; }
busy=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' 'tolower($1) ~ /81:00/ && $3 > 2000' | wc -l)
[ "$busy" = "0" ] || { log "GPU0 has $busy compute processes over 2 GiB; refusing to start"; exit 8; }
$H/userguard.sh none > /dev/null || { log "user container on GPU0; refusing"; exit 8; }
docker ps --format '{{.Names}}' | grep -qx "$CN" && { log "container $CN exists"; exit 7; }
grep MemAvailable /proc/meminfo > $R/mem_before.txt
BASE=(-e GLM53_MODE=nvme3 -e GLM53_EC_MAX_SLOTS=1376 -e GLM53_MAIN_CPUS=24 -e TORCH_CUDA_ARCH_LIST=8.6 -e GLM53_K_OVL=0
      -e GLM53_NV_VRING=24 -e GLM53_NV_PREFETCH=0 -e GLM53_EC_RESERVE_GB=1.5 -e GLM53_MAX_RQ_TOKENS=4096
      -e GLM53_N134_CAP=/out/cap)
ENVS=("${BASE[@]}")
for kv in "$@"; do ENVS+=(-e "$kv"); done
MOUNTS=(-v /home/sero/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw:/models:ro -v $OUT:/out -v /mnt/nvx/glm53:/nvx:ro
        -v $N/glm53cap:/opt/glm53/glm53:ro -v $S/build/nv:/opt/glm53/build/nv
        -v $N/build/nv2cap:/opt/glm53/build/nv2 -v $S/docker/entrypoint.sh:/opt/glm53/docker/entrypoint.sh:ro)
echo "docker run --name $CN --gpus device=0 --memory ${MEM:-55g} --memory-swap ${MEM:-55g} --cpuset-cpus 2-39 -e PORT=$PORT ${ENVS[*]} ${MOUNTS[*]} $IMG" > $R/cmd.txt
docker run --rm --name $CN --gpus '"device=0"' --memory ${MEM:-55g} --memory-swap ${MEM:-55g} --shm-size 1g --ulimit memlock=-1 --cpuset-cpus 2-39 \
  --network host -e PORT=$PORT "${ENVS[@]}" "${MOUNTS[@]}" $IMG > $R/server.log 2>&1 &
SP=$!
( n=0; while kill -0 $SP 2>/dev/null; do sleep 30; kill -0 $SP 2>/dev/null || break; n=$((n+1))
    $H/userguard.sh $CN > $R/userguard_live.txt 2>&1 || { echo "[$(date +%T)] USER ON GPU0 ($(cat $R/userguard_live.txt)): stopping $CN" | tee -a $R/arm.log; docker stop -t 30 $CN >/dev/null 2>&1; touch $N/STOP; break; }
    [ $((n % 2)) = 0 ] && { $K > $R/kguard_live.txt 2>&1 || { echo "[$(date +%T)] GUARD TRIP: stopping $CN" | tee -a $R/arm.log; cat $R/kguard_live.txt >> $R/arm.log; docker stop -t 30 $CN >/dev/null 2>&1; touch $N/STOP $N/GUARD_TRIP; break; }; }
    [ $((n % 20)) = 0 ] && { du -sb $OUT/cap 2>/dev/null | awk '{printf "%.1f GB\n", $1/1e9}' > $R/cap_size.txt; }
  done ) &
GP=$!
ok=0
for i in $(seq 1 240); do
  curl -s -m 5 $URL/health >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $SP 2>/dev/null || { log "server died during start"; break; }
  sleep 5
done
if [ $ok = 1 ]; then
  sleep 3; log "ready: $(grep -o 'RAM tier [0-9]* slots ([0-9.]* GiB)' $R/server.log | head -1); $(grep -o 'N134 capture.*' $R/server.log | head -1)"
  curl -s -m 30 $URL/stats > $R/stats_ready.json
  log "capture client (target ${TARGET:-200000})"
  timeout ${CLIENT_TIMEOUT_S:-43200} taskset -c 0-1 python3 $N/cap_client.py --url $URL --prompts ${PROMPTS:-$N/prompts.jsonl} \
     ${PANEL---panel $HOME/freetoken-exl3/runs/2026-09-29-G001-ref-panel/ref_panel.json} --code-dir $N/glm53cap --log $R/requests.jsonl \
     --target ${TARGET:-200000} ${MAX_PASSES:+--max-passes $MAX_PASSES} --stop-file $N/STOP ${ONLY_FIRST:+--only-first $ONLY_FIRST} > $R/client.log 2>&1
  log "client exit $?: $(tail -1 $R/client.log)"
  sleep 5
  curl -s -m 60 $URL/stats > $R/stats_end.json
  grep -o '"n134_capture": {[^}]*}[^}]*}' $R/stats_end.json | head -c 600 | tee -a $R/arm.log; echo >> $R/arm.log
fi
docker stop -t 60 $CN >/dev/null 2>&1
wait $SP; sleep 1; kill $GP 2>/dev/null
du -sb $OUT/cap | awk '{printf "capture dir %.1f GB\n", $1/1e9}' | tee -a $R/arm.log
$K > $R/kcheck_after.txt 2>&1 || log "kcheck TRIPPED after stop"
tail -1 $R/kcheck_after.txt >> $R/arm.log
log "ARM_DONE"
