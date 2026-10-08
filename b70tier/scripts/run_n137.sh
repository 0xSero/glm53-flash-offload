#!/bin/bash
# N137 step 3 arm on the 3090 (slot A, hold ~/gpu3090.lock outside): shipped v4.2-nvme image (GLM53_MODE=nvme, 55 GiB)
# with the N137 repo copy mounted over /opt/glm53 and the B70 ring at /ring. The B70 server (n137-b70srv) must already
# run for GLM53_B70=1 arms (scripts/b70_server_up.sh). usage: run_n137.sh <tag> [ENV=val ...]
#   STEPS ("panel sweep verify" | "kl"), SW_PREFILL, SW_CONC, SW_EXTRA, KL_VARIANTS, MEM (55g)
set -u
TAG=$1; shift
N=$HOME/freetoken-exl3/runs/N137-b70tier; H=$HOME/freetoken-exl3/runs/N116-glm53-nvme/s2/hom272; R=$N/runs/$TAG; mkdir -p $R $N/build/nv
PORT=${PORT:-30337}; URL=http://127.0.0.1:$PORT; CN=n137-$TAG
IMG=4732a063fa9e     # ghcr.io/sybil-solutions/glm53-flash-offload v4.2-nvme (commit 3bdb502)
K=$H/kguard.sh
log() { echo "[$(date +%T)] $*" | tee -a $R/arm.log; }
( cd $N/repo && git rev-parse HEAD && git diff --stat ) > $R/code_rev.txt
$K > $R/kcheck_before.txt 2>&1 || { log "kcheck TRIPPED before start"; exit 9; }
busy=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', ' 'tolower($1) ~ /81:00/ && $3 > 2000' | wc -l)
[ "$busy" = "0" ] || { log "GPU0 has $busy compute processes over 2 GiB; refusing to start"; exit 8; }
docker ps --format '{{.Names}}' | grep -q '^omarchy-local-ai' && { log "user container present; refusing"; exit 8; }
ENVS=(-e GLM53_MODE=nvme -e GLM53_EC_MAX_SLOTS=1376 -e TRITON_CACHE_DIR=/opt/glm53/build/triton_cache -e TORCH_CUDA_ARCH_LIST=8.6 -e PORT=$PORT -e GLM53_NV_VERIFY_START=1
      -e GLM53_KL_VARIANTS=${KL_VARIANTS:-control_off,cpu_default,cpu_b70,b70_only})
for kv in "$@"; do ENVS+=(-e "$kv"); done
MOUNTS=(-v /home/sero/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw:/models:ro -v $R:/out -v /mnt/nvx/glm53:/nvx:ro
        -v $N/repo:/opt/glm53:ro -v $N/build:/opt/glm53/build -v /dev/shm/n137:/ring)
ARGS=(); [ "${STEPS:-}" = kl ] && ARGS=(decode-kl --kl-out /out/decode_kl.json)
echo "docker run --name $CN --gpus device=0 --memory ${MEM:-55g} --memory-swap ${MEM:-55g} --shm-size 1g --ulimit memlock=-1 --cpuset-cpus 2-39 --network host ${ENVS[*]} ${MOUNTS[*]} $IMG ${ARGS[*]}" > $R/cmd.txt
docker run --rm --name $CN --gpus '"device=0"' --memory ${MEM:-55g} --memory-swap ${MEM:-55g} --shm-size 1g --ulimit memlock=-1 \
  --cpuset-cpus 2-39 --network host "${ENVS[@]}" "${MOUNTS[@]}" $IMG "${ARGS[@]}" > $R/server.log 2>&1 &
SP=$!
( n=0; while kill -0 $SP 2>/dev/null; do sleep 30; kill -0 $SP 2>/dev/null || break; n=$((n+1))
    $H/userguard.sh $CN > $R/userguard_live.txt 2>&1 || { echo "[$(date +%T)] USER ON GPU0: stopping $CN" | tee -a $R/arm.log; docker stop -t 30 $CN >/dev/null 2>&1; touch $N/STOP; break; }
    [ $((n % 2)) = 0 ] && { $K > $R/kguard_live.txt 2>&1 || { echo "[$(date +%T)] GUARD TRIP: stopping $CN" | tee -a $R/arm.log; docker stop -t 30 $CN >/dev/null 2>&1; touch $N/STOP $N/GUARD_TRIP; break; }; }
    [ -f $N/TRIPPED ] && { echo "[$(date +%T)] B70 guard tripped: stopping $CN" | tee -a $R/arm.log; docker stop -t 30 $CN >/dev/null 2>&1; touch $N/STOP; break; }
  done ) &
GP=$!
if [ "${STEPS:-}" = kl ]; then
  wait $SP; log "decode-kl exit $?"; grep -E "^VARIANT|DECODE_KL_DONE|Error|error" $R/server.log | tail -12 | tee -a $R/arm.log
else
  ok=0
  for i in $(seq 1 240); do
    curl -s -m 5 $URL/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $SP 2>/dev/null || { log "server died during start"; break; }
    sleep 5
  done
  if [ $ok = 1 ]; then
    sleep 3; log "ready"; grep -E "nv2 B70|post-warm|startup verify|selftest" $R/server.log | tee -a $R/arm.log
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > $R/vram_ready.txt
    curl -s -m 30 $URL/stats > $R/stats_ready.json
    for st in ${STEPS:-panel sweep verify}; do
      case $st in
        panel)  log "panel"; timeout 3600 taskset -c 40-42 python3 $HOME/freetoken-exl3/bench/score_ref_panel.py --url $URL --panel $HOME/freetoken-exl3/runs/2026-09-29-G001-ref-panel/ref_panel.json --out $R/score.json > $R/score.log 2>&1; tail -1 $R/score.log | tee -a $R/arm.log ;;
        verify) log "verify"; curl -s -m 1200 "$URL/nv_verify?n=128" > $R/verify.json; cat $R/verify.json | tee -a $R/arm.log; echo >> $R/arm.log ;;
        sweep)  log "sweep"; curl -s -m 30 $URL/stats > $R/stats_before_sweep.json
                timeout 14400 taskset -c 40-42 python3 $H/sweep_marks.py --url $URL --card rtx3090 --template glm --config "HOM-272 N137 $TAG $*" --out $R/sweep.json --prefill ${SW_PREFILL:-8192 32768} --conc ${SW_CONC:-1 2 4} --reps 3 --dec-reps 2 --no-early-exit ${SW_EXTRA:-} > $R/sweep.log 2>&1
                echo "sweep exit $?" >> $R/sweep.log; tail -8 $R/sweep.log | tee -a $R/arm.log
                curl -s -m 30 $URL/stats > $R/stats_after_sweep.json ;;
      esac
      $K > $R/kcheck_mid.txt 2>&1 || { log "kcheck TRIPPED after $st"; break; }
      [ -f $N/TRIPPED ] && { log "B70 guard tripped after $st"; break; }
      curl -s -m 600 $URL/health >/dev/null 2>&1 || { log "SERVER DOWN after $st"; break; }
    done
    curl -s -m 30 $URL/stats > $R/stats_end.json
  fi
  docker stop -t 30 $CN >/dev/null 2>&1
  wait $SP
fi
kill $GP 2>/dev/null
$K > $R/kcheck_after.txt 2>&1 || log "kcheck TRIPPED after stop"
log "ARM_DONE"
