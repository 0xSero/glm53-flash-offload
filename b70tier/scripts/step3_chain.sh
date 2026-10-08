#!/bin/bash
# N137 step 3 chain (joint window): B70 server up (under the B70 lock, guarded), then on the 3090 (gpu3090.lock +
# nvx_bench.lock per arm): s3b70 (GLM53_B70=1: panel, full sweep, verify) -> s3ctl (same session, B70 off: full sweep)
# -> s3kl (paired decode-KL: control_off, cpu_default, cpu_b70, b70_only). Stops on any guard trip / STOP file.
set -u
N=$HOME/freetoken-exl3/runs/N137-b70tier; cd $N; rm -f STOP TRIPPED GUARD_TRIP
log() { echo "[$(date '+%F %T')] $*" | tee -a $N/step3.log; }
nohup bash scripts/n137_b70_lock.sh n137-step3 bash scripts/b70_server_up.sh > $N/b70srv_up.log 2>&1 &
for i in $(seq 1 60); do docker logs n137-b70srv 2>&1 | grep -q "ready for LOAD" && break; sleep 5; done
docker logs n137-b70srv 2>&1 | grep -q "ready for LOAD" || { log "B70 server not up"; cat $N/b70srv_up.log; touch $N/STOP_B70; exit 1; }
log "B70 server up"
arm() { local tag=$1 steps=$2; shift 2; [ -f $N/STOP ] || [ -f $N/TRIPPED ] && { log "stop flag before $tag"; return 1; }
  log "$tag $* (waiting for gpu3090.lock + nvx_bench.lock)"
  flock $HOME/gpu3090.lock flock $HOME/nvx_bench.lock env STEPS="$steps" MEM="${MEM:-55g}" SW_PREFILL="${SW_PREFILL:-8192 32768}" SW_EXTRA="${SW_EXTRA:-}" bash scripts/run_n137.sh $tag "$@" > $N/runs/$tag.outer.log 2>&1
  grep -E "ready|nv2 B70|^decode|prefill|panel|top1|VARIANT|ARM_DONE|TRIP|died|DOWN|refus" $N/runs/$tag/arm.log | tail -20 | sed "s/^/   $tag: /" | tee -a $N/step3.log
  grep -qE "TRIP|died|DOWN|refus" $N/runs/$tag/arm.log && return 1; return 0; }
mkdir -p runs
arm s3b70 "panel sweep verify" GLM53_B70=1 \
 && arm s3ctl "sweep" GLM53_B70=0 \
 && arm s3kl kl GLM53_B70=1 \
 && [ "${FULLRAM:-1}" = 1 ] \
 && MEM=120g SW_PREFILL=8192 SW_EXTRA=--no-32k-decode arm s3frb70 "sweep" GLM53_B70=1 \
 && MEM=120g SW_PREFILL=8192 SW_EXTRA=--no-32k-decode arm s3frctl "sweep" GLM53_B70=0
log "chain end"; touch $N/STOP_B70
for i in $(seq 1 30); do docker ps --format '{{.Names}}' | grep -q '^n137-b70srv$' || break; sleep 2; done
log "B70 server stopped; STEP3_DONE"
