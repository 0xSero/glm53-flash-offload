#!/bin/bash
# glm53-flash-offload container entrypoint: preflight checks, model resolve/download, mode defaults, exec the server.
#   docker run ... IMAGE [extra serve.py args]      extra args are appended (argparse: the last value wins)
#   docker run ... IMAGE bash                        any command that is not an option runs as-is
set -euo pipefail
ROOT=${GLM53_ROOT:-/opt/glm53}

SCRIPT=glm53/serve.py
if [ "${1:-}" = "decode-kl" ]; then   # paired decode-KL check (bench/decode_kl.py) with the serving config
    SCRIPT=bench/decode_kl.py; shift
elif [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    exec "$@"
fi

log() { echo "[glm53] $*" >&2; }
die() { echo "[glm53] ERROR: $*" >&2; exit 1; }

# ---- mode defaults; every value can be overridden with -e ---------------------------------------------------------
#   fast  (default) run G067: GPU expert cache + zero-copy + AVX2 CPU tier for cold decode misses, batched decode,
#                   fused hyper-connection sites, fast tier split, shared expert on a side stream
#   exact           run G066a: cache + zero-copy only, no CPU tier, no side-stream shared expert; bit-exact with
#                   stock exllamav3, ~12.6 tok/s decode
MODE=${GLM53_MODE:-fast}
case "$MODE" in
    fast)  D_TIER=1; D_RES=1.5; D_K=1; MODE_ARGS=(-ambs 4) ;;
    exact) D_TIER=0; D_RES=1.0; D_K=0; MODE_ARGS=() ;;
    *) die "GLM53_MODE must be fast or exact (got $MODE)" ;;
esac
export GLM53_K_HCFUSE=${GLM53_K_HCFUSE:-$D_K}
export GLM53_K_FTSPLIT=${GLM53_K_FTSPLIT:-$D_K}
export GLM53_K_OVL=${GLM53_K_OVL:-$D_K}
export GLM53_ZC_VRAM=${GLM53_ZC_VRAM-0}
export GLM53_ZC_STATS=${GLM53_ZC_STATS:-$ROOT/data/stats_own_dec.json}
export GLM53_EC=${GLM53_EC-1}
export GLM53_EC_WARM=${GLM53_EC_WARM:-$ROOT/data/stats_own_dec.json}
export GLM53_EC_RESERVE_GB=${GLM53_EC_RESERVE_GB:-$D_RES}
export GLM53_EC_STAGE_GB=${GLM53_EC_STAGE_GB:-2.6}
export GLM53_EC_ELASTIC_GB=${GLM53_EC_ELASTIC_GB:-10}
export GLM53_CPU_TIER=${GLM53_CPU_TIER:-$D_TIER}
export GLM53_CT_CPUS=${GLM53_CT_CPUS:-auto}
export GLM53_CT_THREADS=${GLM53_CT_THREADS:-0}
export GLM53_TRITON_PIN=${GLM53_TRITON_PIN-$ROOT/data/triton_pin_exact.json}
export EXLLAMAV3_TUNE_CACHE=${EXLLAMAV3_TUNE_CACHE:-$ROOT/data/coop_autotune_1gpu.bin}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ROOT/triton_cache}
export PYTHONUNBUFFERED=1
for k in GLM53_ZC_VRAM GLM53_EC GLM53_TRITON_PIN; do [ -n "${!k}" ] || unset "$k"; done

MODEL=${GLM53_MODEL_DIR:-/models}
REPO=${GLM53_MODEL_REPO:-turboderp/GLM-5.3-Flash-exl3}
REV=${GLM53_MODEL_REVISION:-332ab457b709b7ba30dd9a448be5de03b80a7ac9}   # branch 3.05bpw
PORT=${PORT:-30000}

# ---- preflight --------------------------------------------------------------------------------------------------
command -v nvidia-smi >/dev/null && timeout 20 nvidia-smi -L >&2 || die "no NVIDIA GPU visible (run with --gpus)"
NG=$(timeout 20 nvidia-smi -L | wc -l)
[ "$NG" -eq 1 ] || log "WARNING: $NG GPUs visible; this build serves on ONE GPU (pass --gpus '\"device=N\"')"
VRAM=$(timeout 20 nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "${VRAM:-0}" -ge 23000 ] || die "GPU has ${VRAM} MiB; the measured configuration needs a 24 GB card"

ML=$(ulimit -l)
[ "$ML" = "unlimited" ] || log "WARNING: locked-memory limit is $ML KiB; run with --ulimit memlock=-1 (pinned host experts)"

if [ "$GLM53_CPU_TIER" = "1" ]; then
    for f in avx2 fma f16c; do
        grep -qw "$f" /proc/cpuinfo || die "CPU lacks $f; the CPU tier needs AVX2+FMA+F16C (use -e GLM53_MODE=exact)"
    done
fi
AVAIL_GB=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
NEED_GB=$([ "$GLM53_CPU_TIER" = "1" ] && echo 222 || echo 114)
if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != "max" ]; then
    LIM_GB=$(( $(cat /sys/fs/cgroup/memory.max) / 1073741824 ))
    [ "$LIM_GB" -ge "$NEED_GB" ] || die "container memory limit ${LIM_GB} GiB < ~${NEED_GB} GiB needed (raise --memory or use -e GLM53_MODE=exact)"
fi
[ "$AVAIL_GB" -ge "$NEED_GB" ] || log "WARNING: MemAvailable ${AVAIL_GB} GiB < ~${NEED_GB} GiB this configuration pins/allocates; expect OOM"
log "mode $MODE: host $(nproc) CPUs visible, MemAvailable ${AVAIL_GB} GiB, GPU ${VRAM} MiB, CPU tier ${GLM53_CPU_TIER}"

# ---- model ------------------------------------------------------------------------------------------------------
if [ ! -f "$MODEL/config.json" ]; then
    [ "${GLM53_MODEL_DOWNLOAD:-1}" = "1" ] || die "no model at $MODEL (mount the 3.05bpw checkpoint there)"
    mkdir -p "$MODEL" && [ -w "$MODEL" ] || die "$MODEL is not writable; mount a host directory with ~126 GB free"
    FREE_GB=$(df -Pk "$MODEL" | awk 'NR==2{printf "%d", $4/1048576}')
    [ "$FREE_GB" -ge 120 ] || die "only ${FREE_GB} GiB free at $MODEL; the checkpoint is 117 GiB (125.3 GB)"
    log "downloading $REPO @ $REV into $MODEL (125.3 GB)"
    python3 - "$REPO" "$REV" "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], max_workers=8)
EOF
fi
[ -f "$MODEL/quantization_config.json" ] || log "WARNING: $MODEL has no quantization_config.json; expected the EXL3 3.05bpw checkpoint"

ARGS=(-m "$MODEL" -cs 131072 --max-batch-size 8 -chunk_size 8192 "${MODE_ARGS[@]}" --host 0.0.0.0 --port "$PORT"
      --served-name "${SERVED_NAME:-glm-5.3-flash}")
# shellcheck disable=SC2206
[ -n "${GLM53_ARGS:-}" ] && ARGS+=($GLM53_ARGS)
log "$SCRIPT ${ARGS[*]} $*"
exec python3 "$ROOT/$SCRIPT" "${ARGS[@]}" "$@"
