#!/bin/bash
# Image-build step (also usable on a host with exllamav3 1.5.1 installed): unpack the Triton cache of the measured
# run and pre-build the two torch extensions (GPU expert cache kernels, CPU tier host + GPU kernels), so the first
# start compiles nothing. Needs nvcc + ninja (no GPU needed). The extensions are built for $TORCH_CUDA_ARCH_LIST.
#   scripts/install.sh [ROOT]     ROOT defaults to the checkout this script lives in
set -euo pipefail
ROOT=${1:-$(cd "$(dirname "$0")/.." && pwd)}
mkdir -p "$ROOT/triton_cache"
tar -xzf "$ROOT/data/triton_cache_c056.tar.gz" -C "$ROOT/triton_cache"
if [ "$ROOT/triton_cache" != /opt/glm53/triton_cache ]; then   # Triton group files record absolute child paths
    grep -rl /opt/glm53/triton_cache "$ROOT/triton_cache" | xargs -r sed -i "s#/opt/glm53/triton_cache#$ROOT/triton_cache#g"
fi
echo "triton cache: $(find "$ROOT/triton_cache" -name '*.autotune.json' | wc -l) autotune records, $(ls "$ROOT/triton_cache" | wc -l) entries"
cd "$ROOT"
MAX_JOBS=${MAX_JOBS:-4} python3 - <<'EOF'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "glm53"))
import expert_cache, cpu_tier
expert_cache._ext()
cpu_tier._build()
print("extensions built:", expert_cache._EXT.__file__, cpu_tier._H.__file__, cpu_tier._CU.__file__)
EOF
