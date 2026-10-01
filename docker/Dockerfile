# Local build of the glm53-flash-offload image from this checkout:
#   docker build -f docker/Dockerfile -t glm53-flash-offload .
# The published image (ghcr.io/0xsero/glm53-flash-offload) is built from the same steps by
# github.com/0xSero/local-ai-images (directory glm53-flash-offload), which fetches this repo at a pinned commit.
#
# Base = the stack the numbers were measured on: lmsysorg/sglang v0.5.20 (CUDA 13.0, torch, Triton, nvcc) +
# exllamav3 v1.5.1 built from source for TORCH_CUDA_ARCH_LIST (8.6 = RTX 3090; sm_86 binaries also run on sm_89).
# For Blackwell (sm_120) rebuild with --build-arg TORCH_CUDA_ARCH_LIST="8.6;12.0" (not measured).
FROM lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f

ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ninja-build \
 && rm -rf /var/lib/apt/lists/*

# ExLlamaV3 v1.5.1 (MIT, turboderp), exactly the tag the measurements used.
ARG EXLLAMAV3_COMMIT=958ec933361b24eb8426ec7222e5b0062a679dcd
ARG MAX_JOBS=4
RUN git clone https://github.com/turboderp-org/exllamav3 /opt/exllamav3-src \
 && cd /opt/exllamav3-src && git checkout -q ${EXLLAMAV3_COMMIT} \
 && test "$(git rev-parse HEAD)" = "${EXLLAMAV3_COMMIT}" \
 && MAX_JOBS=${MAX_JOBS} pip install --no-build-isolation --no-deps -v . 2>&1 | tail -3 \
 && pip install --no-deps marisa-trie \
 && pip show exllamav3 >/dev/null \
 && rm -rf /opt/exllamav3-src

COPY . /opt/glm53
RUN chmod 0755 /opt/glm53/docker/entrypoint.sh /opt/glm53/scripts/install.sh \
 && MAX_JOBS=${MAX_JOBS} /opt/glm53/scripts/install.sh /opt/glm53

# Data paths; the tuning defaults per mode (fast = G067 with the CPU tier, exact = G066a without) live in the entrypoint.
ENV GLM53_MODE=fast \
    GLM53_ZC_STATS=/opt/glm53/data/stats_own_dec.json GLM53_EC_WARM=/opt/glm53/data/stats_own_dec.json \
    GLM53_TRITON_PIN=/opt/glm53/data/triton_pin_exact.json \
    EXLLAMAV3_TUNE_CACHE=/opt/glm53/data/coop_autotune_1gpu.bin TRITON_CACHE_DIR=/opt/glm53/triton_cache
EXPOSE 30000
ENTRYPOINT ["/opt/glm53/docker/entrypoint.sh"]
