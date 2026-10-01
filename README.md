# glm53-flash-offload

Serve **GLM-5.3-Flash** (EXL3 3.05 bpw, 125.3 GB, 288 routed experts x 42 MoE layers) from **one 24 GB RTX 3090**
plus host DDR4, with an OpenAI-compatible API. All routed experts live in pinned host RAM; the GPU keeps an elastic
cache of hot experts, reads the rest over PCIe (zero-copy), and an AVX2 CPU kernel computes the coldest cache misses
during decode on the host cores.

It is a thin layer of monkeypatches over **stock exllamav3 1.5.1** (no exllamav3 source edits) plus a few small
custom kernels. This is the exact code and configuration of run G067 of the FreeToken-EXL3 campaign.

Two modes, one switch (`GLM53_MODE`):

| mode | what runs | quality vs stock exllamav3 | host RAM |
|---|---|---|---|
| `fast` (default) | GPU expert cache + zero-copy misses + **AVX2 CPU tier** for cold decode misses, batched decode (`-ambs 4`), fused decode kernels | prefill exact; decode **not exact** (see Quality) | ~218 GiB (234 GB) |
| `exact` (`-e GLM53_MODE=exact`) | GPU expert cache + zero-copy misses, nothing else | bit-exact (panel top-1 1.0000, KL 0) | ~111 GiB (119 GB) |

## Measured

Host: AMD EPYC 7443P (24 cores, SMT on), 8-channel DDR4 (503 GiB), 1x RTX 3090 24 GB on PCIe 4.0 x16, driver
610.57, Samsung 990 PRO NVMe. Stack: lmsysorg/sglang v0.5.20 (CUDA 13.0) + exllamav3 v1.5.1 built for sm_86.
Protocol (campaign `sweep.py --template glm --prefill 8192 32768 --conc 1 2 4 --reps 3 --dec-reps 2`): prefill =
median of 3 fresh random-token prompts, tokens / time-to-first-token; decode = greedy chat completions run to their
natural end, C simultaneous streams, aggregate = all completion tokens / wall time, median of 2 rounds.

**`fast` mode (G067)**: `-cs 131072 --max-batch-size 8 -chunk_size 8192 -ambs 4`; expert cache reserve 1.5 GB,
prefill staging 2 x 2.6 GB, elastic 10 GB; CPU tier on 22 threads (cpus 2-23); `GLM53_K_HCFUSE=1 GLM53_K_FTSPLIT=1
GLM53_K_OVL=1`.

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | 710 tok/s | 28.15 tok/s | 1 | 131,072 tokens | 1 |
| 32,768 | 951 tok/s | 26.89 tok/s (C1 at 32k context) | 1 | 131,072 tokens | 1 |
| - | - | 31.57 tok/s aggregate (15.78 per stream) | 2 | 131,072 tokens | 1 |
| - | - | 33.98 tok/s aggregate (8.62 per stream) | 4 | 131,072 tokens | 1 |

Raw sweep / panel JSONs for every table: [`results/`](results/). Earlier steps of the same configuration: C056 (staging 2.0 GB, no fused kernels) prefill 626 / 808, decode C1 27.20 /
C2 30.72 / C4 34.72 aggregate, C1 at 32k 26.15; C052c (C056 without `-ambs 4`, streams then decode one after
another) C1 27.32 / C2 27.06 / C4 27.45.

**`exact` mode (G066a)**: `-cs 131072 --max-batch-size 8 -chunk_size 8192`; reserve 1.0 GB, staging 2 x 2.6 GB,
elastic 10 GB; no CPU tier, no fused kernels.

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | 697 tok/s | 12.57 tok/s | 1 | 131,072 tokens | 1 |
| 32,768 | 945 tok/s | 11.86 tok/s (C1 at 32k context) | 1 | 131,072 tokens | 1 |
| - | - | 12.62 tok/s aggregate | 2 | 131,072 tokens | 1 |
| - | - | 12.64 tok/s aggregate | 4 | 131,072 tokens | 1 |

For reference, stock exllamav3 with the experts it cannot fit in VRAM on its CPU worker decodes this model at
7.5-8.4 tok/s on the same box.

Published image smoke test (the README command on the measured host): see [Image smoke](#image-smoke).

## Quality

Reference: a teacher-forced panel (8 prompts, 2,154 scored positions) from stock exllamav3 1.5.1 on the same
checkpoint; metric = top-1 agreement and mean KL over the reference top-20.

- **Prefill / teacher-forced path, both modes: top-1 1.0000, KL 0** (G067, C056, G066a). The expert cache, zero-copy
  reads and prefill staging only change *where* the bytes are read from; the arithmetic is exllamav3's own kernels.
  The CPU tier and the side-stream shared expert are only active for decode batches (<= 4 / <= 8 tokens), so the
  panel does not exercise them.
- **Decode in `fast` mode is not exact.** Paired check (C052e, same process, same token sequence, CPU tier off vs
  on, 6 prompts, 1,288 decode positions): **mean KL 0.0053 (p99 0.086), top-1 agreement 0.977** vs the exact
  GPU-only path (an earlier paired run, C052d: top-1 0.9837). The CPU kernel itself measures ~1e-3 relative RMS error
  per expert vs an fp64 reference (startup self-test: EXACT 9.9e-4, AFFINE 1.05e-3, I16 1.05e-3), the same class as
  exllamav3's own fused GPU MoE kernel (1.1e-3), which alone would predict a much smaller KL; a decode-path bug
  (handoff / masking / combine) is suspected and being diagnosed. `GLM53_K_OVL=1` (shared expert on a side stream)
  is also not bitwise (~2e-3 relative per layer output, the same as exllamav3's non-fused shared-expert path).
  `GLM53_K_HCFUSE` and `GLM53_K_FTSPLIT` are bit-identical to the kernels they replace. A fixed CPU tier ships as a
  new image digest; until then use `GLM53_MODE=exact` when outputs must match exllamav3.
- The Triton autotune picks of the KDA (linear attention) prefill kernels change rounding: a different pick set
  measured top-1 0.989 / KL 0.0034 on the same panel. The image pins the reference pick set
  (`data/triton_pin_exact.json`, `glm53/triton_pin.py`): a fresh container with an empty Triton cache compiles those
  configs and never benchmarks, and stays exact.

## Host requirements

| | requirement | measured on |
|---|---|---|
| GPU | 1x NVIDIA 24 GB, sm_86 (RTX 3090 / A5000 class). sm_86 binaries also run on sm_89 (RTX 4090), not measured. Blackwell (sm_120) needs a rebuild with `TORCH_CUDA_ARCH_LIST="8.6;12.0"`, not measured | RTX 3090, driver 610.57 |
| driver | CUDA 13.0 capable (the base image is CUDA 13.0.3) | 610.57 |
| PCIe | 4.0 x16 recommended; cache misses are read over the link (~25 GB/s) | 4.0 x16 |
| host RAM, `fast` | **~218 GiB (~234 GB) for the process today**: pinned home copy of all routed experts (42 x 288 x 9.44 MB = 114.2 GB) + the CPU tier's block-contiguous second copy (114.2 GB). Measured MemAvailable drop from start to ready: 213.2 GiB / 228.9 GB (G067), 218.3 GiB / 234.4 GB (C056); process RSS 216.9 GiB. 256 GiB total is the bare minimum with nothing else running; 320 GB+ recommended. With `--memory`, allow at least 230g (the campaign used 250g). A single-copy fast mode (target ~170 GB) is in progress and will ship as a new image digest / env setting | 503 GiB |
| host RAM, `exact` | **~111 GiB (~119 GB)**: the pinned home copy only (measured drop 111.1 GiB / 119.3 GB, G066a); 128 GiB total is the minimum, 160 GB+ recommended. This is the low-RAM option today | |
| CPU, `fast` | x86-64 with **AVX2 + FMA + F16C**; ~24 physical cores; decode speed scales with cores x per-core decode throughput (~3.6 GB/s per core under all-core load) | EPYC 7443P 24C |
| memory bandwidth | 8-channel DDR4 (or better) recommended: the CPU tier streams ~80 GB/s while PCIe reads ~25 GB/s from the same DRAM (measured 136 GB/s read at 22 threads) | 8ch DDR4 |
| disk | 117 GiB (125.3 GB) checkpoint, NVMe recommended (load 103-113 s from page cache) | 990 PRO |
| limits | `--ulimit memlock=-1` (pinned host memory) | |

## Run

```bash
# 1. the checkpoint (branch 3.05bpw), or leave the mounted directory empty and the container downloads it
hf download turboderp/GLM-5.3-Flash-exl3 --revision 3.05bpw --local-dir /data/GLM-5.3-Flash-exl3-3.05bpw

# 2. serve on GPU 0, port 30000 (OpenAI API at /v1)
docker run -d --name glm53 --gpus '"device=0"' --ulimit memlock=-1 --shm-size 16g -p 30000:30000 \
  -v /data/GLM-5.3-Flash-exl3-3.05bpw:/models \
  ghcr.io/0xsero/glm53-flash-offload@sha256:<digest>

docker logs -f glm53          # ready after ~2 min ("loaded in ... s")
curl -s localhost:30000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Hello!"}],"max_tokens":512}'
```

Exact mode: add `-e GLM53_MODE=exact`. Extra arguments after the image name are appended to the server command
(the last value wins), e.g. `... IMAGE -cs 65536` for a 64k KV cache (more expert-cache slots).

Endpoints: `/v1/chat/completions` (stream or not; `tools` -> OpenAI `tool_calls`, parsed from GLM's
`<tool_call>` format; reasoning in `reasoning_content`; disable thinking with
`"chat_template_kwargs": {"enable_thinking": false}`), `/v1/completions`, `/v1/models`, `/health`, `/stats` (cache
hit rate, CPU tier counters), `/server_info`, and SGLang-shaped `/generate` + `/tokenize`. Text only (the
checkpoint's vision tower is not loaded). Requests are not capped: `max_tokens` defaults to the remaining context.

Without Docker: exllamav3 1.5.1 built for your GPU, then `scripts/install.sh` (builds the extensions) and `docker/entrypoint.sh` with `GLM53_ROOT=$PWD GLM53_MODEL_DIR=/path/to/model`.

## Reproduce

Everything the numbers depend on, pinned:

| piece | pin |
|---|---|
| model | `turboderp/GLM-5.3-Flash-exl3`, branch `3.05bpw`, revision `332ab457b709b7ba30dd9a448be5de03b80a7ac9` (125.3 GB) |
| image | `ghcr.io/0xsero/glm53-flash-offload@sha256:<digest>` (see [Image smoke](#image-smoke)), built by [0xSero/local-ai-images](https://github.com/0xSero/local-ai-images) `glm53-flash-offload/Dockerfile` |
| base | `lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f` (v0.5.20, SGLang commit 94602c9; CUDA 13.0.3, torch 2.13.0+cu130, Triton 3.7.1, transformers 5.12.1) |
| exllamav3 | v1.5.1 = commit `958ec933361b24eb8426ec7222e5b0062a679dcd`, built with `TORCH_CUDA_ARCH_LIST=8.6` |
| this repo | the commit the image's `GLM53_COMMIT` names (also in `/opt/glm53/COMMIT` inside the image) |
| Triton picks | `data/triton_pin_exact.json` (7 FLA/KDA kernels) |
| host | AMD EPYC 7443P 24 cores (AVX2, no AVX-512), 8-channel DDR4 503 GiB (136-140 GB/s measured read), RTX 3090 24 GB PCIe 4.0 x16 (~25 GB/s host-to-device), Samsung 990 PRO 4 TB NVMe, NVIDIA driver 610.57.04, Linux 7.2 |

Run the server as in [Run](#run), then from a clone of this repo (Python 3, standard library only):

```bash
# quality: teacher-forced panel vs the exllamav3 reference (expect top-1 1.0000, KL 0 in both modes)
python3 bench/score_ref_panel.py --url http://127.0.0.1:30000 --panel reference/glm-5.3-flash-exl3-ref-panel.json --out score.json
# speed: the protocol behind every table here (P2)
python3 bench/sweep.py --url http://127.0.0.1:30000 --card rtx-3090-24gb/glm-5.3-flash-offload --template glm \
  --config "<digest> fast" --prefill 8192 32768 --conc 1 2 4 --reps 3 --dec-reps 2 --no-early-exit --out sweep.json
# decode quality of the CPU tier: paired tier-off / tier-on decode in one process (stop the server first; ~20 min)
docker run --rm --gpus '"device=0"' --ulimit memlock=-1 --shm-size 16g -v <model dir>:/models -v $PWD:/out \
  ghcr.io/0xsero/glm53-flash-offload@sha256:<digest> decode-kl --kl-out /out/decode_kl.json
```

`results/` holds the raw outputs of the campaign runs behind the tables: `G067/` (shipped `fast` defaults), `C056/`
(previous fast config), `G066a/` (`exact` mode) with `sweep.json`, `score.json`, `server_info.json`, the launch
(`launch.txt`: argv + env; `/w` was this repo mounted in the campaign container) and MemAvailable / VRAM at start and
ready; `C052e/decode_kl.json` is the paired decode check. The reference panel (`reference/`) was produced by stock
exllamav3 1.5.1 on the same checkpoint (8 prompts, top-20 logprobs at every position).

## Image smoke

The published image is tested on the measured host exactly as in [Run](#run) (clean pull by digest, bridge
network, model mounted at `/models`): health, chat, tool call, the teacher-forced panel and the P2 sweep. Results are
recorded here per digest.

| digest | panel top-1 / KL | prefill 8k / 32k | decode C1 / C2 / C4 (aggregate) | C1 at 32k |
|---|---|---|---|---|
| pending | | | | |

## Configuration

| env | default (`fast` / `exact`) | meaning |
|---|---|---|
| `GLM53_MODE` | `fast` | `fast` = G067 defaults, `exact` = G066a defaults |
| `GLM53_MODEL_DIR` | `/models` | checkpoint directory; downloaded there if `config.json` is missing |
| `GLM53_MODEL_REPO` / `GLM53_MODEL_REVISION` | `turboderp/GLM-5.3-Flash-exl3` / `332ab457...` (branch 3.05bpw) | download source |
| `GLM53_CPU_TIER` | `1` / `0` | AVX2 CPU tier for cold decode misses |
| `GLM53_CT_CPUS` | `auto` | CPUs for the CPU tier pool; `auto` = one logical CPU per physical core, minus the first 2 cores (left to the Python thread that drives the GPU and the OS); explicit list like `2-23` |
| `GLM53_CT_THREADS` | number of `GLM53_CT_CPUS` | CPU tier threads (one per listed CPU) |
| `GLM53_CT_SWZ` | `1` | `1` = the CPU tier reads its own block-contiguous copy of the experts (+114.2 GB RAM, the measured setup); `0` = it reads the pinned home copy in the native layout (single copy, ~120 GiB total, but ~40 % slower per CPU expert in the kernel benchmark; not measured end to end) |
| `GLM53_CT_A`, `GLM53_CT_B`, `GLM53_CT_TZC` | `0.11`, `0.104`, `0.40` (ms) | cost model: CPU job = A + B x experts; one zero-copy miss = TZC on the GPU |
| `GLM53_EC_RESERVE_GB` | `1.5` / `1.0` | VRAM left free after the expert cache takes the rest |
| `GLM53_EC_STAGE_GB` | `2.6` | prefill staging buffer size (2 buffers, overlap copy with compute) |
| `GLM53_K_HCFUSE`, `GLM53_K_FTSPLIT`, `GLM53_K_OVL` | `1` / `0` | fused hyper-connection decode sites; fast CPU-tier split kernel; shared expert on a side stream |
| `GLM53_EC_ELASTIC_GB` | `10` | cache memory handed back for prefill activations and staging, re-taken for decode |
| `GLM53_TRITON_PIN` | `data/triton_pin_exact.json` | fixed KDA autotune picks (empty = autotune) |
| `PORT`, `SERVED_NAME`, `GLM53_ARGS` | `30000`, `glm-5.3-flash`, empty | server port, model id, extra args |

**Sizing the CPU tier for another CPU.** The tier is limited by per-core decode throughput, not DRAM (SMT, prefetch
distance and 23-46 threads all measured flat). Give it one thread per physical core and leave 2 cores free; `auto`
does that inside the container's cpuset (24 cores -> cpus 2-23, 22 threads). The per-layer split between CPU and
PCIe uses `GLM53_CT_B` = ms per expert on the CPU at your thread count; it was measured at 22 threads, so for N
threads set roughly `GLM53_CT_B = 0.104 x 22 / N` (e.g. 16 cores -> 14 threads -> 0.16). Fewer than ~12 cores: the
split sends most misses back over PCIe and decode approaches `exact`-mode speed; there `exact` is the better choice.

**Other GPUs.** The expert cache sizes itself from free VRAM after load (minus `GLM53_EC_RESERVE_GB`), so a 32 GB or
48 GB card simply gets more slots (higher hit rate, faster decode); keep the reserve at 1.5 GB unless long prefills
OOM. RTX 4090 (sm_89) runs the sm_86 build unchanged; Blackwell needs the image rebuilt with
`--build-arg TORCH_CUDA_ARCH_LIST="8.6;12.0"`, and a new Triton pin may be needed there (`glm53/make_triton_pin.py`
turns a Triton cache into a pin file; check the panel stays at 1.0000). A slower PCIe link (3.0, or 4.0 x8) costs
decode speed roughly in proportion to the cache-miss traffic.

## Registry and kit

- local-ai-registry recipe: [`rtx-3090-24gb/glm-5.3-flash.exllamav3.128k`](https://github.com/0xSero/local-ai-registry/blob/main/registry/recipes/nvidia/rtx-3090-24gb/glm-5.3-flash.exllamav3.128k.json)
  (launch `registry/launches/exllamav3-glm-5.3-flash-exl3-3.05bpw-offload-128k-rtx-3090-24gb.json`; candidate until
  the CPU-tier decode fix lands)
- local-ai-recipe-kit target: [`targets/glm-5.3-flash-offload.md`](https://github.com/0xSero/local-ai-recipe-kit/blob/main/targets/glm-5.3-flash-offload.md)
  (submit your own measurements of this setup there)

## How it works

- **Zero-copy home** (`glm53/exl3_tiers.py`): at load, every routed expert's trellis goes into one exact-size
  pinned host arena per layer (`cudaHostRegister`), and exllamav3's per-expert pointer tables point at its UVA alias.
  Its MoE kernels can read any expert over PCIe with no copy and no host sync.
- **Elastic CLOCK expert cache** (`glm53/expert_cache.py`): one VRAM slot arena (all free VRAM minus a reserve,
  1,376 slots = 13.0 GB in `fast` mode) shared by all MoE layers. Per layer and decode step a one-block kernel
  dedups the router picks, marks hits, picks CLOCK victims for misses and repoints the tables; a gather kernel
  copies the admitted experts. exllamav3's kernels run unchanged against the live tables, so results are bit-exact.
  Warm-started from routing statistics (`data/stats_own_dec.json`).
- **Prefill staging**: for chunks of >= 8,192 picks per layer, each layer's non-cached experts are copied to a VRAM
  staging buffer on a copy stream one layer ahead, overlapping the previous layer's compute. Elastic mode hands
  10 GB of cache slots back to torch for prefill activations and staging, and re-takes them for decode.
- **CPU tier** (`glm53/cpu_tier.py`, `kernels/cpu_avx2/`): in decode (<= 4 tokens), after routing, a GPU kernel
  splits the misses by a cost model: the warmest go zero-copy and are admitted to the cache, the coldest (~4 per
  layer at C1, more when batched) are published through pinned mapped memory to a C++ worker that computes them with an AVX2 MUL1/K=3 trellis
  kernel (fp32/int16 activations, no int8 quantisation) from a block-contiguous host copy; the GPU removes those
  picks from its own MoE and adds the CPU's fp32 partial back after its experts (device-side flag wait, no host sync).
- **Batched decode**: `-ambs 4` gives the recurrent (KDA) cache 4 slots so up to 4 streams decode in one step and
  share the per-layer expert union.
- **Decode kernels** (`fast` mode): the 4 launches of each mHC hyper-connection site fused into 2 (`k_hcfuse.py`,
  bit-exact), a faster CPU-tier split kernel (`kernels/k110/`, bit-identical), and the shared expert moved out of the
  fused MoE kernel onto a side stream so it overlaps the PCIe miss gather and the CPU wait (`k_overlap.py`).

## Layout

```
glm53/serve.py          OpenAI-compatible server over exllamav3 AsyncGenerator
glm53/exl3_tiers.py     pinned zero-copy home copy of all routed experts
glm53/expert_cache.py   elastic CLOCK GPU expert cache + prefill staging (CUDA source inline)
glm53/cpu_tier.py       CPU tier wiring, cost-model policy, self-test, auto CPU sizing
glm53/triton_pin.py     Triton autotune pinning for the KDA kernels (+ make_triton_pin.py)
glm53/k_hcfuse.py       fused hyper-connection decode sites (CUDA source inline)
glm53/k_ftsplit.py      fast CPU-tier split kernel loader (kernels/k110/ft_split_fast.cu)
glm53/k_overlap.py      shared expert on a side stream in decode
kernels/cpu_avx2/       ft_core.h (AVX2 kernels, pool), ft_tier_ext.cpp (host worker), ft_tier_cu.cu (GPU split/combine)
data/                   routing stats, exllamav3 MoE tune cache, Triton autotune pins
docker/                 Dockerfile (local build), entrypoint.sh
scripts/install.sh      pre-build the extensions
bench/                  sweep.py (speed protocol), score_ref_panel.py (quality), decode_kl.py (paired CPU-tier decode check)
reference/              exllamav3 reference panel for GLM-5.3-Flash 3.05bpw
results/                raw JSONs of the measured runs (G067, C056, G066a, C052e)
```

Updating the image: a fix here is a new commit; the image build in
[0xSero/local-ai-images](https://github.com/0xSero/local-ai-images) (`glm53-flash-offload/`) pins this repo by
commit, so bumping that pin produces a new attested digest. To try a checkout without rebuilding, mount it over
`/opt/glm53` (the extensions rebuild on first start; nvcc and ninja are in the image).

## Credits

- [exllamav3](https://github.com/turboderp-org/exllamav3) and the EXL3 format by turboderp (MIT): all model code,
  kernels and the quantised checkpoint [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3);
  the CPU kernel reproduces exllamav3's MUL1 codebook decode.
- FreeToken (FlashML): the idea of serving cold MoE experts from host memory with a GPU-side hot cache and a CPU
  compute tier.
- [SGLang](https://github.com/sgl-project/sglang) (`lmsysorg/sglang` v0.5.20) as the CUDA/torch/Triton base image;
  the server endpoints follow SGLang's `/generate` shape.
- GLM-5.3-Flash by Z.ai.

License: MIT (this repository). The model weights carry their own license.
