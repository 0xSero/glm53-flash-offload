# glm53-flash-offload

Serve **GLM-5.3-Flash** (EXL3 3.05 bpw, 125.3 GB, 288 routed experts x 42 MoE layers) from **one 24 GB RTX 3090**
plus host DDR4, with an OpenAI-compatible API. All routed experts live in pinned host RAM; the GPU keeps an elastic
cache of hot experts, reads the rest over PCIe (zero-copy), and an AVX2 CPU kernel computes the coldest cache misses
during decode on the host cores.

It is a thin layer of monkeypatches over **stock exllamav3 1.5.1** (no exllamav3 source edits) plus two small
custom extensions. This is the exact code and configuration of run C056 of the FreeToken-EXL3 campaign.

Two modes, one switch (`GLM53_MODE`):

| mode | what runs | decode quality vs stock exllamav3 | host RAM |
|---|---|---|---|
| `fast` (default) | GPU expert cache + zero-copy misses + **AVX2 CPU tier** for cold misses, batched decode (`-ambs 4`) | prefill exact; decode **not exact** (see Quality) | ~218 GiB |
| `exact` | GPU expert cache + zero-copy misses, no CPU tier | bit-exact (panel top-1 1.0000, KL 0) | ~111 GiB |

## Measured

Host: AMD EPYC 7443P (24 cores, SMT on), 8-channel DDR4 (503 GiB), 1x RTX 3090 24 GB on PCIe 4.0 x16, driver
610.57, Samsung 990 PRO NVMe. Image stack: lmsysorg/sglang v0.5.20 (CUDA 13.0) + exllamav3 v1.5.1 built for sm_86.
Protocol (campaign `sweep.py --template glm --prefill 8192 32768 --conc 1 2 4 --reps 3 --dec-reps 2`): prefill =
median of 3 fresh random-token prompts, tokens / time-to-first-token; decode = greedy chat completions run to their
natural end, C simultaneous streams, aggregate = all completion tokens / wall time, median of 2 rounds.

**`fast` mode (C056)**: `-cs 131072 --max-batch-size 8 -chunk_size 8192 -ambs 4`, cache reserve 1.5 GB, staging
2 x 2.0 GB, elastic 10 GB, CPU tier on 22 threads (cpus 2-23).

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | 626 tok/s | 27.20 tok/s | 1 | 131,072 tokens | 1 |
| 32,768 | 808 tok/s | 26.15 tok/s (C1 at 32k context) | 1 | 131,072 tokens | 1 |
| - | - | 30.72 tok/s aggregate (15.43 per stream) | 2 | 131,072 tokens | 1 |
| - | - | 34.72 tok/s aggregate (8.71 per stream) | 4 | 131,072 tokens | 1 |

Same configuration without `-ambs 4` (C052c): prefill 629 / 812, decode C1 27.32, C2 27.06, C4 27.45 aggregate
(the generator then runs streams one after another), C1 at 32k 26.45.

**`exact` mode (G066a)**: `-cs 131072 --max-batch-size 8 -chunk_size 8192`, cache reserve 1.0 GB, staging 2 x 2.6 GB,
elastic 10 GB, no CPU tier. (G066a ran with the campaign's reference Triton cache; the image pins the C056 pick set,
whose prefill path measured the same top-1 1.0000 / KL 0.)

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | 697 tok/s | 12.57 tok/s | 1 | 131,072 tokens | 1 |
| 32,768 | 945 tok/s | 11.86 tok/s (C1 at 32k context) | 1 | 131,072 tokens | 1 |
| - | - | 12.62 tok/s aggregate | 2 | 131,072 tokens | 1 |
| - | - | 12.64 tok/s aggregate | 4 | 131,072 tokens | 1 |

For reference, stock exllamav3 with the experts it cannot fit in VRAM on its CPU worker decodes this model at
7.5-8.4 tok/s on the same box.

## Quality

Reference: a teacher-forced panel (8 prompts, 2,154 scored positions) from stock exllamav3 1.5.1 on the same
checkpoint; metric = top-1 agreement and mean KL over the reference top-20.

- **Prefill / teacher-forced path, both modes: top-1 1.0000, KL 0** (C056, C052c, G066a, G060a). The expert cache,
  zero-copy reads and prefill staging only change *where* the bytes are read from; the arithmetic is exllamav3's own
  kernels. The CPU tier is off for forwards larger than 4 tokens, so the panel does not exercise it.
- **Decode with the CPU tier (`fast` mode) is not exact.** Paired check (C052e, same process, same token sequence,
  tier off vs on, 6 prompts, 1,288 decode positions): **mean KL 0.0053 (p99 0.086), top-1 agreement 0.977** vs the
  exact GPU-only path. An earlier paired run (C052d) gave top-1 0.9837 on the same positions. The CPU kernel itself
  measures ~1e-3 relative RMS error per expert vs an fp64 reference (startup self-test: EXACT 9.9e-4, AFFINE 1.05e-3,
  I16 1.05e-3), the same class as exllamav3's own fused GPU MoE kernel (1.1e-3), which by itself would predict a much
  smaller KL; a decode-path bug (handoff / masking / combine) is suspected and being diagnosed. A fixed CPU tier ships
  as a new image digest; until then, use `GLM53_MODE=exact` when you need outputs identical to exllamav3.
- The Triton autotune picks of the KDA (linear attention) prefill kernels change rounding: a different pick set
  measured top-1 0.989 / KL 0.0034 on the same panel. The image ships the measured run's Triton cache and pins its
  picks (`data/triton_pin_c056.json`, `glm53/triton_pin.py`), so a fresh container does not re-benchmark.

## Host requirements

| | requirement | measured on |
|---|---|---|
| GPU | 1x NVIDIA 24 GB, sm_86 (RTX 3090 / A5000 class). sm_86 binaries also run on sm_89 (RTX 4090), not measured. Blackwell (sm_120) needs a rebuild with `TORCH_CUDA_ARCH_LIST="8.6;12.0"`, not measured | RTX 3090, driver 610.57 |
| driver | CUDA 13.0 capable (the base image is CUDA 13.0.3) | 610.57 |
| PCIe | 4.0 x16 recommended; cache misses are read over the link (~25 GB/s) | 4.0 x16 |
| host RAM, `fast` | **~218 GiB for the process**: pinned home copy of all routed experts (42 x 288 x 9.44 MB = 114.2 GB) + the CPU tier's block-contiguous copy (114.2 GB). Measured MemAvailable drop from start to ready: 218.3 GiB (C056), process RSS 216.9 GiB. 256 GiB total is the bare minimum with nothing else running; 320 GB+ recommended. With `--memory`, allow at least 230g (the campaign used 250g) | 503 GiB |
| host RAM, `exact` | ~111 GiB (measured drop 111.1 GiB, G066a); 128 GiB total is the minimum, 160 GB+ recommended | |
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

Endpoints: `/v1/chat/completions` (stream or not; reasoning in `reasoning_content`; disable thinking with
`"chat_template_kwargs": {"enable_thinking": false}`), `/v1/completions`, `/v1/models`, `/health`, `/stats` (cache
hit rate, CPU tier counters), `/server_info`, and SGLang-shaped `/generate` + `/tokenize`. No tool-call parsing.
Requests are not capped: `max_tokens` defaults to the remaining context.

Without Docker: exllamav3 1.5.1 built for your GPU, then `scripts/install.sh` (unpacks the Triton cache, builds the
extensions) and `docker/entrypoint.sh` with `GLM53_ROOT=$PWD GLM53_MODEL_DIR=/path/to/model`.

## Configuration

| env | default (`fast` / `exact`) | meaning |
|---|---|---|
| `GLM53_MODE` | `fast` | `fast` = C056 defaults, `exact` = G066a defaults |
| `GLM53_MODEL_DIR` | `/models` | checkpoint directory; downloaded there if `config.json` is missing |
| `GLM53_MODEL_REPO` / `GLM53_MODEL_REVISION` | `turboderp/GLM-5.3-Flash-exl3` / `332ab457...` (branch 3.05bpw) | download source |
| `GLM53_CPU_TIER` | `1` / `0` | AVX2 CPU tier for cold decode misses |
| `GLM53_CT_CPUS` | `auto` | CPUs for the CPU tier pool; `auto` = one logical CPU per physical core, minus the first 2 cores (left to the Python thread that drives the GPU and the OS); explicit list like `2-23` |
| `GLM53_CT_THREADS` | number of `GLM53_CT_CPUS` | CPU tier threads (one per listed CPU) |
| `GLM53_CT_A`, `GLM53_CT_B`, `GLM53_CT_TZC` | `0.11`, `0.104`, `0.40` (ms) | cost model: CPU job = A + B x experts; one zero-copy miss = TZC on the GPU |
| `GLM53_EC_RESERVE_GB` | `1.5` / `1.0` | VRAM left free after the expert cache takes the rest |
| `GLM53_EC_STAGE_GB` | `2.0` / `2.6` | prefill staging buffer size (2 buffers, overlap copy with compute) |
| `GLM53_EC_ELASTIC_GB` | `10` | cache memory handed back for prefill activations and staging, re-taken for decode |
| `GLM53_TRITON_PIN` | `data/triton_pin_c056.json` | fixed KDA autotune picks (empty = autotune) |
| `PORT`, `SERVED_NAME`, `GLM53_ARGS` | `30000`, `glm-5.3-flash`, empty | server port, model id, extra args |

**Sizing the CPU tier for another CPU.** The tier is limited by per-core decode throughput, not DRAM (SMT, prefetch
distance and 23-46 threads all measured flat). Give it one thread per physical core and leave 2 cores free; `auto`
does that inside the container's cpuset (24 cores -> cpus 2-23, 22 threads). The per-layer split between CPU and
PCIe uses `GLM53_CT_B` = ms per expert on the CPU at your thread count; it was measured at 22 threads, so for N
threads set roughly `GLM53_CT_B = 0.104 x 22 / N` (e.g. 16 cores -> 14 threads -> 0.16). Fewer than ~12 cores: the
split sends most misses back over PCIe and decode approaches `exact`-mode speed; there `exact` is the better choice.

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

## Layout

```
glm53/serve.py          OpenAI-compatible server over exllamav3 AsyncGenerator
glm53/exl3_tiers.py     pinned zero-copy home copy of all routed experts
glm53/expert_cache.py   elastic CLOCK GPU expert cache + prefill staging (CUDA source inline)
glm53/cpu_tier.py       CPU tier wiring, cost-model policy, self-test, auto CPU sizing
glm53/triton_pin.py     Triton autotune pinning for the KDA kernels (+ make_triton_pin.py)
kernels/cpu_avx2/       ft_core.h (AVX2 kernels, pool), ft_tier_ext.cpp (host worker), ft_tier_cu.cu (GPU split/combine)
data/                   routing stats, exllamav3 MoE tune cache, Triton cache + pins of the measured run
docker/                 Dockerfile (local build), entrypoint.sh
scripts/install.sh      unpack Triton cache, pre-build extensions
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
