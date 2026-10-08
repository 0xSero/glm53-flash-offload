# glm53-flash-offload

Serves **GLM-5.3-Flash** (turboderp EXL3 3.05 bpw, 125.3 GB, 288 routed experts x 42 MoE layers) from **one 24 GB
RTX 3090** with an OpenAI-compatible API. The GPU holds a cache of hot experts, host RAM holds a second tier, and in
the NVMe modes everything else is read from a packed expert store on local NVMe. It is a thin layer of patches and
small kernels over stock exllamav3 1.5.1, shipped as one Docker image.

**Result: GLM-5.3-Flash on one RTX 3090 with 55 GB of RAM and 4 NVMe drives: 18 tok/s decode** (18.23 tok/s at C1,
19.59 tok/s aggregate at C4, 8k prefill 650 tok/s; local-ai-registry lab acceptance 17.3 tok/s, all six gates pass).

## Quick start

### Easiest: Omarchy Local AI

```bash
omarchy plugin add https://github.com/sybil-solutions/omarchy-local-ai --enable
```

Open Local AI in the bar, click **Set up Local AI** once, then on the models tab pick **GLM-5.3-Flash, EXL3 3.05 bpw
(55 GB RAM, experts on NVMe)** on the RTX 3090 and **Start** it. The plugin downloads the pinned weights, packs and
verifies the NVMe store, starts the image below and opens Claude Code, Codex, OpenCode, pi and others on it. Its models
folder (`~/.cache/omarchy/local-ai/models`, may be a symlink to another disk) must be on NVMe with 243 GB free; the
recipe is offered only when the machine has the RAM, disk and 40 CPU threads it needs.

### Docker

Needs Docker with the NVIDIA container toolkit, the `hf` CLI (`pip install -U huggingface_hub`), and the hardware in
[Hardware requirements](#hardware-requirements).

```bash
W=/data/GLM-5.3-Flash-exl3-3.05bpw   # checkpoint, 125.3 GB
S=/mnt/nvx/glm53                     # expert store, 117.3 GB, on fast local NVMe (xfs or ext4)
IMG=ghcr.io/sybil-solutions/glm53-flash-offload@sha256:99b8926e01480b1acc35251a60f61cba6a2ad6309491a6ab2bb6b5688f843c16

# 1. weights (branch 3.05bpw, pinned revision)
hf download turboderp/GLM-5.3-Flash-exl3 --revision 332ab457b709b7ba30dd9a448be5de03b80a7ac9 --local-dir "$W"

# 2. once: pack the routed experts into the NVMe store (CPU only), then re-check it (sha256 of every record)
mkdir -p "$S"
docker run --rm -v "$W":/models:ro -v "$S":/nvx "$IMG" pack-store
docker run --rm -v "$W":/models:ro -v "$S":/nvx:ro "$IMG" verify-store

# 3. serve on GPU 0, capped at 55 GiB of host RAM
docker run -d --name glm53 --gpus '"device=0"' \
  --memory 55g --memory-swap 55g --shm-size 1g --ulimit memlock=-1 --cpuset-cpus 2-39 \
  -p 127.0.0.1:30000:30000 \
  -e GLM53_MODE=nvme -e GLM53_EC_MAX_SLOTS=1376 -e GLM53_MODEL_DOWNLOAD=0 \
  -v "$W":/models:ro -v "$S":/nvx:ro "$IMG"
until curl -sf localhost:30000/health >/dev/null; do sleep 5; done   # ~80 s; docker logs glm53 shows "loaded in ... s"

# 4. test
curl -s localhost:30000/v1/chat/completions -H 'content-type: application/json' -d '{"model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": "Say hello."}], "chat_template_kwargs": {"enable_thinking": false}}'
```

- **Fewer than 40 CPU threads:** drop `--cpuset-cpus 2-39` or give your own range; the entrypoint then derives the
  thread layout from the container's CPU set and logs it (speed with a derived layout is not measured).
- **Bit-exact output:** `-e GLM53_MODE=nvme-exact` (8.3 tok/s). **All experts in RAM (~238 GB free):** see
  [Pick your mode](#pick-your-mode).
- **16 GB of RAM:** from the image after this repo's small-host change (next tag), the same command with
  `--memory 16g --memory-swap 16g` runs: below a 24 GiB cap the entrypoint applies a small-host budget (`-rcs 1`,
  margin 1.5 GiB, 128-slot prefill ring, 48 NVMe readers x 1 MiB; each overridable) and the RAM tier sizes itself to
  ~969 experts. NVMe modes now refuse only caps under 15 GiB (`GLM53_NV_MIN_GB`). Older images (v4.1-nvme and before)
  stop at 24 GiB. The digest and lab result land here when the 16 GB acceptance passes.

**Client settings.** Base URL `http://127.0.0.1:30000/v1`, model `glm-5.3-flash`, no API key is checked (send any
string). The server speaks OpenAI Chat Completions (`/v1/chat/completions`, streaming, `tools` -> `tool_calls`,
reasoning in `reasoning_content`), plus `/v1/completions` and `/v1/models`; it has no Anthropic `/v1/messages` or
OpenAI `/v1/responses` endpoint, so Claude Code and Codex connect through the Omarchy Local AI gateway, which speaks
both. Text only. The server has no authentication: keep the port on 127.0.0.1.

## Pick your mode

One image switch, `GLM53_MODE`. Decode at 8k context, prefill at 8k / 32k, all on the host below.

| host RAM free | mode | image | decode | prefill 8k / 32k | quality vs stock exllamav3 |
|---|---|---|---|---|---|
| ~238 GB | `fast` (all experts in RAM, AVX2 CPU tier) | `0xsero@bb633b0b` | 28.2 tok/s C1, 34.0 C4 | 710 / 951 tok/s | decode KL 0.0053 (top-1 0.977); prefill exact |
| ~120 GB | `exact` (all experts in RAM) | `0xsero@bb633b0b` | 12.6 tok/s C1 | 697 / 945 tok/s | bit-exact |
| 55 GiB cap + NVMe store | **`nvme`** | `sybil-solutions@99b8926e` | 18.2 tok/s C1, 19.6 C4 (screen); 14.9 / 16.3 (full sweep) | 650 / 812 tok/s | decode KL 0.0043-0.0051 (top-1 0.983-0.986); prefill exact |
| 16 GiB cap + NVMe store | `nvme` (16 GB settings) | campaign build, no image yet | 12.9 tok/s C1 (screen) | 628 / not run | same CPU lane as `nvme` |
| 55 GiB cap + NVMe store | `nvme-exact` | `sybil-solutions@99b8926e` | 8.3 tok/s C1 | 564 / 806 tok/s | bit-exact |

KL = mean KL divergence of the next-token distribution vs the exact GPU path on the same tokens (paired, same
process). The teacher-forced panel (2,154 positions) is top-1 1.0000 / KL 0 in every mode.

All-RAM (`fast`; add `-e GLM53_MODE=exact` for `exact`), from the registry launch
`exllamav3-glm-5.3-flash-exl3-3.05bpw-offload-128k-rtx-3090-24gb`:

```bash
docker run -d --name glm53 --gpus '"device=0"' --ulimit memlock=-1 --shm-size 16g -p 127.0.0.1:30000:30000 \
  -e GLM53_MODEL_DOWNLOAD=0 -v "$W":/models:ro \
  ghcr.io/0xsero/glm53-flash-offload@sha256:bb633b0bcb85573ad40b6c408af5e1036ae062c593e42479f4e4d2ab521e551d
```

## Hardware requirements

| | NVMe modes (`nvme`, `nvme-exact`) | all-RAM modes (`fast`, `exact`) |
|---|---|---|
| GPU | 1x NVIDIA 24 GB, sm_86 build (RTX 3090 / A5000 class; RTX 4090 runs it, not measured). Driver with CUDA 13.0 support (image base CUDA 13.0.3; registry minimum driver 580). One GPU per container | same |
| CPU | x86-64 with AVX2 + FMA + F16C (`nvme-exact` needs no AVX2). 40 logical CPUs for the measured thread layout (`--cpuset-cpus 2-39`: CPU lane 22 threads on 2-23, GPU-feeding main thread on 24, controller on 25, 16 NVMe readers on 26-39) | AVX2 + FMA + F16C for `fast`; ~24 physical cores (CPU tier: one thread per core minus 2) |
| RAM | `--memory 55g --memory-swap 55g` (55 GiB = 59.1 GB; measured peak 51.3 GiB). 16 GB settings: 14.4 GiB used of a 16 GiB cap (campaign build only) | `fast` ~238 GB free (two copies of the experts); `exact` ~120 GB |
| NVMe | Local NVMe for the store, read at ~20 GB/s during C1 decode at the 16 GB cap and ~7 GB/s averaged over the 55 GB sweep. Measured on 4x Samsung 9100 PRO 1 TB in md RAID0 (512k chunk) on a PCIe 4.0 x16 card, ~26 GB/s. One drive is not measured. Filesystem must support `O_DIRECT`: plain xfs or ext4 on the drives, not a loop file on btrfs or inside a LUKS container | weights on local NVMe (read once at load) |
| disk | 125.3 GB checkpoint + 117.3 GB store = 242.6 GB | 125.3 GB |
| limits | `--ulimit memlock=-1` (RAM tier is pinned), `--shm-size 1g` (tmpfs counts against the cap) | `--ulimit memlock=-1` recommended, `--shm-size 16g` |

## Measured results

Protocol: `bench/sweep.py --template glm --prefill 8192 32768 --conc 1 2 4 --reps 3 --dec-reps 2`. Prefill = median of
3 fresh prompts, tokens / time to first token. Decode = greedy completions run to their natural end, C simultaneous
streams, aggregate tok/s (per-stream in brackets), median of 2 rounds. "Screen" runs measure 8k C1 (+ C4) only.
Every row: KV cache 131,072 tokens, 1 GPU.

**`nvme`, 55 GiB, prefetch on, no read cap (screen n55_pfon_nocap, 2026-10-08)**

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | 649.8 tok/s | 18.23 tok/s | 1 | 131,072 | 1 |
| 8,192 | not run | not run | 2 | 131,072 | 1 |
| 8,192 | not run | 19.59 tok/s (5.44 per stream) | 4 | 131,072 | 1 |
| 32,768 | not run | not run | 1 | 131,072 | 1 |
| 32,768 | not run | not run | 2 | 131,072 | 1 |

1x RTX 3090 (PCIe 4.0 x16), AMD EPYC 7443P (24C/48T), 8-channel DDR4, 4x Samsung 9100 PRO RAID0 (xfs).
Lab acceptance of image `99b8926e` (registry PR #187, 2026-10-08, with an 8 GB/s NVMe read cap that costs ~8 % C1
decode and ~26 % prefill in a same-session pair): decode 17.3 tok/s, prefill 844 tok/s on a 93,437-token prompt.

**`nvme`, 55 GiB, full sweep (campaign arm S3b2, the shipped defaults, 2026-10-07)**

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | 583 tok/s | 14.85 tok/s | 1 | 131,072 | 1 |
| 8,192 | not run | 15.62 tok/s (7.87 per stream) | 2 | 131,072 | 1 |
| 8,192 | not run | 16.27 tok/s (4.63 per stream) | 4 | 131,072 | 1 |
| 32,768 | 812 tok/s | 13.46 tok/s | 1 | 131,072 | 1 |
| 32,768 | not run | not run | 2 | 131,072 | 1 |

1x RTX 3090 (PCIe 4.0 x16), AMD EPYC 7443P (24C/48T), 8-channel DDR4, 4x Samsung 9100 PRO RAID0 (xfs).

**`nvme`, 16 GiB (screen s16_r48, campaign build, 2026-10-08)**: `--memory 16g --memory-swap 16g`,
`GLM53_NV_MARGIN_GB=1.5 GLM53_NV_PF_RING=128 GLM53_NV_THREADS=48 GLM53_NV_PIECE_KB=1024`, `-rcs 1`; RAM tier 970
experts (8.5 GiB).

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | 628 tok/s | 12.91 tok/s | 1 | 131,072 | 1 |
| 8,192 | not run | not run | 2 | 131,072 | 1 |
| 8,192 | not run | not run | 4 | 131,072 | 1 |
| 32,768 | not run | not run | 1 | 131,072 | 1 |
| 32,768 | not run | not run | 2 | 131,072 | 1 |
| 16 GB lab acceptance | lab run in progress | lab run in progress | - | 131,072 | 1 |

1x RTX 3090 (PCIe 4.0 x16), AMD EPYC 7443P (24C/48T), 8-channel DDR4, 4x Samsung 9100 PRO RAID0 (xfs). NVMe read
19.9 GB/s during C1 decode (array ceiling ~26 GB/s); container memory 14.4 GiB of 16.

**`nvme-exact`, 55 GiB (campaign arm S2a, 2026-10-07)**: concurrent streams run one after another.

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | 564 tok/s | 8.28 tok/s | 1 | 131,072 | 1 |
| 8,192 | not run | 8.23 tok/s (serial) | 2 | 131,072 | 1 |
| 8,192 | not run | 8.28 tok/s (serial) | 4 | 131,072 | 1 |
| 32,768 | 806 tok/s | 7.74 tok/s | 1 | 131,072 | 1 |
| 32,768 | not run | not run | 2 | 131,072 | 1 |

1x RTX 3090 (PCIe 4.0 x16), AMD EPYC 7443P (24C/48T), 8-channel DDR4, 4x Samsung 9100 PRO RAID0 (xfs).

**`fast`, all experts in RAM (campaign run G067, 2026-10-01)**

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | 710 tok/s | 28.15 tok/s | 1 | 131,072 | 1 |
| 8,192 | not run | 31.57 tok/s (15.78 per stream) | 2 | 131,072 | 1 |
| 8,192 | not run | 33.98 tok/s (8.62 per stream) | 4 | 131,072 | 1 |
| 32,768 | 951 tok/s | 26.89 tok/s | 1 | 131,072 | 1 |
| 32,768 | not run | not run | 2 | 131,072 | 1 |

1x RTX 3090 (PCIe 4.0 x16), AMD EPYC 7443P (24C/48T), 8-channel DDR4 (503 GiB), weights on Samsung 990 PRO.

**`exact`, all experts in RAM (campaign run G066a, 2026-10-01)**

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | 697 tok/s | 12.57 tok/s | 1 | 131,072 | 1 |
| 8,192 | not run | 12.62 tok/s | 2 | 131,072 | 1 |
| 8,192 | not run | 12.64 tok/s | 4 | 131,072 | 1 |
| 32,768 | 945 tok/s | 11.86 tok/s | 1 | 131,072 | 1 |
| 32,768 | not run | not run | 2 | 131,072 | 1 |

1x RTX 3090 (PCIe 4.0 x16), AMD EPYC 7443P (24C/48T), 8-channel DDR4 (503 GiB), weights on Samsung 990 PRO.

Raw files: [`results/`](results/) (G067, G066a, N119-S2a-55g, N119-S3b2-55g, image smokes P001-P003). Every other run,
image smoke and quality check: [docs/reference.md](docs/reference.md).

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `docker run` fails naming `/dev/nvidia1` (or another GPU device that is not there) | stale CDI spec after a GPU or driver change: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` |
| `[glm53] ERROR: /nvx (...) does not support O_DIRECT reads/writes` | the store is on a filesystem that refuses `O_DIRECT`. Put it on xfs or ext4 on a local NVMe drive and mount that directory at `/nvx`. `pack-store`, `verify-store` and the server all run this probe first |
| `NVMe store not found: /nvx/glm53_flash_exl3_3.05bpw_experts.bin` | run `pack-store` (step 2) and mount the same directory at `/nvx` |
| `GLM53_MODE=nvme sizes its RAM tier from the container memory cap` | add `--memory 55g --memory-swap 55g` |
| `container memory limit N GiB < ~24 GiB needed` | an image from before the small-host change (v4.1-nvme or older): use a newer tag, or a cap of at least 24 GiB. Newer images refuse NVMe modes only under 15 GiB (`< ~15 GiB needed`) |
| `WARNING: memory.swap.max is ...`, or the container is OOM-killed (exit 137) | `--memory-swap` must equal `--memory`. With the default, pages swap out (zram, swap files) and escape the cap. Keep `--shm-size 1g`: shared memory counts against the cap |
| CUDA out of memory in `graph.cu` mid-run while a desktop uses the same GPU | keep `-e GLM53_EC_MAX_SLOTS=1376`; free VRAM, or raise `GLM53_EC_RESERVE_GB` |
| decode far below the tables | the store is on a slow drive. Decode reads ~20 GB/s at the 16 GB cap; an 8 GB/s read cap alone cost ~8 % C1, ~33 % C4 and ~26 % prefill. Check reads with `iostat -x 1` during a request; use a RAID0 of NVMe drives. Also check the startup line `nv2 CPU layout`: a derived layout (fewer than 40 CPUs) is not measured |
| `CPU lacks avx2` | use `-e GLM53_MODE=nvme-exact` (no CPU lane) |
| `no NVIDIA GPU visible` / `WARNING: N GPUs visible` | pass `--gpus '"device=0"'` (one GPU) |

## How it works

- **VRAM expert cache.** All free VRAM after load (minus a 1.5 GB reserve, capped at 1,376 slots = 13 GB) holds
  experts in one CLOCK cache shared by all MoE layers. A small kernel dedups each layer's router picks, marks hits and
  repoints exllamav3's expert tables, so exllamav3's own MoE kernels run unchanged.
- **RAM tier.** A pinned arena sized from the container memory cap (4,831 experts = 42.5 GiB at 55 GiB). Experts
  evicted from VRAM are written back into it, so VRAM and RAM hold different experts (~6,100 of 12,096).
- **NVMe record store.** `pack-store` writes the 12,384 routed experts (42 MoE layers + MTP, 9.47 MB each, 4096-aligned,
  sha256 per record) into one file. A 16-thread `O_DIRECT` reader pool fills misses into fresh RAM slots; the GPU
  waits for the reply on the device, with no host sync.
- **CPU tier.** In decode, cold experts that are already in RAM are computed on the host by an AVX2 kernel (22
  threads) instead of being copied over PCIe; the GPU adds the CPU's partial result back. This is the only part that
  is not bit-exact (`nvme-exact` turns it off).
- **Prefetch.** Each layer's likely picks are read from NVMe a layer ahead; prefill stages each layer's experts
  through a pinned ring that a reader fills several layers ahead.

The placement model and simulator behind these tiers: [moetier](https://github.com/sybil-solutions/moetier).
Registry evidence: launch
[`exllamav3-glm-5.3-flash-exl3-3.05bpw-nvme-55g-128k-rtx-3090-24gb`](https://github.com/sybil-solutions/local-ai-registry/blob/main/registry/launches/exllamav3-glm-5.3-flash-exl3-3.05bpw-nvme-55g-128k-rtx-3090-24gb.json),
recipe [`rtx-3090-24gb/glm-5.3-flash.exllamav3-nvme.128k`](https://github.com/sybil-solutions/local-ai-registry/blob/main/registry/recipes/nvidia/rtx-3090-24gb/glm-5.3-flash.exllamav3-nvme.128k.json),
lab run [`...20261008T083936.json`](https://github.com/sybil-solutions/local-ai-registry/blob/main/lab/runs/rtx-3090-24gb.glm-5.3-flash.exllamav3-glm-5.3-flash-exl3-3.05bpw-nvme-55g-128k-rtx-3090-24gb.128k.20261008T083936.json)
(PR [#187](https://github.com/sybil-solutions/local-ai-registry/pull/187)). Image built by
[local-ai-images](https://github.com/sybil-solutions/local-ai-images) run
[37628249109](https://github.com/sybil-solutions/local-ai-images/actions/runs/37628249109) from this repo at `7f1ee89`.

Full reference (every configuration variable, quality method, image smoke, reproduction pins, code layout):
[docs/reference.md](docs/reference.md).

## Credits

[exllamav3](https://github.com/turboderp-org/exllamav3) and the EXL3 checkpoint
[turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) by turboderp (MIT); FreeToken
(FlashML) for the host-memory expert tier idea; [SGLang](https://github.com/sgl-project/sglang) v0.5.20 as the base
image; GLM-5.3-Flash by Z.ai. License: MIT (this repository); the model weights carry their own license.
