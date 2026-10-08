# glm53-flash-offload

Run **GLM-5.3-Flash** (125 GB, EXL3 3.05 bpw) on **one 24 GB RTX 3090** with an OpenAI-compatible API, using RAM and
NVMe for the experts that do not fit on the GPU.

| RAM for the server | mode | decode, 1 user | prefill 8k / 32k | output vs stock exllamav3 |
|---|---|---|---|---|
| 218 GiB free | `fast` | 28.2 tok/s | 710 / 951 tok/s | near-exact (decode KL 0.005) |
| **55 GiB cap** + NVMe | **`nvme`** | **17.3 tok/s** | 662 / 965 tok/s | near-exact (decode KL 0.005) |
| 16 GiB cap + NVMe | `nvme` | 12.9 tok/s | 628 / not run | near-exact (same CPU lane) |
| 55 GiB cap + NVMe | `nvme-exact` | 8.3 tok/s | 564 / 806 tok/s | bit-exact |

Measured on 1x RTX 3090 (PCIe 4.0 x16), AMD EPYC 7443P (24 cores), 8-channel DDR4, and 4x Samsung 9100 PRO in RAID0.
Decode runs each answer to its natural end. Full tables, dates and raw files: [docs/results.md](docs/results.md).

## Run it

**Omarchy Local AI.** Run the line below. In Local AI, pick **GLM-5.3-Flash, EXL3 3.05 bpw (55 GB RAM, experts on
NVMe)** and press **Start**. The plugin downloads the weights, builds the store and starts the server.

```bash
omarchy plugin add https://github.com/sybil-solutions/omarchy-local-ai --enable
```

**Docker.** Needs the NVIDIA container toolkit, the `hf` CLI, 125 GB for the weights, and 117 GB for the expert store
on a fast local NVMe filesystem (xfs or ext4).

```bash
IMG=ghcr.io/sybil-solutions/glm53-flash-offload@sha256:4732a063fa9e28d4d5dc7b2c3b57cb7ed84ecfff40caeb4b5bc59d71be1882b3
hf download turboderp/GLM-5.3-Flash-exl3 --revision 332ab457b709b7ba30dd9a448be5de03b80a7ac9 --local-dir /data/glm53
docker run --rm -v /data/glm53:/models:ro -v /mnt/nvme/glm53:/nvx "$IMG" pack-store     # once: 117 GB expert store
docker run -d --name glm53 --gpus '"device=0"' --memory 55g --memory-swap 55g --shm-size 1g --ulimit memlock=-1 \
  -e GLM53_MODE=nvme -e GLM53_EC_MAX_SLOTS=1376 -p 127.0.0.1:30000:30000 -v /data/glm53:/models:ro -v /mnt/nvme/glm53:/nvx:ro "$IMG"
```

The server is ready in about 80 s at `http://127.0.0.1:30000/v1`, model `glm-5.3-flash`. For 16 GB of RAM use
`--memory 16g --memory-swap 16g`. For bit-exact output use `GLM53_MODE=nvme-exact`. The all-RAM `fast` mode uses an
older image; see [docs/results.md](docs/results.md).

## How it works

The model has 12,096 routed experts (42 MoE layers x 288), 9.4 MB each, and each token uses 8 per layer. Each expert
is kept in one of three tiers. Two lanes compute them at the same time:

```text
   GPU lane                                       CPU lane
  +-------------------------------+              +---------------------------+
  | VRAM expert cache, ~1,300     |              | 22 AVX2 threads compute   |
  | hottest experts (12.4 GB)     |              | RAM experts in place      |
  +---------------^---------------+              +-------------^-------------+
                  | PCIe copy when admitted                    | reads
  +---------------+--------------------------------------------+-------------+
  | RAM tier, pinned, sized from --memory: ~4,800 experts at 55 GiB          |
  | holds only experts that are NOT in VRAM                                  |
  +---------------^----------------------------------------------------------+
                  | O_DIRECT reads, deep queue, layer-ahead prefetch
  +---------------+----------------------------------------------------------+
  | NVMe store: every expert, 117 GB, one 4K-aligned record each             |
  +--------------------------------------------------------------------------+
```

- **Expert cache in VRAM.** The VRAM left after load holds the hottest experts. exllamav3's own kernels read them
  through repointed tables. On the all-RAM path, cache plus direct reads took decode from 7.5-8.4 tok/s (stock
  exllamav3) to 12.6.
- **RAM tier with no duplicates.** An expert evicted from VRAM moves into RAM, so VRAM and RAM never hold the same
  expert. 55 GiB of RAM then covers ~6,100 different experts instead of ~4,800. Per token at C1: ~120 picks hit
  VRAM, ~175 hit RAM and ~23 come from NVMe.
- **NVMe record store.** Each expert is one 4K-aligned record, read with `O_DIRECT` (no page cache) by 16 threads
  across a 4-drive RAID0. Capping reads at 8 GB/s cost 8 % of C1 decode, 33 % of C4 and 27 % of prefill.
- **CPU tier.** Cold experts already in RAM are computed by the CPU where they sit, instead of being copied over PCIe.
  This runs in parallel with the GPU, which adds the CPU's partial result back. Adding it, together with prefetch,
  took NVMe decode from 8.3 to 15.3 tok/s, at a decode KL of 0.005.
- **Host engine without stalls.** One controller thread plans each layer, issues the reads and answers the GPU
  through mapped memory. The GPU waits on per-expert "landed" flags, with no host sync per layer. Together with the
  no-duplicates RAM tier, this took exact decode from 7.3 to 8.3 tok/s.
- **Layer-ahead prefetch.** The next layer's router runs on the current layer's input, and its NVMe experts are read
  one layer early. NVMe misses fall from 40 to 22 per token, and it is 13 % faster in the same session. An earlier
  "prefetch off is faster" result was drift between sessions.
- **Victim ring.** Evicted experts are parked in a 24-slot VRAM ring. A copy engine drains them to RAM off the
  critical path (~7 GB/s of write-backs at C1).

**Where the time goes now** (C1, 62 ms per token): copying admitted experts to VRAM 28 %, non-MoE GPU work 18 %,
NVMe waits 16 %, waiting for the CPU lane 15 %, the MoE kernel 14 %, host gaps 8 %. No resource is saturated: each is
idle 35-75 % of the time, because every layer waits on its slowest lane. Better scheduling alone would reach about
20 tok/s. Every lane at its hardware floor would reach about 43. DRAM traffic caps it at about 52 unless fewer bytes
move per token. Details: [docs/how-it-works.md](docs/how-it-works.md).

## Tune it for your machine

Point your coding agent (Claude Code, Codex, others) at
[`skills/glm53-offload-setup/SKILL.md`](skills/glm53-offload-setup/SKILL.md). It checks your GPU, RAM, CPU and NVMe,
picks the mode, builds the store, measures, and tunes one setting at a time.

## More

- [docs/results.md](docs/results.md): every measured table, per mode, with dates and raw files
- [docs/how-it-works.md](docs/how-it-works.md): the mechanisms, how each tier is sized, and every setting
- [docs/troubleshooting.md](docs/troubleshooting.md): error messages and fixes
- [docs/reference.md](docs/reference.md): the full campaign record (image smokes, quality method, reproduction pins)

Credits: [exllamav3](https://github.com/turboderp-org/exllamav3) and the
[EXL3 checkpoint](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) by turboderp; FreeToken (FlashML) for the
host-memory expert tier idea; [SGLang](https://github.com/sgl-project/sglang) as the base image; GLM-5.3-Flash by
Z.ai. MIT license (this repository); the weights carry their own license.
