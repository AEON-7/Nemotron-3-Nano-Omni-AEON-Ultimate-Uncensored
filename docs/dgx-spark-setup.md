# DGX Spark deployment guide — Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored

Step-by-step guide for serving the AEON-7 abliterated Nemotron-3-Nano-Omni model on a DGX Spark (GB10 / sm_121a / 128 GB unified memory).

The bench numbers and method writeup live on the [HF model cards](https://huggingface.co/AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-BF16). This document is the operator's runbook.

---

## 0. Pre-flight (one-time host setup)

| Component | Required version | Check command |
|---|---|---|
| GPU | NVIDIA GB10 (DGX Spark) | `nvidia-smi --query-gpu=name --format=csv` |
| NVIDIA driver | ≥ 580.x | `nvidia-smi` (header line) |
| CUDA host runtime | 13.0+ | `nvcc --version` (or rely on the container's bundled CUDA 13.2) |
| Docker | ≥ 25.x with `nvidia-container-toolkit` | `docker info \| grep -i runtime` |
| Free disk | ≥ 60 GB (for image + NVFP4 weights) or ≥ 100 GB (for BF16) | `df -h $HOME` |

If `docker info` doesn't show `Runtimes: ... nvidia ...`, install the toolkit:
```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Authenticate to HuggingFace so the container can pull the (currently public) model weights:
```bash
hf auth login   # paste your hf_*** token
```

---

## 1. Pull the image

```bash
docker pull ghcr.io/aeon-7/vllm-nemotron-omni-aeon-ultimate:v1
# Anonymous pull is supported — no ghcr login required.
```

The image is ~9 GB compressed (~33 GB uncompressed) and contains:
- vLLM v0.20.0 release commit (`88d34c6409`, 2026-04-27)
- FlashInfer v0.6.9 stable (b12x SM121 NVFP4 GEMM backend)
- TurboQuant (AEON-7 fork with CUDA-graph-safe fix)
- transformers 5.7.0, librosa 0.11.0, soundfile 0.13.1, scipy 1.11+
- 3 sm_121a hybrid-arch patches (kv_cache_utils None-safe filter, RTLD_LAZY for sm_121a optional-import, defensive cudagraph alignment)
- Build-time verification that `NemotronH_Nano_Omni_Reasoning_V3` is wired through to `nano_nemotron_vl.NemotronH_Nano_VL_V2` (multimodal: RADIO vision + Parakeet audio)

---

## 2. Choose your variant + serve

### Recommended path: NVFP4 (3× compression, full multimodal)

```bash
docker run -d --rm --name nemotron-omni-aeon \
  --gpus all --shm-size=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_TEST_FORCE_FP8_MARLIN=0 \
  -p 8000:8000 \
  ghcr.io/aeon-7/vllm-nemotron-omni-aeon-ultimate:v1 \
  vllm serve AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-NVFP4 \
    --trust-remote-code \
    --max-model-len 200000 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16
```

Expect:
- Cold start: 4–7 minutes (HF download ~22 GB → load → compile cudagraphs)
- GPU memory after warmup: ~22 GiB
- Available headroom for KV cache + concurrent sequences: ~55 GiB

### BF16 variant (full precision, 3× larger)

```bash
docker run -d --rm --name nemotron-omni-aeon \
  --gpus all --shm-size=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -p 8000:8000 \
  ghcr.io/aeon-7/vllm-nemotron-omni-aeon-ultimate:v1 \
  vllm serve AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-BF16 \
    --trust-remote-code \
    --max-model-len 200000 \
    --gpu-memory-utilization 0.85
```

Expect:
- Cold start: 8–15 minutes (HF download ~66 GB)
- GPU memory after warmup: ~64 GiB
- Available headroom: ~25 GiB

### docker-compose alternative

```bash
curl -fsSL https://raw.githubusercontent.com/AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored/main/examples/docker-compose.yml \
  -o docker-compose.yml
docker compose up -d
docker compose logs -f
```

---

## 3. Verify it's up

```bash
# Wait for "Application startup complete" in the logs
docker logs -f nemotron-omni-aeon | grep -m1 "Application startup complete"

# Health probe
curl -fsS http://localhost:8000/health && echo " ok"

# Models endpoint
curl -fsS http://localhost:8000/v1/models | python3 -m json.tool
```

---

## 4. Smoke tests

### Text + reasoning

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-NVFP4",
    "messages": [{"role":"user","content":"What is 47 * 83? Show your reasoning."}],
    "max_tokens": 1200,
    "temperature": 0
  }' | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

> Use `max_tokens >= 1200` for reasoning-mode workloads. The model thinks inside `<think>...</think>` and the final answer follows. Lower budgets can length-truncate the reasoning trace and produce empty post-`</think>` answers.

### Vision (image-in)

```bash
B64=$(base64 -w0 /path/to/your/image.png)
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-NVFP4\",
    \"messages\": [{
      \"role\":\"user\",
      \"content\": [
        {\"type\":\"text\",\"text\":\"Describe this image briefly.\"},
        {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${B64}\"}}
      ]
    }],
    \"max_tokens\": 200, \"temperature\": 0
  }" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

### Audio (Parakeet path)

vLLM accepts `audio_url` chat content blocks for the Parakeet encoder. See the upstream `nano_nemotron_vl` test suite for end-to-end examples.

---

## 5. Performance expectations (DGX Spark, NVFP4)

Single-stream / concurrent throughput (vLLM streaming, 32k context, max-num-seqs=8):

| Mode | TTFT | Single decode | C=4 aggregate | C=8 aggregate |
|---|---|---|---|---|
| **CUTLASS MoE** (default) | 62–68 ms | 71.7 c/s | 163 c/s | 224 c/s |
| **MARLIN MoE** (`VLLM_TEST_FORCE_FP8_MARLIN=1`) | 60–64 ms | 73.0 c/s | 162 c/s | 245 c/s |

(c/s = streaming chunks per second, ≈ tokens per second at this output style)

**Rule of thumb:** CUTLASS for interactive (lower TTFT, especially at moderate concurrency). MARLIN for batch-heavy throughput (~7–10% more aggregate at C=8). Both are CUTLASS for Linear ops — only the MoE backend differs.

---

## 6. Tuning knobs

| Flag | Default in v1 | Effect |
|---|---|---|
| `--max-model-len` | 32 768 (here 200 000) | Per-sequence context budget. NemotronH supports 256k upstream — 200k leaves headroom for the KV cache slack. |
| `--gpu-memory-utilization` | 0.85 | Higher = more KV cache. Spark unified memory **caps cleanly at 0.88**; 0.90+ thrashes. |
| `--max-num-seqs` | 16 | Concurrent sequences in the engine. 8–16 is the sweet spot for chat UX; 32+ for pure throughput at the cost of TTFT. |
| `--enable-prefix-caching` | off | Off in our default to avoid surprises during validation. Enable for repeated-prefix workloads (RAG, agents) for big TTFT gains. |
| `-e VLLM_TEST_FORCE_FP8_MARLIN` | `0` (CUTLASS) | Set `1` to flip MoE to MARLIN backend. See Performance section. |
| `-e HF_TOKEN` | required if model is private | Public weights need no token; we currently keep these public. |

---

## 7. Troubleshooting

### `Application startup complete` never prints
Most common cause: HF download is slow on first run. The 22 GB NVFP4 (or 66 GB BF16) is downloaded into the mounted `~/.cache/huggingface`. Watch logs for `Loading safetensors checkpoint shards`. If it stalls before that, check `HF_TOKEN` and network egress.

### `Detected ModelOpt NVFP4 checkpoint. Please note that the format is experimental and could change in future.`
This is a vLLM upstream warning, not an error. Safe to ignore.

### `Your GPU does not have native support for FP4 computation` (when MARLIN forced)
Cosmetic. vLLM's compute-capability check doesn't recognize sm_121a; Marlin's "weight-only FP4" path runs fine on Spark. Linear ops use FlashInferCutlass which is the native path.

### Empty responses after `</think>` in reasoning mode
The model exhausted `max_tokens` while still thinking. Bump `max_tokens` to 1200+ for reasoning workloads. (The 100-prompt refusal bench at `max_tokens=600` shows ~18% of `thinking=True` answers run over budget — these are not refusals, just truncations.)

### Out-of-memory on cold start
Spark has 128 GB unified memory. NVFP4 needs ~22 GiB GPU resident + ~50 GiB peak during compile/cudagraph capture. If you've co-located other GPU workloads, stop them. `--gpu-memory-utilization 0.78` is a safer floor while debugging.

### `cudaErrorIllegalAddress` or process crash
Check `dmesg | grep -i nvidia` for ECC or driver events. On Spark, this almost always means another process spiked GPU memory. Restart docker.

---

## 8. Public model URLs

- BF16: https://huggingface.co/AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-BF16
- NVFP4: https://huggingface.co/AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-NVFP4
- Container: https://github.com/AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored (this repo)
- Image: https://ghcr.io/aeon-7/vllm-nemotron-omni-aeon-ultimate
