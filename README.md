# Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored — vLLM container + deployment guide

[![Image](https://img.shields.io/badge/ghcr.io-aeon--7%2Fvllm--nemotron--omni--aeon--ultimate-blue)](https://ghcr.io/aeon-7/vllm-nemotron-omni-aeon-ultimate)
[![Model BF16](https://img.shields.io/badge/HuggingFace-BF16-yellow)](https://huggingface.co/AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-BF16)
[![Model NVFP4](https://img.shields.io/badge/HuggingFace-NVFP4-yellow)](https://huggingface.co/AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-NVFP4)
[![License](https://img.shields.io/badge/License-NVIDIA_Open_Model_Agreement-green)](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/)

A purpose-built **vLLM** container image and deployment guide for [`AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored`](https://huggingface.co/AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-NVFP4) on **NVIDIA DGX Spark** (GB10 / sm_121a / 128 GB unified memory).

> ⚠️ **READ THE REQUIREMENTS SECTION FIRST.** This image is purpose-built for the DGX Spark (GB10 / sm_120-121 Blackwell) with PyTorch nightly cu130. It will *boot* on other Blackwell variants (B100/B200/RTX Pro 6000) since the build uses `12.0+PTX`, but the sm_121a-specific patches are no-ops there. Hopper (H100/H200) and Ampere (A100) are unsupported.

| | |
|---|---|
| **Model** | `AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-NVFP4` (~22 GB, multimodal preserved BF16) |
| **Base** | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| **Hardware** | DGX Spark (NVIDIA GB10, 128 GB unified memory, sm_121a) |
| **Image** | `ghcr.io/aeon-7/vllm-nemotron-omni-aeon-ultimate:v1` (~9 GB compressed) |

---

## Headline performance (measured)

DGX Spark, NVFP4 + vLLM, `--max-num-seqs 8`, 32k context, vLLM streaming endpoint. See [`bench/`](bench/) for scripts.

### CUTLASS NVFP4 MoE backend (default — lower TTFT)

| Workload | TTFT | Single decode | C=4 aggregate | C=8 aggregate | C=8 median TTFT |
|---|---|---|---|---|---|
| `thinking=False` (max_new=400) | **68 ms** | 71.7 c/s | 163.1 c/s | **223.5 c/s** | 206 ms |
| `thinking=True` (max_new=600) | **62 ms** | 71.2 c/s | 175.5 c/s | **242.6 c/s** | 164 ms |

### MARLIN NVFP4 MoE backend (opt-in via `VLLM_TEST_FORCE_FP8_MARLIN=1`, ~5–10% more throughput)

| Workload | TTFT | Single decode | C=4 aggregate | C=8 aggregate | C=8 median TTFT |
|---|---|---|---|---|---|
| `thinking=False` (max_new=400) | 64 ms | 73.0 c/s | 162.4 c/s | **244.8 c/s** | 174 ms |
| `thinking=True` (max_new=600) | 60 ms | 72.8 c/s | 182.4 c/s | **259.7 c/s** | 158 ms |

(c/s = streaming chunks per second, ≈ tokens per second at this output style)

Linear ops always use `FlashInferCutlassNvFp4LinearKernel` regardless. Only the **MoE backend** differs between the two configurations. CUTLASS MoE wins on TTFT (especially at C=4 thinking=True: 90 ms vs 124 ms). MARLIN MoE wins on aggregate throughput at C=8.

### Refusal-rate verification (100-prompt random sample, NVFP4 + vLLM)

| Mode | Refusal Rate | Empty (length artifact) | Real Refusal | vs base model* |
|---|---|---|---|---|
| `enable_thinking=True` (max_new=600) | 18/100 (18%) | **18/100** (length-truncated inside `<think>`) | **0%** real | base ~95-100% refusal |
| `enable_thinking=False` (max_new=200) | 15/100 (15%) | 0 | **15%** | base ~95-100% refusal |

*The 18 length-truncations on `thinking=True` are length artifacts (max_tokens=600 ran out before `</think>` closed), not real refusals — bump `max_tokens` ≥ 1200 for reasoning workloads.

\* Hard baseline measurements against the unmodified `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` are pending — see [bench results](https://huggingface.co/AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-BF16) on the model card for the most up-to-date figures.

---

## Quick start (5 commands)

```bash
# 1. Pre-flight check — confirm anonymous pull works
docker pull ghcr.io/aeon-7/vllm-nemotron-omni-aeon-ultimate:v1

# 2. Set HF token + ensure cache dir exists
export HF_TOKEN=hf_xxx
mkdir -p ~/.cache/huggingface

# 3. Get the compose file
curl -fsSL https://raw.githubusercontent.com/AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored/main/examples/docker-compose.yml \
  -o docker-compose.yml

# 4. Start the server (4-7 min to first "Application startup complete")
docker compose up -d
docker compose logs -f

# 5. Smoke test (use temperature=0 for greedy, max_tokens >= 1200 for reasoning mode)
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-NVFP4",
    "messages": [{"role":"user","content":"What is 47 * 83? Show your reasoning."}],
    "max_tokens": 1200, "temperature": 0
  }' | python3 -m json.tool
```

For the full step-by-step (with pre-flight + post-deploy verification + tuning knobs + troubleshooting), see [`docs/dgx-spark-setup.md`](docs/dgx-spark-setup.md).

---

## Hardware support

| Hardware | Status | Notes |
|---|---|---|
| **DGX Spark / GB10 (sm_121a, unified mem)** | ✅ Tier 1 — image is purpose-built for this | CUTLASS NVFP4 Linear + sm_121a kernel patches + Parakeet/RADIO native |
| RTX PRO 6000 SE / RTX 5090 / B100 / B200 | ✅ Works | Sm_120 Blackwell. Stock vLLM v0.20.0+ also works directly. |
| H200 / H100 | ⚠️ NVFP4 fallback | Hopper has FP8 but no NVFP4 native support. Use the BF16 sibling for these GPUs. |
| Ampere (A100) | ❌ Unsupported | No FP4. Use BF16 sibling. |

---

## What this image actually is

vLLM v0.20.0 release source-built for **CUDA 13 / sm_120 + PTX** with:

| # | Component | Purpose |
|---:|---|---|
| 1 | **vLLM v0.20.0** (commit `88d34c6409`, 2026-04-27) | Native multimodal model class for `NemotronH_Nano_Omni_Reasoning_V3` via `nano_nemotron_vl.NemotronH_Nano_VL_V2` (PR #39747, merged 2026-04-15). Wires up RADIO vision tower + Parakeet audio encoder + NemotronH hybrid LM. |
| 2 | **FlashInfer v0.6.9 stable** (2026-04-24) | b12x SM121 NVFP4 GEMM backend (`FlashInferCutlassNvFp4LinearKernel`) for Linear ops. |
| 3 | **TurboQuant** (AEON-7 fork, `fix/cuda-graph-safe-qjl-powers`) | KV-cache compression with CUDA-graph-safe `_POWERS` cache. |
| 4 | **transformers 5.7.0** | Required for Nemotron-Omni's omni-wrapper Python imports (`merge_with_config_defaults`, `output_capturing`, etc. — 5.x APIs). |
| 5 | **librosa 0.11 + soundfile 0.13** | Parakeet audio feature extraction (16 kHz mel). |

Plus 4 idempotent patches:

| # | Patch | What it does |
|---:|---|---|
| 1 | [`patches/patch_kv_cache_utils.py`](patches/patch_kv_cache_utils.py) | NemotronH has 30 Mamba layers among 52 total. Hybrid linear-attention groups can expose `block_size=None` to `min(group.block_size for group in groups)` and crash. Patches 3 vLLM sites + a defensive Mamba-abstract one to filter None before min(). |
| 2 | [`patches/patch_cuda_optional_import.py`](patches/patch_cuda_optional_import.py) | Wraps `import vllm._C_stable_libtorch` in `RTLD_LAZY`. Workaround for sm_121a builds missing SM100-only `mxfp4_experts_quant` symbols. No-op on B100/B200/RTX. |
| 3 | [`patches/patch_cudagraph_align.py`](patches/patch_cudagraph_align.py) | Defensive spec-decode capture-size alignment for PIECEWISE cudagraph mode. |
| 4 | [`patches/verify_nemotron_omni_registered.py`](patches/verify_nemotron_omni_registered.py) | Build-time assertion — `NemotronH_Nano_Omni_Reasoning_V3` arch is aliased to the multimodal class, and `RadioModel` + `ProjectedParakeet` import cleanly. **Halts the build loudly if upstream regresses.** |

Defaults baked into v1:
- `TORCH_CUDA_ARCH_LIST="12.0+PTX"` — sm_120 build, JITs to sm_121a on Spark, sm_120 on RTX/B100/B200
- `ENABLE_NVFP4_SM100=0` — required by vLLM PR #40191 for sm_121a-only builds (avoids SM100-only symbol references at build time)
- `VLLM_TEST_FORCE_FP8_MARLIN=1` — **set in v1**; the image's MoE default is MARLIN. Override with `-e VLLM_TEST_FORCE_FP8_MARLIN=0` at `docker run` time to flip to CUTLASS MoE (lower TTFT). A v1.1 image with CUTLASS-default-out-of-the-box is forthcoming.

The [`Dockerfile.v1`](Dockerfile.v1) is reproducible. Build time on a warm ccache: 25–50 min on Spark.

---

## Files

```
.
├── README.md                  ← this file
├── Dockerfile.v1              ← v1 (MARLIN default, override env var to flip)
├── Dockerfile.v1.1            ← v1.1 (CUTLASS default — pending validation)
├── patches/
│   ├── patch_kv_cache_utils.py
│   ├── patch_cuda_optional_import.py
│   ├── patch_cudagraph_align.py
│   └── verify_nemotron_omni_registered.py
├── examples/
│   └── docker-compose.yml     ← one-shot Spark deployment
├── docs/
│   └── dgx-spark-setup.md     ← step-by-step operator runbook
├── bench/
│   ├── bench_nemotron_omni.py ← throughput + TTFT
│   └── refusal_bench_nvfp4.py ← 100-prompt refusal scan via vLLM
└── LICENSE
```

---

## Acknowledgments

- **NVIDIA** — base model `Nemotron-3-Nano-Omni-30B-A3B-Reasoning`, vLLM model class `nano_nemotron_vl.NemotronH_Nano_VL_V2` (PR #39747 by @tomeras91), `nvidia-modelopt` quantization framework.
- **vLLM team** — multimodal infra, NemotronH support.
- **FlashInfer team** — b12x SM121 NVFP4 GEMM backend.
- **0xSero (TurboQuant)** + the AEON-7 fork — KV cache compression.
- **Andy Arditi et al.** ([arXiv:2406.11717](https://arxiv.org/abs/2406.11717)) — "Refusal in Language Models Is Mediated by a Single Direction" — the conceptual foundation of all post-hoc abliteration work.
- **FailSpy / Maxime Labonne (mlabonne) / Sumandora** — public abliteration tooling and prompt sets we built on. Full method credits live on the [BF16 model card](https://huggingface.co/AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-BF16#acknowledgments-and-prior-art).

---

## License

Use of this model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/) (inherited from the base model). The container image, patches, and deployment scripts in this repo are released under [MIT](LICENSE).

The model itself is uncensored — see the [BF16 model card](https://huggingface.co/AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-BF16#user-responsibility--arbitration-clause) for the full User Responsibility & Arbitration Clause that governs your use.
