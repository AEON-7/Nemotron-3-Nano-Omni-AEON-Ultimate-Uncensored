#!/usr/bin/env python3
"""Build-time assertion that NemotronH_Nano_Omni_Reasoning_V3 is registered in vLLM.

vLLM v0.20.0 (PR #39747, merged 2026-04-15) aliased the NemotronH_Nano_Omni_Reasoning_V3
and NemotronH_Super_Omni_Reasoning_V3 architectures to the existing NemotronH_Nano_VL_V2
class in vllm/model_executor/models/nano_nemotron_vl.py. That class already wires up:
  - vision_model.* (RADIO encoder via vllm/model_executor/models/radio.py)
  - sound_encoder.* + sound_projection.* (Parakeet via vllm/model_executor/models/parakeet.py)
  - language_model.backbone.* (NemotronH hybrid Mamba2+Attention+MoE)
  - mlp1.* (vision projection)

This patch is a fail-fast verification at image build time so we don't ship a broken image.
If vLLM ever removes the alias, the build halts loudly instead of producing a non-functional image.
"""
import sys
from importlib import import_module


def main() -> int:
    # 1) Verify the registry has both omni archs aliased to the VL class
    reg = import_module("vllm.model_executor.models.registry")
    table = reg._MULTIMODAL_MODELS  # type: ignore[attr-defined]

    expected = {
        "NemotronH_Nano_Omni_Reasoning_V3":  ("nano_nemotron_vl", "NemotronH_Nano_VL_V2"),
        "NemotronH_Super_Omni_Reasoning_V3": ("nano_nemotron_vl", "NemotronH_Nano_VL_V2"),
    }
    missing = [k for k in expected if k not in table]
    if missing:
        print(f"[verify] FAIL: missing arch entries: {missing}")
        print(f"[verify]   PR #39747 may have been reverted upstream.")
        print(f"[verify]   Manual fix: edit registry.py and add the entries above.")
        return 1

    for arch, expected_pair in expected.items():
        actual_pair = table[arch]
        # vLLM's table values are sometimes wrapped in dataclasses; coerce to tuple
        if hasattr(actual_pair, "module_name") and hasattr(actual_pair, "class_name"):
            actual_pair_tuple = (actual_pair.module_name, actual_pair.class_name)
        else:
            actual_pair_tuple = tuple(actual_pair)
        if actual_pair_tuple != expected_pair:
            print(f"[verify] WARN: {arch} → {actual_pair_tuple} (expected {expected_pair})")
            print(f"[verify]   non-standard alias — probably still works but flagging")

    print(f"[verify] OK: NemotronH_Nano_Omni_Reasoning_V3 registered → "
          f"{table['NemotronH_Nano_Omni_Reasoning_V3']}")

    # 2) Verify the underlying VL class can be imported (catches install corruption)
    try:
        nano_vl = import_module("vllm.model_executor.models.nano_nemotron_vl")
        cls = getattr(nano_vl, "NemotronH_Nano_VL_V2")
        print(f"[verify] OK: {cls.__module__}.{cls.__name__}")
    except (ImportError, AttributeError) as e:
        print(f"[verify] FAIL: cannot import NemotronH_Nano_VL_V2: {e}")
        return 1

    # 3) Verify the vision + audio sub-modules are importable
    try:
        radio = import_module("vllm.model_executor.models.radio")
        getattr(radio, "RadioModel")
        print(f"[verify] OK: vllm.model_executor.models.radio.RadioModel")
    except (ImportError, AttributeError) as e:
        print(f"[verify] FAIL: cannot import RadioModel: {e}")
        return 1

    try:
        parakeet = import_module("vllm.model_executor.models.parakeet")
        getattr(parakeet, "ProjectedParakeet")
        getattr(parakeet, "ParakeetExtractor")
        print(f"[verify] OK: vllm.model_executor.models.parakeet.{{ProjectedParakeet,ParakeetExtractor}}")
    except (ImportError, AttributeError) as e:
        print(f"[verify] FAIL: cannot import Parakeet classes: {e}")
        return 1

    print("[verify] all Nemotron-Omni components present and registered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
