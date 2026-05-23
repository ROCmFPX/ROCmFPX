# ROCmFP4 + MTP Comparison

Date: 2026-05-23
Hardware: Framework AMD Strix Halo 395+ desktop, 128 GB unified RAM
Merged tree: `llama.cpp-mtp-rocmfp4`
Branch: `mtp-rocmfp4-strix`
Build: `b9174-22c4f7450`

## What Was Tested

This compares Qwen3.6 27B MTP against Qwen3.6 27B MTP. It does not compare Qwen to Gemma.

The same merged binary was used for both model files:

`/home/caf/strix-fp4/llama.cpp-mtp-rocmfp4/build-strix-rocmfp4/bin/llama-cli`

Models:

- ROCmFP4: `/home/caf/strix-fp4/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-MTP-BF16-to-ROCmFP4-STRIX_LEAN.gguf`
- Baseline Q5: `/home/caf/llm-builds/models/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf`

Shared settings:

- Context window: `262144`
- Reasoning: off
- Tools: off
- MTP: on, `draft-mtp`
- KV cache: `q4_0` for K and V
- Prompt tokens/s and generation tokens/s were reported by `llama-cli -st`

Shared flags:

```bash
-ngl 999 -c 262144 -b 512 -ub 512 -fa on \
-ctk q4_0 -ctv q4_0 --no-mmap --jinja -cnv -st \
--reasoning off \
--spec-type draft-mtp \
--spec-draft-ngl all \
--spec-draft-type-k q4_0 \
--spec-draft-type-v q4_0 \
--spec-draft-n-max 3 \
--spec-draft-n-min 0 \
--spec-draft-p-min 0.0 \
--seed 123 --temp 0.2 --top-k 20 --top-p 0.9 \
--no-display-prompt -n 128
```

Device-specific flags:

- ROCm: `-dev ROCm0 --spec-draft-device ROCm0`
- Vulkan: `-dev Vulkan0 --spec-draft-device Vulkan0`

## Results

| Model | Backend | Context | MTP | Reasoning | Tools | Prompt tok/s | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B-MTP ROCmFP4 STRIX_LEAN | ROCm0 | 262144 | on | off | off | 87.3 | 23.7 |
| Qwen3.6-27B-MTP UD-Q5_K_XL | ROCm0 | 262144 | on | off | off | 34.1 | 15.7 |
| Qwen3.6-27B-MTP ROCmFP4 STRIX_LEAN | Vulkan0 | 262144 | on | off | off | 77.1 | 22.8 |
| Qwen3.6-27B-MTP UD-Q5_K_XL | Vulkan0 | 262144 | on | off | off | 81.6 | 16.3 |

## Notes

On these controlled 262k-context runs with the same merged binary, ROCmFP4 was faster than the Q5 baseline:

- ROCm0: `23.7 tok/s` vs `15.7 tok/s`, about 51% faster decode.
- Vulkan0: `22.8 tok/s` vs `16.3 tok/s`, about 40% faster decode.

After each run, `rocm-smi --showpids` showed no KFD processes running, so VRAM was released.
