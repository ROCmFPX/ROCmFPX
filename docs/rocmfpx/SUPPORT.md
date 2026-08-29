# AMD support tiers

ROCmFPX keeps normal llama.cpp Vulkan support enabled. Vulkan is the portable
default; HIP is an additional backend, and the special ROCmI4/W4A4 build is a
separate opt-in profile.

| GPU generation | Examples | Tier | Qualification expectation |
| --- | --- | --- | --- |
| RDNA 1 | gfx101x | compatible | CPU reference plus Vulkan smoke |
| RDNA 2 | gfx103x | supported | CPU, Vulkan, HIP build/runtime |
| RDNA 3 | gfx110x | supported | CPU, Vulkan, HIP build/runtime |
| RDNA 3.5 | gfx115x | primary | full correctness, Qwen, MTP, throughput |
| RDNA 4 | gfx120x | supported | CPU, Vulkan, HIP build/runtime |
| future RDNA | unknown | provisional | no claim until identified and tested |

“Compatible” is best-effort and not a promise that every ROCm release exposes a
HIP target for that device. Release notes must name the OS, driver, ROCm/Vulkan
versions, exact GPU, commit, model hash, and test matrix. Backend fallbacks must
be visible in logs; a fallback result must not be reported as a native-kernel
benchmark.
