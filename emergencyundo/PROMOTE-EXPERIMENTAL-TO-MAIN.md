# Emergency Undo: Experimental-to-Main Promotion

## Protected recovery points

- Old GitHub `main`: `5b3956605309dd3e6beed49c8f3a41423ba71d25`
- Tested experimental source:
  `a6a93765f7ce9779c13f9881164a65f7a9f31198`
- Remote archive branch:
  `archive/main-before-experimental-2026-07-11`
- Local full backup: `/home/caf/ROCmFPXMAIN`
- Local complete-history bundle:
  `/home/caf/ROCmFPXMAIN/ROCmFPX-main-2026-07-11.bundle`

## Pre-promotion regression fixes and validation

The first promotion candidate was withheld after the Qwen coherency fixture
exposed GPU corruption. The candidate was repaired before publication:

- ROCm restores the missing `mmq.cuh` tile-stride portion of ciru-ai's
  `ac9cdf3ba` fix.
- Vulkan converts packed ROCmFPX Q6 data during asynchronous uploads,
  including 1 MiB loader chunks that split a 26-byte quantization block.

The repaired tree was then rebuilt from an empty build directory and passed:

- Qwen coding, summary, and strict-JSON coherency probes on ROCm and Vulkan.
- Focused ROCmFPX matrix suites: ROCm 82/82 and Vulkan 52/52.
- CPU release suite: 2,156/2,156.
- ROCmFPX reference, ranked-policy, and model-architecture checks.
- ROCm and Vulkan copy gates, plus ROCm FlashAttention performance gates.
- Real-model Q6 Vulkan, ROCm MoE, and MTP smoke tests.

## Draft PR #27 CI repair validation

The promotion remains a draft while GitHub runs the full platform matrix.
Before the repaired branch was pushed, it also passed these follow-up gates:

- PR #25's WebUI repair and PR #26's ROCm 7.2 negative-infinity repair were
  integrated as their original commits, without squashing their authorship.
- A clean CPU build with `LLAMA_FATAL_WARNINGS=ON` completed successfully.
- The production WebUI build completed successfully with its locked npm
  dependencies.
- Python 3.11 scoped lint, type checking, converter syntax, and the full
  isolated `check-requirements.sh` import suite passed.
- The combined HIP and Vulkan build completed successfully, including the
  ROCm 7.2-affected softmax, top-k MoE, cross-entropy, and DeepSeek kernels.
- The mixed WebGPU snapshot was restored to the coherent upstream tree present
  at `5fd2dc2c`; the exact CI Dawn package compiled and linked
  `test-backend-ops` successfully.
- Untouched upstream reproduced 48 quantized FlashAttention failures for the
  tile path on this RADV GPU. The candidate now keeps the verified vector path
  and conservatively falls back to CPU for unverified quantized tile layouts
  and Q8 set-rows parity edge cases.
- Native WebGPU validation then passed 913/913 matrix-multiply cases and
  2,803/2,803 copy, set-rows, and FlashAttention cases on the Radeon 8060S.
- The Snapdragon/Hexagon source snapshot's omitted `htp-opnode.h` was restored
  byte-for-byte from its matching upstream revision.
- `test-llama-archs` passed across ROCm, Vulkan, CPU, and Meta backends.
- The CPU ROCmFPX gate passed 2,156/2,156 tests.
- A real 35B MoE ROCm benchmark completed at 67.15 generation tokens/second.
- Real Qwable MTP decode completed cleanly at 33.8 tokens/second on ROCm and
  32.9 tokens/second on Vulkan.

The next GitHub matrix exposed several source directories that had been copied
from mismatched upstream revisions. The follow-up candidate therefore also:

- Restores the complete Hexagon and OpenCL trees from upstream snapshot
  `5fd2dc2c`. Every file hash and both directory file lists were checked
  locally before one extra blank line at EOF was normalized; the new files
  shown by Git are intentional parts of those trees.
- Completes the matching upstream chat/PEG parser changes for whitespace,
  streaming, and LFM2/LFM2.5 tool parsing, and skips unsupported Responses API
  tools that have no Chat Completions equivalent.
- Restores the optional multimodal GQA default `n_head_kv = n_head`, fixing the
  TinyGemma vision-server startup assertion.
- Uses CPU fallback for non-contiguous WebGPU normalization views that fail
  parity on Apple's WebGPU/Metal implementation.
- Removes two CUDA const-correctness warnings that are errors in the CUDA CI
  configuration.
- Uses the Windows CRT environment API for the HIP queue workaround and routes
  the MTP speculative-step setter through the exported staging API, avoiding
  the two Windows compile/link failures from the previous matrix.
- Adds exact HIP-quality exceptions for three pre-existing main-branch kernels
  and eight gfx908-only ROCmFPX FP3/FP6 FlashAttention instantiations. The
  quality gate remains active for every other symbol; no wildcard exemption is
  used.

Validation of this follow-up tree includes:

- A fresh Release CPU build with `LLAMA_FATAL_WARNINGS=ON`, including
  `test-chat`, `test-llama-archs`, `test-backend-ops`, and `llama-server`.
- The complete Release and Debug chat suites, including the restored LFM and
  Responses cases.
- The architecture round-trip suite.
- TinyGemma vision-server startup with the same model and projector used by
  CI, returning a healthy status.
- WebGPU NORM, L2_NORM, and RMS_NORM correctness: 61/61 supported cases.
- Same-toolchain gfx908 compile-metrics builds of backed-up `main` and the
  candidate, followed by validation of the exact ROCm 7.2.1 CI exception list.
- A rebuilt Vulkan MTP smoke test using the exported speculative-step API,
  completing cleanly at 42.6 generation tokens/second.

This machine does not have OpenCL development headers, and it cannot reproduce
the Snapdragon/Hexagon toolchain. The exact-tree restorations must therefore
pass their GitHub platform jobs before the PR can leave draft status. Do not
treat hash verification alone as a successful backend build.

Do not merge draft PR #27 while any required GitHub check is pending or red.

## Important rule

Do not reset or force-push the shared `main` branch. Undo the promotion with a
new reviewed revert pull request so history and author credit remain intact.

## Find the completed promotion

```bash
PROMOTION_PR=$(gh pr list \
  --repo charlie12345/ROCmFPX \
  --state merged \
  --head agent/promote-experimental-to-main-2026-07-11 \
  --json number \
  --jq '.[0].number')

MERGE_SHA=$(gh pr view "$PROMOTION_PR" \
  --repo charlie12345/ROCmFPX \
  --json mergeCommit \
  --jq '.mergeCommit.oid')

printf 'promotion PR=%s merge=%s\n' "$PROMOTION_PR" "$MERGE_SHA"
```

Confirm that both values are non-empty before continuing.

## Create the rollback pull request

```bash
git clone https://github.com/charlie12345/ROCmFPX.git \
  /home/caf/ROCmFPX-PROMOTION-ROLLBACK
cd /home/caf/ROCmFPX-PROMOTION-ROLLBACK
git switch main
git pull --ff-only origin main
git switch -c emergency/revert-experimental-promotion

# The promotion is merged with GitHub's merge-commit method. Parent 1 is the
# previous main line, so -m 1 restores the pre-promotion tree.
git revert -m 1 "$MERGE_SHA"

scripts/check-rocmfpx-reference.sh
scripts/check-rocmfpx-ranked-policy.sh
cmake -S . -B build-rollback -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF
cmake --build build-rollback --target test-backend-ops -j 2
./build-rollback/bin/test-backend-ops test \
  -o MUL_MAT,GET_ROWS,CPY,SET_ROWS -b CPU

git push -u origin emergency/revert-experimental-promotion
gh pr create \
  --repo charlie12345/ROCmFPX \
  --base main \
  --head emergency/revert-experimental-promotion \
  --title "revert: experimental-to-main promotion" \
  --body "Reverts the promotion merge after local validation."
```

Wait for GitHub checks and merge that rollback PR using **Create a merge
commit**. The remote archive and local bundle are recovery references, not a
reason to rewrite `main`.
