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
- `test-llama-archs` passed across ROCm, Vulkan, CPU, and Meta backends.
- The CPU ROCmFPX gate passed 2,156/2,156 tests.
- A real 35B MoE ROCm benchmark completed at 67.15 generation tokens/second.
- Real Qwable MTP decode completed cleanly at 33.8 tokens/second on ROCm and
  32.9 tokens/second on Vulkan.

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
