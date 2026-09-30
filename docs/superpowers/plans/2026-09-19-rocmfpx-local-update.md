# ROCmFPX local update implementation plan

Goal: integrate the qualified ROCmI4 fixes into Clean Upstream and merge current llama.cpp locally without losing quantization support.

Architecture: preserve existing authored ancestry and local edits; use separate commits for the recovered local baseline and the two GPU fixes. Qualify a true upstream merge in an isolated worktree before advancing the Clean checkout. Keep the installed working binary as fallback.

Constraints: local only; no push or PR. No GGUF ABI/type-ID/weight changes. Do not run model files from USB. Keep ComfyUI untouched. Hermes, OMP and the resident llama server remain stopped during qualification.

- [x] Preserve dirty patch, original head, launcher and working binary identity. Record existing edits as a separate baseline commit.
- [x] Apply the eight-line W4A4 scatter fix and remove the four-line ROCmI4 MMVQ bypass, exactly matching the previously qualified source. Commit separately with assistance disclosure.
- [x] Create /mnt/seconddrive/rocmfpx-clean-update-20260919 as an isolated integration worktree. Merge pinned origin/master with real ancestry, resolving only required integration conflicts.
- [x] Build normal HIP and W4A4 HIP profiles with gfx1151. Run existing CPU quant/reference tests and representative GPU matmul tests across ROCmFPX, ROCmFP4, ROCmI4 and standard quants. Validate ABI diffs and inspect compiler failures rather than removing quant support.
- [x] Run bounded sequential Flash Next coding, tool, retrieval and vision checks against the candidate, comparing with saved baseline results. Keep graphs off, CPU PLE, 256K context, and MTP3/ngram settings in the launch profile.
- [x] Independently review the integration diff; fix material defects and run the relevant checks. Only fast-forward Clean to the tested candidate if those gates pass. Preserve evidence and report any limits; leave requested services stopped.

Review focus: upstream changes to shared quant dispatch; quant type IDs and layouts; multi-token W4A4 packing; qwen4exp PLE/MTP/vision integration; build defaults versus the optional speed profile.

Completion: Clean was fast-forwarded to the true upstream merge 3db88caaa. Both build profiles compiled. Seven API/vision cases and three target/MTP control prompts passed; generated primality code passed 1,006 inputs. Quant formats stayed unchanged. GPU comparisons retained the pre-existing failures; these were not represented as passing correctness tests. The old serving binary remains selected because its matched prefill was about 4% faster. Services remain stopped.

Evidence and limitations: `/home/caf/llm-builds/rocmfpx-clean-update-20260919/`; report: `/home/caf/Downloads/ROCMFPX_CLEAN_LOCAL_UPDATE_20260919.md`.
