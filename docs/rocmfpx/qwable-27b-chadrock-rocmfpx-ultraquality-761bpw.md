# Qwable 27B Chadrock-ROCmFPX-ULTRAQUALITY-7.61BPW

This lane identifies the Qwable 27B Chadrock ROCmFPX high-quality restore profile
validated as the leave-16 ranked-attention policy with the Strix `mmid5`
runtime profile.

## Lane

- Public name: `Qwable 27B Chadrock-ROCmFPX-ULTRAQUALITY-7.61BPW`
- Policy: ranked attention leave-16
- Runtime profile: `rocmfpx-strix-mmid5`
- GGUF artifact basename:
  `Qwen3.6-27B-MTP-Q6ONLY_ROCMFPX_FFNALL_ATTNRANKLEAVE16_Q6K_SPLICE.gguf`
- Measured tensor BPW: `7.613087`
- Measured file BPW: `7.616306`

The `7.61BPW` suffix is the release-lane class name. The exact measured BPW
values above are from the GGUF tensor inventory.

## Reproduction Mapping

Generate the tensor policy with the ranked policy helper:

```bash
python3 scripts/rocmfpx-ranked-policy.py \
  --rank-csv attention-rank.csv \
  --leave-count 16 \
  --base-type q6_0_rocmfpx \
  --restore-type q6_k \
  --output tensor-type.rankleave16.txt
```

Build or run with the Strix `mmid5` ROCmFPX decode profile:

```bash
scripts/rocmfp4-decode-tune-flags.sh rocmfpx-strix-mmid5
```

## Speed Evidence

Stored served-MTP speed rows:

| row | profile | prompt tokens | generated tokens | TG tok/s | PP tok/s | wall s | TTFP s | draft accepted/total |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1959 | `rocmfpx_rankleave16_mmid5_mtp4k` | 3598 | 512 | 22.020 | 223.629 | 39.356 | 16.097 | 416/570 |
| 1960 | `rocmfpx_rankleave16_mmid5_mtp4k_repeat2` | 3598 | 512 | 22.968 | 244.294 | 37.030 | 14.736 | 418/558 |

## PPL and KLD Evidence

Stored PPL/KLD row, 16 chunks, `n_ctx=512`:

| PPL Q | PPL base | delta PPL | PPL ratio | mean KLD | P99 KLD | P99.9 KLD | max KLD | same top |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7.314177 | 7.330096 | -0.015920 | 0.997828 | 0.002434 | 0.017315 | 0.106374 | 2.166316 | 97.794% |

## Notes

This lane is stricter than the leave-32 balanced lane by restoring a larger
quality tail to `q6_k`. In local measurements it traded a small amount of speed
for a lower high-tail KLD than the Q6_K_XL comparator.
