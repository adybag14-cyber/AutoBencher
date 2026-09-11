# DynV32O28 OCTAV-aware refinement — measured checkpoint

**Independent Unsloth Dynamic-v3-inspired LiteRT adaptation. Not official Unsloth; not a losslessness claim.**

## Completed calibration

52 benchmark-independent calibration sequences, 31,522 total tokens: 48 BF16-generated assistant continuations and four documents of 2,048 / 4,096 / 8,192 / 12,288 tokens. Actual OCTAV block-32 and channelwise INT8 reconstruction were measured; prior INT8 protections were preserved. The new map keeps all 42 attention blocks, 28 of 42 MLP blocks, embedding and output head at INT8. Remaining MLP blocks use OCTAV INT4.

## Held-out quantization surrogate

All profiles use the same 16 unseen prompts and BF16 reference, with 255 teacher-forced prediction positions and up to 32 greedy generated tokens per prompt. Weights are dequantized into PyTorch BF16. **These are not LiteRT runtime measurements, official Divergence-300 results, full benchmark scores, or long-context-retention tests.**

| Profile | Mean KL, nats ↓ | Teacher-forced argmax agreement ↑ | Greedy token agreement ↑ | Exact greedy trajectories |
|---|---:|---:|---:|---:|
| DynV3 | 0.327760 | 82.75% | 20.40% | 1/16 |
| DynV31A | 0.169551 | 85.88% | 24.65% | 1/16 |
| DynV32O28 | 0.102394 | 89.80% | 32.12% | 1/16 |
| INT8Control | 0.010225 | 99.22% | 73.32% | 6/16 |

DynV32O28 reduces mean KL by 68.8% relative to old DynV3 and 39.6% relative to DynV31A on this small held-out set. However, exact 32-token trajectory retention remains 1/16. Uniform INT8 is substantially closer to BF16 than either mixed-precision refinement. We do not hide that stronger control or interpret token differences as equivalent to answer-accuracy loss.

Token-agreement denominators include the larger candidate/reference length for each prompt, so early EOS and excessive continuation are not silently discarded. The raw per-case data and corpus hash are included.

## Artifact and runtime status

The separate `DynV31A` 16K/32K/64K family was built, tested with the actual Rust AutoBencher on matched IFEval and GPQA development items, and published under `experimental/dynv31a-2026-09-11` in `Tdamre/MiniCPM5-2B-LiteRT-LongContext`. Every published hash and size was checked against Hugging Face metadata. See the adjacent DynV31A report for the crucial GPQA output-cap caveat.

At this checkpoint DynV32O28 export is in progress. It is not presented as a completed, runtime-validated release. The unrelated existing context-aware `DynV32` artifacts are not overwritten or conflated with it.

## Full-precision-control bug

The old wrapper converted `NONE` to Python `None`; the converter then discarded that argument and selected its default INT8 recipe. The process-local corrected wrapper passes the explicit string `none`, disables FP16/mixed precision, and passed the installed configuration/dispatcher check. A previously generated file whose name contains `FP32Ref` is therefore classified as an INT8 control, not an unquantized reference.

## Tests

2 actual-quantizer tests passed in the WSL quantization environment; 4 precision-selection and 3 full-precision-control argument tests passed locally. GitHub CI independently passed the 7 dependency-free selection/control tests.
