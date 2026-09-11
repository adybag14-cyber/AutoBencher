# MiniCPM5 Dynamic-v3 LiteRT — BF16-relative AutoBencher report

**Date:** 2026-09-11
**Target:** `Tdamre/MiniCPM5-2B-LiteRT-LongContext`
**Purpose:** quantify degradation relative to the original BF16 `openbmb/MiniCPM5-2B`, not merely report standalone quant scores.

## Main finding

The Dynamic-v3-inspired LiteRT artifacts are **not lossless relative to BF16**.

On short-context frozen source-success checks, all three variants retained the BF16 result on six selected checks: LiveCodeBench v6, MATH-500, IFEval, MMLU-Pro, BFCL v4, and GAIA Text-103 (**6/6 retained for each artifact**). However, a fresh matched AutoBencher GPQA-Diamond diagnostic found a clear reasoning regression: BF16 solves the sampled item with a 2,048-token ceiling, while all three quantized variants terminate naturally on the same wrong answer.

Long-context retrieval shows a stronger degradation signal. On the selected NoLiMa-Hard case, BF16 succeeds at approximately 12K, 24K, and 40K rendered tokens. The 16K and 32K Dynamic-v3 artifacts return the wrong character at their corresponding long-context lengths; the 64K artifact also returns the wrong character on the supplemental ~12K test, while its ~40K test exceeds LiteRT-LM 0.17's hard 10-minute CPU session deadline.

## Fresh matched AutoBencher comparison

Settings shared within each score comparison: EvalScope 1.11.1, `enable_thinking=false`, greedy decoding (`temperature=0`, `top_p=1`), batch size 1, identical benchmark item and evaluator. BF16 used vLLM 0.29.0; LiteRT used LiteRT-LM 0.17.0 with the compatibility proxy that maps `max_tokens` to `max_completion_tokens`.

| Benchmark | BF16 | DynV3 16K | DynV3 32K | DynV3 64K | Interpretation |
|---|---:|---:|---:|---:|---|
| IFEval item 0, 512-token ceiling | **100%** | **100%** | **100%** | **100%** | retained |
| GPQA-Diamond item 0, 2,048-token ceiling | **100%** | **0%** | **0%** | **0%** | **-100 pp regression** |

The two-item matched diagnostic therefore retains **1/2 BF16-passing items (50%)** for each quant. This percentage is a diagnostic, not a benchmark-wide estimate; the item count is too small for a leaderboard-style claim.

### Why the GPQA rerun matters

The first BF16 GPQA attempt used a 512-token ceiling and was truncated at exactly 512 tokens, so its initial 0% result was not a valid degradation baseline. The corrected BF16 run used a 2,048-token ceiling and:

- scored **100%**;
- stopped naturally after **1,447 output tokens**;
- selected the correct `10^-4 eV` answer;
- AutoBencher run ID: `df9a43a7-2c0b-4252-85a6-8eb647aaf6e4`.

All three quantized reruns used the same 2,048-token ceiling and still:

- scored **0%**;
- stopped naturally after **401 output tokens** rather than hitting the ceiling;
- produced the same wrong output byte-for-byte (`SHA256 97329a9a098dece14e6afd9ac6c9ea94861b5352eaa4afdf707ffd5dd904a038`).

Matched quant run IDs:

- 16K: `543456ae-19e4-48e7-a316-76c5e4c9ba2f`
- 32K: `f7cecbf8-62bd-4baf-9c4c-2ad8c91b7b02`
- 64K: `a89fc032-4008-4a6b-811a-b9a3773af1f4`

This is a clean quantization-fidelity failure on the sampled GPQA problem rather than a timeout or output-cap artifact.

## BF16-first frozen short-context retention suite

Before inspecting the quant outputs, BF16 was scanned and source-passing items were frozen. The following six clean source-success items were then replayed identically on all three LiteRT artifacts.

| Frozen check | BF16 | DynV3 16K | DynV3 32K | DynV3 64K |
|---|---:|---:|---:|---:|
| LiveCodeBench v6 | 1 | 1 | 1 | 1 |
| MATH-500 | 1 | 1 | 1 | 1 |
| IFEval | 1 | 1 | 1 | 1 |
| MMLU-Pro | 1 | 1 | 1 | 1 |
| BFCL v4 | 1 | 1 | 1 | 1 |
| GAIA Text-103 | 1 | 1 | 1 | 1 |
| **Source-success retention** | **6/6** | **6/6** | **6/6** | **6/6** |

SWE-bench Verified is excluded from that denominator because the bounded BF16 semantic-patch proxy itself scored 0. Claw-Gym is also excluded because it is a bounded output-content proxy and BF16 scored 8/9 checks rather than a clean pass. Their quant scores matched the BF16 proxy scores, but neither is suitable as a clean source-success fidelity item.

The frozen suite and the fresh GPQA result are not contradictory: they show that these mixed-precision quants preserve several selected short-context behaviors while still having at least one reproducible reasoning regression.

## Long-context BF16 degradation comparison

Selected NoLiMa-Hard case: `0408Inv_T04_C02_twohop` — question: **Which character has been to Madrid?** Gold: **Yuki**.

| Comparison | BF16 | Dynamic-v3 | Delta / result |
|---|---|---|---|
| ~12K rendered tokens: BF16 vs 16K artifact | `Yuki` ✅ | `Rebecca` ❌ | **-100 pp** |
| ~24K rendered tokens: BF16 vs 32K artifact | `Yuki` ✅ | `Rebecca` ❌ | **-100 pp** |
| ~40K rendered tokens: BF16 vs 64K artifact | `Yuki` ✅ | 10-minute LiteRT deadline | not scored; practical runtime failure |
| ~12K supplemental: BF16 vs 64K artifact | `Yuki` ✅ | `Rebecca` ❌ | **-100 pp** |

Rendered prompt token counts were 12,114 / 24,114 / 40,114 for the corresponding BF16 tests. The selected case was chosen from a BF16-only scan before quant outputs were inspected.

This is the strongest current evidence of degradation: **BF16 retrieval succeeds where the quantized models return the wrong entity**, and the largest LiteRT context configuration additionally becomes impractical on the current CPU runtime at ~40K tokens.

## Latency is not a fair BF16-vs-quant comparison

Do **not** infer that BF16 is intrinsically faster from these runs. BF16 was served through vLLM on an RTX 4090, while the LiteRT benchmark claims used XNNPACK CPU because the available WSL WebGPU path selected Mesa `llvmpipe` rather than the RTX 4090. The scores and deterministic outputs are useful for fidelity comparison; cross-backend latency is not.

Within the LiteRT artifacts themselves, the same short prompts became progressively slower as the configured context capacity increased. That remains a useful practical observation for these artifacts/runtime combinations.

## Artifact identities

| Variant | Context | SHA-256 |
|---|---:|---|
| `MiniCPM5-2B-LiteRT-DynV3Mixed-16k.litertlm` | 16,384 | `deb0a0f15027542f98079f3d515d42254dc4dd7aeb3dc49cbb2bd733ada1c8be` |
| `MiniCPM5-2B-LiteRT-DynV3Mixed-32k.litertlm` | 32,768 | `67416815468538c24d92249f62a86feab0d1ace2086eda79fce12c2296264c82` |
| `MiniCPM5-2B-LiteRT-DynV3Mixed-64k.litertlm` | 65,536 | `f403c5cc32da5464b866595c4ddb71ffdffc3571c45bf2047c639737984f841e` |

## Scope

These are targeted quantization-fidelity diagnostics, **not full leaderboard reproductions**. The correct conclusion is not that the quants are globally 50% worse or that GPQA is 0% overall. The evidence supports a narrower but important statement:

> The Dynamic-v3-inspired LiteRT variants preserve all six selected frozen short-context BF16-success cases, but measurable degradation exists: all three lose a BF16-correct GPQA-Diamond sample, and long-context NoLiMa retrieval degrades substantially relative to BF16.

Machine-readable evidence is in `autobencher_dynamic_v3_results.json`. The earlier full retention details remain in the Hugging Face repository's `dynamic_v3_benchmark_results.json`.
