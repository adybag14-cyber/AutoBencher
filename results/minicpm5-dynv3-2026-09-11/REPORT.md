# AutoBencher Dynamic-v3 LiteRT sampled integration results

**Date:** 2026-09-11
**Target:** `Tdamre/MiniCPM5-2B-LiteRT-LongContext`
**Scope:** real-use sampled integration test, **not** a formal leaderboard reproduction.

## Protocol

- AutoBencher base: `14ae60899ddcf04d57b7c69a9b4a5612dc71d3ed` plus the LiteRT compatibility changes in this result update.
- EvalScope 1.11.1; LiteRT-LM 0.17.0; WSL2 CPU/XNNPACK; 32 CPU threads.
- `thinking=false`, greedy (`temperature=0`, `top_p=1`), seed 42, evaluator batch size 1.
- `--limit 1`: exactly one item per EvalScope subset. IFEval and GPQA-Diamond each expose one selected subset here.
- 512-token completion cap. A local compatibility proxy mirrors EvalScope/OpenAI `max_tokens` to LiteRT-LM `max_completion_tokens`.

## Results

| Variant | IFEval | IFEval latency | GPQA-Diamond | GPQA latency | Run ID |
|---|---:|---:|---:|---:|---|
| 16K | 100% | 92.500s | 0% | 91.188s | `6d9c67f3-983c-4c17-97fe-4632cb6c9bd9` |
| 32K | 100% | 133.562s | 0% | 142.000s | `ee2b4179-a527-4d1e-924b-0fd5ea928c57` |
| 64K | 100% | 231.860s | 0% | 215.890s | `9eb75ebd-17bd-4e9f-bbb8-cb7101c3019b` |

The IFEval response used 410 output tokens and the GPQA-Diamond response used 401 output tokens for every variant, both below the enforced 512-token cap. Reasoning-token accounting was 0 because server-side thinking was disabled.

### Cross-variant determinism

For each benchmark, the complete generated text was byte-identical across 16K, 32K and 64K. The SHA-256 hashes were:

- IFEval: `d37834c4483ca6d5c736251a7163af8e1be8522c0fb33d23dcab56721d45c27c`
- GPQA-Diamond: `97329a9a098dece14e6afd9ac6c9ea94861b5352eaa4afdf707ffd5dd904a038`

This is strong evidence that the three context-capacity artifacts preserve the same short-context behavior under this deterministic protocol. It is **not** evidence that their full-benchmark accuracy is 100%/0%; each score is from one sampled item only.

### Runtime scaling

The same byte-identical outputs became slower as the configured context capacity increased:

- 16K: IFEval 92.500s; GPQA 91.188s
- 32K: IFEval 133.562s; GPQA 142.000s
- 64K: IFEval 231.860s; GPQA 215.890s

That makes context capacity a meaningful CPU runtime/memory trade-off even when model behavior is unchanged on short prompts.

## Registry coverage

A full AutoBencher dry-run was also performed for each variant. Each traversed all 34 MiniCPM5 model-card rows: 20 have bundled runnable plans and 14 remain explicitly blocked where exact public reproduction prerequisites are unavailable. There were no dry-run infrastructure failures/timeouts.

Dry-run IDs:
- 16K: `040f10f8-130c-46e9-a0ad-758e282c4abc`
- 32K: `b17f6e34-798f-4831-866b-5cc015b759da`
- 64K: `2e64bee4-6893-4f08-bfea-02b97c77179f`

## LiteRT compatibility finding

LiteRT-LM 0.17.0 reads `max_completion_tokens` in its OpenAI-compatible handler, while EvalScope supplies the legacy `max_tokens` parameter. Direct LiteRT serving therefore ignored the intended cap. The added local proxy mirrors `max_tokens` into `max_completion_tokens`; a validation request capped at 16 tokens returned exactly 16 completion tokens.

An earlier MATH-500 diagnostic launched before this translation existed generated for more than ten minutes and was cancelled. It is intentionally excluded from scored results.

## Artifact identity

- 16K (16,384 tokens): `MiniCPM5-2B-LiteRT-DynV3Mixed-16k.litertlm` — SHA-256 `deb0a0f15027542f98079f3d515d42254dc4dd7aeb3dc49cbb2bd733ada1c8be`
- 32K (32,768 tokens): `MiniCPM5-2B-LiteRT-DynV3Mixed-32k.litertlm` — SHA-256 `67416815468538c24d92249f62a86feab0d1ace2086eda79fce12c2296264c82`
- 64K (65,536 tokens): `MiniCPM5-2B-LiteRT-DynV3Mixed-64k.litertlm` — SHA-256 `f403c5cc32da5464b866595c4ddb71ffdffc3571c45bf2047c639737984f841e`
