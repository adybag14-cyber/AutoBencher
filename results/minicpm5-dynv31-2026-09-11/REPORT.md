# DynV31A experimental artifacts â€” 11 September 2026

**Experimental; not a replacement for existing root artifacts. Near-lossless quality has not been established.**

Dynamic-v3-inspired LiteRT adaptation, not an official Unsloth quantization or GGUF. All attention blocks, 20 selected MLP blocks, the vocabulary embedding and output head use INT8. Remaining MLP blocks use block-32 OCTAV INT4.

This is a two-item development regression diagnostic, not full IFEval or GPQA leaderboard accuracy. Each percentage describes one selected item. The BF16 controls are earlier matched AutoBencher runs, with exact prompt and output-receipt identity checked by the accompanying audit.

| Context | IFEval candidate / BF16 | GPQA candidate / BF16 | GPQA candidate tokens / allowance |
|---|---:|---:|---:|
| 16k | 100% / 100% | 100% / 100% | 2048 / 2048 |
| 32k | 100% / 100% | 100% / 100% | 2048 / 2048 |
| 64k | 100% / 100% | 100% / 100% | 2048 / 2048 |

## Completion and formatting audit

16k: output at cap = **True**; adapter-reported stop reason = `stop`; explicit `ANSWER: [LETTER]` line = **False**. The score is the evaluator's extracted-choice score, not a separate formatting score.

32k: output at cap = **True**; adapter-reported stop reason = `stop`; explicit `ANSWER: [LETTER]` line = **False**. The score is the evaluator's extracted-choice score, not a separate formatting score.

64k: output at cap = **True**; adapter-reported stop reason = `stop`; explicit `ANSWER: [LETTER]` line = **False**. The score is the evaluator's extracted-choice score, not a separate formatting score.

An output exactly at its allowance is not silently treated as a complete natural termination, even when the adapter reports `stop`.

Matched protocol: seed 42, temperature 0, top-p 1, thinking disabled, evaluation batch size 1. IFEval allows 512 generated tokens and GPQA 2048. LiteRT-LM uses CPU inference with 32 threads and AutoBencher's token-cap compatibility proxy.

Long-context NoLiMa retention has not been established for this family. Running a short question in a 64K-capacity artifact does not validate 64K retrieval. The previous DynV3 family had demonstrated long-context failures. End-to-end differences versus BF16 can include conversion, tokenizer and runtime effects, not solely weight quantization. BF16 GPU and LiteRT CPU latency must not be compared as a quantization speedup.

DynV32 context-aware calibration is a separate candidate family. Its results must not be substituted for these measurements.

## Frozen artifact hashes

```text
743f73e2917f90fa4229504363c6191ae25e8e6e1fff98380ac6c4991600c2a8  MiniCPM5-2B-LiteRT-DynV31A-16k.litertlm
e0f52b6223f6e8dcc015471287732721b8c8471848204ba2f2741e2a2e611bff  MiniCPM5-2B-LiteRT-DynV31A-32k.litertlm
b301387dcc67c78b26f100868e650e0bb18a8459926f67ed3cae69e6f4ef36ba  MiniCPM5-2B-LiteRT-DynV31A-64k.litertlm
```
