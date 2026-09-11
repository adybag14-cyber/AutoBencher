# MiniCPM5 LiteRT dynamic-quantization refinement

This work is an **Unsloth Dynamic-v3-inspired LiteRT adaptation**, not an official Unsloth quantization, GGUF conversion, or faithful reproduction of Unsloth's proprietary calibration data. It is post-training quantization, not quantization-aware training. The software additions are Apache-2.0, consistent with this repository. Model and benchmark licenses remain their own.

## Keep three families separate

| Family | Attention INT8 blocks | MLP INT8 blocks | Other MLP weights | Status interpretation |
|---|---:|---:|---|---|
| Existing DynV3 | See original manifest | See original manifest | Blockwise INT4 | Existing control, never overwritten here |
| DynV31A | 42 | 20 | Block-32 OCTAV INT4 | Independently exported at 16K, 32K, 64K; separate experimental diagnostic |
| DynV32 conservative | 42 | 32 | Block-32 OCTAV INT4 | New calibration plus a no-demotion precision floor; export completion is not quality acceptance |

Both new families retain INT8 vocabulary embeddings and output heads. Do not substitute one family's results or sensitivity map for another. More INT8 weights can reduce a weight-error proxy but do not prove better end-to-end accuracy.

## Completed context-aware calibration

The original independent synthetic corpus has 72 user/assistant dialogues, covering mathematics, coding, agent/API planning, structured conversation, multilingual text and reasoning. The completed default run processed 7,244 short tokens and separate long documents of 12,288, 24,576 and 40,960 tokens for 16K, 32K and 64K capacities respectively. Long documents contain fictional maintenance records, not NoLiMa evaluation text or needles. No benchmark questions, answers or existing model evaluation responses were used to create the corpus.

Seed: `730219`. Corpus SHA-256: `54b63eb37d9eb1182d2f7b5e6135ab607e493e0ee7b65eb32913f5eb145b30f5`.

Input second moments are recorded at 294 linear modules across 42 layers. A 50/50 mixture of the diverse short-dialogue moments and a capacity-specific long-document moment prevents the long sample from overwhelming all short categories merely by token count. Layer selection uses diagonal activation-weighted local quantization-error reduction normalized by signal energy. This is a sensitivity proxy, **not measured model loss, KL divergence, or an end-to-end benchmark**.

The INT4 simulator models OCTAV clipping, signed [-8,7] quantization and BF16-to-FP16 scale storage; INT8 uses symmetric narrow-range [-127,127] per-output-channel quantization. Zero/underflowed blocks are guarded. Exact implementation parity tests have not been completed; final exports use the real installed quantizer.

The independent context measurements selected the same 20 MLP blocks:

```text
0, 5, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 41
```

The conservative export map takes the union of this set and the frozen DynV31A INT8 set. It protects 32 MLP blocks and never demotes an existing DynV31A INT8 block. The calibration-only selections remain in the manifest for traceability. Previously inspected IFEval/GPQA cases are **development regression cases**, not fresh held-out validation for this refinement.

## Execution in the existing Windows/WSL workspace

These scripts intentionally target the established Devbox installation. They are not a standalone installer for a fresh machine. They depend on the already-present MiniCPM5 checkpoint, its original chat template, the patched `hf-to-litertlm` converter, `export_minicpm_dynv31.py`, and the zero-scale repair helper. Inspect those dependencies before reproduction elsewhere.

```bash
ROOT=/home/tdamre/minicpm5-litert-20260910
REPO=/mnt/c/Users/adyba/docker-chatgpt-devbox/workspace/AutoBencher
"$ROOT/.venv/bin/python" "$REPO/scripts/minicpm5_calibrate_refinement.py" \
  --model "$ROOT/model/MiniCPM5-2B" --out "$ROOT/dynv32"
bash "$REPO/scripts/build_minicpm5_refinement_all.sh"
```

The builder exports each capacity independently, uses its saved calibration, retains prefill signatures 1024/256/64/16/4/1, repairs invalid zero scales with the existing converter helper, and records final SHA-256 and byte length. It refuses to overwrite a partial raw export and verifies existing completed exports before reusing them. A completed export says nothing by itself about answer quality or long-context retention.

Toolchain source revisions recorded in the existing workspace:

```text
litert_torch=6d4c622c9a3aade4d411a1e859c6bbe571d38ee0
litert_lm=0d43b55250dfd2a69d64b761e238f2fc4b12c959
```

## Evaluation and acceptance boundaries

Use AutoBencher with matched prompts, token budgets, tokenizer/chat template, temperature, seed, thinking configuration and evaluator versions. Preserve real outputs and failure statuses. Compare against the original local BF16 checkpoint as well as the older quantization. In particular, do not reuse the earlier truncated 512-token GPQA BF16 run; the valid diagnostic control allows 2,048 output tokens.

One selected IFEval item and one GPQA item are useful regression checks but do not establish full-benchmark performance. A sample exactly at the output limit is not assumed to be a natural completed answer merely because an adapter reports `stop`. Extracted-choice correctness and requested final-answer formatting are separate observations.

The 64K artifact passing a short question does not validate 64K retrieval. Runtime failures must remain unscored, not converted into wrong-answer zeros. Fresh held-out evaluation, long-context retrieval at multiple depths, and broader model-card benchmark coverage are still required before a near-lossless claim. BF16 GPU latency and LiteRT CPU latency are not comparable as quantization speedups, especially when exports run concurrently.

## Safe publication

Publish experimental artifacts separately from existing release paths. Hash the exact immutable copy served during evaluation, not a source output path another concurrent conversion might rebuild. Attach comparison JSON, per-context hashes and explicit limitations. Use a parent-commit guard for Hugging Face model-card edits and never log credentials or commit token files.


<!-- REFINEMENT_TRANSPORT_AND_AUDIT -->
## Resumable diagnostic transport and evidence checks

`benchmark_minicpm5_refinement.py --resume` retains completed runs and repeats only unfinished checks. It serves the frozen experiment imports, verifies their pinned SHA-256 hashes and uses a 3,600-second evaluator request timeout for slow CPU inference. The generation budgets remain 512 tokens for IFEval and 2,048 for GPQA. Timeout changes are transport changes, not a model-quality improvement.

`audit_minicpm5_comparison.py` compares exact prompt hashes, input token counts, frozen model identities and saved BF16 output hashes. It reports output-cap hits independently from adapter stop reasons and distinguishes extracted-choice correctness from the requested final-answer format.

`publish_minicpm5_experimental.py` requires completed, audited inference before publishing a separate experimental directory. It refuses unaudited data, mixed model families, missing contexts and unsupported near-lossless claims. A wrong answer can be published transparently; a failed transport request cannot be silently scored as a wrong answer.

The bounded localhost proxy explicitly rejects conflicting output-token allowances and unsupported streaming requests. `test_refinement_transport_policy.py` exercises this control-plane behavior with a mock backend; those tests do not establish quantizer simulator parity or model accuracy.

The export scheduler can checkpoint after the 16K candidate by writing `hold` to `dynv32/export_scheduler_gate.txt`. Subsequent context invocations exit with status 75 before touching their output directories. Writing `continue` permits the remaining contexts to resume. This is a scheduling checkpoint, not a claim that deferred artifacts exist.
