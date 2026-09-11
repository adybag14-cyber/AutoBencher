# OCTAV-aware dynamic LiteRT refinement

This is an **independent Unsloth Dynamic-v3-inspired LiteRT adaptation**, not an official Unsloth artifact or a reproduction of an unpublished recipe. It uses post-training weight quantization. It does not perform QAT, QAD, or claim losslessness.

## What changes

The original calibration ranked a blockwise min/max INT4 approximation on 48 short input prompts. The new implementation generates actual BF16 assistant continuations, adds four independent long documents, and measures projection error using the installed AI Edge Quantizer's OCTAV block-32 INT4 and channelwise INT8 implementations. This includes the exporter's scale rounding. Sampled activation vectors retain channel covariance; relative projection error avoids simply preferring layers with larger residual-stream magnitudes.

The completed September 11, 2026 calibration contains **52 sequences and 31,522 tokens**, including documents of 2,048, 4,096, 8,192 and 12,288 tokens. It is a small synthetic calibration corpus, not a broad population-quality guarantee. No benchmark questions or answer keys are read by the calibration or selection scripts.

`build_dynamic_litert.py` preserves every INT8 layer in the supplied frozen precision floor, protects boundary layers, and adds calibration-ranked MLP blocks up to an explicit quota. The experimental DynV32O28 configuration uses all 42 attention blocks, 28 MLP blocks, embedding and output projection at INT8; remaining MLP weights use block-32 INT4 plus OCTAV. The recorded `selection.json` is authoritative.

## Separate evidence levels

A completed export is not a successful runtime test. Runtime initialization is not benchmark accuracy. Passing a small regression set is not benchmark-wide parity. Short-context parity does not establish 32K or 64K retrieval fidelity.

The scripts record these separately:

- `calibrate_dynamic_litert.py`: calibration corpus, hash, per-projection errors and selection proposal.
- `build_dynamic_litert.py`: floor-preserving selection, input/toolchain fingerprints, per-context artifact hashes and build status. Runtime/benchmark status remains pending until measured.
- `validate_dynamic_surrogate.py`: 16 unseen prompts, up to 256 teacher-forced positions, BF16-relative KL divergence, teacher-forced argmax agreement, and 32-token greedy trajectory comparisons. It runs dequantized weights in **PyTorch BF16**, not LiteRT, and is explicitly a surrogate diagnostic.
- `bench_litert_candidate.py`: runs the actual Rust AutoBencher executable against an isolated LiteRT server/proxy, checks the requested model alias, records artifact identity and a real inference smoke response, and stops its own serving children afterward.

## Example WSL workflow

Use the same environment as the working LiteRT exporter. The supplied exporter and zero-scale-fixer paths must be reviewed, versioned, and compatible with the model. Their hashes are recorded; neither is silently downloaded.

```bash
python scripts/calibrate_dynamic_litert.py \
  --model /path/to/MiniCPM5-2B \
  --prompt-source /path/to/previous-calibration.py \
  --output /path/to/calibration

python scripts/build_dynamic_litert.py \
  --model /path/to/MiniCPM5-2B \
  --sensitivity /path/to/calibration/sensitivity.json \
  --floor config/minicpm5-dynv32-floor.json \
  --exporter /path/to/export_minicpm_dynv31.py \
  --zero-scale-fixer /path/to/fix_zero_scales_inplace.py \
  --source-root /path/to/hf-to-litertlm \
  --output /path/to/out_dynv32_octav28 \
  --contexts 16384,32768,65536 --mlp-int8 28

python scripts/validate_dynamic_surrogate.py \
  --model /path/to/MiniCPM5-2B \
  --profile previous=/path/to/previous-selection.json \
  --profile candidate=/path/to/out_dynv32_octav28/selection.json \
  --output /path/to/heldout

python scripts/bench_litert_candidate.py \
  --artifact /path/to/out_dynv32_octav28/16k/MiniCPM5-2B-LiteRT-DynV32O28-16k.litertlm \
  --label minicpm5-dynv32o28-16k --context 16384 \
  --only gpqa-diamond --limit 1 --max-tokens 2048
```

The benchmark helper currently targets WSL with the existing **Windows Rust AutoBencher executable** in `target/release/autobencher.exe` and a provisioned AutoBencher EvalScope environment. It is not a portable Linux binary installer. Use a distinct free port pair with `--port` when another run is active. For matched IFEval diagnostics use `--only ifeval --limit 1 --max-tokens 512`. Larger evaluation samples require equivalent BF16 runs; historical one-item controls cannot establish parity for a larger new sample.

## Safety and reproducibility

Existing artifacts are never replaced. Completed builds can be retained using `--resume` only when fingerprints match. Partial raw exports are not automatically deleted. The build hashes every safetensors weight file, model configuration, exporter, zero-scale fixer and selected recipe, and records toolchain versions. Benchmark aliases cannot be repointed silently. No script here reads a Hugging Face token or performs an upload.

Tests:

```bash
python scripts/test_dynamic_quantizer.py
python scripts/test_dynamic_selection.py
```

The quantizer tests cover dense and zero-row reconstruction and the fixed-fixture INT8/INT4 error ordering. Selection tests cover the frozen floor, invalid quotas, input immutability and incomplete rankings.

Official methodology context: https://unsloth.ai/docs/basics/dynamic-3.0-ggufs

All new implementation files use the repository's Apache-2.0 license. External toolchains and model weights retain their respective licenses.

## Full-precision control correction

The existing exporter translates `NONE` into Python `None`, but the installed LiteRT export entry point drops `None` arguments and defaults to INT8. `export_full_precision_control.py` fixes this in its own process by passing the explicit string `none`, which the installed quantization dispatcher recognizes. It also disables FP16/mixed-precision conversion. Its `--check-only` mode verifies the configuration and dispatcher; exported tensor dtypes still require a separate audit. An artifact named FP32 is not proof of full precision.
