# LiteRT model validation on an Android device

The scripts in this repository provide a pinned Android inference runner, a
localhost OpenAI-compatible bridge, and guarded artifact preparation. A model
loading successfully is only an initialization check. Usable output, actual
hardware delegation, and accuracy against the original checkpoint are separate
acceptance checks.

## Build and start the device runner

For a manual Gallery import, scroll to the bottom of the import dialog and
select both CPU and GPU under **Compatible accelerators**. The initially visible
form does not show that setting. Enable thinking support if you want the
thinking toggle, then disable thinking for the included comparison protocol.
If you replace an imported entry with the same filename, restart Gallery to
reload its capability declaration before checking its accelerator controls.

Use JDK 21 and an Android SDK containing platform 36. The builder downloads
LiteRT-LM Android 0.17.0, its runtime dependencies, and R8, and checks every pinned
SHA-256 before compiling. It uses internal JNI methods from this exact version;
other SDK versions require a reviewed adapter and another device test.

```bash
python scripts/build_litert_android_runner.py \
  --android-sdk /path/to/android-sdk --jdk /path/to/jdk21 \
  --output /path/to/runner-build

adb -s SERIAL shell mkdir -p /data/local/tmp/litert-validation
adb -s SERIAL push /path/to/runner-build/jni017 /data/local/tmp/litert-validation/
adb -s SERIAL push /path/to/runner-build/device_runner_jni /data/local/tmp/litert-validation/
adb -s SERIAL shell chmod 755 /data/local/tmp/litert-validation/device_runner_jni
adb -s SERIAL push model.litertlm /data/local/tmp/litert-validation/model.litertlm
```

Compute the model's SHA-256 on the host, then provide it to the bridge:

```bash
python scripts/litert_android_server.py \
  --adb /path/to/adb --serial SERIAL \
  --model-path /data/local/tmp/litert-validation/model.litertlm \
  --sha256 MODEL_SHA256 --model-id minicpm5-2b-android-gpu \
  --runner /data/local/tmp/litert-validation/device_runner_jni \
  --cache-dir /data/local/tmp/litert-validation/cache-gpu \
  --backend gpu --port 9396 --evidence device-evidence
```

The bridge independently hashes the device file before launching the engine.
It binds to `127.0.0.1`, verifies the requested model ID, and accepts only the
supported greedy text-generation settings. Each request creates a fresh
conversation. Native output, token counts, timing, model identity, and process
ownership are recorded. Retain `native-stderr.log` to confirm GPU delegation
(`LITERT_CL`) rather than inferring GPU execution from a configuration string.
The engine remains resident across requests so initialization is not repeated
for every question.

Run the complete evaluation in a second terminal:

```bash
autobencher --config autobencher-gpqa-full-android.toml run \
  --only gpqa-diamond --strict
```

Use the same pinned EvalScope version, dataset revision, tokenizer/template,
question order, decoding configuration, and answer extractor for BF16 and the
device candidate. The included comparison profiles use EvalScope 1.11.1, which
loads the 198-question GPQA Diamond split. Record the cached dataset fingerprint
and adapter hash alongside the run. Calibration data must be independent of
these benchmark questions.

Once both runs are complete, compare their saved reviews and predictions:

```bash
python scripts/compare_gpqa_runs.py --bf16 .autobencher/runs/BF16_RUN_ID \
  --candidate .autobencher/runs/DEVICE_RUN_ID --output comparison.json
```

The comparison rejects partial splits, duplicate IDs, differing prompts or
answer ordering, evaluator errors, and mismatched generation protocols. It
reports paired improvements/regressions, answer agreement, truncations, token
counts, and a descriptive paired question bootstrap interval. Its per-question
records contain hashes and answer letters, without copying benchmark questions.

The bridge exits after its maximum lifetime (eight hours by default). For an
early controlled shutdown, POST `/shutdown` with the `X-Task-Stop-Key` stored in
the local evidence directory's `owner.json`. This nonce is not served by
`/health`. Stop or checkpoint the evaluator first. The bridge closes its native
runner and verifies the captured command, executable, and start time before
any fallback termination. Never restart a shared ADB daemon to stop a test.

## Bounded prefill for mobile GPUs

The original MiniCPM5 64k bundle includes a `prefill_1024` graph that requests a
2,147,483,648-byte FP16 attention allocation on Adreno 830. The tested device
reports a 1,073,741,824-byte maximum allocation. Reducing the prefill ladder
avoids that per-buffer limit. Total memory and model correctness still need
separate testing.

```bash
python scripts/prepare_litert_mobile.py --help
```

The preparation script prunes unsupported prefill signatures and their graph
closures without moving external weight buffers. It remaps retained composite
subgraph references, reparses the result, and verifies that every byte outside
its recorded patches remains identical. It rejects unsupported control-flow
and custom graph structures rather than guessing reference semantics. Use a
new output path and retain the generated preparation report.

This graph edit does not improve weight quantization. A preserved but inaccurate
model remains inaccurate after its GPU allocation problem is fixed.

## Calibrated MiniCPM5 exports and precision checks

`export_minicpm_mobile.py` accepts an explicit precision-selection JSON. It
protects embeddings and the output head at INT8, applies the selected INT8
attention/MLP layers, and uses block-32 OCTAV INT4 only for remaining layers.
It preserves the original HF tokenizer and Jinja chat template, uses FP16
activations, and exports a single bounded prefill signature plus decode.

```bash
python scripts/export_minicpm_mobile.py --model /path/to/BF16-checkpoint \
  --selection /path/to/selection.json --context 32768 \
  --output /path/to/empty-export-directory
```

The original checkpoint must match the intended HF revision and safetensors
hash. The recipe is scoped to MiniCPM5-2B. Reserve at least 18 GiB free scratch
space for one export and run exports serially on a constrained filesystem.
The output provenance records package versions, context, selected precision,
artifact hash, and an explicit pending device-validation status.

The strict calibration policy used for the current quality candidate selected
INT8 for every attention and MLP group. It must therefore be described as an
INT8 candidate. A selection method inspired by data-dependent quantization does
not establish that a particular export is lossless or an official Unsloth build.
Inspect the stored tensors after export; a requested rule alone is not proof
that it matched the intended weights.

`repair_litert_zero_blocks.py` addresses a narrower INT4 quantizer defect. It
checks the original BF16 shard and clears nonzero codes only in MLP blocks whose
entire source range lies below half the smallest positive scale value. It also
repairs the corresponding invalid scales. Representable nonzero source weights
with invalid scales cause a hard failure. All other bytes remain unchanged.
This rounds extremely small BF16 values to zero and does not assert whole-model
accuracy. The tested repaired INT4 candidate still failed a simple arithmetic
control, so this repair alone did not establish an acceptable conversion.

The converter environment used during development pins LiteRT-Torch source
`6d4c622c9a3aade4d411a1e859c6bbe571d38ee0`, LiteRT converter 0.4.0,
ai-edge-litert-nightly 2.3.0.dev20260909, ai-edge-quantizer-nightly
0.10.0.dev20260910, torch 2.14.0+cu130, transformers 5.17.0, and
LiteRT-LM/builder 0.17.0. Recheck installed versions against each artifact's own
provenance before claiming a reproduced result.

## NPU acceptance

NPU compilation is specific to the SoC and vendor/runtime ABI. Record the SoC,
compiler, Qualcomm SDK, dispatch library, native runner, and all artifact/library
hashes. Require both actual HTP dispatch and correct output on the physical
device. Passing a Python CPU reference executor or creating a QNN context does
not verify a usable NPU model. NPU compatibility is not implied by the GPU
runner's supported SDK version.

The optional C API source in `scripts/android/device_runner.c` implements the
same three-record request protocol for a separately pinned C API library. It
exists to isolate SDK/runtime issues; its output must not be combined with a
different runtime's benchmark without retaining that difference in provenance.
