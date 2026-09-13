# AutoBencher

**AutoBencher** is a Rust orchestration layer for reproducible LLM evaluation. It was created around the complete evaluation table on the [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) model card and is designed to make repeated benchmarking of original, converted, fine-tuned, and quantized checkpoints fast and auditable.

Licensed under the **Apache License 2.0**.

## What it does

Running AutoBencher with no subcommand starts the full configured suite:

```bash
cargo run --release
```

The default configuration targets `openbmb/MiniCPM5-2B`, auto-detects an existing OpenAI-compatible endpoint or managed vLLM backend, bootstraps the required isolated EvalScope environments, verifies the current model card against the pinned benchmark registry, runs the suite, and writes durable results. On Windows, if native vLLM is unavailable but WSL2 is present, AutoBencher automatically provisions a cached WSL vLLM environment and uses the NVIDIA GPU exposed to WSL.

For an already-running model server:

```bash
cargo run --release -- run \
  --model /models/MiniCPM5-2B-DynV3-32k \
  --engine external \
  --api-base http://127.0.0.1:8000
```

For a quick pipeline check:

```bash
cargo run --release -- run --limit 5 \
  --only mmlu-pro,math-500,ifeval,livecodebench-v6
```

GPQA Diamond contains **198 questions**. Run its complete split without a sample
limit:

```bash
cargo run --release -- --config autobencher-gpqa-full-bf16.toml run \
  --only gpqa-diamond --strict
```

This profile expects a separately started BF16 server. The matching
`autobencher-gpqa-full-android.toml` profile targets the pinned Android LiteRT
runner. Both use greedy decoding, thinking disabled, a 2,048-token output cap,
seed 42, and one request at a time. These settings support a paired conversion
comparison; they do not reproduce the model card's thinking-enabled score.
See [Android setup and artifact validation](docs/litert-mobile-validation.md).

AutoBencher checks GPQA's requested, successful, scored, and distinct reviewed
question counts. A successful evaluator exit with fewer than 198 questions is
`failed` unless a smaller `--limit` was explicitly requested. An intentional
sample is labelled `sampled`; the report includes its actual question count.
Missing reports, duplicate question IDs, and evaluator errors fail full coverage.

To continue an interrupted evaluation without regenerating completed answers:

```bash
cargo run --release -- --config autobencher-gpqa-full-bf16.toml run \
  --only gpqa-diamond --resume RUN_ID --strict
```

Resume validates the saved model, endpoint, seed, generation settings, sample
limit, EvalScope requirement, and saved benchmark definitions before updating
the harness. Partial EvalScope predictions are reused with `--use-cache`.
Earlier attempts and configuration snapshots are retained. Increase
`benchmark_timeout_minutes` for a slow device; changing the evaluation protocol
requires a new run. The latest attempt duration and the total across attempts
are reported separately.

To see the complete command plan without downloading dependencies or starting a model:

```bash
cargo run --release -- run --dry-run
```

## Complete MiniCPM5-2B model-card registry

AutoBencher pins all **34 rows across 9 categories** from the model card:

- **Code Reasoning:** LiveCodeBench v6; LCB-Pro 25Q2 Easy; LCB-Pro 25Q2 Medium; OJBench; SciCode (wbg)
- **Math Reasoning:** AIME 2025; AIME 2026; HMMT Feb 2026; MATH-500
- **Instruction Following:** IFBench; IFEval; Multi-IF
- **General Knowledge:** MMLU-Pro; MMLU-Redux; HLE; GPQA-Diamond; SuperGPQA
- **Long Context:** AA-LCR; NoLiMa; LongBenchPro; LongBench v2
- **Tool Use:** τ³-Bench Banking; τ²-Bench Telecom; BFCL v4
- **Coding Agent:** SWE-bench Verified; SWE-bench Pro; Terminal-Bench v2.1
- **Search Agent:** BrowseComp-ZH; BrowseComp Top100; GAIA Text-103
- **General Agent:** GDPval-AA v2; Claw-Gym; WildClaw; QwenClaw

The registry is validated by tests so an accidental row deletion, duplicate ID, or category omission fails CI.

## Scientific-integrity rule

A model-card row is not necessarily the same thing as a publicly reproducible one-command benchmark. Some rows depend on encrypted/restricted data, separately licensed assets, a web/search provider, an external user simulator, a proprietary judge/sandbox, or evaluation details that are not published on the model card.

AutoBencher therefore **does not fake unavailable rows with home-grown proxy scores**. It uses three behaviors:

1. `evalscope` — runs a maintained EvalScope adapter against the same OpenAI-compatible model endpoint.
2. `native` — uses a benchmark-specific upstream recipe when a reproducible command is available.
3. `blocked` — records the row, reason, provenance and run status without turning infrastructure unavailability into a zero model score.

A fast-moving native harness can be enabled without recompiling AutoBencher:

```bash
export AUTOBENCHER_NATIVE_CMD_NOLIMA='your pinned official NoLiMa command using ${MODEL} ${API_BASE} ${RESULT_DIR}'
```

Supported placeholders are `${MODEL}`, `${API_BASE}`, `${API_KEY}`, `${RESULT_DIR}`, `${WORKSPACE}`, and `${LIMIT}`. On Windows, use the equivalent environment-variable mechanism.

This makes the benchmark registry stable while allowing exact upstream commands/revisions to evolve independently.

## Commands

```text
autobencher                         Run the full suite using autobencher.toml
autobencher run [OPTIONS]           Run/resume selected benchmarks
autobencher list                    List all 34 benchmark rows
autobencher doctor                  Check local tooling and registry integrity
autobencher setup [--force]         Bootstrap/update isolated EvalScope env
autobencher verify-card             Verify registry names against live model card
autobencher report [--run-id ID]    Rebuild JSON/CSV/Markdown reports
```

Useful run options:

```text
--model MODEL            Hugging Face ID or local checkpoint
--engine auto|vllm|wsl-vllm|external
                           Auto-detect, force native/WSL vLLM, or use an existing endpoint
--api-base URL           OpenAI-compatible server root
--only ID,ID,...         Benchmark IDs only
--category NAME          Category filter
--limit N                Smoke-test sample limit
--parallel N             Max concurrent benchmark processes
--resume UUID            Continue an existing run
--dry-run                Persist an execution plan only
--strict                 Non-zero exit if blocked/failed/timed-out rows exist
--no-setup               Do not bootstrap EvalScope
--no-card-check          Skip online registry-drift verification
```

## Result layout

Every run is immutable-by-ID and stores machine-readable and human-readable output:

```text
.autobencher/
├── autobencher.sqlite3
└── runs/<run-id>/
    ├── manifest.json
    ├── summary.json
    ├── results.json
    ├── results.csv
    ├── report.md
    ├── provenance/
    │   ├── model-card.md
    │   └── model-card-check.json
    └── <benchmark-id>/
        ├── stdout.log
        ├── stderr.log
        ├── result.json
        └── ... native harness outputs ...
```

The SQLite database uses WAL mode and stores each benchmark result independently, so a crash or machine reboot does not erase completed work. `--resume` skips terminal rows and continues the remaining work.

## Model serving

### Model server managed by AutoBencher

The default `autobencher.toml` contains:

```toml
model = "openbmb/MiniCPM5-2B"
engine = "auto"
api_base = "http://127.0.0.1:8000"

[vllm]
port = 8000
tensor_parallel_size = 1
gpu_memory_utilization = 0.90
max_model_len = 131072
```

With `engine = "auto"`, AutoBencher first reuses an already-running endpoint. Otherwise it uses native vLLM when available; on Windows it falls back to a managed WSL2 vLLM installation (currently pinned to vLLM 0.29.0 and cached under the WSL user cache). AutoBencher waits for `/v1/models` before launching evaluation processes and terminates a model server that it started when the run ends. MiniCPM5 tool calling is enabled with the `minicpm5` parser in the default vLLM arguments. On WSL2, AutoBencher forces vLLM's V1 model runner because the current V2 runner requires UVA/pinned memory that WSL2 may not expose reliably; it also disables the optional FlashInfer top-k/top-p sampler so an incompatible system CUDA compiler cannot break server startup, while vLLM can still use GPU FlashAttention for attention.

### External server

Any OpenAI-compatible endpoint can be used, including vLLM, SGLang, llama.cpp server, a custom quantized runtime bridge, or a remote service:

```bash
autobencher run --engine external --api-base http://host:8000 --model served-model-name
```

If a key is required, set the environment variable named by `api_key_env` in `autobencher.toml` (default `AUTOBENCHER_API_KEY`). Secrets are passed to subprocesses and are not intentionally written into reports.

### LiteRT-LM Dynamic-v3 example

The repository includes a tested profile for the Dynamic-v3-inspired MiniCPM5 LiteRT artifacts published at `Tdamre/MiniCPM5-2B-LiteRT-LongContext`. Import the three artifacts into LiteRT-LM using the aliases expected by `config/litert-dynv3-server.json`:

```bash
litert-lm import /path/MiniCPM5-2B-LiteRT-DynV3Mixed-16k.litertlm minicpm5-dynv3-16k
litert-lm import /path/MiniCPM5-2B-LiteRT-DynV3Mixed-32k.litertlm minicpm5-dynv3-32k
litert-lm import /path/MiniCPM5-2B-LiteRT-DynV3Mixed-64k.litertlm minicpm5-dynv3-64k
```

Start LiteRT-LM and the small OpenAI-compatibility proxy in separate terminals:

```bash
litert-lm --config config/litert-dynv3-server.json serve --host 127.0.0.1 --port 9379
python scripts/litert_openai_proxy.py --listen-port 9380 --backend http://127.0.0.1:9379
```

Then run a sampled comparison, for example:

```bash
autobencher --config autobencher-litert-dynv3.toml run \
  --model minicpm5-dynv3-16k \
  --engine external \
  --api-base http://127.0.0.1:9380 \
  --only ifeval,gpqa-diamond --limit 1 --no-setup --no-card-check
```

Run the BF16 source with the matched deterministic profile before interpreting quant scores:

```bash
autobencher --config autobencher-bf16-parity.toml run \
  --only ifeval,gpqa-diamond --limit 1 --no-setup --no-card-check
```



LiteRT-LM 0.17.0 reads the OpenAI field `max_completion_tokens`, while the EvalScope OpenAI client used by this project supplies the legacy `max_tokens` field. `scripts/litert_openai_proxy.py` mirrors that field so completion limits are actually enforced without changing benchmark prompts. The profile also sets `eval_batch_size = 1`, which is important for a single local LiteRT engine; the general AutoBencher default remains 8.

The real-use 16K/32K/64K results are reported relative to the BF16 source in [`results/minicpm5-dynv3-2026-09-11/REPORT.md`](results/minicpm5-dynv3-2026-09-11/REPORT.md). The current evidence includes six BF16-first frozen short-context source-success checks retained by every quant, a matched GPQA-Diamond diagnostic that regresses from BF16 100% to 0% for all three quants, and NoLiMa long-context cases where BF16 succeeds but the quantized artifacts fail retrieval or hit the LiteRT CPU deadline. These are fidelity diagnostics, not full leaderboard reproductions.

## EvalScope isolation

`autobencher setup` creates isolated environments below `.autobencher/envs/` for mutually incompatible evaluator families (core, sandboxed code, IFEval, IFBench, Multi-IF, BFCL, SWE-bench, and Terminal-Bench). This separation is required because some current optional evaluator dependencies conflict—for example BFCL and Terminal-Bench cannot be resolved safely into one Python environment. `uv` is preferred and shares its package cache across those environments; Python `venv` + pip is the fallback.

The default suite uses EvalScope where it has maintained adapters, including several benchmark families that otherwise require substantially different harnesses. AutoBencher selects the correct isolated evaluator environment per benchmark automatically. Each benchmark still gets its own directory, timeout, logs and result record.

## Score extraction

AutoBencher never discards upstream artifacts. It records stdout/stderr and recursively inspects bounded JSON/JSONL result files for common primary metrics such as `score`, `acc`, `accuracy`, `prompt_level_strict`, `resolved_rate`, `pass@1`, `reward`, and `success_rate`.

If a process exits successfully but the result format is ambiguous, the row remains `completed` with an unset primary score and a note instructing the operator to inspect the raw report. It is not guessed.

Model-card reference values are stored on a 0–100 scale. Accuracy metrics emitted
on a 0–1 scale are normalized; performance metrics retain their original units.
Reference deltas are disabled by default. Set `compare_model_card_reference = true`
only after establishing that your full run matches the reference methodology.
The published reference remains visible for provenance when comparison is disabled.

## Reproducibility and provenance

Each run captures:

- model ID/path, engine, endpoint and seed
- exact selected benchmark IDs
- host OS/architecture/CPU/RAM snapshot
- Rust, Git, Python, uv, Docker and NVIDIA tool versions when available
- the command line used to start AutoBencher
- a snapshot and SHA-256 of the live MiniCPM5-2B model card
- per-benchmark commands, timestamps, exit codes and durations
- effective configuration and selected benchmark definitions, excluding API-key values
- GPQA question coverage and archived attempts for resumed evaluations
- all parsed metric candidates and their source file/log
- benchmark upstream URLs and licensing notes in the registry

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
cargo build --release
```

The CI workflow performs formatting, clippy, tests, and a release build on Linux, Windows, and macOS where applicable.

The Android HTTP bridge uses only Python's standard library:

```bash
python -m unittest discover -s scripts -p test_litert_android_server.py
```

The artifact preparation and INT4 repair tests additionally require the pinned
LiteRT conversion environment described in the device-validation guide.

## Licensing and third-party benchmarks

AutoBencher itself is Apache-2.0. Benchmark datasets and external harnesses retain **their own licenses and terms**. AutoBencher does not relicense or redistribute them. In particular, NoLiMa's official evaluation assets use an Adobe Research license with non-commercial restrictions, and OJBench's repository currently uses AGPL-3.0. Review each upstream benchmark's terms before downloading or executing it.

## Status philosophy

`completed` means the harness process completed and any implemented coverage gate
passed; it does not automatically imply official leaderboard comparability.
GPQA Diamond has an explicit full-split gate. Other benchmark splits remain
`unverified` until their own count/identity gates are implemented. `blocked`
means prerequisites or an exact public recipe were unavailable. `timed_out` and
`failed` are execution states and are deliberately not converted into model
correctness scores.


### Reference-score comparability

The MiniCPM5-2B model card marks some rows with a dagger as results taken from
Artificial Analysis, while the remaining rows are reported as OpenBMB internal
reproductions. A public adapter marked `verified=true` establishes the adapter's
identity, not identical prompting, decoding, judges, or leaderboard methodology.
Model-card comparison therefore requires an explicit configuration opt-in;
sampled and incomplete GPQA runs never receive a reference delta. Rows lacking
the exact slice, judge, simulator, search stack, or assets remain `blocked`.

That distinction is central to AutoBencher: **benchmark failures and model failures are not the same thing.**
