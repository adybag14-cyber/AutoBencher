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

## EvalScope isolation

`autobencher setup` creates isolated environments below `.autobencher/envs/` for mutually incompatible evaluator families (core, sandboxed code, IFEval, IFBench, Multi-IF, BFCL, SWE-bench, and Terminal-Bench). This separation is required because some current optional evaluator dependencies conflict—for example BFCL and Terminal-Bench cannot be resolved safely into one Python environment. `uv` is preferred and shares its package cache across those environments; Python `venv` + pip is the fallback.

The default suite uses EvalScope where it has maintained adapters, including several benchmark families that otherwise require substantially different harnesses. AutoBencher selects the correct isolated evaluator environment per benchmark automatically. Each benchmark still gets its own directory, timeout, logs and result record.

## Score extraction

AutoBencher never discards upstream artifacts. It records stdout/stderr and recursively inspects bounded JSON/JSONL result files for common primary metrics such as `score`, `acc`, `accuracy`, `prompt_level_strict`, `resolved_rate`, `pass@1`, `reward`, and `success_rate`.

If a process exits successfully but the result format is ambiguous, the row remains `completed` with an unset primary score and a note instructing the operator to inspect the raw report. It is not guessed.

Model-card reference values are stored on a 0–100 scale. Metrics emitted on a 0–1 scale are normalized before reference deltas are computed.

## Reproducibility and provenance

Each run captures:

- model ID/path, engine, endpoint and seed
- exact selected benchmark IDs
- host OS/architecture/CPU/RAM snapshot
- Rust, Git, Python, uv, Docker and NVIDIA tool versions when available
- the command line used to start AutoBencher
- a snapshot and SHA-256 of the live MiniCPM5-2B model card
- per-benchmark commands, timestamps, exit codes and durations
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

## Licensing and third-party benchmarks

AutoBencher itself is Apache-2.0. Benchmark datasets and external harnesses retain **their own licenses and terms**. AutoBencher does not relicense or redistribute them. In particular, NoLiMa's official evaluation assets use an Adobe Research license with non-commercial restrictions, and OJBench's repository currently uses AGPL-3.0. Review each upstream benchmark's terms before downloading or executing it.

## Status philosophy

`completed` means the harness process completed; it does not automatically imply official leaderboard comparability. `blocked` means prerequisites or an exact public recipe were unavailable. `timed_out` and `failed` are infrastructure/execution states and are deliberately not converted into model correctness scores.


### Reference-score comparability

The MiniCPM5-2B model card marks some rows with a dagger as results taken from Artificial Analysis, while the remaining rows are reported as OpenBMB internal reproductions. AutoBencher stores those model-card values as provenance/reference targets, not as a guarantee that a public local harness is byte-for-byte identical to the unpublished evaluation environment. A public adapter can therefore be `verified=true` while its delta to the model-card reference remains informational. Rows for which the exact reported slice, judge, simulator, search stack or assets cannot be reproduced are held as `blocked` rather than silently scored with a different task.

That distinction is central to AutoBencher: **benchmark failures and model failures are not the same thing.**
