mod bootstrap;
mod card;
mod config;
mod coverage;
mod doctor;
mod engine;
mod metrics;
mod native;
mod process;
mod registry;
mod report;
mod resume;
mod runner;
mod store;
mod types;

use anyhow::{Context, Result, bail};
use chrono::Utc;
use clap::{Args, Parser, Subcommand};
use config::AppConfig;
use engine::EngineGuard;
use std::{collections::HashSet, fs, path::PathBuf};
use store::Store;
use types::{BenchStatus, RunManifest};
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(
    name = "autobencher",
    version,
    about = "Automatic, reproducible LLM benchmark orchestration"
)]
struct Cli {
    #[arg(long, default_value = "autobencher.toml", global = true)]
    config: PathBuf,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the complete MiniCPM5-2B model-card suite (default command).
    Run(RunArgs),
    /// List every registered benchmark and execution status.
    List(ListArgs),
    /// Install/update the isolated EvalScope environment.
    Setup {
        #[arg(long)]
        force: bool,
    },
    /// Inspect local runtime prerequisites and registry integrity.
    Doctor,
    /// Re-check that the model-card benchmark names still match the registry.
    VerifyCard,
    /// Regenerate reports for an existing run (latest by default).
    Report {
        #[arg(long)]
        run_id: Option<String>,
    },
}

#[derive(Debug, Args, Clone, Default)]
struct RunArgs {
    /// Hugging Face model id or local model path.
    #[arg(long)]
    model: Option<String>,
    /// vllm (AutoBencher starts it) or external (already-running OpenAI API).
    #[arg(long)]
    engine: Option<String>,
    #[arg(long)]
    api_base: Option<String>,
    /// Restrict to benchmark ids; repeat or comma-separate.
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,
    /// Restrict to one or more categories.
    #[arg(long, value_delimiter = ',')]
    category: Vec<String>,
    /// Limit samples per EvalScope/native adapter for smoke runs.
    #[arg(long)]
    limit: Option<usize>,
    /// Maximum parallel benchmark processes. Exclusive agent/code benches serialize automatically.
    #[arg(long)]
    parallel: Option<usize>,
    /// Resume a previous run id; terminally completed/blocked rows are retained.
    #[arg(long)]
    resume: Option<String>,
    /// Do not create/update the isolated EvalScope environment.
    #[arg(long)]
    no_setup: bool,
    /// Build and persist the complete execution plan without launching model or benchmarks.
    #[arg(long)]
    dry_run: bool,
    /// Return a non-zero exit code if anything is blocked, failed or timed out.
    #[arg(long)]
    strict: bool,
    /// Skip live model-card drift verification.
    #[arg(long)]
    no_card_check: bool,
}

#[derive(Debug, Args)]
struct ListArgs {
    #[arg(long)]
    category: Option<String>,
    #[arg(long)]
    verified_only: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "autobencher=info".into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let command = cli.command.unwrap_or(Command::Run(RunArgs::default()));
    let cfg = AppConfig::load(&cli.config)?;
    let registry = registry::load_builtin()?;

    match command {
        Command::Run(args) => run(cfg, registry, args).await,
        Command::List(args) => {
            list(&registry, args);
            Ok(())
        }
        Command::Setup { force } => {
            cfg.ensure_dirs()?;
            bootstrap::ensure_all(&cfg, force)?;
            println!("All isolated EvalScope harness environments are ready.");
            Ok(())
        }
        Command::Doctor => doctor_cmd(&cfg, &registry),
        Command::VerifyCard => verify_card_cmd(&cfg, &registry).await,
        Command::Report { run_id } => report_cmd(&cfg, &registry, run_id),
    }
}

async fn run(mut cfg: AppConfig, registry: types::SuiteRegistry, args: RunArgs) -> Result<()> {
    if let Some(model) = args.model.clone() {
        cfg.model = model;
    }
    if let Some(engine) = args.engine.clone() {
        cfg.engine = engine;
    }
    if let Some(api_base) = args.api_base.clone() {
        cfg.api_base = api_base;
    }
    if let Some(p) = args.parallel {
        cfg.parallel = p.max(1);
    }
    cfg.ensure_dirs()?;

    let selected = select(&registry, &args)?;
    if selected.is_empty() {
        bail!("no benchmarks matched the selection");
    }
    let selected_ids = selected.iter().map(|b| b.id.clone()).collect::<Vec<_>>();
    let run_id = args
        .resume
        .clone()
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let run_dir = cfg.workspace.join("runs").join(&run_id);
    fs::create_dir_all(&run_dir)?;
    fs::create_dir_all(run_dir.join("provenance"))?;

    let store = Store::open(&cfg.workspace)?;
    let previous = if args.resume.is_some() {
        store.load_results(&run_id)?
    } else {
        vec![]
    };
    if args.resume.is_some() {
        let path = run_dir.join("provenance/config.json");
        let old_cfg: AppConfig = serde_json::from_slice(&fs::read(&path)
            .context("resume requires saved configuration provenance; use a new run for an older run without it")?)?;
        resume::validate(&old_cfg, &cfg, &previous, args.limit)?;
        let registry_path = run_dir.join("provenance/benchmarks.json");
        if registry_path.is_file() {
            let old_specs: serde_json::Value = serde_json::from_slice(&fs::read(&registry_path)?)?;
            if old_specs != serde_json::to_value(&selected)? {
                bail!("resume requires the same selected benchmark definitions");
            }
        }
        let limit_path = run_dir.join("provenance/sample-limit.json");
        if limit_path.is_file() {
            let old_limit: Option<usize> = serde_json::from_slice(&fs::read(&limit_path)?)?;
            if old_limit != args.limit {
                bail!("resume requires the same sample limit");
            }
        }
        fs::copy(
            &path,
            run_dir.join("provenance").join(format!(
                "config-before-resume-{}.json",
                Utc::now().format("%Y%m%dT%H%M%S%f")
            )),
        )?;
    }
    // AppConfig records the key's environment-variable name, never its value.
    fs::write(
        run_dir.join("provenance/config.json"),
        serde_json::to_vec_pretty(&cfg)?,
    )?;
    fs::write(
        run_dir.join("provenance/benchmarks.json"),
        serde_json::to_vec_pretty(&selected)?,
    )?;
    fs::write(
        run_dir.join("provenance/sample-limit.json"),
        serde_json::to_vec_pretty(&args.limit)?,
    )?;

    if !args.no_card_check {
        match card::verify(&cfg.model_card_url, &registry, &run_dir.join("provenance")).await {
            Ok(check) if check.missing_registry_names.is_empty() => println!(
                "Model-card registry check: OK (SHA256 {})",
                &check.sha256[..16]
            ),
            Ok(check) => {
                tracing::warn!(missing=?check.missing_registry_names, "model-card registry drift detected")
            }
            Err(e) => {
                tracing::warn!(error=%e, "model-card verification unavailable; continuing with pinned registry")
            }
        }
    }

    let needs_evalscope = selected
        .iter()
        .any(|b| b.runner == types::RunnerKind::Evalscope);
    if needs_evalscope && !args.no_setup && !args.dry_run {
        bootstrap::ensure_evalscope_set(&cfg, &selected, false)?;
    }

    if args.resume.is_some() {
        let old: RunManifest = serde_json::from_slice(&fs::read(run_dir.join("manifest.json"))?)?;
        if old.model != cfg.model || old.seed != cfg.seed || old.api_base != cfg.api_base {
            bail!(
                "resume requires the same model, seed and endpoint; use a new run for a changed setup"
            );
        }
        fs::write(
            run_dir.join("provenance").join(format!(
                "manifest-before-resume-{}.json",
                Utc::now().format("%Y%m%dT%H%M%S")
            )),
            serde_json::to_vec_pretty(&old)?,
        )?;
    }
    let terminal: HashSet<_> = previous
        .iter()
        .filter(|r| {
            matches!(
                r.status,
                BenchStatus::Completed | BenchStatus::Blocked | BenchStatus::Skipped
            )
        })
        .map(|r| r.benchmark_id.clone())
        .collect();
    let pending = selected
        .into_iter()
        .filter(|b| !terminal.contains(&b.id))
        .collect::<Vec<_>>();

    let mut manifest = RunManifest {
        run_id: run_id.clone(),
        suite_id: registry.suite.id.clone(),
        suite_name: registry.suite.name.clone(),
        model: cfg.model.clone(),
        engine: cfg.engine.clone(),
        api_base: cfg.api_base.clone(),
        seed: cfg.seed,
        started_at: Utc::now().to_rfc3339(),
        finished_at: None,
        host: doctor::snapshot(),
        selected_benchmarks: selected_ids,
        command_line: std::env::args().collect(),
    };
    fs::write(
        run_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    store.create_run(&manifest)?;

    let mut engine = if args.dry_run || pending.is_empty() {
        None
    } else {
        Some(EngineGuard::start(&cfg, &cfg.model, &cfg.engine, &cfg.api_base).await?)
    };

    let ctx = runner::RunContext {
        run_id: run_id.clone(),
        cfg: cfg.clone(),
        store: store.clone(),
        model: cfg.model.clone(),
        api_base: cfg.api_base.clone(),
        limit: args.limit,
        run_dir: run_dir.clone(),
        dry_run: args.dry_run,
        reuse_predictions: args.resume.is_some(),
    };
    let _new = runner::run_all(ctx, pending, cfg.parallel).await?;
    if let Some(g) = &mut engine {
        g.stop().await;
    }

    manifest.finished_at = Some(Utc::now().to_rfc3339());
    fs::write(
        run_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    store.finish_run(&manifest)?;
    let results = store.load_results(&run_id)?;
    report::write_all(&run_dir, &manifest, &registry, &results)?;

    println!("\nRun {} complete", run_id);
    println!("Results: {}", run_dir.display());
    println!("Report:  {}", run_dir.join("report.md").display());
    println!(
        "Scored: {}/{}; blocked: {}; failed/timeouts: {}",
        results
            .iter()
            .filter(|r| r.status == BenchStatus::Completed && r.primary_score.is_some())
            .count(),
        manifest.selected_benchmarks.len(),
        results
            .iter()
            .filter(|r| r.status == BenchStatus::Blocked)
            .count(),
        results
            .iter()
            .filter(|r| matches!(r.status, BenchStatus::Failed | BenchStatus::TimedOut))
            .count()
    );

    if args.strict
        && results.iter().any(|r| {
            matches!(
                r.status,
                BenchStatus::Blocked | BenchStatus::Failed | BenchStatus::TimedOut
            )
        })
    {
        bail!("strict mode: one or more benchmarks were blocked, failed or timed out");
    }
    Ok(())
}

fn select(registry: &types::SuiteRegistry, args: &RunArgs) -> Result<Vec<types::BenchmarkSpec>> {
    let known: HashSet<_> = registry.benchmark.iter().map(|b| b.id.as_str()).collect();
    for id in &args.only {
        if !known.contains(id.as_str()) {
            bail!("unknown benchmark id: {id}");
        }
    }
    let cats = args
        .category
        .iter()
        .map(|x| x.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    Ok(registry
        .benchmark
        .iter()
        .filter(|b| {
            (args.only.is_empty() || args.only.contains(&b.id))
                && (cats.is_empty() || cats.contains(&b.category.to_ascii_lowercase()))
        })
        .cloned()
        .collect())
}

fn list(registry: &types::SuiteRegistry, args: ListArgs) {
    println!(
        "{} — {} benchmarks",
        registry.suite.name,
        registry.benchmark.len()
    );
    println!(
        "{:<28} {:<24} {:<10} {:<9} NAME",
        "ID", "CATEGORY", "RUNNER", "VERIFIED"
    );
    for b in &registry.benchmark {
        if args.verified_only && !b.verified {
            continue;
        }
        if let Some(cat) = &args.category
            && !b.category.eq_ignore_ascii_case(cat)
        {
            continue;
        }
        println!(
            "{:<28} {:<24} {:<10} {:<9} {}",
            b.id,
            b.category,
            format!("{:?}", b.runner).to_lowercase(),
            b.verified,
            b.name
        );
    }
}

fn doctor_cmd(cfg: &AppConfig, registry: &types::SuiteRegistry) -> Result<()> {
    registry::validate(registry)?;
    let host = doctor::snapshot();
    println!(
        "AutoBencher doctor\n  registry: {}/{} rows OK\n  host: {} / {} / {} CPUs / {:.1} GiB RAM",
        registry.benchmark.len(),
        registry.suite.expected_count,
        host.os,
        host.arch,
        host.cpu_count,
        host.total_memory_bytes as f64 / 1024f64.powi(3)
    );
    for (name, value) in [
        ("rustc", host.rustc),
        ("git", host.git),
        ("python", host.python),
        ("uv", host.uv),
        ("docker", host.docker),
        ("GPU", host.nvidia_smi),
    ] {
        println!(
            "  {name:<8}: {}",
            value
                .unwrap_or_else(|| "NOT FOUND".into())
                .replace('\n', " | ")
        );
    }
    println!("  Harness environments:");
    for line in bootstrap::status_lines(cfg) {
        println!("    {line}");
    }
    println!(
        "  verified recipes: {}/{}",
        registry.benchmark.iter().filter(|b| b.verified).count(),
        registry.benchmark.len()
    );
    Ok(())
}

async fn verify_card_cmd(cfg: &AppConfig, registry: &types::SuiteRegistry) -> Result<()> {
    cfg.ensure_dirs()?;
    let out = card::verify(
        &cfg.model_card_url,
        registry,
        &cfg.workspace.join("provenance"),
    )
    .await?;
    println!("model card SHA256: {}", out.sha256);
    if out.missing_registry_names.is_empty() {
        println!(
            "all {} registered benchmark names are present",
            out.registry_count
        );
        Ok(())
    } else {
        bail!(
            "model card drift: missing names: {}",
            out.missing_registry_names.join(", ")
        )
    }
}

fn report_cmd(
    cfg: &AppConfig,
    registry: &types::SuiteRegistry,
    run_id: Option<String>,
) -> Result<()> {
    let store = Store::open(&cfg.workspace)?;
    let id = run_id.or(store.latest_run_id()?).context("no runs found")?;
    let run_dir = cfg.workspace.join("runs").join(&id);
    let manifest: RunManifest = serde_json::from_slice(&fs::read(run_dir.join("manifest.json"))?)?;
    let results = store.load_results(&id)?;
    report::write_all(&run_dir, &manifest, registry, &results)?;
    println!("{}", run_dir.join("report.md").display());
    Ok(())
}
