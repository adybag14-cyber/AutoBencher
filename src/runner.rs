use crate::{
    config::AppConfig,
    metrics,
    native::{self, NativePlan},
    process::{self, ProcessSpec},
    store::Store,
    types::{BenchStatus, BenchmarkResult, BenchmarkSpec, RunnerKind},
};
use anyhow::{Context, Result};
use chrono::Utc;
use futures::{StreamExt, stream::FuturesUnordered};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{RwLock, Semaphore};

#[derive(Clone)]
pub struct RunContext {
    pub run_id: String,
    pub cfg: AppConfig,
    pub store: Store,
    pub model: String,
    pub api_base: String,
    pub limit: Option<usize>,
    pub run_dir: PathBuf,
    pub dry_run: bool,
}

pub async fn run_all(
    ctx: RunContext,
    specs: Vec<BenchmarkSpec>,
    parallel: usize,
) -> Result<Vec<BenchmarkResult>> {
    let sem = Arc::new(Semaphore::new(parallel.max(1)));
    let isolation = Arc::new(RwLock::new(()));
    let mut tasks = FuturesUnordered::new();

    for spec in specs {
        let ctx = ctx.clone();
        let sem = Arc::clone(&sem);
        let isolation = Arc::clone(&isolation);
        tasks.push(tokio::spawn(async move {
            let _permit = sem
                .acquire_owned()
                .await
                .context("scheduler semaphore closed")?;
            if spec.exclusive {
                let _guard = isolation.write().await;
                run_one(&ctx, &spec).await
            } else {
                let _guard = isolation.read().await;
                run_one(&ctx, &spec).await
            }
        }));
    }

    let mut out = Vec::new();
    while let Some(joined) = tasks.next().await {
        match joined {
            Ok(Ok(result)) => out.push(result),
            Ok(Err(e)) => {
                tracing::error!(error=%e, "benchmark task failed before result persistence")
            }
            Err(e) => tracing::error!(error=%e, "benchmark task panicked"),
        }
    }
    out.sort_by(|a, b| a.benchmark_id.cmp(&b.benchmark_id));
    Ok(out)
}

async fn run_one(ctx: &RunContext, spec: &BenchmarkSpec) -> Result<BenchmarkResult> {
    let result_dir = ctx.run_dir.join(&spec.id);
    tokio::fs::create_dir_all(&result_dir).await?;
    let started = Utc::now();
    let tick = Instant::now();
    tracing::info!(benchmark=%spec.name, id=%spec.id, "starting benchmark");

    let plan = match spec.runner {
        RunnerKind::Evalscope => NativePlan::Command(native::evalscope_compat(
            &ctx.cfg,
            spec,
            &ctx.model,
            &ctx.api_base,
            &result_dir,
            ctx.limit,
        )),
        RunnerKind::Native => native::plan(
            spec,
            &ctx.cfg,
            &ctx.model,
            &ctx.api_base,
            &result_dir,
            ctx.limit,
        ),
    };

    if let NativePlan::Blocked(reason) = plan {
        let result = make_terminal_result(
            ctx,
            spec,
            started.to_rfc3339(),
            tick.elapsed().as_secs_f64(),
            BenchStatus::Blocked,
            None,
            None,
            None,
            Some(reason),
            &result_dir,
            vec![],
        );
        persist(ctx, &result_dir, &result).await?;
        tracing::warn!(benchmark=%spec.name, "blocked: {}", result.error.as_deref().unwrap_or("unknown"));
        return Ok(result);
    }

    let NativePlan::Command(command) = plan else {
        unreachable!()
    };
    let persisted_command = command.display.clone();
    if ctx.dry_run {
        let result = make_terminal_result(
            ctx,
            spec,
            started.to_rfc3339(),
            tick.elapsed().as_secs_f64(),
            BenchStatus::Skipped,
            Some(persisted_command.clone()),
            None,
            None,
            None,
            &result_dir,
            vec!["dry-run: command was not executed".into()],
        );
        persist(ctx, &result_dir, &result).await?;
        return Ok(result);
    }

    let stdout_log = result_dir.join("stdout.log");
    let stderr_log = result_dir.join("stderr.log");
    let proc_spec = ProcessSpec {
        invocation: command.invocation.clone(),
        display_command: persisted_command.clone(),
        cwd: result_dir.clone(),
        timeout: Duration::from_secs(spec.timeout_minutes * 60),
        env: vec![
            ("AUTOBENCHER_MODEL".into(), ctx.model.clone()),
            ("AUTOBENCHER_API_BASE".into(), ctx.api_base.clone()),
            ("AUTOBENCHER_RUN_ID".into(), ctx.run_id.clone()),
            ("AUTOBENCHER_CHILD_API_KEY".into(), ctx.cfg.api_key()),
            ("PYTHONUNBUFFERED".into(), "1".into()),
        ],
        stdout_log: stdout_log.clone(),
        stderr_log: stderr_log.clone(),
    };

    let outcome = process::run_process(proc_spec).await;
    let duration = tick.elapsed().as_secs_f64();
    let finished = Utc::now().to_rfc3339();
    let result = match outcome {
        Ok(outcome) => {
            let (mut m, candidates) =
                metrics::extract(&outcome.stdout_tail, &outcome.stderr_tail, &result_dir);
            normalize_metric_scale(&mut m, spec.reference_score);
            let primary = metrics::choose_primary(&m);
            let status = if outcome.timed_out {
                BenchStatus::TimedOut
            } else if outcome.exit_code == Some(0) {
                BenchStatus::Completed
            } else {
                BenchStatus::Failed
            };
            let mut notes = Vec::new();
            if status == BenchStatus::Completed && primary.is_none() {
                notes.push("process completed successfully, but no unambiguous primary score was parsed; inspect raw logs/reports".into());
            }
            if !spec.verified {
                notes.push("registry marks this recipe as provisional/unverified against the exact model-card methodology".into());
            }
            if let Some(note) = &spec.license_note {
                notes.push(note.clone());
            }
            BenchmarkResult {
                run_id: ctx.run_id.clone(),
                benchmark_id: spec.id.clone(),
                benchmark_name: spec.name.clone(),
                category: spec.category.clone(),
                runner: spec.runner,
                status,
                started_at: started.to_rfc3339(),
                finished_at: finished,
                duration_seconds: duration,
                exit_code: outcome.exit_code,
                primary_score: primary,
                reference_score: spec.reference_score,
                delta_from_reference: primary.map(|p| p - spec.reference_score),
                metrics: m,
                metric_candidates: candidates,
                command: Some(persisted_command.clone()),
                stdout_log: Some(path_string(&stdout_log)),
                stderr_log: Some(path_string(&stderr_log)),
                result_dir: path_string(&result_dir),
                error: if outcome.timed_out {
                    Some(format!("timeout after {} minutes", spec.timeout_minutes))
                } else if outcome.exit_code != Some(0) {
                    Some(format!("process exited with {:?}", outcome.exit_code))
                } else {
                    None
                },
                notes,
            }
        }
        Err(e) => make_terminal_result(
            ctx,
            spec,
            started.to_rfc3339(),
            duration,
            BenchStatus::Failed,
            Some(persisted_command),
            None,
            None,
            Some(format!("{e:#}")),
            &result_dir,
            vec![],
        ),
    };
    persist(ctx, &result_dir, &result).await?;
    tracing::info!(benchmark=%spec.name, status=%result.status.as_str(), score=?result.primary_score, elapsed=result.duration_seconds, "benchmark finished");
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn make_terminal_result(
    ctx: &RunContext,
    spec: &BenchmarkSpec,
    started_at: String,
    duration: f64,
    status: BenchStatus,
    command: Option<String>,
    exit_code: Option<i32>,
    score: Option<f64>,
    error: Option<String>,
    result_dir: &Path,
    notes: Vec<String>,
) -> BenchmarkResult {
    BenchmarkResult {
        run_id: ctx.run_id.clone(),
        benchmark_id: spec.id.clone(),
        benchmark_name: spec.name.clone(),
        category: spec.category.clone(),
        runner: spec.runner,
        status,
        started_at,
        finished_at: Utc::now().to_rfc3339(),
        duration_seconds: duration,
        exit_code,
        primary_score: score,
        reference_score: spec.reference_score,
        delta_from_reference: score.map(|s| s - spec.reference_score),
        metrics: Default::default(),
        metric_candidates: vec![],
        command,
        stdout_log: None,
        stderr_log: None,
        result_dir: path_string(result_dir),
        error,
        notes,
    }
}

async fn persist(ctx: &RunContext, dir: &Path, result: &BenchmarkResult) -> Result<()> {
    let json = serde_json::to_vec_pretty(result)?;
    tokio::fs::write(dir.join("result.json"), json).await?;
    ctx.store.save_result(result)?;
    Ok(())
}

fn normalize_metric_scale(metrics: &mut std::collections::BTreeMap<String, f64>, reference: f64) {
    if reference <= 1.0 {
        return;
    }
    for v in metrics.values_mut() {
        if (0.0..=1.0).contains(v) {
            *v *= 100.0;
        }
    }
}

fn path_string(p: &Path) -> String {
    p.to_string_lossy().to_string()
}
