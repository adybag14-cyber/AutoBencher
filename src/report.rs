use crate::types::{BenchStatus, BenchmarkResult, RunManifest, SuiteRegistry};
use anyhow::Result;
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Debug, Serialize)]
struct Summary<'a> {
    run_id: &'a str,
    model: &'a str,
    total_registered: usize,
    attempted: usize,
    completed: usize,
    scored: usize,
    failed: usize,
    timed_out: usize,
    blocked: usize,
    skipped: usize,
    score_mean: Option<f64>,
    reference_mean_same_rows: Option<f64>,
    category_means: BTreeMap<String, f64>,
}

pub fn write_all(
    run_dir: &Path,
    manifest: &RunManifest,
    registry: &SuiteRegistry,
    results: &[BenchmarkResult],
) -> Result<()> {
    let scored: Vec<_> = results
        .iter()
        .filter(|r| r.status == BenchStatus::Completed && r.primary_score.is_some())
        .collect();
    let score_mean = mean(scored.iter().filter_map(|r| r.primary_score));
    let reference_mean_same_rows = mean(scored.iter().map(|r| r.reference_score));
    let mut cats: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for r in &scored {
        cats.entry(r.category.clone())
            .or_default()
            .push(r.primary_score.unwrap());
    }
    let category_means = cats
        .into_iter()
        .filter_map(|(k, v)| mean(v.into_iter()).map(|m| (k, m)))
        .collect();
    let summary = Summary {
        run_id: &manifest.run_id,
        model: &manifest.model,
        total_registered: registry.benchmark.len(),
        attempted: results.len(),
        completed: count(results, BenchStatus::Completed),
        scored: scored.len(),
        failed: count(results, BenchStatus::Failed),
        timed_out: count(results, BenchStatus::TimedOut),
        blocked: count(results, BenchStatus::Blocked),
        skipped: count(results, BenchStatus::Skipped),
        score_mean,
        reference_mean_same_rows,
        category_means,
    };
    fs::write(
        run_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    fs::write(
        run_dir.join("results.json"),
        serde_json::to_vec_pretty(results)?,
    )?;
    fs::write(
        run_dir.join("report.md"),
        markdown(manifest, registry, results, &summary),
    )?;
    write_csv(&run_dir.join("results.csv"), results)?;
    Ok(())
}

fn markdown(
    manifest: &RunManifest,
    registry: &SuiteRegistry,
    results: &[BenchmarkResult],
    summary: &Summary<'_>,
) -> String {
    let mut s = format!(
        "# AutoBencher report\n\n- Run: `{}`\n- Model: `{}`\n- Engine: `{}`\n- Suite: {} ({} registered rows)\n- Selected: {}\n- Scored: {}/{} selected\n- Blocked: {} · Failed: {} · Timed out: {}\n\n",
        manifest.run_id,
        manifest.model,
        manifest.engine,
        registry.suite.name,
        registry.benchmark.len(),
        manifest.selected_benchmarks.len(),
        summary.scored,
        manifest.selected_benchmarks.len(),
        summary.blocked,
        summary.failed,
        summary.timed_out
    );
    if let Some(m) = summary.score_mean {
        s.push_str(&format!("Mean over scored rows: **{m:.3}**\n\n"));
    }
    s.push_str("| Category | Benchmark | Status | Score | Reference | Delta |\n|---|---|---:|---:|---:|---:|\n");
    let mut rows: Vec<_> = results.iter().collect();
    rows.sort_by_key(|r| (&r.category, &r.benchmark_name));
    for r in rows {
        s.push_str(&format!(
            "| {} | {} | {} | {} | {:.1} | {} |\n",
            r.category,
            r.benchmark_name,
            r.status.as_str(),
            fmt(r.primary_score),
            r.reference_score,
            fmt(r.delta_from_reference)
        ));
    }
    s.push_str("\n> A blocked row is deliberately not scored. AutoBencher does not replace unavailable official harnesses with proxy metrics.\n");
    s
}

fn write_csv(path: &Path, results: &[BenchmarkResult]) -> Result<()> {
    let mut out = String::from(
        "category,benchmark_id,benchmark,status,score,reference,delta,duration_seconds\n",
    );
    for r in results {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{:.3}\n",
            csv(&r.category),
            csv(&r.benchmark_id),
            csv(&r.benchmark_name),
            r.status.as_str(),
            r.primary_score.map(|x| x.to_string()).unwrap_or_default(),
            r.reference_score,
            r.delta_from_reference
                .map(|x| x.to_string())
                .unwrap_or_default(),
            r.duration_seconds
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn csv(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}
fn fmt(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "—".into())
}
fn count(r: &[BenchmarkResult], s: BenchStatus) -> usize {
    r.iter().filter(|x| x.status == s).count()
}
fn mean<I: Iterator<Item = f64>>(it: I) -> Option<f64> {
    let v: Vec<_> = it.collect();
    if v.is_empty() {
        None
    } else {
        Some(v.iter().sum::<f64>() / v.len() as f64)
    }
}
