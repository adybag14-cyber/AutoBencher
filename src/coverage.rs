use crate::types::{BenchmarkSpec, EvaluationCoverage};
use serde_json::Value;
use std::{collections::BTreeSet, fs, path::Path};
use walkdir::WalkDir;

/// GPQA Diamond's complete public split contains 198 distinct questions.
/// A successful evaluator process (or a 100% progress bar) is not evidence
/// that the complete split was evaluated.
pub fn inspect(spec: &BenchmarkSpec, dir: &Path, limit: Option<usize>) -> EvaluationCoverage {
    let mut coverage = EvaluationCoverage {
        expected_samples: (spec.id == "gpqa-diamond").then_some(198),
        requested_limit: limit,
        ..Default::default()
    };
    let mut reports = Vec::new();
    for entry in WalkDir::new(dir.join("reports"))
        .max_depth(4)
        .into_iter()
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() || entry.path().extension().is_none_or(|x| x != "json") {
            continue;
        }
        let Ok(raw) = fs::read_to_string(entry.path()) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        if value.get("dataset_name").and_then(Value::as_str) == Some(spec.dataset.as_str()) {
            reports.push((entry.path().to_path_buf(), value));
        }
    }
    reports.sort_by(|a, b| a.0.cmp(&b.0));
    if reports.len() == 1 {
        let (path, report) = &reports[0];
        coverage.evidence.push(path.display().to_string());
        coverage.scored_samples = count(report.get("num"));
        let execution = report.get("execution_summary");
        coverage.requested_samples = count(execution.and_then(|v| v.get("requested")));
        coverage.succeeded_samples = count(execution.and_then(|v| v.get("succeeded")));
        coverage.errored_samples = count(execution.and_then(|v| v.get("errored")));
        if execution
            .and_then(|v| v.get("incomplete"))
            .and_then(Value::as_bool)
            == Some(true)
        {
            coverage
                .issues
                .push("evaluator reports incomplete execution".into());
        }
        if let (Some(scored), Some(succeeded)) =
            (coverage.scored_samples, coverage.succeeded_samples)
            && scored != succeeded
        {
            coverage.issues.push(format!(
                "aggregate count {scored} differs from succeeded count {succeeded}"
            ));
        }
    } else if coverage.expected_samples.is_some() {
        coverage.issues.push(format!(
            "expected one matching aggregate report, found {}",
            reports.len()
        ));
    }

    let mut unique = BTreeSet::new();
    let mut rows = 0usize;
    let mut files = 0usize;
    for entry in WalkDir::new(dir.join("reviews"))
        .max_depth(4)
        .into_iter()
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if !entry.file_type().is_file()
            || path.extension().is_none_or(|x| x != "jsonl")
            || !path.file_name().is_some_and(|x| {
                x.to_string_lossy()
                    .starts_with(&format!("{}_", spec.dataset))
            })
        {
            continue;
        }
        let Ok(raw) = fs::read_to_string(path) else {
            coverage
                .issues
                .push(format!("cannot read review file {}", path.display()));
            continue;
        };
        files += 1;
        coverage.evidence.push(path.display().to_string());
        for line in raw.lines().filter(|line| !line.trim().is_empty()) {
            rows += 1;
            let id = serde_json::from_str::<Value>(line)
                .ok()
                .and_then(|v| v.pointer("/sample_score/sample_id").cloned());
            match id {
                Some(id) if id.is_string() || id.is_number() => {
                    unique.insert(id.to_string());
                }
                _ => coverage
                    .issues
                    .push(format!("review row {rows} has no valid sample ID")),
            }
        }
    }
    if files > 0 {
        coverage.unique_samples = Some(unique.len());
        if rows != unique.len() {
            coverage.issues.push(format!(
                "{rows} review rows contain {} unique sample IDs",
                unique.len()
            ));
        }
    }
    classify(&mut coverage);
    coverage
}

fn count(value: Option<&Value>) -> Option<usize> {
    value
        .and_then(Value::as_u64)
        .and_then(|n| n.try_into().ok())
}

fn classify(c: &mut EvaluationCoverage) {
    let Some(full) = c.expected_samples else {
        c.scope = if c.requested_limit.is_some() {
            "sampled"
        } else {
            "unverified"
        }
        .into();
        return;
    };
    let requested = c.requested_limit.unwrap_or(full).min(full);
    if requested == 0 {
        c.issues
            .push("requested sample limit must be positive".into());
    }
    for (name, count) in [
        ("requested", c.requested_samples),
        ("scored", c.scored_samples),
        ("succeeded", c.succeeded_samples),
        ("distinct reviewed", c.unique_samples),
    ] {
        if count != Some(requested) {
            c.issues.push(format!(
                "expected {requested} {name} questions, observed {count:?}"
            ));
        }
    }
    if c.errored_samples != Some(0) {
        c.issues.push(format!(
            "expected zero evaluator errors, observed {:?}",
            c.errored_samples
        ));
    }
    c.scope = if !c.issues.is_empty() {
        "incomplete"
    } else if requested < full {
        "sampled"
    } else {
        "full"
    }
    .into();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completed(n: usize, limit: Option<usize>) -> EvaluationCoverage {
        EvaluationCoverage {
            expected_samples: Some(198),
            requested_limit: limit,
            requested_samples: Some(n),
            scored_samples: Some(n),
            succeeded_samples: Some(n),
            unique_samples: Some(n),
            errored_samples: Some(0),
            ..Default::default()
        }
    }
    #[test]
    fn one_question_cannot_pass_as_full_diamond() {
        let mut c = completed(1, None);
        classify(&mut c);
        assert_eq!(c.scope, "incomplete");
    }
    #[test]
    fn intentional_sample_is_labelled_sampled() {
        let mut c = completed(1, Some(1));
        classify(&mut c);
        assert_eq!(c.scope, "sampled");
    }
    #[test]
    fn all_198_distinct_questions_are_required() {
        let mut c = completed(198, None);
        classify(&mut c);
        assert_eq!(c.scope, "full");
        c.unique_samples = Some(197);
        classify(&mut c);
        assert_eq!(c.scope, "incomplete");
    }
    #[test]
    fn full_limit_is_full_and_missing_counts_fail_closed() {
        let mut c = completed(198, Some(198));
        classify(&mut c);
        assert_eq!(c.scope, "full");
        c.errored_samples = None;
        classify(&mut c);
        assert_eq!(c.scope, "incomplete");
    }

    #[test]
    fn aggregate_report_and_review_ids_must_agree() {
        let dir = tempfile::tempdir().unwrap();
        let spec = crate::registry::load_builtin()
            .unwrap()
            .benchmark
            .into_iter()
            .find(|s| s.id == "gpqa-diamond")
            .unwrap();
        fs::create_dir_all(dir.path().join("reports/model")).unwrap();
        fs::create_dir_all(dir.path().join("reviews/model")).unwrap();
        fs::write(dir.path().join("reports/model/gpqa_diamond.json"),
            r#"{"dataset_name":"gpqa_diamond","num":198,"execution_summary":{"requested":198,"succeeded":198,"errored":0,"incomplete":false}}"#).unwrap();
        let review = dir.path().join("reviews/model/gpqa_diamond_diamond.jsonl");
        let rows = (0..198)
            .map(|id| format!("{{\"sample_score\":{{\"sample_id\":{id}}}}}\n"))
            .collect::<String>();
        fs::write(&review, &rows).unwrap();
        assert_eq!(inspect(&spec, dir.path(), None).scope, "full");
        fs::write(
            &review,
            rows.replace("\"sample_id\":197", "\"sample_id\":196"),
        )
        .unwrap();
        let c = inspect(&spec, dir.path(), None);
        assert_eq!(c.scope, "incomplete");
        assert_eq!(c.unique_samples, Some(197));
        assert!(c.issues.iter().any(|s| s.contains("198 review rows")));
        fs::remove_file(&review).unwrap();
        assert_eq!(inspect(&spec, dir.path(), None).scope, "incomplete");
    }
}
