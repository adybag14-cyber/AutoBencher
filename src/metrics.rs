use crate::types::MetricCandidate;
use regex::Regex;
use serde_json::Value;
use std::{collections::BTreeMap, fs, path::Path};
use walkdir::WalkDir;

const SCORE_KEYS: &[&str] = &[
    "primary_score",
    "score",
    "acc",
    "accuracy",
    "overall_accuracy",
    "prompt_level_strict",
    "resolved_rate",
    "resolve_rate",
    "pass@1",
    "pass_at_1",
    "reward",
    "success_rate",
    "mean",
    "average",
    "avg_score",
];

pub fn extract(
    stdout: &str,
    stderr: &str,
    result_dir: &Path,
) -> (BTreeMap<String, f64>, Vec<MetricCandidate>) {
    let mut candidates = Vec::new();
    extract_text(stdout, "stdout", &mut candidates);
    extract_text(stderr, "stderr", &mut candidates);

    for ent in WalkDir::new(result_dir)
        .max_depth(5)
        .into_iter()
        .filter_entry(|entry| !(entry.file_type().is_dir() && entry.file_name() == "attempts"))
        .filter_map(Result::ok)
    {
        if !ent.file_type().is_file() {
            continue;
        }
        let path = ent.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext, "json" | "jsonl") {
            continue;
        }
        let Ok(meta) = fs::metadata(path) else {
            continue;
        };
        if meta.len() > 32 * 1024 * 1024 {
            continue;
        }
        let Ok(raw) = fs::read_to_string(path) else {
            continue;
        };
        if ext == "json" {
            if let Ok(v) = serde_json::from_str::<Value>(&raw) {
                // A resumed run can still have our previous result.json here.
                // It is orchestration metadata, not fresh evaluator evidence.
                if v.get("run_id").is_some()
                    && v.get("benchmark_id").is_some()
                    && v.get("result_dir").is_some()
                {
                    continue;
                }
                let source = path.display().to_string();
                extract_known_report_shapes(&v, &source, &mut candidates);
                extract_json(&v, "", &source, &mut candidates);
            }
        } else {
            for (i, line) in raw.lines().take(100_000).enumerate() {
                if let Ok(v) = serde_json::from_str::<Value>(line) {
                    extract_json(
                        &v,
                        "",
                        &format!("{}:line{}", path.display(), i + 1),
                        &mut candidates,
                    );
                }
            }
        }
    }

    dedup(&mut candidates);
    let mut metrics = BTreeMap::new();
    for c in &candidates {
        metrics.entry(c.key.clone()).or_insert(c.value);
    }
    (metrics, candidates)
}

fn extract_known_report_shapes(v: &Value, source: &str, out: &mut Vec<MetricCandidate>) {
    // EvalScope >=1.11 report schema: the top-level metrics array contains the
    // aggregate benchmark score. Prefer the metric matching
    // primary_metric_identity; falling back to the first metric is safe for
    // single-metric reports and avoids accidentally selecting a subset score.
    let Some(metrics) = v.get("metrics").and_then(Value::as_array) else {
        return;
    };
    let primary_name = v
        .get("primary_metric_identity")
        .and_then(|x| x.get("name"))
        .and_then(Value::as_str);
    let selected = primary_name
        .and_then(|name| {
            metrics.iter().find(|metric| {
                metric
                    .get("identity")
                    .and_then(|x| x.get("name"))
                    .and_then(Value::as_str)
                    == Some(name)
            })
        })
        .or_else(|| metrics.first());
    if let Some(score) = selected
        .and_then(|metric| metric.get("score"))
        .and_then(Value::as_f64)
    {
        out.push(MetricCandidate {
            key: "primary_score".into(),
            value: score,
            source: source.into(),
        });
    }
}

fn extract_text(text: &str, source: &str, out: &mut Vec<MetricCandidate>) {
    let re = Regex::new(r"(?i)(accuracy|acc|score|resolved[_ -]?rate|pass@1|reward|success[_ -]?rate|prompt[_ -]?level[_ -]?strict|average|mean)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)(%)?").unwrap();
    for cap in re.captures_iter(text) {
        let mut value = cap[2].parse::<f64>().unwrap_or(f64::NAN);
        if cap.get(3).is_some() {
            value /= 100.0;
        }
        if value.is_finite() {
            out.push(MetricCandidate {
                key: normalize(&cap[1]),
                value,
                source: source.into(),
            });
        }
    }
}

fn extract_json(v: &Value, prefix: &str, source: &str, out: &mut Vec<MetricCandidate>) {
    match v {
        Value::Object(map) => {
            for (k, child) in map {
                let path = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                if let Some(n) = child.as_f64() {
                    let nk = normalize(k);
                    if SCORE_KEYS.iter().any(|needle| nk == normalize(needle)) {
                        out.push(MetricCandidate {
                            key: path.clone(),
                            value: n,
                            source: source.into(),
                        });
                    }
                }
                extract_json(child, &path, source, out);
            }
        }
        Value::Array(items) => {
            for child in items.iter().take(10_000) {
                extract_json(child, prefix, source, out);
            }
        }
        _ => {}
    }
}

fn normalize(s: &str) -> String {
    s.to_ascii_lowercase().replace([' ', '-'], "_")
}

fn dedup(items: &mut Vec<MetricCandidate>) {
    let mut seen = std::collections::HashSet::new();
    items.retain(|m| seen.insert((m.key.clone(), m.value.to_bits(), m.source.clone())));
}

pub fn choose_primary(metrics: &BTreeMap<String, f64>) -> Option<f64> {
    for key in [
        "primary_score",
        "score",
        "accuracy",
        "acc",
        "overall_accuracy",
        "prompt_level_strict",
        "resolved_rate",
        "pass@1",
        "reward",
        "success_rate",
        "average",
        "mean",
    ] {
        if let Some(v) = metrics.get(key) {
            return Some(*v);
        }
        if let Some((_, v)) = metrics
            .iter()
            .find(|(k, _)| k.ends_with(&format!(".{key}")))
        {
            return Some(*v);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evalscope_primary_metric_beats_subset_scores() {
        let v: Value = serde_json::json!({
            "primary_metric_identity": {"name": "accuracy"},
            "metrics": [{
                "identity": {"name": "accuracy"},
                "score": 0.5714,
                "categories": [{"subsets": [{"score": 1.0}]}]
            }]
        });
        let mut out = Vec::new();
        extract_known_report_shapes(&v, "test", &mut out);
        assert_eq!(out[0].key, "primary_score");
        assert!((out[0].value - 0.5714).abs() < 1e-9);
    }

    #[test]
    fn previous_autobencher_results_do_not_contaminate_resumed_scores() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("result.json"), r#"{"run_id":"old","benchmark_id":"gpqa-diamond","result_dir":"old","primary_score":100}"#).unwrap();
        let report = dir.path().join("reports");
        std::fs::create_dir(&report).unwrap();
        std::fs::write(report.join("gpqa.json"), r#"{"primary_metric_identity":{"name":"accuracy"},"metrics":[{"identity":{"name":"accuracy"},"score":0.5}]}"#).unwrap();
        let (metrics, _) = extract("", "", dir.path());
        assert_eq!(choose_primary(&metrics), Some(0.5));
    }
}
