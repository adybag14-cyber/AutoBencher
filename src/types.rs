use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SuiteRegistry {
    pub suite: SuiteMeta,
    #[serde(default)]
    pub benchmark: Vec<BenchmarkSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SuiteMeta {
    pub id: String,
    pub name: String,
    pub source_url: String,
    pub expected_count: usize,
    pub reference_model: String,
    pub reference_average: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BenchmarkSpec {
    pub id: String,
    pub name: String,
    pub category: String,
    pub runner: RunnerKind,
    pub dataset: String,
    #[serde(default)]
    pub evalscope_args: Vec<String>,
    pub reference_score: f64,
    pub timeout_minutes: u64,
    pub exclusive: bool,
    pub verified: bool,
    pub upstream: Option<String>,
    pub license_note: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunnerKind {
    Evalscope,
    Native,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BenchStatus {
    Pending,
    Running,
    Completed,
    Failed,
    TimedOut,
    Blocked,
    Skipped,
}

impl BenchStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
            Self::Blocked => "blocked",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricCandidate {
    pub key: String,
    pub value: f64,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub run_id: String,
    pub benchmark_id: String,
    pub benchmark_name: String,
    pub category: String,
    pub runner: RunnerKind,
    pub status: BenchStatus,
    pub started_at: String,
    pub finished_at: String,
    pub duration_seconds: f64,
    #[serde(default)]
    pub total_attempt_duration_seconds: Option<f64>,
    pub exit_code: Option<i32>,
    pub primary_score: Option<f64>,
    pub reference_score: f64,
    pub delta_from_reference: Option<f64>,
    pub metrics: BTreeMap<String, f64>,
    pub metric_candidates: Vec<MetricCandidate>,
    pub command: Option<String>,
    pub stdout_log: Option<String>,
    pub stderr_log: Option<String>,
    pub result_dir: String,
    pub error: Option<String>,
    pub notes: Vec<String>,
    #[serde(default)]
    pub coverage: EvaluationCoverage,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvaluationCoverage {
    pub expected_samples: Option<usize>,
    pub requested_limit: Option<usize>,
    pub requested_samples: Option<usize>,
    pub succeeded_samples: Option<usize>,
    pub scored_samples: Option<usize>,
    pub unique_samples: Option<usize>,
    pub errored_samples: Option<usize>,
    pub scope: String,
    pub evidence: Vec<String>,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub run_id: String,
    pub suite_id: String,
    pub suite_name: String,
    pub model: String,
    pub engine: String,
    pub api_base: String,
    pub seed: u64,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub host: HostSnapshot,
    pub selected_benchmarks: Vec<String>,
    pub command_line: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HostSnapshot {
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub cpu_count: usize,
    pub total_memory_bytes: u64,
    pub rustc: Option<String>,
    pub git: Option<String>,
    pub python: Option<String>,
    pub uv: Option<String>,
    pub docker: Option<String>,
    pub nvidia_smi: Option<String>,
}
