use crate::{config::AppConfig, types::BenchmarkResult};
use anyhow::{Result, bail};
use serde_json::Value;

/// Validate inputs that can change cached predictions before replacing any
/// provenance file or starting/updating a harness environment.
pub fn validate(
    old: &AppConfig,
    next: &AppConfig,
    previous: &[BenchmarkResult],
    limit: Option<usize>,
) -> Result<()> {
    if old.model != next.model || old.seed != next.seed || old.api_base != next.api_base {
        bail!(
            "resume requires the same model, seed and endpoint; use a new run for a changed setup"
        );
    }
    let old_generation: Value = serde_json::from_str(&old.generation_config)?;
    let new_generation: Value = serde_json::from_str(&next.generation_config)?;
    if old_generation != new_generation {
        bail!("resume requires the same decoding configuration");
    }
    if old.evalscope_spec != next.evalscope_spec {
        bail!("resume requires the same EvalScope requirement");
    }
    if previous.iter().any(|r| r.coverage.requested_limit != limit) {
        bail!("resume requires the same sample limit");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AppConfig {
        toml::from_str(include_str!("../autobencher-gpqa-full-bf16.toml")).unwrap()
    }

    #[test]
    fn changed_predictions_are_rejected_but_timeout_can_increase() {
        let old = config();
        let mut new = old.clone();
        new.benchmark_timeout_minutes = Some(480);
        assert!(validate(&old, &new, &[], None).is_ok());
        new.generation_config = r#"{"temperature":1}"#.into();
        assert!(validate(&old, &new, &[], None).is_err());
        let mut new = old.clone();
        new.model = "another-model".into();
        assert!(validate(&old, &new, &[], None).is_err());
        let mut new = old.clone();
        new.evalscope_spec = "evalscope==0.0.1".into();
        assert!(validate(&old, &new, &[], None).is_err());
    }

    #[test]
    fn json_whitespace_does_not_change_the_protocol() {
        let old = config();
        let mut new = old.clone();
        let value: Value = serde_json::from_str(&new.generation_config).unwrap();
        new.generation_config = serde_json::to_string_pretty(&value).unwrap();
        assert!(validate(&old, &new, &[], None).is_ok());
    }
}
