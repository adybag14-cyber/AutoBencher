use crate::types::SuiteRegistry;
use anyhow::{Context, Result, bail};

pub fn load_builtin() -> Result<SuiteRegistry> {
    let raw = include_str!("../config/minicpm5-2b.toml");
    let registry: SuiteRegistry =
        toml::from_str(raw).context("parse built-in benchmark registry")?;
    validate(&registry)?;
    Ok(registry)
}

pub fn validate(registry: &SuiteRegistry) -> Result<()> {
    if registry.benchmark.len() != registry.suite.expected_count {
        bail!(
            "registry count mismatch: expected {}, found {}",
            registry.suite.expected_count,
            registry.benchmark.len()
        );
    }
    let mut ids = std::collections::HashSet::new();
    for b in &registry.benchmark {
        if !ids.insert(&b.id) {
            bail!("duplicate benchmark id: {}", b.id);
        }
        if b.timeout_minutes == 0 {
            bail!("benchmark {} has zero timeout", b.id);
        }
    }
    Ok(())
}
