use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    pub model: String,
    pub engine: String,
    pub api_base: String,
    pub api_key_env: String,
    pub workspace: PathBuf,
    pub parallel: usize,
    pub seed: u64,
    pub model_card_url: String,
    pub evalscope_spec: String,
    pub generation_config: String,
    pub vllm: VllmConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VllmConfig {
    pub executable: String,
    pub port: u16,
    pub host: String,
    pub tensor_parallel_size: usize,
    pub gpu_memory_utilization: f64,
    pub max_model_len: usize,
    #[serde(default)]
    pub extra_args: Vec<String>,
}

impl AppConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let absolute_config = if p.is_absolute() {
            p.to_path_buf()
        } else {
            std::env::current_dir()
                .context("resolve current directory for config")?
                .join(p)
        };
        let raw = fs::read_to_string(&absolute_config)
            .with_context(|| format!("read config {}", absolute_config.display()))?;
        let mut cfg: Self = toml::from_str(&raw)
            .with_context(|| format!("parse config {}", absolute_config.display()))?;
        if cfg.workspace.is_relative() {
            let base = absolute_config
                .parent()
                .context("config path has no parent directory")?;
            cfg.workspace = base.join(&cfg.workspace);
        }
        if cfg.parallel == 0 {
            cfg.parallel = 1;
        }
        Ok(cfg)
    }

    pub fn ensure_dirs(&self) -> Result<()> {
        for d in [
            self.workspace.clone(),
            self.workspace.join("runs"),
            self.workspace.join("harnesses"),
            self.workspace.join("cache"),
        ] {
            fs::create_dir_all(&d).with_context(|| format!("create {}", d.display()))?;
        }
        Ok(())
    }

    pub fn api_key(&self) -> String {
        std::env::var(&self.api_key_env).unwrap_or_else(|_| "EMPTY".to_string())
    }
}
