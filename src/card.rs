use crate::types::SuiteRegistry;
use anyhow::{Context, Result, anyhow};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Duration};
use tokio::time::sleep;

#[derive(Debug, serde::Serialize)]
pub struct CardCheck {
    pub url: String,
    pub sha256: String,
    pub bytes: usize,
    pub registry_count: usize,
    pub missing_registry_names: Vec<String>,
}

pub async fn verify(url: &str, registry: &SuiteRegistry, snapshot_dir: &Path) -> Result<CardCheck> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(45))
        .build()
        .context("build model-card HTTP client")?;
    let mut last_error = None;
    let mut downloaded = None;
    for attempt in 0..4_u32 {
        let response = client.get(url).send().await;
        match response {
            Ok(resp) => match resp.error_for_status() {
                Ok(resp) => match resp.text().await {
                    Ok(text) => {
                        downloaded = Some(text);
                        break;
                    }
                    Err(err) => last_error = Some(anyhow!(err).context("read model card")),
                },
                Err(err) => last_error = Some(anyhow!(err).context("model-card HTTP status")),
            },
            Err(err) => last_error = Some(anyhow!(err).context("download model card")),
        }
        if attempt < 3 {
            sleep(Duration::from_secs(1_u64 << attempt)).await;
        }
    }
    let text = downloaded.ok_or_else(|| {
        last_error.unwrap_or_else(|| anyhow!("model-card download failed without an error"))
    })?;
    fs::create_dir_all(snapshot_dir)?;
    fs::write(snapshot_dir.join("model-card.md"), &text)?;
    let sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
    let missing_registry_names = registry
        .benchmark
        .iter()
        .filter(|b| !text.contains(&b.name))
        .map(|b| b.name.clone())
        .collect::<Vec<_>>();
    let check = CardCheck {
        url: url.into(),
        sha256,
        bytes: text.len(),
        registry_count: registry.benchmark.len(),
        missing_registry_names,
    };
    fs::write(
        snapshot_dir.join("model-card-check.json"),
        serde_json::to_vec_pretty(&check)?,
    )?;
    Ok(check)
}
