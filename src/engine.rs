use crate::config::AppConfig;
use anyhow::{Context, Result, bail};
use reqwest::Client;
use std::{path::Path, process::Stdio, time::Duration};
use tokio::{
    process::{Child, Command},
    time::sleep,
};

pub struct EngineGuard {
    child: Option<Child>,
    wsl_pid_file: Option<String>,
}

impl EngineGuard {
    pub async fn start(cfg: &AppConfig, model: &str, engine: &str, api_base: &str) -> Result<Self> {
        if wait_ready(api_base, Duration::from_millis(500))
            .await
            .is_ok()
        {
            tracing::info!(%api_base, "using already-running model endpoint");
            return Ok(Self {
                child: None,
                wsl_pid_file: None,
            });
        }

        match engine {
            "external" => {
                wait_ready(api_base, Duration::from_secs(30)).await?;
                Ok(Self {
                    child: None,
                    wsl_pid_file: None,
                })
            }
            "vllm" => Self::start_native_vllm(cfg, model, api_base).await,
            "wsl-vllm" => Self::start_wsl_vllm(cfg, model, api_base).await,
            "auto" => {
                if which::which(&cfg.vllm.executable).is_ok() {
                    Self::start_native_vllm(cfg, model, api_base).await
                } else if cfg!(windows) && which::which("wsl.exe").is_ok() {
                    tracing::info!("native vLLM not found; using managed WSL vLLM");
                    Self::start_wsl_vllm(cfg, model, api_base).await
                } else {
                    bail!(
                        "no model server found: install vLLM, use --engine external with an OpenAI-compatible endpoint, or on Windows enable WSL2"
                    )
                }
            }
            other => bail!("unknown engine {other:?}; expected auto, vllm, wsl-vllm or external"),
        }
    }

    async fn start_native_vllm(cfg: &AppConfig, model: &str, api_base: &str) -> Result<Self> {
        let exe = which::which(&cfg.vllm.executable).with_context(|| {
            format!(
                "{} not found; use --engine auto/wsl-vllm/external or install vLLM",
                cfg.vllm.executable
            )
        })?;
        let mut cmd = Command::new(exe);
        cmd.arg("serve")
            .arg(model)
            .arg("--host")
            .arg(&cfg.vllm.host)
            .arg("--port")
            .arg(cfg.vllm.port.to_string())
            .arg("--tensor-parallel-size")
            .arg(cfg.vllm.tensor_parallel_size.to_string())
            .arg("--gpu-memory-utilization")
            .arg(cfg.vllm.gpu_memory_utilization.to_string())
            .arg("--max-model-len")
            .arg(cfg.vllm.max_model_len.to_string())
            .arg("--served-model-name")
            .arg(model)
            .args(&cfg.vllm.extra_args)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let child = cmd.spawn().context("start native vLLM server")?;
        let mut guard = Self {
            child: Some(child),
            wsl_pid_file: None,
        };
        if let Err(e) = guard
            .wait_ready_or_exit(api_base, Duration::from_secs(900))
            .await
        {
            guard.stop().await;
            return Err(e);
        }
        Ok(guard)
    }

    async fn start_wsl_vllm(cfg: &AppConfig, model: &str, api_base: &str) -> Result<Self> {
        if !cfg!(windows) {
            bail!("wsl-vllm is only available from a Windows AutoBencher binary");
        }
        ensure_wsl_vllm().await?;
        ensure_wsl_gpu().await?;
        let wsl_model = model_for_wsl(model).await?;
        let pid_file = format!(
            "/tmp/autobencher-vllm-{}-{}.pid",
            std::process::id(),
            cfg.vllm.port
        );
        let mut argv = vec![
            "serve".to_string(),
            wsl_model.clone(),
            "--host".to_string(),
            "0.0.0.0".to_string(),
            "--port".to_string(),
            cfg.vllm.port.to_string(),
            "--tensor-parallel-size".to_string(),
            cfg.vllm.tensor_parallel_size.to_string(),
            "--gpu-memory-utilization".to_string(),
            cfg.vllm.gpu_memory_utilization.to_string(),
            "--max-model-len".to_string(),
            cfg.vllm.max_model_len.to_string(),
            "--served-model-name".to_string(),
            model.to_string(),
        ];
        argv.extend(cfg.vllm.extra_args.clone());
        let args = argv
            .iter()
            .map(|s| bash_quote(s))
            .collect::<Vec<_>>()
            .join(" ");
        let script = format!(
            "set -euo pipefail; root=\"${{XDG_CACHE_HOME:-$HOME/.cache}}/autobencher/vllm\"; echo $$ > {}; export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0; exec \"$root/bin/vllm\" {}",
            bash_quote(&pid_file),
            args
        );
        let mut cmd = Command::new("wsl.exe");
        cmd.args(["-e", "bash", "-lc", &script])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let child = cmd.spawn().context("start managed WSL vLLM server")?;
        let mut guard = Self {
            child: Some(child),
            wsl_pid_file: Some(pid_file),
        };
        if let Err(e) = guard
            .wait_ready_or_exit(api_base, Duration::from_secs(1200))
            .await
        {
            guard.stop().await;
            return Err(e);
        }
        Ok(guard)
    }

    async fn wait_ready_or_exit(&mut self, base: &str, timeout: Duration) -> Result<()> {
        let client = Client::builder().timeout(Duration::from_secs(5)).build()?;
        let url = format!("{}/v1/models", base.trim_end_matches('/'));
        let started = std::time::Instant::now();
        loop {
            if let Some(child) = &mut self.child
                && let Some(status) = child.try_wait().context("poll model-server child")?
            {
                bail!("model server exited before readiness with {status}");
            }
            if let Ok(resp) = client.get(&url).send().await
                && resp.status().is_success()
            {
                return Ok(());
            }
            if started.elapsed() >= timeout {
                bail!("model endpoint did not become ready: {url}");
            }
            sleep(Duration::from_secs(2)).await;
        }
    }

    pub async fn stop(&mut self) {
        if let Some(pid_file) = self.wsl_pid_file.take() {
            let script = format!(
                "if [ -s {p} ]; then pid=$(cat {p}); kill $pid 2>/dev/null || true; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 0.2; done; kill -9 $pid 2>/dev/null || true; rm -f {p}; fi",
                p = bash_quote(&pid_file)
            );
            let _ = Command::new("wsl.exe")
                .args(["-e", "bash", "-lc", &script])
                .status()
                .await;
        }
        if let Some(child) = &mut self.child {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        self.child = None;
    }
}

async fn ensure_wsl_vllm() -> Result<()> {
    let script = r#"set -euo pipefail
root="${XDG_CACHE_HOME:-$HOME/.cache}/autobencher/vllm"
uv_bin="$(command -v uv || true)"
if [ -z "$uv_bin" ] && [ -x "$HOME/.local/bin/uv" ]; then uv_bin="$HOME/.local/bin/uv"; fi
if [ -z "$uv_bin" ]; then
  echo 'AutoBencher requires uv inside WSL to provision vLLM automatically.' >&2
  exit 41
fi
marker="$root/.autobencher-vllm-spec"
spec='vllm==0.29.0'
if [ ! -x "$root/bin/python" ]; then
  "$uv_bin" venv --python 3.12 "$root"
fi
if [ ! -x "$root/bin/vllm" ] || [ ! -f "$marker" ] || [ "$(cat "$marker" 2>/dev/null || true)" != "$spec" ]; then
  "$uv_bin" pip install --upgrade --python "$root/bin/python" "$spec"
  printf '%s\n' "$spec" > "$marker"
fi
"$root/bin/vllm" --version
"#;
    let status = Command::new("wsl.exe")
        .args(["-e", "bash", "-lc", script])
        .status()
        .await
        .context("provision WSL vLLM environment")?;
    if !status.success() {
        bail!("WSL vLLM provisioning failed with {status}");
    }
    Ok(())
}

async fn ensure_wsl_gpu() -> Result<()> {
    let status = Command::new("wsl.exe")
        .args([
            "-e",
            "bash",
            "-lc",
            "command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null",
        ])
        .status()
        .await
        .context("check WSL NVIDIA GPU visibility")?;
    if !status.success() {
        bail!("WSL is available but no NVIDIA GPU is visible to it");
    }
    Ok(())
}

async fn model_for_wsl(model: &str) -> Result<String> {
    if !Path::new(model).exists() {
        return Ok(model.to_string());
    }
    let out = Command::new("wsl.exe")
        .args(["-e", "wslpath", "-a", "-u", model])
        .output()
        .await
        .context("translate local model path for WSL")?;
    if !out.status.success() {
        bail!(
            "failed to translate Windows model path for WSL: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn bash_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

async fn wait_ready(base: &str, timeout: Duration) -> Result<()> {
    let client = Client::builder().timeout(Duration::from_secs(5)).build()?;
    let url = format!("{}/v1/models", base.trim_end_matches('/'));
    let started = std::time::Instant::now();
    loop {
        if let Ok(resp) = client.get(&url).send().await
            && resp.status().is_success()
        {
            return Ok(());
        }
        if started.elapsed() >= timeout {
            bail!("model endpoint did not become ready: {url}");
        }
        sleep(Duration::from_secs(2)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bash_quote_handles_single_quote() {
        assert_eq!(bash_quote("a'b"), "'a'\"'\"'b'");
    }
}
