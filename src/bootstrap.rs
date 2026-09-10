use crate::{config::AppConfig, types::BenchmarkSpec};
use anyhow::{Context, Result, bail};
use std::{collections::BTreeSet, fmt, fs, path::PathBuf, process::Command};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HarnessEnv {
    Core,
    CodeSandbox,
    IfEval,
    IfBench,
    MultiIf,
    Bfcl,
    SweBench,
    TerminalBench,
}

impl HarnessEnv {
    pub const ALL: [Self; 8] = [
        Self::Core,
        Self::CodeSandbox,
        Self::IfEval,
        Self::IfBench,
        Self::MultiIf,
        Self::Bfcl,
        Self::SweBench,
        Self::TerminalBench,
    ];

    pub fn slug(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::CodeSandbox => "code-sandbox",
            Self::IfEval => "ifeval",
            Self::IfBench => "ifbench",
            Self::MultiIf => "multi-if",
            Self::Bfcl => "bfcl",
            Self::SweBench => "swe-bench",
            Self::TerminalBench => "terminal-bench",
        }
    }

    fn extras(self) -> &'static str {
        match self {
            Self::Core => "",
            Self::CodeSandbox => "sandbox",
            Self::IfEval => "ifeval",
            Self::IfBench => "ifbench",
            Self::MultiIf => "multi_if",
            Self::Bfcl => "bfcl",
            Self::SweBench => "sandbox,swe_bench",
            Self::TerminalBench => "terminal_bench",
        }
    }
}

impl fmt::Display for HarnessEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

pub fn env_for(spec: &BenchmarkSpec) -> HarnessEnv {
    match spec.id.as_str() {
        "livecodebench-v6" | "scicode-wbg" => HarnessEnv::CodeSandbox,
        "ifeval" => HarnessEnv::IfEval,
        "ifbench" => HarnessEnv::IfBench,
        "multi-if" => HarnessEnv::MultiIf,
        "bfcl-v4" => HarnessEnv::Bfcl,
        "swe-bench-verified" | "swe-bench-pro" => HarnessEnv::SweBench,
        "terminal-bench-v2-1" => HarnessEnv::TerminalBench,
        _ => HarnessEnv::Core,
    }
}

pub fn package_spec(cfg: &AppConfig, env: HarnessEnv) -> String {
    if env == HarnessEnv::Core || env.extras().is_empty() {
        return cfg.evalscope_spec.clone();
    }
    let suffix = cfg
        .evalscope_spec
        .strip_prefix("evalscope")
        .unwrap_or(cfg.evalscope_spec.as_str());
    format!("evalscope[{}]{}", env.extras(), suffix)
}

pub fn env_dir(cfg: &AppConfig, env: HarnessEnv) -> PathBuf {
    cfg.workspace.join("envs").join(env.slug())
}

pub fn python_path(cfg: &AppConfig, env: HarnessEnv) -> PathBuf {
    let root = env_dir(cfg, env);
    #[cfg(windows)]
    {
        root.join("Scripts").join("python.exe")
    }
    #[cfg(not(windows))]
    {
        root.join("bin").join("python")
    }
}

pub fn evalscope_path(cfg: &AppConfig, env: HarnessEnv) -> PathBuf {
    let root = env_dir(cfg, env);
    #[cfg(windows)]
    {
        root.join("Scripts").join("evalscope.exe")
    }
    #[cfg(not(windows))]
    {
        root.join("bin").join("evalscope")
    }
}

pub fn evalscope_path_for(cfg: &AppConfig, spec: &BenchmarkSpec) -> PathBuf {
    evalscope_path(cfg, env_for(spec))
}

pub fn required_envs(specs: &[BenchmarkSpec]) -> BTreeSet<HarnessEnv> {
    specs
        .iter()
        .filter(|b| b.runner == crate::types::RunnerKind::Evalscope)
        .map(env_for)
        .collect()
}

pub fn ensure_evalscope_set(cfg: &AppConfig, specs: &[BenchmarkSpec], force: bool) -> Result<()> {
    for env in required_envs(specs) {
        ensure_evalscope_env(cfg, env, force)?;
    }
    Ok(())
}

pub fn ensure_all(cfg: &AppConfig, force: bool) -> Result<()> {
    for env in HarnessEnv::ALL {
        ensure_evalscope_env(cfg, env, force)?;
    }
    Ok(())
}

pub fn ensure_evalscope_env(cfg: &AppConfig, env: HarnessEnv, force: bool) -> Result<()> {
    let evalscope = evalscope_path(cfg, env);
    let venv = env_dir(cfg, env);
    let marker = venv.join(".autobencher-spec");
    let spec = package_spec(cfg, env);
    let marker_matches = fs::read_to_string(&marker)
        .map(|s| s.trim() == spec.trim())
        .unwrap_or(false);
    if evalscope.exists() && marker_matches && !force {
        return Ok(());
    }

    cfg.ensure_dirs()?;
    println!("Bootstrapping harness env `{env}` with {spec}");
    let py = python_path(cfg, env);
    if !py.exists() {
        if which::which("uv").is_ok() {
            let status = Command::new("uv")
                .args(["venv", "--python", "3.12"])
                .arg(&venv)
                .status()
                .with_context(|| format!("create {env} venv with uv"))?;
            if !status.success() {
                bail!("uv venv failed for {env} with {status}");
            }
        } else {
            let bootstrap_python = which::which("python3")
                .or_else(|_| which::which("python"))
                .context("Python 3.12+ or uv is required")?;
            let status = Command::new(&bootstrap_python)
                .args(["-m", "venv"])
                .arg(&venv)
                .status()
                .with_context(|| format!("create {env} Python venv"))?;
            if !status.success() {
                bail!("python -m venv failed for {env} with {status}");
            }
        }
    }

    if which::which("uv").is_ok() {
        let status = Command::new("uv")
            .args(["pip", "install", "--upgrade", "--python"])
            .arg(&py)
            .arg(&spec)
            .status()
            .with_context(|| format!("install {spec} with uv"))?;
        if !status.success() {
            bail!("EvalScope install failed for {env} with {status}");
        }
    } else {
        let status = Command::new(&py)
            .args(["-m", "pip", "install", "--upgrade", "pip"])
            .status()
            .context("upgrade pip")?;
        if !status.success() {
            bail!("pip upgrade failed for {env} with {status}");
        }
        let status = Command::new(&py)
            .args(["-m", "pip", "install", "--upgrade"])
            .arg(&spec)
            .status()
            .with_context(|| format!("install {spec} with pip"))?;
        if !status.success() {
            bail!("pip install failed for {env} with {status}");
        }
    }

    if !evalscope.exists() {
        bail!(
            "EvalScope executable was not created for {env} at {}",
            evalscope.display()
        );
    }
    fs::write(&marker, format!("{spec}\n"))?;
    Ok(())
}

pub fn status_lines(cfg: &AppConfig) -> Vec<String> {
    HarnessEnv::ALL
        .into_iter()
        .map(|env| {
            let path = evalscope_path(cfg, env);
            let state = if path.exists() { "ready" } else { "missing" };
            format!("{:<15} {:<7} {}", env.slug(), state, package_spec(cfg, env))
        })
        .collect()
}
