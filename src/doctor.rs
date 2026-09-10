use crate::types::HostSnapshot;
use std::process::Command;
use sysinfo::System;

pub fn snapshot() -> HostSnapshot {
    let mut sys = System::new_all();
    sys.refresh_all();
    HostSnapshot {
        hostname: hostname::get()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".into()),
        os: format!(
            "{} {}",
            System::name().unwrap_or_else(|| std::env::consts::OS.into()),
            System::os_version().unwrap_or_default()
        )
        .trim()
        .to_string(),
        arch: std::env::consts::ARCH.into(),
        cpu_count: sys.cpus().len(),
        total_memory_bytes: sys.total_memory(),
        rustc: version("rustc", &["--version"]),
        git: version("git", &["--version"]),
        python: version(
            if cfg!(windows) { "python" } else { "python3" },
            &["--version"],
        ),
        uv: version("uv", &["--version"]),
        docker: version("docker", &["--version"]),
        nvidia_smi: version(
            "nvidia-smi",
            &[
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
        ),
    }
}

fn version(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    let text = if out.stdout.is_empty() {
        &out.stderr
    } else {
        &out.stdout
    };
    let s = String::from_utf8_lossy(text).trim().to_string();
    (!s.is_empty()).then_some(s)
}
