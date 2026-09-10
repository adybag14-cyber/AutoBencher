use anyhow::{Context, Result};
use std::{path::PathBuf, process::Stdio, time::Duration};
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncWriteExt},
    process::Command,
    time,
};

#[derive(Debug, Clone)]
pub enum Invocation {
    /// Execute through the platform shell. Reserved for user/native harness recipes.
    Shell(String),
    /// Execute an exact argv vector with no shell re-parsing.
    Direct { program: String, args: Vec<String> },
}

#[derive(Debug, Clone)]
pub struct ProcessSpec {
    pub invocation: Invocation,
    pub display_command: String,
    pub cwd: PathBuf,
    pub timeout: Duration,
    pub env: Vec<(String, String)>,
    pub stdout_log: PathBuf,
    pub stderr_log: PathBuf,
}

#[derive(Debug)]
pub struct ProcessOutcome {
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub stdout_tail: String,
    pub stderr_tail: String,
}

fn shell_command(command: &str) -> Command {
    #[cfg(windows)]
    {
        let mut cmd = Command::new("powershell.exe");
        cmd.args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ]);
        cmd
    }
    #[cfg(not(windows))]
    {
        let mut cmd = Command::new("sh");
        cmd.args(["-lc", command]);
        cmd
    }
}

fn build_command(invocation: &Invocation) -> Command {
    match invocation {
        Invocation::Shell(command) => shell_command(command),
        Invocation::Direct { program, args } => {
            let mut cmd = Command::new(program);
            cmd.args(args);
            cmd
        }
    }
}

async fn pump<R>(mut reader: R, log_path: PathBuf, tail_limit: usize) -> Result<String>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut file = File::create(&log_path)
        .await
        .with_context(|| format!("create {}", log_path.display()))?;
    let mut tail = Vec::with_capacity(tail_limit.min(128 * 1024));
    let mut buf = [0_u8; 16 * 1024];
    loop {
        let n = reader.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).await?;
        tail.extend_from_slice(&buf[..n]);
        if tail.len() > tail_limit {
            let excess = tail.len() - tail_limit;
            tail.drain(..excess);
        }
    }
    file.flush().await?;
    Ok(String::from_utf8_lossy(&tail).into_owned())
}

pub async fn run_process(spec: ProcessSpec) -> Result<ProcessOutcome> {
    tokio::fs::create_dir_all(&spec.cwd).await?;
    if let Some(parent) = spec.stdout_log.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    if let Some(parent) = spec.stderr_log.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut cmd = build_command(&spec.invocation);
    cmd.current_dir(&spec.cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    for (k, v) in &spec.env {
        cmd.env(k, v);
    }

    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn command: {}", spec.display_command))?;
    let stdout = child.stdout.take().context("capture stdout")?;
    let stderr = child.stderr.take().context("capture stderr")?;
    let out_path = spec.stdout_log.clone();
    let err_path = spec.stderr_log.clone();
    let out_task = tokio::spawn(async move { pump(stdout, out_path, 128 * 1024).await });
    let err_task = tokio::spawn(async move { pump(stderr, err_path, 128 * 1024).await });

    let wait = time::timeout(spec.timeout, child.wait()).await;
    let (exit_code, timed_out) = match wait {
        Ok(status) => (status?.code(), false),
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            (None, true)
        }
    };

    let stdout_tail = out_task.await.context("join stdout pump")??;
    let stderr_tail = err_task.await.context("join stderr pump")??;
    Ok(ProcessOutcome {
        exit_code,
        timed_out,
        stdout_tail,
        stderr_tail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_invocation_preserves_json_argument() {
        let json = r#"{"max_tokens":32768,"extra_body":{"chat_template_kwargs":{"enable_thinking":true}}}"#;
        let invocation = Invocation::Direct {
            program: "evalscope".into(),
            args: vec!["--generation-config".into(), json.into()],
        };
        match invocation {
            Invocation::Direct { args, .. } => assert_eq!(args[1], json),
            Invocation::Shell(_) => unreachable!(),
        }
    }
}
