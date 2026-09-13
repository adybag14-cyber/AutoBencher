use crate::{config::AppConfig, process::Invocation, types::BenchmarkSpec};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct PlannedCommand {
    pub invocation: Invocation,
    /// Human-readable command safe to persist in logs/reports.
    pub display: String,
}

impl PlannedCommand {
    fn shell(command: String) -> Self {
        Self {
            display: command.clone(),
            invocation: Invocation::Shell(command),
        }
    }
}

#[derive(Debug, Clone)]
pub enum NativePlan {
    Command(PlannedCommand),
    Blocked(String),
}

pub fn plan(
    spec: &BenchmarkSpec,
    cfg: &AppConfig,
    model: &str,
    api_base: &str,
    result_dir: &Path,
    limit: Option<usize>,
) -> NativePlan {
    let env_key = native_env_key(&spec.id);
    if let Ok(raw) = std::env::var(&env_key) {
        return NativePlan::Command(PlannedCommand::shell(expand(
            &raw, cfg, model, api_base, result_dir, limit,
        )));
    }

    match spec.id.as_str() {
        "ojbench" => NativePlan::Blocked("OJBench requires its official judge/testdata pipeline. Set AUTOBENCHER_NATIVE_CMD_OJBENCH to a pinned upstream command after preparing the official assets and accepting their terms.".into()),
        "nolima" => NativePlan::Blocked("NoLiMa requires separately licensed Adobe Research evaluation assets and a model/run config. AutoBencher does not redistribute or silently accept that license. Set AUTOBENCHER_NATIVE_CMD_NOLIMA after preparing the official checkout.".into()),
        "tau3-banking" => NativePlan::Blocked("The model-card tau3 Banking score is marked as an Artificial Analysis result. Exact reproduction also needs a separately configured user simulator and version-isolated tau3 environment. Pin it with AUTOBENCHER_NATIVE_CMD_TAU3_BANKING.".into()),
        "tau2-telecom" => NativePlan::Blocked("tau2 Telecom requires a user-simulator model in addition to the model under test, and its tau2 package version conflicts with tau3. Pin the simulator and isolated upstream environment with AUTOBENCHER_NATIVE_CMD_TAU2_TELECOM.".into()),
        "browsecomp-zh" => NativePlan::Blocked("BrowseComp-ZH needs the official evaluation data plus a search stack. Set AUTOBENCHER_NATIVE_CMD_BROWSECOMP_ZH after preparing the official data and provider.".into()),
        "browsecomp-top100" => NativePlan::Blocked("The model card reports the Top100 slice while general public runners commonly expose full BrowseComp. AutoBencher will not score the wrong slice. Set AUTOBENCHER_NATIVE_CMD_BROWSECOMP_TOP100 to the exact Top100 recipe.".into()),
        "gaia-text-103" => NativePlan::Blocked("The model card reports the Text-103 slice; generic GAIA runners expose broader configurations and need an agent/search stack. Set AUTOBENCHER_NATIVE_CMD_GAIA_TEXT_103 to the exact Text-103 recipe.".into()),
        "gdpval-aa-v2" => NativePlan::Blocked("EvalScope can generate GDPval submission artifacts, but it does not reproduce the official GDPval-AA v2 judge score shown on the model card. Set AUTOBENCHER_NATIVE_CMD_GDPVAL_AA_V2 only when an exact judge pipeline is available.".into()),
        _ => NativePlan::Blocked(format!("No verified exact public one-command recipe is pinned for {}. The row remains in the complete 34-row registry and is not replaced with a proxy. Set {env_key} to an official/native command to enable it.", spec.name)),
    }
}

pub fn evalscope_compat(
    cfg: &AppConfig,
    spec: &BenchmarkSpec,
    model: &str,
    api_base: &str,
    result_dir: &Path,
    limit: Option<usize>,
    reuse_predictions: bool,
) -> PlannedCommand {
    let exe = crate::bootstrap::evalscope_path_for(cfg, spec);
    let base = api_base.trim_end_matches('/');
    let endpoint = if base.ends_with("/v1") {
        base.to_string()
    } else {
        format!("{base}/v1")
    };

    // EvalScope accepts JSON-valued arguments. Execute it directly so Windows
    // PowerShell cannot re-parse and corrupt nested JSON quoting.
    let mut args = vec![
        "eval".into(),
        "--model".into(),
        model.into(),
        "--api-url".into(),
        endpoint,
        "--api-key".into(),
        cfg.api_key(),
        "--eval-type".into(),
        "openai_api".into(),
        "--datasets".into(),
        spec.dataset.clone(),
        "--seed".into(),
        cfg.seed.to_string(),
        "--work-dir".into(),
        result_dir.display().to_string(),
        "--no-timestamp".into(),
        "--enable-progress-tracker".into(),
        "--eval-batch-size".into(),
        cfg.eval_batch_size.to_string(),
        "--generation-config".into(),
        cfg.generation_config.clone(),
    ];
    args.extend(spec.evalscope_args.clone());
    if reuse_predictions && result_dir.join("predictions").is_dir() {
        args.push("--use-cache".into());
        args.push(result_dir.display().to_string());
    }
    if let Some(n) = limit {
        args.push("--limit".into());
        args.push(n.to_string());
    }

    let program = exe.display().to_string();
    let display = render_direct_command(&program, &args, &cfg.api_key());
    PlannedCommand {
        invocation: Invocation::Direct { program, args },
        display,
    }
}

fn render_direct_command(program: &str, args: &[String], secret: &str) -> String {
    let mut rendered = vec![q(program)];
    for (idx, arg) in args.iter().enumerate() {
        let is_secret = (idx > 0 && args[idx - 1] == "--api-key")
            || (!secret.is_empty() && secret != "EMPTY" && arg == secret);
        if is_secret {
            rendered.push("<redacted>".into());
        } else {
            rendered.push(q(arg));
        }
    }
    rendered.join(" ")
}

fn native_env_key(id: &str) -> String {
    format!(
        "AUTOBENCHER_NATIVE_CMD_{}",
        id.to_ascii_uppercase().replace('-', "_")
    )
}

fn expand(
    template: &str,
    cfg: &AppConfig,
    model: &str,
    api_base: &str,
    result_dir: &Path,
    limit: Option<usize>,
) -> String {
    template
        .replace("${MODEL}", model)
        .replace("${API_BASE}", api_base)
        .replace("${API_KEY}", secret_ref())
        .replace("${RESULT_DIR}", &result_dir.display().to_string())
        .replace("${WORKSPACE}", &cfg.workspace.display().to_string())
        .replace(
            "${LIMIT}",
            &limit.map(|n| n.to_string()).unwrap_or_default(),
        )
}

fn secret_ref() -> &'static str {
    #[cfg(windows)]
    {
        "$env:AUTOBENCHER_CHILD_API_KEY"
    }
    #[cfg(not(windows))]
    {
        "\"${AUTOBENCHER_CHILD_API_KEY}\""
    }
}

pub fn q(s: &str) -> String {
    if !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-/:\\".contains(c))
    {
        return s.to_string();
    }
    #[cfg(windows)]
    {
        format!("'{}'", s.replace('\'', "''"))
    }
    #[cfg(not(windows))]
    {
        format!("'{}'", s.replace('\'', "'\"'\"'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_env_key_is_stable() {
        assert_eq!(
            native_env_key("swe-bench-pro"),
            "AUTOBENCHER_NATIVE_CMD_SWE_BENCH_PRO"
        );
    }

    #[test]
    fn quote_preserves_json_as_one_argument() {
        let raw = r#"{"enabled":true}"#;
        assert!(q(raw).len() > raw.len());
    }

    #[test]
    fn rendered_direct_command_redacts_key() {
        let args = vec![
            "--api-key".into(),
            "super-secret".into(),
            "--limit".into(),
            "1".into(),
        ];
        let display = render_direct_command("evalscope", &args, "super-secret");
        assert!(!display.contains("super-secret"));
        assert!(display.contains("<redacted>"));
    }
}
