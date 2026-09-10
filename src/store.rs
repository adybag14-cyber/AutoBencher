use crate::types::{BenchmarkResult, RunManifest};
use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct Store {
    db_path: PathBuf,
}

impl Store {
    pub fn open(workspace: &Path) -> Result<Self> {
        fs::create_dir_all(workspace)?;
        let db_path = workspace.join("autobencher.sqlite3");
        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            r#"
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, model TEXT NOT NULL, engine TEXT NOT NULL,
              started_at TEXT NOT NULL, finished_at TEXT, manifest_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS benchmark_results (
              run_id TEXT NOT NULL, benchmark_id TEXT NOT NULL, status TEXT NOT NULL,
              score REAL, duration_seconds REAL NOT NULL, result_json TEXT NOT NULL,
              PRIMARY KEY (run_id, benchmark_id)
            );
        "#,
        )?;
        Ok(Self { db_path })
    }

    fn conn(&self) -> Result<Connection> {
        Ok(Connection::open(&self.db_path)?)
    }

    pub fn create_run(&self, m: &RunManifest) -> Result<()> {
        let json = serde_json::to_string(m)?;
        self.conn()?.execute(
            "INSERT OR REPLACE INTO runs(run_id,model,engine,started_at,finished_at,manifest_json) VALUES (?1,?2,?3,?4,?5,?6)",
            params![m.run_id, m.model, m.engine, m.started_at, m.finished_at, json],
        )?;
        Ok(())
    }

    pub fn finish_run(&self, m: &RunManifest) -> Result<()> {
        self.create_run(m)
    }

    pub fn save_result(&self, r: &BenchmarkResult) -> Result<()> {
        let json = serde_json::to_string(r)?;
        self.conn()?.execute(
            "INSERT OR REPLACE INTO benchmark_results(run_id,benchmark_id,status,score,duration_seconds,result_json) VALUES (?1,?2,?3,?4,?5,?6)",
            params![r.run_id, r.benchmark_id, r.status.as_str(), r.primary_score, r.duration_seconds, json],
        )?;
        Ok(())
    }

    pub fn load_results(&self, run_id: &str) -> Result<Vec<BenchmarkResult>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT result_json FROM benchmark_results WHERE run_id=?1 ORDER BY benchmark_id",
        )?;
        let rows = stmt.query_map([run_id], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            let raw = row?;
            out.push(serde_json::from_str(&raw).context("decode stored benchmark result")?);
        }
        Ok(out)
    }

    pub fn latest_run_id(&self) -> Result<Option<String>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1")?;
        let mut rows = stmt.query([])?;
        Ok(rows.next()?.map(|r| r.get(0)).transpose()?)
    }
}
