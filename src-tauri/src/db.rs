use crate::marketplace::Marketplace;
use rusqlite::Connection;
use std::path::PathBuf;

/// Resolve the DocLab data root (`~/.doclab/`), creating it if missing.
pub fn data_root() -> Result<PathBuf, String> {
    let home = dirs::home_dir().ok_or("cannot resolve home directory")?;
    let root = home.join(".doclab");
    std::fs::create_dir_all(&root)
        .map_err(|e| format!("cannot create data root {}: {e}", root.display()))?;
    std::fs::create_dir_all(root.join("datasets"))
        .map_err(|e| format!("cannot create datasets dir: {e}"))?;
    std::fs::create_dir_all(root.join("experiments"))
        .map_err(|e| format!("cannot create experiments dir: {e}"))?;
    Ok(root)
}

pub fn experiments_dir() -> Result<PathBuf, String> {
    Ok(data_root()?.join("experiments"))
}

/// Open `~/.doclab/doclab.db` and ensure M1/M3 tables exist.
pub fn open_db() -> Result<Connection, String> {
    let path = data_root()?.join("doclab.db");
    let conn =
        Connection::open(&path).map_err(|e| format!("cannot open db {}: {e}", path.display()))?;
    create_datasets_table(&conn)?;
    create_experiments_table(&conn)?;
    run_migrations(&conn)?;
    Ok(conn)
}

/// Bring an existing `experiments` table up to the current schema. New columns
/// are added here (guarded) so older `~/.doclab/doclab.db` files keep working
/// without a destructive rebuild. `CREATE TABLE IF NOT EXISTS` covers fresh
/// installs; this covers upgrades.
fn run_migrations(conn: &Connection) -> Result<(), String> {
    add_column_if_missing(conn, "experiments", "is_best", "INTEGER NOT NULL DEFAULT 0")?;
    add_column_if_missing(conn, "experiments", "checkpoint_path", "TEXT")?;
    Ok(())
}

fn add_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
    decl: &str,
) -> Result<(), String> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|e| format!("cannot inspect {table}: {e}"))?;
    let existing: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|e| format!("cannot read {table} columns: {e}"))?
        .filter_map(Result::ok)
        .collect();
    if existing.iter().any(|c| c == column) {
        return Ok(());
    }
    conn.execute(
        &format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"),
        [],
    )
    .map(|_| ())
    .map_err(|e| format!("cannot add column {column} to {table}: {e}"))
}

#[allow(dead_code)]
pub(crate) fn add_column_if_missing_public(
    conn: &Connection,
    table: &str,
    column: &str,
    decl: &str,
) -> Result<(), String> {
    add_column_if_missing(conn, table, column, decl)
}

fn create_datasets_table(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS datasets (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            hf_id       TEXT NOT NULL,
            revision    TEXT NOT NULL,
            data_type   TEXT NOT NULL,
            task_types  TEXT NOT NULL,
            label_column TEXT NOT NULL,
            category    TEXT,
            description TEXT
        );",
    )
    .map_err(|e| format!("cannot create datasets table: {e}"))
}

pub(crate) fn create_experiments_table(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS experiments (
            id              TEXT PRIMARY KEY,
            created_at_ms   INTEGER NOT NULL,
            updated_at_ms   INTEGER NOT NULL,
            status          TEXT NOT NULL,
            goal_text       TEXT NOT NULL,
            dataset_id      TEXT NOT NULL,
            primary_metric  TEXT,
            metric_value    REAL,
            baseline_metric REAL,
            model_type      TEXT,
            framework       TEXT,
            device          TEXT,
            plan_path       TEXT NOT NULL,
            metrics_path    TEXT,
            error_path      TEXT,
            model_card_path TEXT,
            worker_stdout   TEXT,
            worker_stderr   TEXT,
            error_code      TEXT,
            error_message   TEXT,
            is_best         INTEGER NOT NULL DEFAULT 0
        );",
    )
    .map_err(|e| format!("cannot create experiments table: {e}"))
}

/// Idempotent upsert of the curated index into SQLite. Re-running does not
/// duplicate rows. This mirror is for M3+ joins — it is NOT the query hot
/// path, so a failure here is logged by the caller, not fatal.
pub fn mirror_datasets(conn: &Connection, market: &Marketplace) -> Result<(), String> {
    for d in &market.datasets {
        let task_types = d.task_types.join(",");
        conn.execute(
            "INSERT INTO datasets
                (id, name, hf_id, revision, data_type, task_types, label_column, category, description)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(id) DO UPDATE SET
                name=?2, hf_id=?3, revision=?4, data_type=?5,
                task_types=?6, label_column=?7, category=?8, description=?9",
            rusqlite::params![
                d.id, d.name, d.hf_id, d.revision, d.data_type,
                task_types, d.label_column, d.category, d.description,
            ],
        )
        .map_err(|e| format!("cannot mirror dataset '{}': {e}", d.id))?;
    }
    Ok(())
}
