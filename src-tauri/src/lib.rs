mod agent_llm;
mod db;
mod experiments;
mod marketplace;

use experiments::{AgentArtifacts, ExperimentDetail, ExperimentSummary, PlanPreview, WorkerPlan};
use marketplace::{Dataset, Marketplace};
use std::path::PathBuf;
use std::process::Command;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

/// Proves the Rust -> Python plumbing works (M0). Runs the worker's --help
/// from the repo's worker/ dir and returns its stdout.
#[tauri::command]
fn worker_healthcheck() -> Result<String, String> {
    let worker_dir = format!("{}/../worker", env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("python3")
        .args(["-m", "doclab_worker", "--help"])
        .current_dir(&worker_dir)
        .output()
        .map_err(|e| format!("failed to launch python3: {e}"))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

/// Query the curated marketplace (M1). The agent may ONLY surface datasets
/// returned from here — there is no live Hugging Face search.
#[tauri::command]
fn query_datasets(
    state: tauri::State<Marketplace>,
    keyword: Option<String>,
    data_type: Option<String>,
    task_type: Option<String>,
) -> Vec<Dataset> {
    state.query(
        keyword.as_deref(),
        data_type.as_deref(),
        task_type.as_deref(),
    )
}

#[tauri::command]
fn create_plan(
    state: tauri::State<Marketplace>,
    goal_text: String,
    dataset_id: Option<String>,
) -> Result<PlanPreview, String> {
    experiments::create_plan(&state, goal_text, dataset_id)
}

#[tauri::command]
fn run_experiment(
    state: tauri::State<Marketplace>,
    plan: WorkerPlan,
    goal_text: String,
    agent_artifacts: AgentArtifacts,
) -> Result<ExperimentDetail, String> {
    experiments::run_experiment(&state, plan, goal_text, agent_artifacts)
}

#[tauri::command]
fn list_experiments() -> Result<Vec<ExperimentSummary>, String> {
    experiments::list_experiments()
}

#[tauri::command]
fn get_experiment(id: String) -> Result<ExperimentDetail, String> {
    experiments::get_experiment(id)
}

fn marketplace_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("marketplace")
        .join("datasets.yaml")
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // YAML is the source of truth; a bad index is fatal at startup (fail loud).
    let market =
        Marketplace::load(&marketplace_path()).expect("failed to load curated dataset marketplace");

    // SQLite is a mirror for M3+ joins, NOT the query hot path — a failure
    // here is logged but must not block the in-memory marketplace.
    match db::open_db() {
        Ok(conn) => {
            if let Err(e) = db::mirror_datasets(&conn, &market) {
                eprintln!("warning: failed to mirror datasets to sqlite: {e}");
            }
        }
        Err(e) => eprintln!("warning: failed to open doclab.db: {e}"),
    }

    // Fallback A (DEMO.md): a pre-completed run in history, ready to open if a
    // live demo run fails or runs long. Non-fatal — the app works without it.
    if let Err(e) = experiments::seed_demo_experiment() {
        eprintln!("warning: failed to seed demo experiment: {e}");
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(market)
        .invoke_handler(tauri::generate_handler![
            greet,
            worker_healthcheck,
            query_datasets,
            create_plan,
            run_experiment,
            list_experiments,
            get_experiment,
            experiments::run_predict,
            agent_llm::get_agent_status,
            agent_llm::agent_parse_intent,
            agent_llm::agent_pick_dataset
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
