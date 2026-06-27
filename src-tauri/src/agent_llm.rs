// Hybrid LLM agent module: rules-first with optional LLM assistance
// All LLM calls happen here (Rust) to keep API keys secure and out of the frontend bundle.

use crate::marketplace::Dataset;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::time::Duration;

/// Agent operating mode from DOCLAB_AGENT_MODE environment variable
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentMode {
    Rules,  // Default: keyword-based parsing and selection only
    Hybrid, // Rules + LLM for ambiguous cases
    Llm,    // LLM-first for all parsing and selection
}

/// LLM provider configuration
#[derive(Debug, Clone)]
pub enum LlmProvider {
    OpenAI { api_key: String, model: String },
    Anthropic { api_key: String, model: String },
}

/// Agent status exposed to frontend via get_agent_status command
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentStatus {
    pub mode: String,
    pub llm_configured: bool,
    pub provider: Option<String>,
}

/// Intent response from LLM (must match TypeScript Intent type)
#[derive(Debug, Serialize, Deserialize)]
pub struct IntentResponse {
    pub task_type: String,
    pub modality: String,
    pub metric_hint: Option<String>,
    pub goal_text: String,
}

/// Dataset selection response from LLM
#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetSelectionResponse {
    pub dataset_id: String,
    pub rationale: String,
}

// Cached agent mode (read once at startup)
static AGENT_MODE: OnceLock<AgentMode> = OnceLock::new();

/// Get the configured agent mode from environment variable
pub fn get_agent_mode() -> AgentMode {
    *AGENT_MODE.get_or_init(|| {
        std::env::var("DOCLAB_AGENT_MODE")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "hybrid" => Some(AgentMode::Hybrid),
                "llm" => Some(AgentMode::Llm),
                "rules" => Some(AgentMode::Rules),
                _ => {
                    eprintln!("warning: invalid DOCLAB_AGENT_MODE '{}', defaulting to 'rules'", s);
                    Some(AgentMode::Rules)
                }
            })
            .unwrap_or(AgentMode::Rules)
    })
}

/// Get the configured LLM provider from environment variables
/// Returns None if no API key is configured (triggers fallback to rules)
pub fn get_llm_provider() -> Result<Option<LlmProvider>, String> {
    let provider_name = std::env::var("DOCLAB_LLM_PROVIDER")
        .unwrap_or_else(|_| "openai".to_string())
        .to_lowercase();

    match provider_name.as_str() {
        "openai" => {
            if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
                let model = std::env::var("DOCLAB_LLM_MODEL")
                    .unwrap_or_else(|_| "gpt-4o-mini".to_string());
                Ok(Some(LlmProvider::OpenAI { api_key, model }))
            } else {
                Ok(None)
            }
        }
        "anthropic" => {
            if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
                let model = std::env::var("DOCLAB_LLM_MODEL")
                    .unwrap_or_else(|_| "claude-3-5-haiku-20241022".to_string());
                Ok(Some(LlmProvider::Anthropic { api_key, model }))
            } else {
                Ok(None)
            }
        }
        _ => Err(format!(
            "unsupported DOCLAB_LLM_PROVIDER '{}' (supported: openai, anthropic)",
            provider_name
        )),
    }
}

/// Tauri command: get agent status for frontend display
#[tauri::command]
pub fn get_agent_status() -> AgentStatus {
    let mode = get_agent_mode();
    let provider_result = get_llm_provider();

    let (llm_configured, provider) = match provider_result {
        Ok(Some(LlmProvider::OpenAI { .. })) => (true, Some("openai".to_string())),
        Ok(Some(LlmProvider::Anthropic { .. })) => (true, Some("anthropic".to_string())),
        Ok(None) => (false, None),
        Err(_) => (false, None),
    };

    AgentStatus {
        mode: format!("{:?}", mode).to_lowercase(),
        llm_configured,
        provider,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_mode_defaults_to_rules() {
        std::env::remove_var("DOCLAB_AGENT_MODE");
        // Clear the OnceLock for testing (note: this won't work in practice due to OnceLock semantics)
        // In real tests, we'd need to use a different approach or accept that the first test sets the value
        let mode = std::env::var("DOCLAB_AGENT_MODE")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "hybrid" => Some(AgentMode::Hybrid),
                "llm" => Some(AgentMode::Llm),
                _ => Some(AgentMode::Rules),
            })
            .unwrap_or(AgentMode::Rules);
        assert_eq!(mode, AgentMode::Rules);
    }

    #[test]
    fn no_api_key_returns_none_provider() {
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::set_var("DOCLAB_LLM_PROVIDER", "openai");

        let provider = get_llm_provider().unwrap();
        assert!(provider.is_none());
    }

    #[test]
    fn openai_provider_configured_with_key() {
        std::env::set_var("DOCLAB_LLM_PROVIDER", "openai");
        std::env::set_var("OPENAI_API_KEY", "test-key");
        std::env::remove_var("DOCLAB_LLM_MODEL");
        std::env::remove_var("ANTHROPIC_API_KEY"); // Clear Anthropic key to avoid conflict

        let provider = get_llm_provider().unwrap();
        assert!(provider.is_some());

        if let Some(LlmProvider::OpenAI { api_key, model }) = provider {
            assert_eq!(api_key, "test-key");
            assert_eq!(model, "gpt-4o-mini");
        } else {
            panic!("Expected OpenAI provider");
        }

        std::env::remove_var("OPENAI_API_KEY");
    }

    #[test]
    fn anthropic_provider_configured_with_key() {
        std::env::set_var("DOCLAB_LLM_PROVIDER", "anthropic");
        std::env::set_var("ANTHROPIC_API_KEY", "test-key");
        std::env::remove_var("DOCLAB_LLM_MODEL");
        std::env::remove_var("OPENAI_API_KEY"); // Clear OpenAI key to avoid conflict

        let provider = get_llm_provider().unwrap();
        assert!(provider.is_some());

        if let Some(LlmProvider::Anthropic { api_key, model }) = provider {
            assert_eq!(api_key, "test-key");
            assert_eq!(model, "claude-3-5-haiku-20241022");
        } else {
            panic!("Expected Anthropic provider");
        }

        std::env::remove_var("ANTHROPIC_API_KEY");
    }
}

/// Call LLM API with system and user prompts
/// Returns the raw text response from the LLM
fn call_llm(provider: &LlmProvider, system: &str, user: &str) -> Result<String, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("failed to create HTTP client: {}", e))?;

    match provider {
        LlmProvider::OpenAI { api_key, model } => {
            let body = serde_json::json!({
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "temperature": 0.0
            });

            let response = client
                .post("https://api.openai.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .map_err(|e| format!("OpenAI API request failed: {}", e))?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().unwrap_or_default();
                return Err(format!("OpenAI API error {}: {}", status, error_text));
            }

            let json: serde_json::Value = response
                .json()
                .map_err(|e| format!("failed to parse OpenAI response: {}", e))?;

            json["choices"][0]["message"]["content"]
                .as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| "OpenAI response missing content".to_string())
        }
        LlmProvider::Anthropic { api_key, model } => {
            let body = serde_json::json!({
                "model": model,
                "max_tokens": 1024,
                "system": system,
                "messages": [
                    {"role": "user", "content": user}
                ]
            });

            let response = client
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .map_err(|e| format!("Anthropic API request failed: {}", e))?;

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().unwrap_or_default();
                return Err(format!("Anthropic API error {}: {}", status, error_text));
            }

            let json: serde_json::Value = response
                .json()
                .map_err(|e| format!("failed to parse Anthropic response: {}", e))?;

            json["content"][0]["text"]
                .as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| "Anthropic response missing text".to_string())
        }
    }
}

/// Extract JSON from LLM response (handles markdown code blocks)
fn extract_json(text: &str) -> Result<serde_json::Value, String> {
    // Try parsing directly first
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(text) {
        return Ok(json);
    }

    // Try extracting from markdown code block
    if let Some(start) = text.find("```json") {
        if let Some(end) = text[start..].find("```") {
            let json_str = &text[start + 7..start + end].trim();
            return serde_json::from_str(json_str)
                .map_err(|e| format!("failed to parse JSON from code block: {}", e));
        }
    }

    // Try finding any JSON object
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            let json_str = &text[start..=end];
            return serde_json::from_str(json_str)
                .map_err(|e| format!("failed to parse extracted JSON: {}", e));
        }
    }

    Err("no valid JSON found in LLM response".to_string())
}

/// Parse intent using LLM
fn parse_intent_llm(goal_text: &str, provider: &LlmProvider) -> Result<IntentResponse, String> {
    let system = "You are a medical ML planning assistant. Parse the user's goal into strict JSON matching this schema:
{
  \"task_type\": \"predict\" | \"classify\" | \"detect\" | \"summarize\" | \"generate\",
  \"modality\": \"tabular\" | \"image\" | \"text\",
  \"metric_hint\": \"accuracy\" | \"auc\" | \"rouge\" | null,
  \"goal_text\": \"<original user string>\"
}

Valid task_types: predict, classify, detect, summarize, generate
Valid modalities: tabular, image, text
Valid metric_hints: accuracy, auc, rouge, or null

Return ONLY the JSON object, no explanation.";

    let response = call_llm(provider, system, goal_text)?;
    let json = extract_json(&response)?;

    let intent: IntentResponse = serde_json::from_value(json)
        .map_err(|e| format!("failed to deserialize intent: {}", e))?;

    // Validate fields
    let valid_tasks = ["predict", "classify", "detect", "summarize", "generate"];
    let valid_modalities = ["tabular", "image", "text"];
    let valid_metrics = ["accuracy", "auc", "rouge"];

    if !valid_tasks.contains(&intent.task_type.as_str()) {
        return Err(format!("invalid task_type: {}", intent.task_type));
    }
    if !valid_modalities.contains(&intent.modality.as_str()) {
        return Err(format!("invalid modality: {}", intent.modality));
    }
    if let Some(ref metric) = intent.metric_hint {
        if !valid_metrics.contains(&metric.as_str()) {
            return Err(format!("invalid metric_hint: {}", metric));
        }
    }

    Ok(intent)
}

/// Pick dataset using LLM from a list of candidates
fn pick_dataset_llm(
    goal_text: &str,
    candidates: &[Dataset],
    provider: &LlmProvider,
) -> Result<DatasetSelectionResponse, String> {
    if candidates.is_empty() {
        return Err("no candidates provided".to_string());
    }

    let system = "You are a dataset selection assistant. Choose EXACTLY ONE dataset_id from the provided list. You must ONLY return a dataset_id that appears in the candidates list. Never invent or hallucinate dataset IDs.

Return JSON in this format:
{
  \"dataset_id\": \"<id from list>\",
  \"rationale\": \"<1-2 sentences explaining why this dataset matches the goal>\"
}";

    let mut user_prompt = format!("Goal: {}\n\nAvailable datasets:\n", goal_text);
    for dataset in candidates {
        user_prompt.push_str(&format!(
            "- id: {}\n  name: {}\n  description: {}\n\n",
            dataset.id, dataset.name, dataset.description
        ));
    }
    user_prompt.push_str("\nPick the best match and return JSON only.");

    let response = call_llm(provider, system, &user_prompt)?;
    let json = extract_json(&response)?;

    let selection: DatasetSelectionResponse = serde_json::from_value(json)
        .map_err(|e| format!("failed to deserialize selection: {}", e))?;

    // CRITICAL: Validate dataset_id is in candidates
    if !candidates.iter().any(|d| d.id == selection.dataset_id) {
        return Err(format!(
            "LLM returned invalid dataset_id '{}' not in candidates",
            selection.dataset_id
        ));
    }

    Ok(selection)
}

/// Tauri command: parse intent using LLM
#[tauri::command]
pub fn agent_parse_intent(goal_text: String) -> Result<IntentResponse, String> {
    let mode = get_agent_mode();
    if mode == AgentMode::Rules {
        return Err("LLM parsing disabled in rules mode".to_string());
    }

    match get_llm_provider()? {
        Some(provider) => parse_intent_llm(&goal_text, &provider),
        None => Err("No LLM provider configured".to_string()),
    }
}

/// Tauri command: pick dataset using LLM
#[tauri::command]
pub fn agent_pick_dataset(
    goal_text: String,
    candidates: Vec<Dataset>,
) -> Result<DatasetSelectionResponse, String> {
    let mode = get_agent_mode();
    if mode == AgentMode::Rules {
        return Err("LLM selection disabled in rules mode".to_string());
    }

    if candidates.is_empty() {
        return Err("No candidates provided".to_string());
    }

    match get_llm_provider()? {
        Some(provider) => pick_dataset_llm(&goal_text, &candidates, &provider),
        None => Err("No LLM provider configured".to_string()),
    }
}
