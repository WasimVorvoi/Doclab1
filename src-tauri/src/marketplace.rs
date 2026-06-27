use serde::{Deserialize, Serialize};
use std::path::Path;

/// A curated dataset entry. The agent (M5) may only ever return ids that
/// exist here — there is no live Hugging Face search at runtime.
///
/// Read from `datasets.yaml` (snake_case) but serialized to the frontend in
/// camelCase, matching the convention used by the structs in `experiments.rs`
/// and the `Dataset` interface in `src/types/tauri.ts`. Without the camelCase
/// serialize, the UI sees `data_type`/`task_types` as `undefined` and crashes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all(serialize = "camelCase", deserialize = "snake_case"))]
pub struct Dataset {
    pub id: String,
    pub name: String,
    pub hf_id: String,
    /// Pinned commit SHA or tag — never a moving branch like `main`.
    pub revision: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub category: String,
    pub data_type: String,
    pub task_types: Vec<String>,
    #[serde(default)]
    pub modality: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub size: String,
    pub label_column: String,
    #[serde(default)]
    pub limitations: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marketplace {
    pub datasets: Vec<Dataset>,
}

impl Marketplace {
    /// Parse and validate the curated index. Fails loud on a malformed or
    /// incomplete entry — a missing dataset at demo time is worse than a
    /// startup error in dev.
    pub fn load(path: &Path) -> Result<Marketplace, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read marketplace {}: {e}", path.display()))?;
        let market: Marketplace =
            serde_yaml::from_str(&text).map_err(|e| format!("invalid marketplace YAML: {e}"))?;
        market.validate()?;
        Ok(market)
    }

    fn validate(&self) -> Result<(), String> {
        for d in &self.datasets {
            let missing = |f: &str| format!("dataset '{}' missing required field: {f}", d.id);
            if d.id.is_empty() {
                return Err("a dataset entry is missing required field: id".into());
            }
            if d.name.is_empty() {
                return Err(missing("name"));
            }
            if d.hf_id.is_empty() {
                return Err(missing("hf_id"));
            }
            if d.revision.is_empty() {
                return Err(missing("revision"));
            }
            if d.revision == "main" {
                return Err(format!(
                    "dataset '{}' has unpinned revision 'main' — pin a commit SHA/tag",
                    d.id
                ));
            }
            if d.data_type.is_empty() {
                return Err(missing("data_type"));
            }
            if d.task_types.is_empty() {
                return Err(missing("task_types"));
            }
            if d.label_column.is_empty() {
                return Err(missing("label_column"));
            }
        }
        Ok(())
    }

    /// Keyword + optional type filters, ranked by match strength. The agent
    /// can only ever see datasets returned from here.
    pub fn query(
        &self,
        keyword: Option<&str>,
        data_type: Option<&str>,
        task_type: Option<&str>,
    ) -> Vec<Dataset> {
        let kw = keyword.map(|k| k.to_lowercase());
        let mut scored: Vec<(u32, &Dataset)> = self
            .datasets
            .iter()
            .filter(|d| match data_type {
                Some(dt) => d.data_type.eq_ignore_ascii_case(dt),
                None => true,
            })
            .filter(|d| match task_type {
                Some(tt) => d.task_types.iter().any(|t| t.eq_ignore_ascii_case(tt)),
                None => true,
            })
            .filter_map(|d| match &kw {
                Some(k) => {
                    let score = d.keyword_score(k);
                    if score > 0 {
                        Some((score, d))
                    } else {
                        None
                    }
                }
                None => Some((0, d)),
            })
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.into_iter().map(|(_, d)| d.clone()).collect()
    }
}

impl Dataset {
    /// Count of fields whose text contains the (already-lowercased) keyword.
    fn keyword_score(&self, kw: &str) -> u32 {
        let fields = [
            &self.name,
            &self.description,
            &self.category,
            &self.modality,
            &self.id,
        ];
        fields
            .iter()
            .filter(|f| f.to_lowercase().contains(kw))
            .count() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"
datasets:
  - id: diabetes_readmission
    name: Diabetes 130-US Hospitals Readmission
    hf_id: imodels/diabetes-readmission
    revision: 191ab1f0aa68d52f6cd55d68df57849fad1751ca
    description: Predict 30-day hospital readmission.
    category: readmission
    data_type: tabular
    task_types: [predict, classify]
    modality: encounter records
    label_column: readmitted
"#;

    fn parse(yaml: &str) -> Result<Marketplace, String> {
        let m: Marketplace = serde_yaml::from_str(yaml).map_err(|e| format!("yaml: {e}"))?;
        m.validate()?;
        Ok(m)
    }

    #[test]
    fn loads_and_validates() {
        let m = parse(VALID).expect("valid marketplace should load");
        assert_eq!(m.datasets.len(), 1);
        assert_eq!(m.datasets[0].label_column, "readmitted");
    }

    #[test]
    fn query_keyword_hit() {
        let m = parse(VALID).unwrap();
        let hits = m.query(Some("readmission"), None, None);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "diabetes_readmission");
    }

    #[test]
    fn query_keyword_miss_returns_empty() {
        let m = parse(VALID).unwrap();
        assert!(m.query(Some("chest x-ray"), None, None).is_empty());
    }

    #[test]
    fn query_type_filter() {
        let m = parse(VALID).unwrap();
        assert_eq!(m.query(None, Some("tabular"), None).len(), 1);
        assert!(m.query(None, Some("image"), None).is_empty());
    }

    #[test]
    fn missing_label_column_fails() {
        let bad = VALID.replace("    label_column: readmitted\n", "");
        let err = parse(&bad).expect_err("missing label_column must fail");
        assert!(err.contains("label_column"), "got: {err}");
    }

    #[test]
    fn unpinned_revision_fails() {
        let bad = VALID.replace(
            "revision: 191ab1f0aa68d52f6cd55d68df57849fad1751ca",
            "revision: main",
        );
        let err = parse(&bad).expect_err("unpinned revision must fail");
        assert!(err.contains("main"), "got: {err}");
    }

    /// Regression: the frontend (`src/types/tauri.ts`) and the in-app agent
    /// read camelCase keys; if `Dataset` serializes snake_case the Datasets
    /// tab and Plan flow crash on `dataset.dataType`/`taskTypes` being
    /// undefined. YAML must still deserialize from snake_case.
    #[test]
    fn serializes_camel_case_for_frontend() {
        let m = parse(VALID).unwrap();
        let json = serde_json::to_string(&m.datasets[0]).unwrap();
        assert!(json.contains("\"dataType\""), "expected camelCase: {json}");
        assert!(json.contains("\"taskTypes\""), "expected camelCase: {json}");
        assert!(json.contains("\"hfId\""), "expected camelCase: {json}");
        assert!(json.contains("\"labelColumn\""), "expected camelCase: {json}");
        assert!(!json.contains("\"data_type\""), "snake_case leaked: {json}");
    }
}
