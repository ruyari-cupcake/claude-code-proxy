use std::collections::HashSet;

use crate::config;

use super::request::ServiceTier;

pub const ALLOWED_MODELS: &[&str] = &[
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
];

pub const MODEL_ALIASES: &[(&str, &str)] = &[
    ("haiku", "gpt-5.6-luna"),
    ("claude-haiku-4-5", "gpt-5.6-luna"),
    ("claude-haiku-4-5-20251001", "gpt-5.6-luna"),
];

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub model: String,
    pub service_tier: Option<ServiceTier>,
}

fn fast_model_aliases() -> HashSet<String> {
    ALLOWED_MODELS.iter().map(|m| format!("{m}-fast")).collect()
}

fn resolve_fast_model_alias(model: &str) -> ResolvedModel {
    let fast_set = fast_model_aliases();
    if fast_set.contains(model) {
        let base = model.trim_end_matches("-fast");
        ResolvedModel {
            model: base.to_string(),
            service_tier: Some(ServiceTier::Priority),
        }
    } else {
        ResolvedModel {
            model: model.to_string(),
            service_tier: None,
        }
    }
}

pub fn resolve_model_request(model: &str) -> ResolvedModel {
    resolve_model_request_with_config_override(model, true)
}

pub fn resolve_model_request_with_config_override(
    model: &str,
    apply_config_override: bool,
) -> ResolvedModel {
    let alias = MODEL_ALIASES
        .iter()
        .find(|(alias, _)| *alias == model)
        .map(|(_, target)| *target)
        .unwrap_or(model);

    let requested = resolve_fast_model_alias(alias);

    let override_model = apply_config_override.then(config::codex_model).flatten();
    let resolved = match override_model {
        Some(ref val) if !val.is_empty() => resolve_fast_model_alias(val),
        _ => requested.clone(),
    };

    ResolvedModel {
        model: resolved.model,
        service_tier: if requested.service_tier == Some(ServiceTier::Priority)
            || resolved.service_tier == Some(ServiceTier::Priority)
        {
            Some(ServiceTier::Priority)
        } else {
            resolved.service_tier
        },
    }
}

pub fn resolve_model(model: &str) -> String {
    resolve_model_request(model).model
}

#[derive(Debug, Clone)]
pub struct ModelNotAllowedError {
    pub model: String,
}

impl std::fmt::Display for ModelNotAllowedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Model not allowed: {}", self.model)
    }
}

pub fn assert_allowed_model(model: &str) -> Result<(), ModelNotAllowedError> {
    if ALLOWED_MODELS.contains(&model) {
        Ok(())
    } else {
        Err(ModelNotAllowedError {
            model: model.to_string(),
        })
    }
}

pub fn uses_responses_lite(model: &str) -> bool {
    matches!(
        model,
        "gpt-5.6-luna" | "gpt-5.6-sol" | "gpt-5.6-terra" | "gpt-6-astra"
    )
}

/// `gpt-5.6-luna` exists only behind the Responses Lite lane; the full
/// Responses API resolves it to a `-free` variant and returns 404 (Model not
/// found gpt-5.6-luna-free-...). Hosted web_search requests must run on the
/// full lane, so luna is upgraded to its nearest full-lane sibling.
pub fn full_lane_web_search_model(model: &str) -> &str {
    if model == "gpt-5.6-luna" {
        "gpt-5.6-sol"
    } else {
        model
    }
}

pub fn is_valid_model_for_codex(model: &str) -> bool {
    if ALLOWED_MODELS.contains(&model) {
        return true;
    }
    let fast_set = fast_model_aliases();
    if fast_set.contains(model) {
        return true;
    }
    MODEL_ALIASES.iter().any(|(alias, _)| *alias == model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haiku_resolves_to_luna() {
        let r = resolve_model_request("haiku");
        assert_eq!(r.model, "gpt-5.6-luna");
    }

    #[test]
    fn web_search_upgrades_luna_to_full_lane_sibling() {
        assert_eq!(full_lane_web_search_model("gpt-5.6-luna"), "gpt-5.6-sol");
        assert_eq!(full_lane_web_search_model("gpt-5.6-sol"), "gpt-5.6-sol");
        assert_eq!(full_lane_web_search_model("gpt-5.6-terra"), "gpt-5.6-terra");
        assert_eq!(full_lane_web_search_model("gpt-5.4"), "gpt-5.4");
    }

    #[test]
    fn non_haiku_claude_models_are_not_codex_aliases() {
        for model in [
            "sonnet",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "opus",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "fable",
            "claude-fable-5",
            "claude-fable-5-1",
            "mythos",
        ] {
            assert_eq!(
                resolve_model_request_with_config_override(model, false).model,
                model
            );
            assert!(!is_valid_model_for_codex(model));
        }
    }

    #[test]
    fn fast_suffix_adds_priority() {
        let r = resolve_model_request("gpt-5.6-sol-fast");
        assert_eq!(r.model, "gpt-5.6-sol");
        assert_eq!(r.service_tier, Some(ServiceTier::Priority));
    }

    #[test]
    fn allowed_models_accept_base() {
        assert!(assert_allowed_model("gpt-5.4").is_ok());
        assert!(assert_allowed_model("gpt-5.6-sol").is_ok());
        assert!(assert_allowed_model("gpt-5.6-terra").is_ok());
        assert!(assert_allowed_model("gpt-6-astra").is_ok());
        assert!(assert_allowed_model("gpt-5.6-luna").is_ok());
    }

    #[test]
    fn not_allowed_rejected() {
        assert!(assert_allowed_model("gpt-7").is_err());
    }
}
