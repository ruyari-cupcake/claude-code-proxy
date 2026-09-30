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
    "gpt-6.1-sol",
];

pub const MODEL_ALIASES: &[(&str, &str)] = &[
    ("haiku", "gpt-5.6-luna"),
    ("claude-haiku-4-5", "gpt-5.6-luna"),
    ("claude-haiku-4-5-20251001", "gpt-5.6-luna"),
];

/// Codex model ids the OpenAI backend serves. `ALLOWED_MODELS` is the KNOWN list used for
/// discovery (`/v1/models`) and error messages; any other `gpt-*` id is passed through to
/// Codex as well, so a newly released model needs no proxy rebuild — the backend rejects an
/// id it does not know with its own 400. Cupcake patch (2026-09-30): a hard-coded catalogue
/// meant one Rust edit + rebuild + restart per model release.
pub const CODEX_MODEL_PREFIX: &str = "gpt-";

/// Models served on the full Responses API; every other Codex id (the 5.6 family, Astra,
/// GPT-6 / 6.1 Sol and anything newer) lives behind the Responses Lite lane.
pub const FULL_LANE_MODELS: &[&str] = &[
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
];

pub fn is_codex_model_id(model: &str) -> bool {
    model.starts_with(CODEX_MODEL_PREFIX) && model.len() > CODEX_MODEL_PREFIX.len()
}

/// `<codex id>-fast` → `<codex id>`; `None` when the id is not a Codex priority alias.
pub fn strip_fast_alias(model: &str) -> Option<&str> {
    model
        .strip_suffix("-fast")
        .filter(|base| is_codex_model_id(base))
}

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub model: String,
    pub service_tier: Option<ServiceTier>,
}

fn resolve_fast_model_alias(model: &str) -> ResolvedModel {
    match strip_fast_alias(model) {
        Some(base) => ResolvedModel {
            model: base.to_string(),
            service_tier: Some(ServiceTier::Priority),
        },
        None => ResolvedModel {
            model: model.to_string(),
            service_tier: None,
        },
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
    if ALLOWED_MODELS.contains(&model) || is_codex_model_id(model) {
        Ok(())
    } else {
        Err(ModelNotAllowedError {
            model: model.to_string(),
        })
    }
}

pub fn uses_responses_lite(model: &str) -> bool {
    is_codex_model_id(model) && !FULL_LANE_MODELS.contains(&model)
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
    if ALLOWED_MODELS.contains(&model) || is_codex_model_id(model) {
        return true;
    }
    if strip_fast_alias(model).is_some() {
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
        assert!(assert_allowed_model("gpt-6.1-sol").is_ok());
        assert!(uses_responses_lite("gpt-6.1-sol"));
        assert_eq!(
            resolve_model_request("gpt-6.1-sol-fast").model,
            "gpt-6.1-sol"
        );
        assert!(assert_allowed_model("gpt-5.6-luna").is_ok());
    }

    #[test]
    fn not_allowed_rejected() {
        for model in ["o3", "claude-opus-5", "gpt-", "gpt", "kimi-k3"] {
            assert!(assert_allowed_model(model).is_err(), "{model}");
            assert!(!is_valid_model_for_codex(model), "{model}");
        }
    }

    #[test]
    fn unknown_gpt_ids_pass_through_to_codex_on_the_lite_lane() {
        // A model released after this build needs no catalogue edit.
        for model in ["gpt-7", "gpt-6.2-nova", "gpt-6-luna"] {
            assert!(assert_allowed_model(model).is_ok(), "{model}");
            assert!(is_valid_model_for_codex(model), "{model}");
            assert!(uses_responses_lite(model), "{model}");
            let fast = format!("{model}-fast");
            assert!(is_valid_model_for_codex(&fast), "{fast}");
            let r = resolve_model_request(&fast);
            assert_eq!(r.model, model);
            assert_eq!(r.service_tier, Some(ServiceTier::Priority));
        }
        // Known full-lane ids keep the full Responses API.
        for model in FULL_LANE_MODELS {
            assert!(!uses_responses_lite(model), "{model}");
        }
        for model in [
            "gpt-5.6-luna",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-6-astra",
        ] {
            assert!(uses_responses_lite(model), "{model}");
        }
        // `-fast` on a non-Codex id is not an alias.
        assert_eq!(strip_fast_alias("claude-opus-5-fast"), None);
        assert_eq!(strip_fast_alias("gpt-6.1-sol-fast"), Some("gpt-6.1-sol"));
    }
}
