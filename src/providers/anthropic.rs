use crate::{
    anthropic::schema::MessagesRequest,
    provider::{CliHandlers, PassthroughRequest, Provider, RequestContext},
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use axum::{
    body::Body,
    http::{HeaderMap, HeaderName},
    response::Response,
};
use futures_util::StreamExt;
use reqwest::redirect::Policy;
use std::sync::Arc;

pub const PRODUCTION_UPSTREAM: &str = "https://api.anthropic.com";

#[derive(Clone)]
pub struct AnthropicProvider {
    upstream: Arc<str>,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn production() -> Self {
        Self::new(PRODUCTION_UPSTREAM)
    }

    /// Test-only upstream injection is exposed through the registry constructor; production
    /// always calls `production`, keeping the real destination non-configurable.
    #[doc(hidden)]
    pub fn new(upstream: &str) -> Self {
        let client = reqwest::Client::builder()
            .redirect(Policy::none())
            .build()
            .expect("Anthropic HTTP client configuration is valid");
        Self {
            upstream: Arc::from(upstream.trim_end_matches('/')),
            client,
        }
    }

    pub async fn forward(&self, request: PassthroughRequest) -> Response {
        let url = format!("{}{}", self.upstream, request.path_and_query);
        let mut upstream = self.client.post(url).body(request.raw_body);
        for (name, value) in &request.headers {
            if should_forward_request_header(name)
                && !connection_names_header(&request.headers, name)
            {
                upstream = upstream.header(name, value);
            }
        }
        match upstream.send().await {
            Ok(response) => relay_response(response).await,
            Err(error) => crate::anthropic::json_error(
                axum::http::StatusCode::BAD_GATEWAY,
                "api_error",
                format!("Anthropic upstream request failed: {}", error.without_url()),
            ),
        }
    }
}

fn should_forward_request_header(name: &HeaderName) -> bool {
    let value = name.as_str();
    if value.starts_with("x-claude-code-") || is_hop_by_hop(value) {
        return false;
    }
    matches!(
        value,
        "authorization" | "x-api-key" | "content-type" | "accept" | "user-agent"
    ) || value.starts_with("anthropic-")
}

fn is_hop_by_hop(name: &str) -> bool {
    matches!(
        name,
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    ) || name.starts_with("proxy-")
}

async fn relay_response(response: reqwest::Response) -> Response {
    let status = response.status();
    let headers = response.headers().clone();
    let omission = error_body_omission(&headers);
    let mut fields = serde_json::Map::from_iter([
        ("provider".into(), serde_json::json!("anthropic")),
        ("status".into(), serde_json::json!(status.as_u16())),
    ]);
    let body = if status.as_u16() >= 400 && omission.is_none() {
        // Anthropic sends error JSON chunked (no Content-Length), so bound the read
        // incrementally instead of trusting framing headers: stop buffering past the
        // cap and relay the remainder untouched.
        let mut stream = response.bytes_stream();
        let mut bytes = bytes::BytesMut::new();
        let mut read_error = None;
        let mut over_cap = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(chunk) => {
                    bytes.extend_from_slice(&chunk);
                    if bytes.len() as u64 > MAX_BUFFERED_ERROR_BODY_BYTES {
                        over_cap = true;
                        break;
                    }
                }
                Err(error) => {
                    read_error = Some(std::io::Error::other(error));
                    break;
                }
            }
        }
        if over_cap {
            fields.insert("bodyOmitted".into(), serde_json::json!("body_too_large"));
        } else {
            match error_diagnostic_preview(&bytes) {
                Ok(preview) => {
                    fields.insert("errorBody".into(), serde_json::json!(preview));
                }
                Err(reason) => {
                    fields.insert("bodyOmitted".into(), serde_json::json!(reason));
                }
            }
        }
        // Preserve received bytes, any unread remainder, and any terminal read failure,
        // rather than turning a broken upstream body into a successful empty response.
        let buffered = futures_util::stream::iter(
            std::iter::once(Ok(bytes.freeze())).chain(read_error.map(Err)),
        );
        if over_cap {
            Body::from_stream(
                buffered.chain(stream.map(|chunk| chunk.map_err(std::io::Error::other))),
            )
        } else {
            Body::from_stream(buffered)
        }
    } else {
        Body::from_stream(
            response
                .bytes_stream()
                .map(|chunk| chunk.map_err(std::io::Error::other)),
        )
    };
    if status.as_u16() >= 400 {
        if let Some(reason) = omission {
            fields.insert("bodyOmitted".into(), serde_json::json!(reason));
        }
        // Logger applies shared redaction and JSON-encodes one physical line.
        crate::logging::create_logger("server").warn("anthropic_upstream_error", Some(fields));
    }
    let mut builder = Response::builder().status(status);
    for (name, value) in &headers {
        if !is_hop_by_hop(name.as_str())
            && !name.as_str().starts_with("x-claude-code-")
            && !connection_names_header(&headers, name)
        {
            builder = builder.header(name, value);
        }
    }
    builder
        .body(body)
        .expect("upstream response headers are valid")
}

const MAX_BUFFERED_ERROR_BODY_BYTES: u64 = 64 * 1024;

fn error_body_omission(headers: &HeaderMap) -> Option<&'static str> {
    // Intermediary/non-JSON bodies can echo credentials; only provider JSON is
    // eligible for buffering/logging. Size is bounded by the incremental cap in
    // relay_response, so chunked JSON (Anthropic's normal error framing) qualifies;
    // a declared oversize length short-circuits without reading.
    let is_json = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|media_type| media_type.trim().eq_ignore_ascii_case("application/json"));
    if !is_json {
        return Some("non_json");
    }
    match headers
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
    {
        Some(length) if length > MAX_BUFFERED_ERROR_BODY_BYTES => Some("body_too_large"),
        _ => None,
    }
}

const MAX_ERROR_BODY_LOG_BYTES: usize = 600;

fn error_diagnostic_preview(bytes: &[u8]) -> Result<String, &'static str> {
    let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|_| "invalid_json")?;
    let shape_error = "invalid_error_shape";
    if value.get("type").and_then(serde_json::Value::as_str) != Some("error") {
        return Err(shape_error);
    }
    let error = value.get("error").ok_or(shape_error)?;
    let error_type = error
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or(shape_error)?;
    let message = error
        .get("message")
        .and_then(serde_json::Value::as_str)
        .ok_or(shape_error)?;
    // Bound before shared redaction (which has its own larger string cap), and never
    // copy unexpected provider/intermediary fields into the diagnostic document.
    let mut end = message.len().min(MAX_ERROR_BODY_LOG_BYTES);
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    let message =
        crate::logging::redact_value(serde_json::json!(scrub_token_like_runs(&message[..end])));
    let mut diagnostic = serde_json::json!({
        "type": "error",
        "error": {"type": error_type, "message": message},
    });
    if let Some(request_id) = value.get("request_id") {
        diagnostic["request_id"] = serde_json::json!(request_id.as_str().ok_or(shape_error)?);
    }
    Ok(error_body_preview(diagnostic.to_string().as_bytes()))
}

/// The shared key-based redactor passes bare strings through unchanged, and an
/// upstream error message may echo a credential or signature verbatim. Mask any
/// long unbroken token-like run; real Anthropic diagnostics (field paths, model
/// ids, short words) stay below this length while keys/signatures do not.
fn scrub_token_like_runs(message: &str) -> String {
    const MAX_TOKEN_RUN: usize = 28;
    let mut out = String::with_capacity(message.len());
    let mut run = String::new();
    for ch in message.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '+' | '/' | '=') {
            run.push(ch);
            continue;
        }
        if run.chars().count() >= MAX_TOKEN_RUN {
            out.push_str("[redacted-token]");
        } else {
            out.push_str(&run);
        }
        run.clear();
        if ch != ' ' || !out.ends_with(' ') {
            out.push(ch);
        }
    }
    out.trim_end().to_string()
}

fn error_body_preview(bytes: &[u8]) -> String {
    let mut preview = String::new();
    for ch in String::from_utf8_lossy(bytes).chars() {
        let escaped = if ch.is_control() {
            ch.escape_default().to_string()
        } else {
            ch.to_string()
        };
        // Bound the escaped value, without splitting UTF-8 or an escape sequence.
        if preview.len() + escaped.len() > MAX_ERROR_BODY_LOG_BYTES {
            break;
        }
        preview.push_str(&escaped);
    }
    preview
}

struct AnthropicCli;
static ANTHROPIC_CLI: AnthropicCli = AnthropicCli;

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn supported_models(&self) -> Vec<String> {
        crate::registry::ANTHROPIC_STYLE_ALIASES
            .iter()
            .map(|model| (*model).to_string())
            .collect()
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &ANTHROPIC_CLI
    }

    async fn handle_messages(&self, _body: MessagesRequest, _ctx: RequestContext) -> Response {
        missing_raw_request()
    }

    async fn handle_count_tokens(&self, _body: MessagesRequest, _ctx: RequestContext) -> Response {
        missing_raw_request()
    }

    async fn handle_passthrough(&self, request: PassthroughRequest) -> Response {
        self.forward(request).await
    }
}

fn missing_raw_request() -> Response {
    crate::anthropic::json_error(
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        "api_error",
        "Anthropic passthrough envelope missing",
    )
}

// Connection can nominate otherwise end-to-end headers; strip those on both legs.
fn connection_names_header(headers: &HeaderMap, name: &HeaderName) -> bool {
    headers
        .get_all("connection")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|token| token.trim().eq_ignore_ascii_case(name.as_str()))
}

impl CliHandlers for AnthropicCli {
    fn login(&self) -> Result<()> {
        Err(anyhow!("Anthropic authentication is owned by Claude Code"))
    }
    fn device(&self) -> Result<()> {
        self.login()
    }
    fn status(&self) -> Result<()> {
        Err(anyhow!("Anthropic authentication is owned by Claude Code"))
    }
    fn logout(&self) -> Result<()> {
        Err(anyhow!("Anthropic authentication is owned by Claude Code"))
    }
}
