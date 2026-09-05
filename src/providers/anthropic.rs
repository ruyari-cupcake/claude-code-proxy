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
            Ok(response) => relay_response(response),
            Err(_error) => crate::anthropic::json_error(
                axum::http::StatusCode::BAD_GATEWAY,
                "api_error",
                "Anthropic upstream request failed",
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

fn relay_response(response: reqwest::Response) -> Response {
    let status = response.status();
    let headers = response.headers().clone();
    let stream = response
        .bytes_stream()
        .map(|chunk| chunk.map_err(std::io::Error::other));
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
        .body(Body::from_stream(stream))
        .expect("upstream response headers are valid")
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
