use claude_code_proxy::{config::AliasProvider, registry::Registry};

#[test]
fn selective_routing_overrides_alias_configuration_and_affinity() {
    for configured in [AliasProvider::Codex, AliasProvider::Kimi] {
        let registry = Registry::new(configured);
        for (model, expected) in [
            ("claude-fable-5-1", "anthropic"),
            ("claude-fable-5", "anthropic"),
            ("fable", "anthropic"),
            ("opus", "anthropic"),
            ("sonnet", "anthropic"),
            ("mythos", "anthropic"),
            ("claude-opus-5", "anthropic"),
            ("claude-future-99", "anthropic"),
            ("claude-haiku-9", "anthropic"),
            ("haiku", "codex"),
            ("claude-haiku-4-5", "codex"),
            ("claude-haiku-4-5-20251001", "codex"),
            ("gpt-6-astra", "codex"),
            ("gpt-6-astra[1m]", "codex"),
            ("haiku[1m]", "codex"),
            ("claude-haiku-4-5[1m]", "codex"),
            ("claude-haiku-4-5-20251001[1m]", "codex"),
            ("claude-fable-5-1[1m]", "anthropic"),
            ("kimi-k3", "kimi"),
            ("grok-4.6", "grok"),
            ("cursor:opus", "cursor"),
        ] {
            for affinity in [None, Some(&AliasProvider::Kimi)] {
                assert_eq!(
                    registry
                        .provider_for_model(model, affinity)
                        .map(|p| p.name()),
                    Some(expected),
                    "{model}"
                );
            }
        }
    }
}

use axum::{
    Router,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    response::Response,
    routing::post,
};
use claude_code_proxy::server::app;
use http_body_util::BodyExt;
use serde_json::json;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;
use tower::ServiceExt;

#[derive(Clone, Debug)]
struct Seen {
    uri: String,
    headers: HeaderMap,
    body: Bytes,
}

async fn upstream(State(seen): State<Arc<Mutex<Vec<Seen>>>>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let body = body.collect().await.unwrap().to_bytes();
    seen.lock().unwrap().push(Seen {
        uri: parts.uri.to_string(),
        headers: parts.headers,
        body,
    });
    Response::builder()
        .status(StatusCode::IM_A_TEAPOT)
        .header("content-type", "application/json")
        .header("x-request-id", "sentinel-request")
        .header("connection", "close, x-nominated")
        .header("x-nominated", "must-not-relay")
        .header("x-claude-code-secret", "must-not-relay")
        .header("set-cookie", "response-cookie=preserved")
        .body(Body::from(r#"{"sentinel":"unchanged error"}"#))
        .unwrap()
}

async fn mock_upstream() -> (String, Arc<Mutex<Vec<Seen>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let router = Router::new()
        .route("/{*path}", post(upstream))
        .with_state(seen.clone());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (url, seen)
}

fn request(uri: &str, model: &str, raw: &'static str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream")
        .header("authorization", "Bearer oauth-sentinel")
        .header("anthropic-version", "2023-06-01")
        .header("anthropic-beta", "oauth-sentinel-beta")
        .header("anthropic-future", "future-sentinel")
        .header("x-claude-code-session-id", "must-not-forward")
        .header("cookie", "must-not-forward")
        .header("x-api-key", "api-key-sentinel")
        .header("user-agent", "selective-test-client")
        .header("connection", "anthropic-strip, x-api-drop")
        .header("anthropic-strip", "must-not-forward")
        .header("proxy-authorization", "proxy-secret-sentinel")
        .header("x-claude-code-agent-id", "agent-private-sentinel")
        .body(Body::from(raw.replace("$MODEL", model)))
        .unwrap()
}

#[tokio::test]
async fn non_haiku_claude_passthrough_preserves_request_and_response() {
    let (url, seen) = mock_upstream().await;
    let registry = test_registry(&url);
    let app = app(Arc::new(registry));
    let raw = r#"{ "model" : "$MODEL", "messages" : [ { "role":"user", "content":"bytes stay exact" } ] }"#;
    let expected = raw.replace("$MODEL", "claude-fable-5-1");

    let response = app
        .oneshot(request("/v1/messages?beta=true", "claude-fable-5-1", raw))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::IM_A_TEAPOT);
    assert_eq!(response.headers()["x-request-id"], "sentinel-request");
    assert!(response.headers().get("connection").is_none());
    assert!(response.headers().get("x-nominated").is_none());
    assert!(response.headers().get("x-claude-code-secret").is_none());
    assert_eq!(response.headers()["content-type"], "application/json");
    assert_eq!(
        response.headers()["set-cookie"],
        "response-cookie=preserved"
    );
    assert_eq!(
        response.into_body().collect().await.unwrap().to_bytes(),
        r#"{"sentinel":"unchanged error"}"#
    );
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].uri, "/v1/messages?beta=true");
    assert_eq!(seen[0].body, expected);
    assert_eq!(seen[0].headers["authorization"], "Bearer oauth-sentinel");
    assert_eq!(seen[0].headers["anthropic-future"], "future-sentinel");
    assert!(seen[0].headers.get("x-claude-code-session-id").is_none());
    assert!(seen[0].headers.get("cookie").is_none());
    assert_eq!(seen[0].headers["x-api-key"], "api-key-sentinel");
    assert_eq!(seen[0].headers["user-agent"], "selective-test-client");
    assert_eq!(seen[0].headers["accept"], "text/event-stream");
    assert!(seen[0].headers.get("anthropic-strip").is_none());
    assert!(seen[0].headers.get("proxy-authorization").is_none());
    assert!(seen[0].headers.get("x-claude-code-agent-id").is_none());
}

#[tokio::test]
async fn count_tokens_uses_identical_anthropic_passthrough() {
    let (url, seen) = mock_upstream().await;
    let app = app(Arc::new(test_registry(&url)));
    let raw = r#"{"model":"$MODEL","messages":[]}"#;
    let response = app
        .oneshot(request(
            "/v1/messages/count_tokens?beta=true",
            "sonnet",
            raw,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::IM_A_TEAPOT);
    assert_eq!(
        seen.lock().unwrap()[0].uri,
        "/v1/messages/count_tokens?beta=true"
    );
}

#[tokio::test]
async fn oversized_body_preserves_current_400_without_upstream_call() {
    let (url, seen) = mock_upstream().await;
    let app = app(Arc::new(test_registry(&url)));
    let huge =
        json!({"model":"claude-fable-5-1", "messages":[], "padding":"x".repeat(claude_code_proxy::openai_compat::MAX_OPENAI_REQUEST_BYTES + 1)})
            .to_string();
    let response = app
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(huge))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn redirects_are_not_followed() {
    async fn redirect() -> Response {
        Response::builder()
            .status(StatusCode::TEMPORARY_REDIRECT)
            .header("location", "/should-not-run")
            .body(Body::empty())
            .unwrap()
    }
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new().route("/v1/messages", post(redirect)),
        )
        .await
        .unwrap()
    });
    let app = app(Arc::new(test_registry(&url)));
    let response = app
        .oneshot(request(
            "/v1/messages",
            "claude-fable-5-1",
            r#"{"model":"$MODEL","messages":[]}"#,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
}

#[tokio::test]
async fn sse_bytes_and_chunk_timing_are_streamed_without_buffering() {
    use futures_util::StreamExt;
    type Release = Arc<Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>;
    const DEADLOCK_GUARD: std::time::Duration = std::time::Duration::from_secs(30);

    async fn sse(State(release): State<Release>) -> Response {
        let receiver = release.lock().unwrap().take().unwrap();
        let first = futures_util::stream::once(async {
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: first\n\n"))
        });
        let second = futures_util::stream::once(async move {
            receiver
                .await
                .expect("test must explicitly release the second chunk");
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: second\n\n"))
        });
        Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(first.chain(second)))
            .unwrap()
    }

    let (release, receiver) = tokio::sync::oneshot::channel();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new()
                .route("/v1/messages", post(sse))
                .with_state(Arc::new(Mutex::new(Some(receiver)))),
        )
        .await
        .unwrap()
    });
    // The unreleased channel, not elapsed wall-clock time, proves incremental delivery.
    // Bound the entire exchange only to catch a buffering deadlock.
    tokio::time::timeout(DEADLOCK_GUARD, async {
        let response = app(Arc::new(test_registry(&url)))
            .oneshot(request(
                "/v1/messages",
                "claude-fable-5-1",
                r#"{"model":"$MODEL","messages":[],"stream":true}"#,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let mut body = response.into_body();
        let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
        assert_eq!(first, "data: first\n\n");
        release
            .send(())
            .expect("upstream must still be waiting for release");
        let second = body.frame().await.unwrap().unwrap().into_data().unwrap();
        assert_eq!(second, "data: second\n\n");
        assert!(body.frame().await.is_none());
    })
    .await
    .expect("streaming exchange deadlocked before explicit second-chunk release");
}

fn test_registry(url: &str) -> Registry {
    Registry::from_providers(
        AliasProvider::Kimi,
        [
            Arc::new(claude_code_proxy::providers::anthropic::AnthropicProvider::new(url))
                as Arc<dyn claude_code_proxy::provider::Provider>,
            Arc::new(NoCodex),
        ],
    )
}

struct NoCodex;
#[async_trait::async_trait]
impl claude_code_proxy::provider::Provider for NoCodex {
    fn name(&self) -> &'static str {
        "codex"
    }
    fn supported_models(&self) -> Vec<String> {
        vec!["gpt-6-astra".into()]
    }
    fn cli(&self) -> &'static dyn claude_code_proxy::provider::CliHandlers {
        panic!("no CLI in transport tests")
    }
    async fn handle_messages(
        &self,
        _: claude_code_proxy::anthropic::schema::MessagesRequest,
        _: claude_code_proxy::provider::RequestContext,
    ) -> Response {
        panic!("Anthropic request must never fall back to Codex")
    }
    async fn handle_count_tokens(
        &self,
        _: claude_code_proxy::anthropic::schema::MessagesRequest,
        _: claude_code_proxy::provider::RequestContext,
    ) -> Response {
        panic!("Anthropic count_tokens must never fall back to Codex")
    }
}

#[tokio::test]
async fn unavailable_anthropic_does_not_fall_back() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let response = app(Arc::new(test_registry(&url)))
        .oneshot(request(
            "/v1/messages",
            "claude-fable-5-1",
            r#"{"model":"$MODEL","messages":[]}"#,
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert!(!String::from_utf8_lossy(&body).contains(&url));
}

#[tokio::test]
async fn malformed_json_and_missing_model_do_not_reach_upstream() {
    let (url, seen) = mock_upstream().await;
    for raw in ["{", "", r#"{"messages":[]}"#, r#"{"model":42}"#] {
        let response = app(Arc::new(test_registry(&url)))
            .oneshot(request("/v1/messages", "unused", raw))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
    assert!(seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn future_payload_schema_and_model_hint_are_not_translated() {
    let (url, seen) = mock_upstream().await;
    let raw = r#"{ "model":"$MODEL", "messages": {"future":true}, "max_tokens":4294967296, "stream":"future" }"#;
    let model = "claude-fable-5-1[1m]";
    let response = app(Arc::new(test_registry(&url)))
        .oneshot(request("/v1/messages?x=%2F&x=two", model, raw))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::IM_A_TEAPOT);
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0].body, raw.replace("$MODEL", model));
    assert_eq!(seen[0].uri, "/v1/messages?x=%2F&x=two");
}

#[test]
fn discovery_owns_exact_fable_id_only_in_anthropic() {
    let registry = Registry::new(AliasProvider::Kimi);
    assert!(
        registry
            .all_supported_models()
            .contains(&("claude-fable-5-1".into(), "anthropic".into()))
    );
    assert!(
        !registry
            .supported_models_for("codex")
            .iter()
            .any(|m| m.contains("fable"))
    );
    assert!(
        !claude_code_proxy::providers::codex::translate::model_allowlist::is_valid_model_for_codex(
            "claude-fable-5-1"
        )
    );
}
