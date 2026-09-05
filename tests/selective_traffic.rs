use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
    response::Response,
    routing::post,
};
use claude_code_proxy::{
    config::AliasProvider, provider::Provider, providers::anthropic::AnthropicProvider,
    registry::Registry, server::app,
};
use http_body_util::BodyExt;
use std::{path::Path, sync::Arc};
use tower::ServiceExt;

// One test in its own executable: set environment before creating runtime threads.
#[test]
fn passthrough_logs_redact_auth_and_do_not_capture_private_payloads() {
    let dir = tempfile::tempdir().unwrap();
    unsafe {
        std::env::set_var("XDG_STATE_HOME", dir.path());
        std::env::set_var("CCP_CONFIG_DIR", dir.path().join("config"));
        std::env::set_var("CCP_TRAFFIC_LOG", "1");
        std::env::set_var("CCP_AUTO_REVIEW_MODEL", "gpt-6-astra");
    }
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
        async fn upstream(req: Request<Body>) -> Response {
            assert_eq!(req.headers()["authorization"], "Bearer oauth-private-sentinel");
            assert_eq!(req.headers()["x-api-key"], "api-private-sentinel");
            let bytes = req.into_body().collect().await.unwrap().to_bytes();
            assert!(String::from_utf8_lossy(&bytes).contains("request-private-sentinel"));
            Response::builder().status(StatusCode::UNAUTHORIZED)
                .header("content-type", "text/plain")
                .body(Body::from("response-private-sentinel oauth-private-sentinel")).unwrap()
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, Router::new().route("/v1/messages", post(upstream))).await.unwrap(); });
        let registry = Registry::from_providers(AliasProvider::Kimi, [Arc::new(AnthropicProvider::new(&url)) as Arc<dyn Provider>]);
        let raw = r#"{"model":"claude-fable-5-1","messages":[{"role":"user","content":"request-private-sentinel"}],"system":[{"text":"You are a security monitor for autonomous AI coding agents."}],"stream":false}"#;
        let response = app(Arc::new(registry)).oneshot(Request::post("/v1/messages?api_key=query-private-sentinel")
            .header("authorization", "Bearer oauth-private-sentinel")
            .header("x-api-key", "api-private-sentinel")
            .header("proxy-authorization", "proxy-private-sentinel")
            .header("content-type", "application/json")
            .body(Body::from(raw)).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "configured auto-review must not steal Anthropic requests");
        assert_eq!(response.into_body().collect().await.unwrap().to_bytes(), "response-private-sentinel oauth-private-sentinel");
        task.abort();
    });
    fn read_tree(path: &Path, documents: &mut Vec<String>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                read_tree(&path, documents);
            } else {
                documents.push(std::fs::read_to_string(path).unwrap());
            }
        }
    }
    let mut documents = Vec::new();
    read_tree(dir.path(), &mut documents);
    let logs = documents.join("\n");
    assert!(logs.contains("000") || logs.contains("reqId"));
    assert!(logs.contains("anthropic"));
    assert!(logs.contains("[redacted"));
    assert!(dir.path().join("claude-code-proxy/traffic").is_dir());
    for secret in [
        "oauth-private-sentinel",
        "api-private-sentinel",
        "proxy-private-sentinel",
        "query-private-sentinel",
        "request-private-sentinel",
        "response-private-sentinel",
    ] {
        assert!(
            !logs.contains(secret),
            "private value appeared in structured capture: {secret}"
        );
    }
}
