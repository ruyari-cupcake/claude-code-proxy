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
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;

// Isolated executable: configure file logging before starting runtime threads.
#[test]
fn upstream_errors_log_bounded_details_and_relay_exact_bytes() {
    let dir = tempfile::tempdir().unwrap();
    unsafe {
        std::env::set_var("XDG_STATE_HOME", dir.path());
        std::env::set_var("CCP_CONFIG_DIR", dir.path().join("config"));
    }
    let diagnostic = serde_json::json!({"type":"error", "error":{"type":"invalid_request_error", "message":"bad model"}, "request_id":"fixture-request"});
    let mut with_secret = diagnostic.clone();
    with_secret["unexpected"] = serde_json::json!("UNEXPECTED-TOKEN-NOT-LOGGED");
    with_secret["error"]["token"] = serde_json::json!("UNEXPECTED-TOKEN-NOT-LOGGED");
    let small = with_secret.to_string();
    let large = serde_json::json!({"type":"error", "error":{"type":"invalid_request_error", "message":format!("{}TAIL-MUST-NOT-BE-LOGGED", "é".repeat(500))}}).to_string();
    let prefix = r#"{"error":{"message":""#;
    let large_preview = format!("{prefix}{}", "é".repeat((600 - prefix.len()) / 2));
    let controls = serde_json::json!({"type":"error", "error":{"type":"invalid_request_error", "message":"line one\nline two\r\t\0"}}).to_string();
    let masked_input = serde_json::json!({"type":"error", "error":{"type":"invalid_request_error", "message":"signature sk-ant-api03-THISLOOKSLIKEALONGSECRETVALUE0123456789 rejected for messages.1"}}).to_string();
    let masked_expected = serde_json::json!({"type":"error", "error":{"type":"invalid_request_error", "message":"signature [redacted-token] rejected for messages.1"}}).to_string();
    let cases = [
        (StatusCode::BAD_REQUEST, small, Some(diagnostic.to_string())),
        (StatusCode::BAD_REQUEST, masked_input, Some(masked_expected)),
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            large,
            Some(large_preview),
        ),
        (StatusCode::BAD_REQUEST, controls.clone(), Some(controls)),
        (StatusCode::OK, "SUCCESS-NOT-LOGGED".into(), None),
        (
            StatusCode::TEMPORARY_REDIRECT,
            "REDIRECT-NOT-LOGGED".into(),
            None,
        ),
        (
            StatusCode::UNAUTHORIZED,
            "ECHOED-CREDENTIAL-MUST-NOT-BE-LOGGED".into(),
            None,
        ),
        (
            StatusCode::BAD_REQUEST,
            "OVERSIZED-NOT-LOGGED".repeat(4096),
            None,
        ),
        (StatusCode::BAD_REQUEST, "CHUNKED-NOT-LOGGED".into(), None),
        (
            StatusCode::BAD_REQUEST,
            "INVALID-JSON-NOT-LOGGED".into(),
            None,
        ),
        (
            StatusCode::BAD_REQUEST,
            r#"{"unexpected":"WRONG-SHAPE-NOT-LOGGED"}"#.into(),
            None,
        ),
    ];
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
        for (index, (status, body, _)) in cases.iter().enumerate() {
            let content_type = if index == 6 { "text/plain" } else { "application/json; charset=utf-8" };
            let status = *status;
            let expected = body.clone();
            let body = body.clone();
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let router = Router::new().route("/v1/messages", post(move || {
                let body = body.clone();
                async move { Response::builder().status(status).header("content-type", content_type)
                    .header("x-request-id", "error-observability-fixture").body(if index == 8 {
                        Body::from_stream(futures_util::stream::once(async move { Ok::<_, std::io::Error>(body) }))
                    } else { Body::from(body) }).unwrap() }
            }));
            let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
            let registry = Registry::from_providers(AliasProvider::Kimi, [Arc::new(AnthropicProvider::new(&url)) as Arc<dyn Provider>]);
            let response = app(Arc::new(registry)).oneshot(Request::post("/v1/messages")
                .header("authorization", "Bearer REQUEST-AUTH-MUST-NOT-BE-LOGGED")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"claude-fable-5","messages":[{"role":"user","content":"REQUEST-BODY-MUST-NOT-BE-LOGGED"}]}"#)).unwrap()).await.unwrap();
            assert_eq!(response.status(), status);
            assert_eq!(response.headers()["content-type"], content_type);
            assert_eq!(response.headers()["x-request-id"], "error-observability-fixture");
            assert_eq!(response.into_body().collect().await.unwrap().to_bytes(), expected);
            task.abort();
        }
    });
    let logs = std::fs::read_to_string(claude_code_proxy::logging::log_file()).unwrap();
    let events: Vec<Value> = logs
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["msg"] == "anthropic_upstream_error")
        .collect();
    assert_eq!(
        events.len(),
        9,
        "exactly one diagnostic per upstream error: {logs}"
    );
    for (event, (status, _, expected)) in events
        .iter()
        .zip(cases.iter().filter(|(_, _, expected)| expected.is_some()))
    {
        assert_eq!(event["service"], "server");
        assert_eq!(event["level"], "warn");
        assert_eq!(event["fields"]["provider"], "anthropic");
        assert_eq!(event["fields"]["status"], status.as_u16());
        let body = event["fields"]["errorBody"].as_str().unwrap();
        assert_eq!(body, expected.as_ref().unwrap());
        assert!(body.len() <= 600);
        assert!(!body.chars().any(char::is_control));
    }
    assert_eq!(events[4]["fields"]["status"], 401);
    assert_eq!(events[4]["fields"]["bodyOmitted"], "non_json");
    assert_eq!(events[5]["fields"]["bodyOmitted"], "body_too_large");
    assert_eq!(events[6]["fields"]["bodyOmitted"], "invalid_json");
    assert_eq!(events[7]["fields"]["bodyOmitted"], "invalid_json");
    assert_eq!(events[8]["fields"]["bodyOmitted"], "invalid_error_shape");
    for event in &events[4..] {
        assert!(event["fields"].get("errorBody").is_none());
    }
    for forbidden in [
        "UNEXPECTED-TOKEN-NOT-LOGGED",
        "THISLOOKSLIKEALONGSECRETVALUE0123456789",
        "INVALID-JSON-NOT-LOGGED",
        "WRONG-SHAPE-NOT-LOGGED",
        "OVERSIZED-NOT-LOGGED",
        "CHUNKED-NOT-LOGGED",
        "ECHOED-CREDENTIAL-MUST-NOT-BE-LOGGED",
        "TAIL-MUST-NOT-BE-LOGGED",
        "SUCCESS-NOT-LOGGED",
        "REDIRECT-NOT-LOGGED",
        "REQUEST-AUTH-MUST-NOT-BE-LOGGED",
        "REQUEST-BODY-MUST-NOT-BE-LOGGED",
    ] {
        assert!(
            !logs.contains(forbidden),
            "unexpected logged value: {forbidden}"
        );
    }
}
