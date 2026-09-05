//! Spec-derived lifecycle acceptance tests. Transport fixtures follow selective_routing.rs;
//! no monitor/session/compaction behavior is mocked. Unique IDs isolate global session state.
use axum::{
    Router,
    body::{Body, Bytes},
    extract::State,
    http::{Request, StatusCode},
    response::Response,
    routing::post,
};
use claude_code_proxy::{
    MessagesRequest,
    config::AliasProvider,
    monitor::{EndpointKind, MonitorHandle, RequestStatus},
    provider::{CliHandlers, Provider, RequestContext},
    providers::{
        anthropic::AnthropicProvider,
        codex::compaction::{
            CompactionAttempt, activate_compaction, begin_compaction, clear_compaction,
            store_compaction,
        },
    },
    registry::Registry,
    server::app_with_monitor,
    session::existing_session_now,
};
use futures_util::StreamExt;
use http_body_util::BodyExt;
use serde_json::json;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{net::TcpListener, task::JoinHandle};
use tower::ServiceExt;

const CLAUDE: &str = "claude-fable-5-1[1m]";
const CODEX: &str = "gpt-6-astra";
const GUARD: Duration = Duration::from_secs(30);

struct Upstream {
    url: String,
    task: JoinHandle<()>,
}
impl Drop for Upstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn serve(router: Router) -> Upstream {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    Upstream { url, task }
}
async fn fixed(status: StatusCode) -> Upstream {
    serve(Router::new().route(
        "/{*path}",
        post(move || async move {
            Response::builder()
                .status(status)
                .body(Body::from("sentinel-body"))
                .unwrap()
        }),
    ))
    .await
}
fn router(url: &str, monitor: &MonitorHandle) -> Router {
    app_with_monitor(
        Arc::new(Registry::from_providers(
            AliasProvider::Codex,
            [
                Arc::new(AnthropicProvider::new(url)) as Arc<dyn Provider>,
                Arc::new(FakeCodex),
            ],
        )),
        Some(monitor.clone()),
    )
}
fn sid() -> String {
    format!("selective-lifecycle-{}", uuid::Uuid::new_v4())
}
fn request(session: &str, model: &str, endpoint: &str) -> Request<Body> {
    Request::post(endpoint)
        .header("content-type", "application/json")
        .header("x-claude-code-session-id", session)
        .body(Body::from(
            json!({"model":model,"messages":[],"max_tokens":16,"stream":true}).to_string(),
        ))
        .unwrap()
}
fn assert_terminal(
    monitor: &MonitorHandle,
    session: &str,
    seq: u64,
    provider: &str,
    model: &str,
    status: RequestStatus,
    http_status: Option<u16>,
) {
    let state = monitor.snapshot();
    assert!(
        state.active.is_empty(),
        "stale active rows: {:?}",
        state.active
    );
    assert_eq!(
        state.recent.len(),
        seq as usize,
        "one terminal row per request"
    );
    let rows: Vec<_> = state
        .recent
        .iter()
        .filter(|row| row.session_seq == Some(seq))
        .collect();
    assert_eq!(
        rows.len(),
        1,
        "sequence must resolve to exactly one terminal row"
    );
    let row = rows[0];
    assert_eq!(row.session_id.as_deref(), Some(session));
    assert_eq!(row.provider.as_deref(), Some(provider));
    assert_eq!(
        row.model.as_deref(),
        Some(model),
        "retain exact requested model, including hint"
    );
    assert_eq!(row.status, status);
    if let Some(code) = http_status {
        assert_eq!(row.http_status, Some(code));
    }
    assert_eq!(
        existing_session_now(Some(session)).unwrap().seq,
        seq,
        "no double sequence increment"
    );
}

// Hypothesis: an early transport/error return leaves the started row unresolved.
// Existing routing tests assert HTTP behavior only; oracle is terminal row + exact session identity.
#[tokio::test]
async fn upstream_connection_and_http_errors_finish_exactly_once() {
    let failed = fixed(StatusCode::TOO_MANY_REQUESTS).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let refused = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    for (url, expected) in [
        (&refused, StatusCode::BAD_GATEWAY),
        (&failed.url, StatusCode::TOO_MANY_REQUESTS),
    ] {
        let monitor = MonitorHandle::new(10);
        let session = sid();
        assert!(existing_session_now(Some(&session)).is_none());
        assert!(monitor.snapshot().active.is_empty());
        let response = tokio::time::timeout(
            GUARD,
            router(url, &monitor).oneshot(request(&session, CLAUDE, "/v1/messages")),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), expected);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        if expected == StatusCode::TOO_MANY_REQUESTS {
            assert_eq!(body, "sentinel-body");
        }
        assert_terminal(
            &monitor,
            &session,
            1,
            "anthropic",
            CLAUDE,
            RequestStatus::Failed,
            Some(expected.as_u16()),
        );
    }
}

// Hypothesis: cleanup runs only at EOF, not when a partially-read response is dropped.
// An unreleased channel proves the upstream has not completed; no sleeps or timing race.
#[tokio::test]
async fn dropping_client_mid_stream_terminates_the_active_request() {
    type Gate = Arc<Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>;
    async fn stream(State(gate): State<Gate>) -> Response {
        let receiver = gate.lock().unwrap().take().unwrap();
        let first = futures_util::stream::once(async {
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: first\n\n"))
        });
        let tail = futures_util::stream::once(async move {
            let _ = receiver.await;
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: last\n\n"))
        });
        Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(first.chain(tail)))
            .unwrap()
    }
    let (_release, receiver) = tokio::sync::oneshot::channel();
    let upstream = serve(
        Router::new()
            .route("/v1/messages", post(stream))
            .with_state(Arc::new(Mutex::new(Some(receiver)))),
    )
    .await;
    tokio::time::timeout(GUARD, async {
        let monitor = MonitorHandle::new(10);
        let session = sid();
        assert!(existing_session_now(Some(&session)).is_none());
        let response = router(&upstream.url, &monitor)
            .oneshot(request(&session, CLAUDE, "/v1/messages"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body();
        assert_eq!(
            body.frame().await.unwrap().unwrap().into_data().unwrap(),
            "data: first\n\n"
        );
        let before = monitor.snapshot();
        assert_eq!(before.active.len(), 1);
        assert!(before.recent.is_empty());
        let active = before.active.first().unwrap();
        assert_eq!(active.session_seq, Some(1));
        assert_eq!(active.provider.as_deref(), Some("anthropic"));
        assert_eq!(active.model.as_deref(), Some(CLAUDE));
        drop(body);
        assert_terminal(
            &monitor,
            &session,
            1,
            "anthropic",
            CLAUDE,
            RequestStatus::Failed,
            None,
        );
    })
    .await
    .expect("stream blocked before first chunk");
}

// Public compaction API fixture: unconfirmed native compaction, awaiting portable summary.
// This is a legitimate partial state, not a test-only mock or a fabricated validation rule.
fn pending(session: &str) -> CompactionAttempt {
    let attempt = begin_compaction(session, CODEX);
    let history = serde_json::from_value(
        json!([{"type":"compaction","encrypted_content":"opaque-native-history"}]),
    )
    .unwrap();
    assert!(
        store_compaction(session, attempt, history),
        "fixture must exist before dispatch"
    );
    attempt
}
fn activate(session: &str, attempt: CompactionAttempt) -> bool {
    let summary: Vec<claude_code_proxy::providers::codex::translate::request::ResponsesInputItem> = serde_json::from_value(json!([{"type":"message","role":"assistant","content":[{"type":"output_text","text":"A portable summary retaining the conversation and its pending work."}]}])).unwrap();
    activate_compaction(Some(session), Some(attempt), CODEX, &summary)
}
async fn compaction_case(endpoint: &str, retained: bool) {
    let upstream = fixed(StatusCode::OK).await;
    let monitor = MonitorHandle::new(10);
    let app = router(&upstream.url, &monitor);
    let session = sid();
    let other = sid();
    let attempt = pending(&session);
    let untouched = pending(&other);
    for seq in 1..=2 {
        let response = app
            .clone()
            .oneshot(request(&session, "sonnet", endpoint))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.into_body().collect().await.unwrap().to_bytes(),
            "sentinel-body"
        );
        assert_terminal(
            &monitor,
            &session,
            seq,
            "anthropic",
            "sonnet",
            RequestStatus::Completed,
            Some(200),
        );
        let expected = if retained {
            EndpointKind::CountTokens
        } else {
            EndpointKind::Messages
        };
        assert!(
            monitor
                .snapshot()
                .recent
                .iter()
                .all(|row| row.endpoint == expected)
        );
    }
    assert!(
        activate(&other, untouched),
        "another session's compaction must survive"
    );
    let actual = activate(&session, attempt);
    clear_compaction(&session);
    clear_compaction(&other);
    assert_eq!(
        actual, retained,
        "pending compaction retention differs for {endpoint}"
    );
}

// Hypothesis: shared passthrough cleanup erases compaction during auxiliary token probes.
// Existing count_tokens tests check forwarded URI, never re-read pending state.
#[tokio::test]
async fn repeated_count_tokens_preserves_pending_codex_compaction() {
    compaction_case("/v1/messages/count_tokens", true).await;
}

// Hypothesis: early passthrough dispatch skips normal message invalidation, or clears all sessions.
// Before/after activation oracle rejects both stale retained state and over-broad cleanup.
#[tokio::test]
async fn repeated_messages_clears_only_its_sessions_pending_codex_compaction() {
    compaction_case("/v1/messages", false).await;
}

// Hypothesis: provider/sequence fields leak from session affinity across provider switches.
// Existing routing tests don't send alternating providers through one monitored session.
#[tokio::test]
async fn alternating_providers_keep_exact_models_and_monotonic_sequences() {
    let upstream = fixed(StatusCode::OK).await;
    let monitor = MonitorHandle::new(10);
    let app = router(&upstream.url, &monitor);
    let session = sid();
    assert!(existing_session_now(Some(&session)).is_none());
    for (i, (model, provider, expected)) in [
        (CODEX, "codex", "codex-body"),
        (CLAUDE, "anthropic", "sentinel-body"),
        (CODEX, "codex", "codex-body"),
        ("mythos", "anthropic", "sentinel-body"),
    ]
    .into_iter()
    .enumerate()
    {
        assert!(monitor.snapshot().active.is_empty());
        let response = app
            .clone()
            .oneshot(request(&session, model, "/v1/messages"))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.into_body().collect().await.unwrap().to_bytes(),
            expected
        );
        assert_terminal(
            &monitor,
            &session,
            i as u64 + 1,
            provider,
            model,
            RequestStatus::Completed,
            Some(200),
        );
    }
    let state = monitor.snapshot();
    let mut rows: Vec<_> = state
        .recent
        .iter()
        .map(|row| {
            (
                row.session_seq.unwrap(),
                row.provider.as_deref().unwrap(),
                row.model.as_deref().unwrap(),
            )
        })
        .collect();
    rows.sort_unstable();
    assert_eq!(
        rows,
        vec![
            (1, "codex", CODEX),
            (2, "anthropic", CLAUDE),
            (3, "codex", CODEX),
            (4, "anthropic", "mythos")
        ]
    );
}

struct FakeCodex;
#[async_trait::async_trait]
impl Provider for FakeCodex {
    fn name(&self) -> &'static str {
        "codex"
    }
    fn supported_models(&self) -> Vec<String> {
        vec![CODEX.into()]
    }
    fn cli(&self) -> &'static dyn CliHandlers {
        panic!("no CLI in lifecycle tests")
    }
    async fn handle_messages(&self, _: MessagesRequest, _: RequestContext) -> Response {
        Response::new(Body::from("codex-body"))
    }
    async fn handle_count_tokens(&self, _: MessagesRequest, _: RequestContext) -> Response {
        panic!("passthrough token probes must not reach Codex")
    }
}
