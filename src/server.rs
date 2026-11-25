//! OpenAI-compatible HTTP serving layer for Velox (`veloxc::server`).
//!
//! This is the Velox-native replacement for Chimera-Inference's FastAPI
//! server. It keeps the useful API surface — `/health`, `/v1/models`,
//! `/v1/chat/completions` with SSE streaming — without adding a Python
//! runtime or any external crate.
//!
//! Concurrency model: a small fixed thread pool. Each connection is handled
//! by one worker; model execution is serialized behind a mutex because the
//! engine owns mutable KV-cache and activation state. That is the correct
//! default for a single-model, single-GPU/CPU node and avoids corrupting the
//! KV cache with concurrent forwards.

use crate::runtime::{TransformerConfig, VeloxStaticExecutionEngine};
use crate::sampling::{SamplerSession, SamplingParams};
use crate::tokenizer::Tokenizer;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Default bind address. Loopback only: the server has no authentication.
pub const DEFAULT_BIND_ADDR: &str = "127.0.0.1:8080";

/// Maximum request body accepted before responding `413`.
pub const MAX_BODY_BYTES: usize = 1 << 20;

/// Maximum generated tokens per request, independent of client input.
pub const MAX_REQUEST_TOKENS: usize = 2048;

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_addr: String,
    pub weights_path: String,
    pub tokenizer_path: String,
    pub model_file: String,
    pub max_seq_len: usize,
    pub worker_threads: usize,
    pub default_max_tokens: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: DEFAULT_BIND_ADDR.to_string(),
            weights_path: "smollm-135m.safetensors".to_string(),
            tokenizer_path: "tokenizer.json".to_string(),
            model_file: "examples/tinystories_smollm.vlx".to_string(),
            max_seq_len: 2048,
            worker_threads: 4,
            default_max_tokens: 128,
        }
    }
}

/// Shared serving state. The engine is behind a mutex because generation
/// mutates KV-cache and activation buffers.
struct AppState {
    engine: Mutex<VeloxStaticExecutionEngine>,
    tokenizer: Tokenizer,
    model_id: String,
    max_seq_len: usize,
    default_max_tokens: usize,
}

impl AppState {
    fn model_card_json(&self) -> String {
        format!(
            "{{\"id\":\"{}\",\"object\":\"model\",\"created\":{},\"owned_by\":\"velox\"}}",
            json_escape(&self.model_id),
            unix_seconds()
        )
    }
}

/// Loads the model and tokenizer, then serves until the listener errors.
pub fn serve(config: ServerConfig) -> Result<(), String> {
    let model_config = load_model_config(&config.model_file, config.max_seq_len)?;
    let engine = VeloxStaticExecutionEngine::load_safetensors(&config.weights_path, model_config)
        .map_err(|err| format!("failed to load weights '{}': {err}", config.weights_path))?;
    let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
        .map_err(|err| format!("failed to load tokenizer '{}': {err}", config.tokenizer_path))?;

    let model_id = Path::new(&config.weights_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("velox-model")
        .to_string();

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer,
        model_id,
        max_seq_len: config.max_seq_len,
        default_max_tokens: config.default_max_tokens,
    });

    let listener = TcpListener::bind(&config.bind_addr)
        .map_err(|err| format!("failed to bind {}: {err}", config.bind_addr))?;
    eprintln!(
        "velox server listening on http://{} (model={}, workers={})",
        config.bind_addr, state.model_id, config.worker_threads
    );

    let mut workers = Vec::with_capacity(config.worker_threads);
    for _ in 0..config.worker_threads {
        let listener = listener
            .try_clone()
            .map_err(|err| format!("failed to clone listener: {err}"))?;
        let state = Arc::clone(&state);
        workers.push(thread::spawn(move || loop {
            let (stream, peer) = match listener.accept() {
                Ok(pair) => pair,
                Err(err) => {
                    eprintln!("accept failed: {err}");
                    return;
                }
            };
            if let Err(err) = handle_connection(stream, &state) {
                eprintln!("connection from {peer} ended with error: {err}");
            }
        }));
    }

    for worker in workers {
        let _ = worker.join();
    }
    Ok(())
}

fn load_model_config(model_file: &str, max_seq_len: usize) -> Result<TransformerConfig, String> {
    if model_file == "examples/tinystories_smollm.vlx"
        || model_file.ends_with("tinystories_smollm.vlx")
    {
        return Ok(TransformerConfig::smollm_135m(max_seq_len));
    }
    Err(format!(
        "server only knows the SmolLM-135M reference model; got '{model_file}'"
    ))
}

fn handle_connection(mut stream: TcpStream, state: &Arc<AppState>) -> Result<(), String> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(120)));
    let _ = stream.set_nodelay(true);

    let request = match read_request(&mut stream) {
        Ok(request) => request,
        Err(response) => {
            write_response(&mut stream, &response)?;
            return Ok(());
        }
    };

    let response = route(request, state);
    write_response(&mut stream, &response)
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

struct HttpResponse {
    status: u16,
    reason: &'static str,
    content_type: &'static str,
    body: Vec<u8>,
    extra_headers: Vec<(String, String)>,
    chunked: bool,
}

impl HttpResponse {
    fn json(status: u16, body: String) -> Self {
        Self {
            status,
            reason: reason_phrase(status),
            content_type: "application/json",
            body: body.into_bytes(),
            extra_headers: Vec::new(),
            chunked: false,
        }
    }

    fn text(status: u16, body: String) -> Self {
        Self {
            status,
            reason: reason_phrase(status),
            content_type: "text/plain; charset=utf-8",
            body: body.into_bytes(),
            extra_headers: Vec::new(),
            chunked: false,
        }
    }

    fn empty(status: u16) -> Self {
        Self {
            status,
            reason: reason_phrase(status),
            content_type: "text/plain; charset=utf-8",
            body: Vec::new(),
            extra_headers: Vec::new(),
            chunked: false,
        }
    }

    fn stream(self) -> Self {
        let mut response = self;
        response.content_type = "text/event-stream";
        response.chunked = true;
        response
            .extra_headers
            .push(("Cache-Control".to_string(), "no-cache".to_string()));
        response
            .extra_headers
            .push(("Connection".to_string(), "keep-alive".to_string()));
        response
    }
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest, HttpResponse> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|_| HttpResponse::empty(500))?);
    let mut request_line = String::new();
    reader
        .read_line(&mut request_line)
        .map_err(|_| HttpResponse::text(400, "failed to read request line".to_string()))?;
    let request_line = request_line.trim_end_matches(['\r', '\n']);
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();
    if method.is_empty() || path.is_empty() {
        return Err(HttpResponse::text(400, "malformed request line".to_string()));
    }

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .map_err(|_| HttpResponse::text(400, "failed to read headers".to_string()))?;
        if read == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| HttpResponse::text(400, "invalid content-length".to_string()))?;
            }
        }
    }

    if content_length > MAX_BODY_BYTES {
        return Err(HttpResponse::text(
            413,
            format!("body exceeds {MAX_BODY_BYTES} bytes"),
        ));
    }

    let mut body = vec![0u8; content_length];
    reader
        .read_exact(&mut body)
        .map_err(|_| HttpResponse::text(400, "failed to read body".to_string()))?;

    Ok(HttpRequest { method, path, body })
}

fn route(request: HttpRequest, state: &Arc<AppState>) -> HttpResponse {
    let path = request.path.split('?').next().unwrap_or(&request.path);
    match (request.method.as_str(), path) {
        ("GET", "/health") => HttpResponse::json(
            200,
            format!(
                "{{\"status\":\"ok\",\"model\":\"{}\",\"max_seq_len\":{}}}",
                json_escape(&state.model_id),
                state.max_seq_len
            ),
        ),
        ("GET", "/v1/models") => HttpResponse::json(
            200,
            format!("{{\"object\":\"list\",\"data\":[{}]}}", state.model_card_json()),
        ),
        ("POST", "/v1/chat/completions") => handle_chat_completions(request.body, state),
        ("GET", "/") => HttpResponse::text(
            200,
            "Velox native inference server. Try POST /v1/chat/completions.\n".to_string(),
        ),
        _ => HttpResponse::text(404, "not found".to_string()),
    }
}

fn handle_chat_completions(body: Vec<u8>, state: &Arc<AppState>) -> HttpResponse {
    let request = match ChatRequest::from_json(&body) {
        Ok(request) => request,
        Err(message) => return HttpResponse::text(400, message),
    };

    if request.messages.is_empty() {
        return HttpResponse::text(400, "messages must not be empty".to_string());
    }

    let prompt = render_chat_prompt(&request.messages);
    let mut prompt_tokens = state.tokenizer.encode(&prompt);
    let bos = 1u32;
    if !prompt_tokens.starts_with(&[bos]) {
        prompt_tokens.insert(0, bos);
    }

    let max_tokens = request
        .max_tokens
        .unwrap_or(state.default_max_tokens)
        .min(MAX_REQUEST_TOKENS);
    if max_tokens == 0 {
        return HttpResponse::text(400, "max_tokens must be > 0".to_string());
    }
    if prompt_tokens.len().saturating_add(max_tokens) > state.max_seq_len {
        return HttpResponse::text(
            400,
            format!(
                "prompt ({} tokens) + max_tokens ({}) exceeds max_seq_len ({})",
                prompt_tokens.len(),
                max_tokens,
                state.max_seq_len
            ),
        );
    }

    let params = request.sampling_params();
    if let Err(err) = params.validate() {
        return HttpResponse::text(400, format!("invalid sampling parameters: {err}"));
    }

    let mut engine = match state.engine.lock() {
        Ok(engine) => engine,
        Err(_) => return HttpResponse::text(503, "engine lock poisoned".to_string()),
    };

    let history_capacity = prompt_tokens.len().saturating_add(max_tokens);
    let mut session = match SamplerSession::new(params, engine.config.vocab_size, history_capacity) {
        Ok(session) => session,
        Err(err) => return HttpResponse::text(400, format!("sampler init failed: {err}")),
    };

    if request.stream {
        return stream_chat_completion(
            &mut engine,
            &mut session,
            &prompt_tokens,
            max_tokens,
            request.model.as_deref(),
            &state.tokenizer,
        );
    }

    let mut output = vec![0u32; max_tokens];
    let mut decoder = state.tokenizer.streaming_decoder();
    let mut text = String::new();
    let written = engine.generate_sampled_streaming_with_session(
        &prompt_tokens,
        max_tokens,
        &mut session,
        &mut output,
        |_, token| {
            text.push_str(&decoder.decode_token(token));
        },
    );
    text.push_str(&decoder.flush());
    output.truncate(written);

    let finish_reason = if written < max_tokens { "length" } else { "length" };
    let body = format!(
        "{{\"id\":\"chatcmpl-{}\",\"object\":\"chat.completion\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}}},\"finish_reason\":\"{}\"}}],\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}}}",
        unique_id(),
        unix_seconds(),
        json_escape(request.model.as_deref().unwrap_or(&state.model_id)),
        json_escape(&text),
        finish_reason,
        prompt_tokens.len(),
        written,
        prompt_tokens.len() + written,
    );
    HttpResponse::json(200, body)
}

fn stream_chat_completion(
    engine: &mut VeloxStaticExecutionEngine,
    session: &mut SamplerSession,
    prompt_tokens: &[u32],
    max_tokens: usize,
    model: Option<&str>,
    tokenizer: &Tokenizer,
) -> HttpResponse {
    let mut output = vec![0u32; max_tokens];
    let mut decoder = tokenizer.streaming_decoder();
    let mut text = String::new();
    let started = Instant::now();

    let written = engine.generate_sampled_streaming_with_session(
        prompt_tokens,
        max_tokens,
        session,
        &mut output,
        |_, token| {
            text.push_str(&decoder.decode_token(token));
        },
    );
    text.push_str(&decoder.flush());
    output.truncate(written);

    let completion_id = format!("chatcmpl-{}", unique_id());
    let created = unix_seconds();
    let model_id = model.unwrap_or("velox-model");
    let mut body = Vec::new();

    let role_chunk = format!(
        "{{\"id\":\"{completion_id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}",
        json_escape(model_id)
    );
    push_sse(&mut body, &role_chunk);

    // Re-tokenizing the decoded prefix would be expensive and inexact, so the
    // content chunk carries the full generated text followed by a terminal
    // usage chunk. This is valid SSE and keeps the engine single-writer.
    if !text.is_empty() {
        let content_chunk = format!(
            "{{\"id\":\"{completion_id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}},\"finish_reason\":null}}]}}",
            json_escape(model_id),
            json_escape(&text)
        );
        push_sse(&mut body, &content_chunk);
    }

    let finish_chunk = format!(
        "{{\"id\":\"{completion_id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"length\"}}]}}",
        json_escape(model_id)
    );
    push_sse(&mut body, &finish_chunk);

    let usage_chunk = format!(
        "{{\"id\":\"{completion_id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":\"{}\",\"choices\":[],\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{},\"decode_ms\":{:.3}}}}}",
        json_escape(model_id),
        prompt_tokens.len(),
        written,
        prompt_tokens.len() + written,
        started.elapsed().as_secs_f64() * 1000.0
    );
    push_sse(&mut body, &usage_chunk);
    body.extend_from_slice(b"data: [DONE]\n\n");

    HttpResponse::stream(HttpResponse {
        status: 200,
        reason: "OK",
        content_type: "text/event-stream",
        body,
        extra_headers: Vec::new(),
        chunked: false,
    })
}

fn push_sse(body: &mut Vec<u8>, payload: &str) {
    body.extend_from_slice(b"data: ");
    body.extend_from_slice(payload.as_bytes());
    body.extend_from_slice(b"\n\n");
}

#[derive(Debug)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    seed: u64,
    stream: bool,
}

#[derive(Debug)]
struct ChatMessage {
    role: String,
    content: String,
}

fn json_get<'a>(object: &'a [(String, JsonValue)], key: &str) -> Option<&'a JsonValue> {
    object
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

impl ChatRequest {
    fn from_json(body: &[u8]) -> Result<Self, String> {
        let text = std::str::from_utf8(body).map_err(|_| "body must be UTF-8".to_string())?;
        let json = JsonValue::parse(text)?;
        let object = json
            .as_object()
            .ok_or_else(|| "request body must be a JSON object".to_string())?;

        let model = json_get(object, "model")
            .and_then(JsonValue::as_str)
            .map(str::to_string);

        let messages_json = json_get(object, "messages")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| "missing 'messages' array".to_string())?;
        let mut messages = Vec::with_capacity(messages_json.len());
        for item in messages_json {
            let message = item
                .as_object()
                .ok_or_else(|| "each message must be an object".to_string())?;
            let role = json_get(message, "role")
                .and_then(JsonValue::as_str)
                .unwrap_or("user")
                .to_string();
            let content = json_get(message, "content")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string();
            messages.push(ChatMessage { role, content });
        }

        let max_tokens = match json_get(object, "max_tokens").or_else(|| json_get(object, "max_completion_tokens")) {
            Some(JsonValue::Number(n)) if n.fract() == 0.0 && *n >= 0.0 => Some(*n as usize),
            Some(JsonValue::Number(_)) => return Err("max_tokens must be an integer".to_string()),
            _ => None,
        };

        Ok(Self {
            model,
            messages,
            max_tokens,
            temperature: number_or(json_get(object, "temperature"), 0.0)?,
            top_k: match json_get(object, "top_k") {
                Some(JsonValue::Number(n)) if n.fract() == 0.0 && *n >= 0.0 => *n as usize,
                Some(JsonValue::Number(_)) => return Err("top_k must be an integer".to_string()),
                _ => 0,
            },
            top_p: number_or(json_get(object, "top_p"), 1.0)?,
            min_p: number_or(json_get(object, "min_p"), 0.0)?,
            repetition_penalty: number_or(json_get(object, "repetition_penalty"), 1.0)?,
            frequency_penalty: number_or(json_get(object, "frequency_penalty"), 0.0)?,
            presence_penalty: number_or(json_get(object, "presence_penalty"), 0.0)?,
            seed: match json_get(object, "seed") {
                Some(JsonValue::Number(n)) if n.fract() == 0.0 && *n >= 0.0 => *n as u64,
                Some(JsonValue::Number(_)) => return Err("seed must be an integer".to_string()),
                _ => 0x5EED_0F_5EED_0F5E,
            },
            stream: matches!(json_get(object, "stream"), Some(JsonValue::Bool(true))),
        })
    }

    fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
        }
    }
}

fn number_or(value: Option<&JsonValue>, default: f32) -> Result<f32, String> {
    match value {
        Some(JsonValue::Number(n)) => Ok(*n as f32),
        Some(_) => Err("expected a number".to_string()),
        None => Ok(default),
    }
}

fn render_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let role = if message.role.is_empty() {
            "user"
        } else {
            message.role.as_str()
        };
        prompt.push_str(role);
        prompt.push_str(": ");
        prompt.push_str(&message.content);
        prompt.push('\n');
    }
    prompt.push_str("assistant: ");
    prompt
}

fn write_response(stream: &mut TcpStream, response: &HttpResponse) -> Result<(), String> {
    let framing = if response.chunked {
        "Transfer-Encoding: chunked\r\n".to_string()
    } else {
        format!("Content-Length: {}\r\n", response.body.len())
    };
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\n{}",
        response.status, response.reason, response.content_type, framing
    );
    for (name, value) in &response.extra_headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");

    stream
        .write_all(head.as_bytes())
        .map_err(|err| format!("failed to write response head: {err}"))?;
    if response.chunked {
        // A single chunk followed by the terminating zero-length chunk. This
        // keeps the response valid for clients that require chunked SSE.
        write_chunk(stream, &response.body)?;
        stream
            .write_all(b"0\r\n\r\n")
            .map_err(|err| format!("failed to write terminal chunk: {err}"))?;
    } else {
        stream
            .write_all(&response.body)
            .map_err(|err| format!("failed to write response body: {err}"))?;
    }
    stream
        .flush()
        .map_err(|err| format!("failed to flush response: {err}"))
}

fn write_chunk(stream: &mut TcpStream, body: &[u8]) -> Result<(), String> {
    if body.is_empty() {
        return Ok(());
    }
    stream
        .write_all(format!("{:X}\r\n", body.len()).as_bytes())
        .and_then(|_| stream.write_all(body))
        .and_then(|_| stream.write_all(b"\r\n"))
        .map_err(|err| format!("failed to write chunk: {err}"))
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unique_id() -> String {
    format!("{:016x}{:016x}", unix_seconds(), &COUNTER.fetch_add(1, Ordering::Relaxed))
}

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

// ============================================================================
// Minimal JSON parser (objects, arrays, strings, numbers, bools, null)
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    fn parse(input: &str) -> Result<Self, String> {
        let mut parser = JsonParser {
            bytes: input.as_bytes(),
            pos: 0,
        };
        parser.skip_whitespace();
        let value = parser.parse_value()?;
        parser.skip_whitespace();
        if parser.pos != parser.bytes.len() {
            return Err(format!("trailing JSON at byte {}", parser.pos));
        }
        Ok(value)
    }

    fn as_object(&self) -> Option<&Vec<(String, JsonValue)>> {
        match self {
            Self::Object(entries) => Some(entries),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }
}

impl JsonValue {
    fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object().and_then(|entries| {
            entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value)
        })
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl JsonParser<'_> {
    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(JsonValue::String(self.parse_string()?)),
            Some(b't') => self.parse_literal("true", JsonValue::Bool(true)),
            Some(b'f') => self.parse_literal("false", JsonValue::Bool(false)),
            Some(b'n') => self.parse_literal("null", JsonValue::Null),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(other) => Err(format!("unexpected byte '{}' at {}", other as char, self.pos)),
            None => Err("unexpected end of JSON".to_string()),
        }
    }

    fn parse_literal(&mut self, literal: &str, value: JsonValue) -> Result<JsonValue, String> {
        if self.bytes[self.pos..].starts_with(literal.as_bytes()) {
            self.pos += literal.len();
            Ok(value)
        } else {
            Err(format!("invalid literal at byte {}", self.pos))
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.pos += 1; // {
        let mut entries = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(format!("expected object key at byte {}", self.pos));
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' at byte {}", self.pos));
            }
            self.pos += 1;
            let value = self.parse_value()?;
            entries.push((key, value));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(entries));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.pos)),
            }
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.pos += 1; // [
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.pos)),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            let byte = self
                .peek()
                .ok_or_else(|| "unterminated JSON string".to_string())?;
            self.pos += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escape = self
                        .peek()
                        .ok_or_else(|| "unterminated escape".to_string())?;
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hex = self
                                .bytes
                                .get(self.pos..self.pos + 4)
                                .ok_or_else(|| "short \\u escape".to_string())?;
                            let code = u32::from_str_radix(
                                std::str::from_utf8(hex).map_err(|_| "bad \\u escape".to_string())?,
                                16,
                            )
                            .map_err(|_| "bad \\u escape".to_string())?;
                            self.pos += 4;
                            out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                        }
                        other => return Err(format!("bad escape '\\{}'", other as char)),
                    }
                }
                _ => {
                    // Re-decode the UTF-8 sequence starting at byte-1.
                    let start = self.pos - 1;
                    let width = utf8_width(byte);
                    let end = start + width;
                    let slice = self
                        .bytes
                        .get(start..end)
                        .ok_or_else(|| "truncated UTF-8".to_string())?;
                    out.push_str(
                        std::str::from_utf8(slice)
                            .map_err(|_| "invalid UTF-8 in string".to_string())?,
                    );
                    self.pos = end;
                }
            }
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid number".to_string())?;
        text.parse::<f64>()
            .map(JsonValue::Number)
            .map_err(|_| format!("invalid number '{text}'"))
    }
}

fn utf8_width(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_chat_request_parsing() {
        let body = br#"{
            "model": "smollm",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hello"}
            ],
            "max_tokens": 8,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": true
        }"#;
        let request = ChatRequest::from_json(body).unwrap();
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.max_tokens, Some(8));
        assert_eq!(request.temperature, 0.7);
        assert!(request.stream);
    }

    #[test]
    fn test_chat_prompt_rendering() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "be terse".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "hi".to_string(),
            },
        ];
        assert_eq!(render_chat_prompt(&messages), "system: be terse\nuser: hi\nassistant: ");
    }

    #[test]
    fn test_json_escape_roundtrip() {
        let raw = "line\nquote\"slash\\tab\t";
        let escaped = json_escape(raw);
        let parsed = JsonValue::parse(&format!("\"{escaped}\"")).unwrap();
        assert_eq!(parsed.as_str(), Some(raw));
    }
}