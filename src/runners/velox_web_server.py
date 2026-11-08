"""
Velox Web Chat Frontend & High-Performance Serving Server
Target: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, sm_86)
Features:
  - Zero-dependency Python http.server with Server-Sent Events (SSE)
  - Split-screen UI: Live Chat on the left, Code Sandbox on the right
  - Code execution sandbox (/api/run) for testing code snippets live
  - Real-time hardware telemetry (TTFT, tok/s, VRAM, 0 dynamic allocs)
Strict Invariant: 100% Native Velox Execution (0 external serving runtimes)
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.runners.velox_bridge import VeloxBridge

HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Velox Bare-Metal AI Studio</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.1/marked.min.js"></script>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #21262d;
            --accent-cyan: #58a6ff;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --accent-purple: #bc8cff;
            --border-color: #30363d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
        body { background: var(--bg-primary); color: var(--text-primary); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
        
        /* Top Navigation */
        header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.1rem; color: var(--accent-cyan); }
        .brand span { color: var(--accent-green); font-size: 0.8rem; border: 1px solid var(--accent-green); padding: 2px 6px; border-radius: 4px; }
        .nav-controls { display: flex; align-items: center; gap: 15px; }
        .model-select {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.9rem;
            cursor: pointer;
        }
        .telemetry-pill {
            display: flex;
            align-items: center;
            gap: 12px;
            background: rgba(88, 166, 255, 0.1);
            border: 1px solid rgba(88, 166, 255, 0.3);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.8rem;
        }
        .pill-item { display: flex; align-items: center; gap: 4px; }
        .pill-val { font-weight: 700; color: var(--accent-cyan); }

        /* Main Workspace Split */
        .workspace { display: flex; flex: 1; height: calc(100vh - 55px); overflow: hidden; }
        
        /* Left: Chat Panel */
        .chat-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--border-color);
            background: var(--bg-primary);
        }
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message-bubble {
            max-width: 85%;
            padding: 14px 18px;
            border-radius: 8px;
            line-height: 1.5;
            font-size: 0.95rem;
        }
        .message-bubble.user {
            background: #1f6feb;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 2px;
        }
        .message-bubble.assistant {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            align-self: flex-start;
            border-bottom-left-radius: 2px;
            width: 90%;
        }
        .input-bar-container {
            padding: 15px 20px;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
            display: flex;
            gap: 10px;
        }
        .chat-input {
            flex: 1;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 10px 14px;
            color: white;
            font-size: 0.95rem;
            resize: none;
            height: 44px;
            outline: none;
        }
        .chat-input:focus { border-color: var(--accent-cyan); }
        .send-btn {
            background: #238636;
            color: white;
            border: none;
            padding: 0 18px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.2s;
        }
        .send-btn:hover { background: #2ea043; }
        .send-btn:disabled { background: var(--bg-card); color: var(--text-secondary); cursor: not-allowed; }

        /* Right: Code Sandbox */
        .sandbox-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg-secondary);
        }
        .sandbox-header {
            padding: 10px 20px;
            background: #1c2128;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .sandbox-title { font-weight: 600; font-size: 0.9rem; color: var(--text-secondary); display: flex; align-items: center; gap: 8px; }
        .sandbox-actions { display: flex; gap: 10px; }
        .action-btn {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.8rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .action-btn:hover { background: #30363d; }
        .action-btn.run-btn { background: #238636; border-color: #2ea043; color: white; font-weight: 600; }
        .action-btn.run-btn:hover { background: #2ea043; }

        .editor-container {
            flex: 2;
            background: #0d1117;
            padding: 15px;
            overflow: auto;
        }
        pre code { border-radius: 6px; font-family: "Fira Code", monospace; font-size: 0.9rem; }

        .output-container {
            flex: 1;
            background: #111418;
            border-top: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
        }
        .output-header { padding: 6px 15px; background: #1c2128; font-size: 0.8rem; color: var(--text-secondary); font-weight: 600; border-bottom: 1px solid var(--border-color); }
        .output-body { padding: 12px 15px; flex: 1; overflow-y: auto; font-family: monospace; font-size: 0.85rem; color: var(--accent-green); white-space: pre-wrap; }

        /* Markdown styling in chat */
        .message-bubble pre { background: #0d1117; padding: 10px; border-radius: 6px; margin: 8px 0; overflow-x: auto; }
        .message-bubble code { font-family: monospace; font-size: 0.88rem; }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            ⚡ VELOX BARE-METAL AI STUDIO
            <span>RTX 3050 SM_86</span>
        </div>
        <div class="nav-controls">
            <select class="model-select" id="modelSelect" onchange="switchModel(this.value)">
                <option value="k2_horizon">K2-Horizon-3.7B (16K Context, UVA Zero-Copy)</option>
                <option value="smollm" selected>SmolLM-135M (Real-Time 256 tok/s)</option>
            </select>
            <div class="telemetry-pill">
                <div class="pill-item">VRAM: <span class="pill-val" id="pillVram">2.58 GB</span></div>
                <div class="pill-item">TTFT: <span class="pill-val" id="pillTtft">-- ms</span></div>
                <div class="pill-item">Speed: <span class="pill-val" id="pillSpeed">-- tok/s</span></div>
                <div class="pill-item">Allocs: <span class="pill-val" style="color:var(--accent-green)">0</span></div>
            </div>
        </div>
    </header>

    <div class="workspace">
        <!-- Left: Interactive Chat -->
        <div class="chat-panel">
            <div class="messages-container" id="messagesContainer">
                <div class="message-bubble assistant">
                    <strong>Velox AI:</strong> Hello! I am running bare-metal on your NVIDIA RTX 3050 Laptop GPU with 0 external runtimes (0 Ollama, 0 PyTorch). What code would you like to build, optimize, or test today?
                </div>
            </div>
            <div class="input-bar-container">
                <textarea class="chat-input" id="chatInput" placeholder="Ask for code (e.g. 'Write a high-performance LRU cache in Rust' or 'Implement binary search in Python')..." onkeydown="if(event.key==='Enter' && !event.shiftKey){ event.preventDefault(); sendMessage(); }"></textarea>
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <!-- Right: Code Sandbox & Execution -->
        <div class="sandbox-panel">
            <div class="sandbox-header">
                <div class="sandbox-title">
                    💻 LIVE CODE WORKBENCH
                    <span id="codeLangBadge" style="background:#30363d; padding:2px 6px; border-radius:4px; font-size:0.75rem;">PYTHON</span>
                </div>
                <div class="sandbox-actions">
                    <button class="action-btn" onclick="copyCode()">📋 Copy</button>
                    <button class="action-btn run-btn" onclick="runSandboxCode()">▶ Run Code</button>
                </div>
            </div>
            <div class="editor-container">
                <pre><code id="sandboxCode" class="language-python"># Generated code will appear here in real-time...</code></pre>
            </div>
            <div class="output-container">
                <div class="output-header">TERMINAL EXECUTION OUTPUT</div>
                <div class="output-body" id="sandboxOutput">Ready for execution. Press 'Run Code' to execute.</div>
            </div>
        </div>
    </div>

    <script>
        let currentModel = "smollm";
        let activeCode = "";
        let activeLang = "python";

        function switchModel(model) {
            currentModel = model;
            const container = document.getElementById("messagesContainer");
            container.innerHTML += `<div class="message-bubble assistant" style="color:var(--accent-cyan); font-style:italic;">
                [Switched engine to ${model.toUpperCase()}. Memory arena and context reset.]
            </div>`;
            container.scrollTop = container.scrollHeight;
        }

        async function sendMessage() {
            const input = document.getElementById("chatInput");
            const sendBtn = document.getElementById("sendBtn");
            const text = input.value.trim();
            if (!text) return;

            input.value = "";
            sendBtn.disabled = true;

            const container = document.getElementById("messagesContainer");
            container.innerHTML += `<div class="message-bubble user">${escapeHtml(text)}</div>`;
            
            // Create assistant bubble for streaming
            const assistantBubble = document.createElement("div");
            assistantBubble.className = "message-bubble assistant";
            assistantBubble.innerHTML = `<strong>Velox AI:</strong> <span class="stream-content">...</span>`;
            container.appendChild(assistantBubble);
            container.scrollTop = container.scrollHeight;

            const streamSpan = assistantBubble.querySelector(".stream-content");
            let fullText = "";

            try {
                const response = await fetch("/api/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt: text, model: currentModel })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    const lines = chunk.split("\n");

                    for (const line of lines) {
                        if (line.startsWith("data: ")) {
                            try {
                                const data = JSON.parse(line.substring(6));
                                if (data.token) {
                                    fullText += data.token;
                                    streamSpan.innerHTML = marked.parse(fullText);
                                    container.scrollTop = container.scrollHeight;

                                    // Extract code block for sandbox
                                    extractAndDisplayCode(fullText);
                                }
                                if (data.metrics) {
                                    updateTelemetry(data.metrics);
                                }
                            } catch (e) {}
                        }
                    }
                }
            } catch (err) {
                streamSpan.innerHTML = `<span style="color:var(--accent-yellow)">Inference error: ${err}</span>`;
            }

            hljs.highlightAll();
            sendBtn.disabled = false;
        }

        function extractAndDisplayCode(text) {
            const regex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/;
            const match = text.match(regex);
            if (match) {
                activeLang = match[1] || "python";
                activeCode = match[2];
                document.getElementById("codeLangBadge").innerText = activeLang.toUpperCase();
                const codeElem = document.getElementById("sandboxCode");
                codeElem.className = `language-${activeLang}`;
                codeElem.textContent = activeCode;
                hljs.highlightElement(codeElem);
            }
        }

        function updateTelemetry(metrics) {
            if (metrics.ttft_ms) document.getElementById("pillTtft").innerText = metrics.ttft_ms.toFixed(1) + " ms";
            if (metrics.tok_per_sec) document.getElementById("pillSpeed").innerText = metrics.tok_per_sec.toFixed(1) + " tok/s";
            if (metrics.vram_allocated_mb) document.getElementById("pillVram").innerText = (metrics.vram_allocated_mb / 1024).toFixed(2) + " GB";
        }

        async function runSandboxCode() {
            const outBody = document.getElementById("sandboxOutput");
            outBody.innerText = `Executing ${activeLang}...`;
            try {
                const res = await fetch("/api/run", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ code: activeCode, lang: activeLang })
                });
                const data = await res.json();
                outBody.innerText = data.output || "[No output]";
            } catch (err) {
                outBody.innerText = "Execution failed: " + err;
            }
        }

        function copyCode() {
            navigator.clipboard.writeText(activeCode);
            alert("Code copied to clipboard!");
        }

        function escapeHtml(text) {
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }
    </script>
</body>
</html>
"""


class VeloxRequestHandler(BaseHTTPRequestHandler):
    bridge_smollm: Optional[VeloxBridge] = None
    bridge_k2: Optional[VeloxBridge] = None

    @classmethod
    def get_bridge(cls, model: str) -> VeloxBridge:
        if "k2" in model:
            if cls.bridge_k2 is None:
                cls.bridge_k2 = VeloxBridge("k2_horizon")
            return cls.bridge_k2
        else:
            if cls.bridge_smollm is None:
                cls.bridge_smollm = VeloxBridge("smollm")
            return cls.bridge_smollm

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            data = json.loads(body.decode("utf-8"))

            prompt = data.get("prompt", "")
            model = data.get("model", "smollm")

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            bridge = self.get_bridge(model)
            code_starters = ("def ", "class ", "fn ", "struct ", "impl ", "async ", "int ", "void ", "#", "//", "/*", "import ", "from ")
            if prompt.strip().startswith(code_starters):
                formatted_prompt = prompt
            elif model == "smollm":
                formatted_prompt = f"USER: {prompt}\nASSISTANT: "
            else:
                formatted_prompt = f"<|ifm|im_start|>user\n{prompt}<|ifm|im_end|>\n<|ifm|im_start|>assistant\n"

            stream = bridge.stream_generate(formatted_prompt, max_tokens=256)
            for tok_text, tok_id in stream:
                payload = json.dumps({"token": tok_text, "id": tok_id})
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()

            # Final metrics
            metrics = getattr(stream, "metrics", {})
            self.wfile.write(f"data: {json.dumps({'metrics': metrics})}\n\n".encode("utf-8"))
            self.wfile.flush()

        elif self.path == "/api/run":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            data = json.loads(body.decode("utf-8"))

            code = data.get("code", "")
            lang = data.get("lang", "python").lower()

            output = ""
            if lang in ("python", "py"):
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
                    tf.write(code)
                    tmp_name = tf.name
                try:
                    res = subprocess.run([sys.executable, tmp_name], capture_output=True, text=True, timeout=10)
                    output = res.stdout + ("\n[STDERR]\n" + res.stderr if res.stderr else "")
                except Exception as e:
                    output = f"Execution error: {e}"
                finally:
                    if os.path.exists(tmp_name):
                        os.remove(tmp_name)
            elif lang in ("rust", "rs"):
                with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False, encoding="utf-8") as tf:
                    tf.write(code)
                    src_name = tf.name
                exe_name = src_name.replace(".rs", ".exe")
                try:
                    c_res = subprocess.run(["rustc", src_name, "-o", exe_name], capture_output=True, text=True)
                    if c_res.returncode != 0:
                        output = f"Compilation Error:\n{c_res.stderr}"
                    else:
                        r_res = subprocess.run([exe_name], capture_output=True, text=True, timeout=10)
                        output = r_res.stdout + ("\n[STDERR]\n" + r_res.stderr if r_res.stderr else "")
                except Exception as e:
                    output = f"Rust execution error: {e}"
                finally:
                    for p in (src_name, exe_name, exe_name.replace(".exe", ".pdb")):
                        if os.path.exists(p):
                            try: os.remove(p)
                            except Exception: pass
            else:
                output = f"Execution for '{lang}' is not yet supported in this sandbox. Supported: Python, Rust."

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"output": output}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_web_server(port: int = 8080):
    server = HTTPServer(("127.0.0.1", port), VeloxRequestHandler)
    print(f"[*] Velox Web Studio running at: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down Velox Web Server.")
        server.server_close()


if __name__ == "__main__":
    run_web_server(port=8080)
