#!/usr/bin/env python3
"""Mock MAKEFUN_ENV_URL for local testing of main.py (stdlib only).

Simulates 4 providers with different wires, models, capabilities and
failure schedules — no real API keys, no network beyond localhost.

Providers (catalog at GET /catalog):
  nova    OpenAI wire, model nova-pro      q=0.95, tools+vision+json
          -> ALWAYS 402 (free allowance gone). Demonstrates 402 failover.
  orbit   Anthropic wire, model orbit-sonnet q=0.90, tools+json, no vision
          -> 429 with Retry-After: 2 on its FIRST call only, then 200.
          Demonstrates 429 failover / recovery.
  pulse   Gemini wire, model pulse-flash   q=0.75, no tools, no vision
          -> always 200. The fallback that actually answers request 1.
  alpha   OpenAI wire, model alpha-mini   q=0.60, tools+json
          -> always 200.

Deterministic request-1 trace (plain OpenAI chat, no tools/images):
  nova 402 -> orbit 429 -> pulse 200 (Gemini translated to OpenAI).
Request 2 (same): nova 402 -> orbit 200 (429 already spent).
A tools request skips pulse (no tools) and lands on orbit/alpha.

Run:  python3 tests/mock_env.py [--port 18081]
Then:  MAKEFUN_ENV_URL=http://127.0.0.1:18081 python3 main.py --list
       MAKEFUN_ENV_URL=http://127.0.0.1:18081 python3 main.py --port 18080
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 18081
STATE = {"orbit_calls": 0}


def catalog(port):
    base = "http://127.0.0.1:%d" % port
    return {"providers": [
        {"id": "nova", "name": "nova", "api_type": "openai",
         "base_url": base + "/providers/nova",
         "auth": {"header": "X-Nova-Key", "value": "mock-nova-key"},
         "models": [{"id": "nova-pro", "quality": 0.95,
                     "context_window": 128000, "supports_tools": True,
                     "supports_vision": True, "supports_json": True,
                     "input_cost": 0, "output_cost": 0,
                     "remaining_allowance": 1000}]},
        {"id": "orbit", "name": "orbit", "api_type": "anthropic",
         "base_url": base + "/providers/orbit",
         "auth": {"header": "X-Orbit-Key", "value": "mock-orbit-key"},
         "models": [{"id": "orbit-sonnet", "quality": 0.90,
                     "context_window": 200000, "supports_tools": True,
                     "supports_vision": False, "supports_json": True,
                     "input_cost": 0, "output_cost": 0,
                     "remaining_allowance": 1000}]},
        {"id": "pulse", "name": "pulse", "api_type": "gemini",
         "base_url": base + "/providers/pulse",
         "auth": {"header": "X-Pulse-Key", "value": "mock-pulse-key"},
         "models": [{"id": "pulse-flash", "quality": 0.75,
                     "context_window": 1000000, "supports_tools": False,
                     "supports_vision": False, "supports_json": False,
                     "input_cost": 0, "output_cost": 0,
                     "remaining_allowance": 5000}]},
        {"id": "alpha", "name": "alpha", "api_type": "openai",
         "base_url": base + "/providers/alpha",
         "auth": {"header": "X-Alpha-Key", "value": "mock-alpha-key"},
         "models": [{"id": "alpha-mini", "quality": 0.60,
                     "context_window": 32768, "supports_tools": True,
                     "supports_vision": False, "supports_json": True,
                     "input_cost": 0, "output_cost": 0,
                     "remaining_allowance": 5000}]},
    ]}


def openai_ok(model, text):
    return 200, {"id": "chatcmpl-mock", "object": "chat.completion",
                 "created": 0, "model": model,
                 "choices": [{"index": 0,
                              "message": {"role": "assistant",
                                          "content": text},
                              "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 8,
                           "total_tokens": 18}}


def anthropic_ok(model, text):
    return 200, {"id": "msg_mock", "type": "message", "role": "assistant",
                 "model": model,
                 "content": [{"type": "text", "text": text}],
                 "stop_reason": "end_turn", "stop_sequence": None,
                 "usage": {"input_tokens": 10, "output_tokens": 8}}


def gemini_ok(model, text):
    return 200, {"candidates": [{"content": {"role": "model",
                                             "parts": [{"text": text}]},
                                 "finishReason": "STOP"}],
                 "usageMetadata": {"promptTokenCount": 10,
                                   "candidatesTokenCount": 8,
                                   "totalTokenCount": 18}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, obj, extra=None):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode() or "{}")
        except Exception:
            return {}

    def do_GET(self):
        if self.path in ("/catalog", "/providers", "/"):
            self._send(200, catalog(self.server.server_port))
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        body = self._read_json()
        # Streaming: mock answers a non-streaming body, server wraps it.
        # (The mock's job is catalog + failover, not SSE fidelity.)
        if self.path == "/providers/nova/v1/chat/completions":
            self._send(402, {"error": {"message": "free allowance exhausted",
                                       "type": "insufficient_quota",
                                       "code": "402"}})
        elif self.path == "/providers/orbit/v1/messages":
            STATE["orbit_calls"] += 1
            if STATE["orbit_calls"] == 1:
                self._send(429, {"error": {"message": "rate limited",
                                           "type": "rate_limit"}},
                           {"Retry-After": "2"})
            else:
                s, o = anthropic_ok("orbit-sonnet",
                                    "Hello from orbit-sonnet (Anthropic wire).")
                self._send(s, o)
        elif self.path == "/providers/pulse/v1/models/pulse-flash:generateContent":
            s, o = gemini_ok("pulse-flash",
                             "Hello from pulse-flash (Gemini wire).")
            self._send(s, o)
        elif self.path == "/providers/alpha/v1/chat/completions":
            s, o = openai_ok("alpha-mini",
                             "Hello from alpha-mini (OpenAI wire).")
            self._send(s, o)
        else:
            self._send(404, {"error": {"message": "unknown mock route: "
                                        + self.path}})


if __name__ == "__main__":
    port = PORT
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
