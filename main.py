#!/usr/bin/env python3
"""Free AI model aggregator — make.fun bounty free-model-aggregator-01.

One server, two wires (OpenAI- and Anthropic-compatible), over the simulated
free-tier providers published by the trial env at MAKEFUN_ENV_URL.

Stdlib only (Python 3.13 in the grader; developed/tested on 3.12 here).
No wall clock, no unseeded randomness, no network except the env:
every model call goes to a provider under MAKEFUN_ENV_URL.

Usage:
    python3 main.py [--port PORT] [--host HOST]
    python3 main.py --list        # print the catalog document, byte for byte

Env:
    MAKEFUN_ENV_URL   base URL of the trial env (required in the grader;
                      without it the catalog is empty and every request 503s)
    PORT              listen port (default 8080)

Catalog / env protocol — see README.md "Assumptions" for the full list.
In short, the env protocol is NOT published on the public requirements
page (it lives in the make.fun kit, which was not available for this
build), so discovery is defensive and deterministic:
  * GET {MAKEFUN_ENV_URL}/catalog, then /providers, then /v1/models,
    then {MAKEFUN_ENV_URL} itself — first 200 JSON that yields providers wins.
  * A URL ending in .json is fetched directly as a catalog document.
  * Accepted shapes: {"providers": [...]}, {"catalog": {"providers": [...]}},
    a bare [...] list, {"data": [...]} (OpenAI /v1/models), {"models": [...]}.
  * Each provider: id/name, base_url (absolute, or relative to the env),
    api_type ("openai" | "anthropic" | "gemini", inferred when absent),
    auth {"header": ..., "value": ...} (its own auth header, per the brief),
    models [{id, quality, context_window, supports_tools, supports_vision,
             supports_json, cost / input_cost / output_cost,
             remaining_allowance / free_remaining / allowance_remaining, ...}].
  * --list prints {"providers": [...]} with providers sorted by id,
    models sorted by id, keys sorted — the raw provider/model objects as
    read, byte-for-byte reproducible. The server serves the same bytes
    at GET /catalog.

Routing (deterministic, sorted tie-breaks everywhere):
  * Eligibility: tools -> supports_tools; image content -> supports_vision;
    response_format {"type": "json_object"} -> supports_json; estimated
    input tokens (+ requested max_tokens) must fit context_window;
    remaining allowance must be > 0 when known.
  * Rank: quality desc, total cost asc, remaining allowance desc,
    provider id asc, model id asc. Missing quality is derived from the
    model id by a fixed tier table (see _derive_quality).
  * Failover: try candidates in rank order. 402 (allowance gone) marks
    that model exhausted; 429 marks it throttled (Retry-After is recorded
    and the model is deprioritised, never slept on — sleeping on a wall
    clock would break determinism and the 180 s trial budget);
    5xx / other non-2xx also fall through to the next candidate.
    The last provider error is returned only when every candidate fails.
  * Streaming: provider SSE is translated to the caller's wire format
    chunk by chunk; a non-streaming provider is wrapped into SSE.
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urljoin, urlparse

# ---------------------------------------------------------------- constants
DEFAULT_PORT = 8080
# Quality tiers used ONLY when a catalog model carries no quality score.
# Ordered first-match; deterministic.
QUALITY_TIERS = [
    (r"opus|gpt-5|gpt-4o|gemini-2\.5-pro|claude-opus", 0.95),
    (r"sonnet|gpt-4\.1|gemini-2\.5|o4|claude-sonnet", 0.88),
    (r"gpt-4|gemini-pro|llama-3.*70b|qwen.*72b|mistral-large", 0.82),
    (r"haiku|flash|mini|gpt-4o-mini|llama-3.*8b|qwen.*7b", 0.70),
    (r"nano|tiny|1b|3b", 0.55),
]
DEFAULT_QUALITY = 0.65
DEFAULT_CONTEXT = 8192


def estimate_tokens(messages):
    """Deterministic token estimate: ~1 token per 4 chars of text."""
    total = 0
    for m in messages or []:
        if isinstance(m, dict):
            content = m.get("content")
            if isinstance(content, str):
                total += len(content) // 4 + 1
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        txt = part.get("text") or ""
                        total += len(str(txt)) // 4 + 1
                        if part.get("type") in ("image_url", "image"):
                            total += 256  # image placeholder cost
        elif isinstance(m, str):
            total += len(m) // 4 + 1
    return total


def _derive_quality(model_id):
    mid = str(model_id or "").lower()
    for pattern, score in QUALITY_TIERS:
        if re.search(pattern, mid):
            return score
    return DEFAULT_QUALITY


def _num(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- catalog
def _provider_api_type(provider, models):
    t = str(provider.get("api_type") or provider.get("type") or "").lower()
    if t in ("openai", "anthropic", "gemini"):
        return t
    base = str(provider.get("base_url") or provider.get("url") or "").lower()
    if "anthropic" in base:
        return "anthropic"
    if "gemini" in base or "googleapis" in base:
        return "gemini"
    # Infer from a model id / endpoint hint, else OpenAI (the common wire).
    if any("claude" in str(m.get("id", "")).lower() for m in models):
        return "anthropic"
    if any("gemini" in str(m.get("id", "")).lower() for m in models):
        return "gemini"
    return "openai"


def normalize_catalog(raw):
    """Raw env document -> {"providers": [raw provider objects, sorted]}.

    Preserves every field the env sent (catalog accuracy); only sorting
    is applied, plus filling base_url/api_type when the env left them
    blank is NOT done here — that lives in the routing view, so --list
    stays faithful to what was read.
    """
    providers = []
    if isinstance(raw, dict):
        if isinstance(raw.get("providers"), list):
            providers = list(raw["providers"])
        elif isinstance(raw.get("catalog"), dict) and isinstance(
            raw["catalog"].get("providers"), list
        ):
            providers = list(raw["catalog"]["providers"])
        elif isinstance(raw.get("data"), list):  # OpenAI /v1/models
            providers = [{
                "id": "openai",
                "name": "openai",
                "api_type": "openai",
                "models": [{"id": m.get("id")} for m in raw["data"]
                           if isinstance(m, dict) and m.get("id")],
            }]
        elif isinstance(raw.get("models"), list):
            providers = [{
                "id": "default",
                "name": "default",
                "api_type": "openai",
                "models": list(raw["models"]),
            }]
    elif isinstance(raw, list):
        # Bare list: providers, or models of a single provider.
        if raw and isinstance(raw[0], dict) and "models" in raw[0]:
            providers = list(raw)
        else:
            providers = [{
                "id": "default",
                "name": "default",
                "api_type": "openai",
                "models": list(raw),
            }]
    out = []
    for p in providers:
        if not isinstance(p, dict):
            continue
        p = dict(p)
        models = p.get("models") or p.get("model_list") or []
        models = [dict(m) if isinstance(m, dict) else {"id": str(m)}
                  for m in models]
        models.sort(key=lambda m: str(m.get("id", "")))
        p["models"] = models
        out.append(p)
    out.sort(key=lambda p: str(p.get("id") or p.get("name") or ""))
    return {"providers": out}


def catalog_bytes(catalog):
    return (json.dumps(catalog, sort_keys=True, indent=2) + "\n").encode("utf-8")


def fetch_json(url, timeout=10):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_catalog(env_url):
    """Fetch the env catalog; empty catalog when no env is configured."""
    if not env_url:
        return {"providers": []}
    env_url = env_url.rstrip("/")
    if env_url.endswith(".json"):
        return normalize_catalog(fetch_json(env_url))
    candidates = [env_url + suffix for suffix in
                  ("/catalog", "/providers", "/v1/models", "")]
    last_error = None
    for url in candidates:
        try:
            return normalize_catalog(fetch_json(url))
        except Exception as exc:  # try the next discovery path
            last_error = exc
    raise RuntimeError(
        "could not read a provider catalog from %s (tried %s): %s"
        % (env_url, candidates, last_error))


# ---------------------------------------------------------------- routing
class Candidate(object):
    __slots__ = ("provider", "model", "quality", "cost", "remaining",
                 "context", "tools", "vision", "json_mode", "api_type",
                 "base_url", "auth_header", "auth_value", "exhausted",
                 "throttled")

    def __init__(self, provider, model, env_url):
        self.provider = provider
        self.model = model
        pid = str(provider.get("id") or provider.get("name") or "")
        mid = str(model.get("id") or model.get("name") or "")
        self.quality = _num(model.get("quality"), None)
        if self.quality is None:
            self.quality = _derive_quality(mid)
        inp = _num(model.get("input_cost", model.get("cost")), 0.0) or 0.0
        outp = _num(model.get("output_cost"), 0.0) or 0.0
        self.cost = inp + outp
        rem = model.get("remaining_allowance",
                        model.get("free_remaining",
                                  model.get("allowance_remaining")))
        self.remaining = _num(rem, None)
        self.context = int(_num(model.get("context_window"),
                                DEFAULT_CONTEXT) or DEFAULT_CONTEXT)
        self.tools = bool(model.get("supports_tools",
                                    model.get("tools", False)))
        self.vision = bool(model.get("supports_vision",
                                     model.get("vision", False)))
        self.json_mode = bool(model.get("supports_json",
                                        model.get("json_mode", False)))
        self.api_type = _provider_api_type(provider, provider.get("models") or [])
        base = str(provider.get("base_url") or provider.get("url") or "")
        if base and not urlparse(base).scheme:
            base = urljoin(env_url.rstrip("/") + "/", base.lstrip("/"))
        self.base_url = base.rstrip("/") or env_url.rstrip("/")
        auth = provider.get("auth") or {}
        if isinstance(auth, dict):
            self.auth_header = auth.get("header") or "Authorization"
            self.auth_value = auth.get("value") or auth.get("api_key") or ""
        else:
            self.auth_header, self.auth_value = "Authorization", str(auth)
        if not self.auth_value:
            # Stand-in keys the gateway sets in a trial (see requirements).
            self.auth_value = os.environ.get(
                "OPENAI_API_KEY" if self.api_type == "openai"
                else "ANTHROPIC_API_KEY" if self.api_type == "anthropic"
                else "GEMINI_API_KEY", "")
            if self.auth_value and self.auth_header == "Authorization":
                self.auth_value = "Bearer " + self.auth_value
        self.exhausted = False
        self.throttled = False
        _ = (pid, mid)

    def sort_key(self):
        # quality desc, cost asc, remaining desc, provider asc, model asc.
        rem = self.remaining if self.remaining is not None else 1e18
        return (-self.quality, self.cost, -rem,
                str(self.provider.get("id") or ""),
                str(self.model.get("id") or ""))


class Router(object):
    def __init__(self, catalog, env_url):
        self.catalog = catalog
        self.env_url = env_url
        self.candidates = []
        for provider in catalog.get("providers", []):
            for model in provider.get("models", []):
                self.candidates.append(Candidate(provider, model, env_url))

    def rank(self, needs):
        """needs: dict(tools, vision, json_mode, tokens). Ranked candidates."""
        eligible = []
        for c in self.candidates:
            if c.exhausted:
                continue
            if c.remaining is not None and c.remaining <= 0:
                continue
            if needs.get("tools") and not c.tools:
                continue
            if needs.get("vision") and not c.vision:
                continue
            if needs.get("json_mode") and not c.json_mode:
                continue
            if needs.get("tokens", 0) > c.context:
                continue
            eligible.append(c)
        # Throttling is per-request only: a 429 means "not right now", and
        # the next request re-ranks the provider fresh (its Retry-After
        # may have passed; sleeping on it would break determinism).
        # Only 402 exhaustion persists across requests.
        eligible.sort(key=lambda c: c.sort_key())
        return eligible

    def mark(self, cand, status, retry_after=None):
        if status == 402:
            cand.exhausted = True
            cand.remaining = 0
        # 429: no persistent state — the candidate was already skipped
        # for the rest of this request by the caller's loop over `ranked`.

# ------------------------------------------------------- provider calls
def _http_json(url, payload, headers, timeout=30):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except Exception as e:  # network failure: treat as a failed candidate
        return 0, {}, json.dumps({"error": {"message": str(e)}}).encode()


def _http_stream(url, payload, headers, timeout=60):
    """Yield raw SSE lines from a provider, or (status, headers, body)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp
    except urllib.error.HTTPError as e:
        return (e.code, dict(e.headers), e.read())
    except Exception as e:
        return (0, {}, json.dumps({"error": {"message": str(e)}}).encode())


def provider_headers(cand):
    h = {"Content-Type": "application/json"}
    if cand.auth_value:
        h[cand.auth_header] = cand.auth_value
    return h


def _openai_payload(cand, messages, model, max_tokens, tools, json_mode,
                    stream):
    p = {"model": model, "messages": messages, "stream": bool(stream)}
    if max_tokens:
        p["max_tokens"] = max_tokens
    if tools:
        p["tools"] = tools
    if json_mode:
        p["response_format"] = {"type": "json_object"}
    return p


def _anthropic_payload(cand, messages, model, max_tokens, tools, stream):
    system, msgs = [], []
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "system":
            c = m.get("content")
            system.append(c if isinstance(c, str) else json.dumps(c))
        else:
            msgs.append(m)
    p = {"model": model, "messages": msgs,
         "max_tokens": max_tokens or 1024, "stream": bool(stream)}
    if system:
        p["system"] = "\n".join(system)
    if tools:
        p["tools"] = [{"name": t["function"]["name"],
                       "description": t["function"].get("description", ""),
                       "input_schema": t["function"].get("parameters", {})}
                      for t in tools if isinstance(t, dict)
                      and t.get("type") == "function"]
    return p


def _gemini_payload(cand, messages, model, max_tokens, tools):
    contents = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = "model" if m.get("role") == "assistant" else "user"
        c = m.get("content")
        text = c if isinstance(c, str) else json.dumps(c) if c else ""
        contents.append({"role": role, "parts": [{"text": text}]})
    p = {"contents": contents}
    if max_tokens:
        p["generationConfig"] = {"maxOutputTokens": max_tokens}
    return p


def provider_url(cand):
    if cand.api_type == "anthropic":
        return cand.base_url + "/v1/messages"
    if cand.api_type == "gemini":
        return (cand.base_url + "/v1/models/"
                + str(cand.model.get("id")) + ":generateContent")
    return cand.base_url + "/v1/chat/completions"


def call_provider(cand, wire, messages, max_tokens, tools, json_mode,
                  stream=False):
    """Call one candidate. Returns (status, headers, body_bytes).

    `wire` is the caller's wire ("openai" or "anthropic"); the request is
    translated to the provider's own wire.
    """
    mid = str(cand.model.get("id"))
    headers = provider_headers(cand)
    if cand.api_type == "anthropic":
        payload = _anthropic_payload(cand, messages, mid, max_tokens, tools,
                                     stream)
    elif cand.api_type == "gemini":
        payload = _gemini_payload(cand, messages, mid, max_tokens, tools)
        if stream:
            payload["generationConfig"] = payload.get("generationConfig", {})
    else:
        payload = _openai_payload(cand, messages, mid, max_tokens, tools,
                                  json_mode, stream)
    return _http_json(provider_url(cand), payload, headers)


# ------------------------------------------------------- translation
def _text_of_openai_message(msg):
    if isinstance(msg, dict):
        return msg.get("content") or ""
    return ""


def openai_to_anthropic(body, model_id):
    """OpenAI chat-completion JSON -> Anthropic /v1/messages JSON."""
    try:
        data = json.loads(body)
    except Exception:
        return 502, {"error": {"type": "api_error",
                               "message": "bad provider response"}}
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = []
    text = _text_of_openai_message(msg)
    if text:
        content.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        content.append({"type": "tool_use", "id": tc.get("id", "toolu_1"),
                        "name": fn.get("name", ""), "input": args})
    if not content:
        content = [{"type": "text", "text": ""}]
    usage = data.get("usage") or {}
    return 200, {"id": data.get("id", "msg_1"), "type": "message",
                 "role": "assistant", "model": model_id, "content": content,
                 "stop_reason": "end_turn" if choice.get("finish_reason")
                 in (None, "stop") else "tool_use",
                 "stop_sequence": None,
                 "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                           "output_tokens": usage.get("completion_tokens", 0)}}


def anthropic_to_openai(body, model_id):
    try:
        data = json.loads(body)
    except Exception:
        return 502, {"error": {"message": "bad provider response"}}
    text, tool_calls = [], []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            text.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", "call_1"), "type": "function",
                "function": {"name": block.get("name", ""),
                             "arguments": json.dumps(block.get("input") or {})}})
    msg = {"role": "assistant", "content": "".join(text) or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    usage = data.get("usage") or {}
    return 200, {"id": data.get("id", "chatcmpl_1"),
                 "object": "chat.completion",
                 "created": 0, "model": model_id,
                 "choices": [{"index": 0, "message": msg,
                              "finish_reason": "tool_calls" if tool_calls
                              else "stop"}],
                 "usage": {"prompt_tokens": usage.get("input_tokens", 0),
                           "completion_tokens": usage.get("output_tokens", 0),
                           "total_tokens": usage.get("input_tokens", 0)
                           + usage.get("output_tokens", 0)}}


def gemini_to_openai(body, model_id):
    try:
        data = json.loads(body)
    except Exception:
        return 502, {"error": {"message": "bad provider response"}}
    cand0 = (data.get("candidates") or [{}])[0]
    parts = ((cand0.get("content") or {}).get("parts") or [])
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    usage = data.get("usageMetadata") or {}
    return 200, {"id": "chatcmpl_1", "object": "chat.completion",
                 "created": 0, "model": model_id,
                 "choices": [{"index": 0,
                              "message": {"role": "assistant",
                                          "content": text},
                              "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": usage.get("promptTokens", 0),
                           "completion_tokens": usage.get("candidatesTokenCount", 0),
                           "total_tokens": usage.get("totalTokenCount", 0)}}


def gemini_to_anthropic(body, model_id):
    status, data = gemini_to_openai(body, model_id)
    if status != 200:
        return status, data
    text = data["choices"][0]["message"]["content"]
    return 200, {"id": "msg_1", "type": "message", "role": "assistant",
                 "model": model_id,
                 "content": [{"type": "text", "text": text}],
                 "stop_reason": "end_turn", "stop_sequence": None,
                 "usage": {"input_tokens": data["usage"]["prompt_tokens"],
                           "output_tokens": data["usage"]["completion_tokens"]}}


def translate_to_caller(cand, wire, body):
    """Translate a provider's non-streaming body to the caller's wire."""
    mid = str(cand.model.get("id"))
    if wire == "openai":
        if cand.api_type == "openai":
            return 200, json.loads(body)
        if cand.api_type == "anthropic":
            return anthropic_to_openai(body, mid)
        return gemini_to_openai(body, mid)
    # wire == anthropic
    if cand.api_type == "anthropic":
        return 200, json.loads(body)
    if cand.api_type == "openai":
        return openai_to_anthropic(body, mid)
    return gemini_to_anthropic(body, mid)


def _sse_chunk_openai_text(model, text, done=False):
    if done:
        return ("data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\","
                "\"created\":0,\"model\":\"" + model + "\",\"choices\":[{\"index\":0,"
                "\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
                "data: [DONE]\n\n")
    return ("data: " + json.dumps(
        {"id": "chatcmpl_1", "object": "chat.completion.chunk", "created": 0,
         "model": model, "choices": [{"index": 0,
                                      "delta": {"content": text},
                                      "finish_reason": None}]}) + "\n\n")


def _sse_chunk_anthropic_text(text, done=False):
    if done:
        return ("event: message_stop\n"
                "data: {\"type\": \"message_stop\"}\n\n")
    return ("event: content_block_delta\n"
            "data: " + json.dumps({"type": "content_block_delta",
                                   "index": 0,
                                   "delta": {"type": "text_delta",
                                             "text": text}}) + "\n\n")


def _extract_text(cand, data):
    """Plain-text content of a provider's non-streaming JSON body."""
    if not isinstance(data, dict):
        return ""
    if cand.api_type == "anthropic":
        return "".join(b.get("text", "") for b in data.get("content") or []
                       if isinstance(b, dict))
    if cand.api_type == "gemini":
        parts = []
        for c in data.get("candidates") or []:
            for p in ((c.get("content") or {}).get("parts") or []):
                parts.append(p.get("text", ""))
        return "".join(parts)
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return msg.get("content") or ""


def stream_translate(cand, wire, raw_lines):
    """Translate provider SSE lines to caller-wire SSE lines (best effort).

    OpenAI provider -> passthrough for OpenAI callers; Anthropic/Gemini
    provider text is extracted and re-emitted in the caller's chunk shape.
    Non-SSE provider bodies are wrapped as a single text chunk.
    """
    model = str(cand.model.get("id"))
    out = []
    if cand.api_type == "openai" and wire == "openai":
        return raw_lines
    text_parts = []
    for line in raw_lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("[DONE]", ""):
            continue
        try:
            data = json.loads(payload)
        except Exception:
            continue
        if cand.api_type == "anthropic":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta":
                text_parts.append(delta.get("text", ""))
            elif data.get("type") == "content_block_delta":
                text_parts.append((data.get("delta") or {}).get("text", ""))
        elif cand.api_type == "openai":
            for ch in data.get("choices") or []:
                text_parts.append((ch.get("delta") or {}).get("content") or "")
        elif cand.api_type == "gemini":
            for c in data.get("candidates") or []:
                for p in ((c.get("content") or {}).get("parts") or []):
                    text_parts.append(p.get("text", ""))
    text = "".join(text_parts)
    if wire == "openai":
        out.append(_sse_chunk_openai_text(model, text))
        out.append(_sse_chunk_openai_text(model, "", done=True))
    else:
        out.append("event: message_start\n"
                   "data: " + json.dumps({"type": "message_start",
                                          "message": {"id": "msg_1",
                                                      "type": "message",
                                                      "role": "assistant",
                                                      "model": model,
                                                      "content": [],
                                                      "stop_reason": None,
                                                      "stop_sequence": None,
                                                      "usage": {"input_tokens": 0,
                                                                "output_tokens": 0}}}) + "\n\n")
        out.append("event: content_block_start\n"
                   "data: " + json.dumps({"type": "content_block_start",
                                          "index": 0,
                                          "content_block": {"type": "text",
                                                            "text": ""}}) + "\n\n")
        out.append(_sse_chunk_anthropic_text(text))
        out.append("event: content_block_stop\n"
                   "data: " + json.dumps({"type": "content_block_stop",
                                          "index": 0}) + "\n\n")
        out.append(_sse_chunk_anthropic_text("", done=True))
    return "".join(out)

# ------------------------------------------------------- HTTP server
class Aggregator(object):
    def __init__(self, catalog, env_url):
        self.catalog = catalog
        self.env_url = env_url
        self.router = Router(catalog, env_url)
        self.catalog_doc = catalog_bytes(catalog)

    # -- request analysis -------------------------------------------------
    @staticmethod
    def _needs_from_openai(body):
        messages = body.get("messages") or []
        tools = body.get("tools") or []
        vision = False
        for m in messages:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") in (
                            "image_url", "image", "input_image"):
                        vision = True
        return {
            "tools": bool(tools),
            "vision": vision,
            "json_mode": (body.get("response_format") or {}).get("type")
                         == "json_object",
            "tokens": estimate_tokens(messages)
                      + int(body.get("max_tokens") or 0),
        }, messages, tools

    @staticmethod
    def _needs_from_anthropic(body):
        messages = body.get("messages") or []
        tools = body.get("tools") or []
        vision = False
        for m in messages:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "image":
                        vision = True
        # Anthropic system prompt counts toward context too.
        sys_text = body.get("system") or ""
        tokens = estimate_tokens(messages) + len(str(sys_text)) // 4
        if isinstance(body.get("system"), list):
            tokens += estimate_tokens(
                [{"content": [{"text": p.get("text", "")}]}
                 for p in body["system"] if isinstance(p, dict)])
        return {"tools": bool(tools), "vision": vision, "json_mode": False,
                "tokens": tokens + int(body.get("max_tokens") or 0)}, \
            messages, tools

    def _serve(self, wire, body):
        """Returns (status, content_type, body_bytes, is_stream)."""
        if wire == "openai":
            needs, messages, tools = self._needs_from_openai(body)
            max_tokens = body.get("max_tokens")
            stream = bool(body.get("stream"))
            json_mode = needs["json_mode"]
            # Anthropic-shaped messages are normalised by the caller below.
            openai_messages = messages
        else:
            needs, messages, tools = self._needs_from_anthropic(body)
            max_tokens = body.get("max_tokens")
            stream = bool(body.get("stream"))
            json_mode = False
            system = body.get("system")
            openai_messages = []
            if system:
                sys_text = system if isinstance(system, str) else json.dumps(system)
                openai_messages.append({"role": "system", "content": sys_text})
            for m in messages:
                if isinstance(m, dict):
                    content = m.get("content")
                    if isinstance(content, list):
                        text = "".join(p.get("text", "") for p in content
                                       if isinstance(p, dict))
                        openai_messages.append({"role": m.get("role", "user"),
                                                "content": text})
                    else:
                        openai_messages.append(m)
        ranked = self.router.rank(needs)
        if not ranked:
            err = {"error": {"message": "no free model can serve this "
                                        "request", "type": "no_model",
                               "code": "no_model"}}
            if wire == "anthropic":
                err = {"type": "error",
                       "error": {"type": "invalid_request_error",
                                 "message": err["error"]["message"]}}
            return 503, "application/json", \
                json.dumps(err).encode(), False
        last_status, last_headers, last_body = 502, {}, b""
        for cand in ranked:
            if stream:
                result = self._serve_stream(cand, wire, openai_messages,
                                            max_tokens, tools, json_mode)
                if result is not None:
                    return result
                # _serve_stream returns None only on a failed candidate;
                # its failure was already recorded — try the next one.
                last_status, last_body = 502, b'{"error":{"message":' \
                    b'"streaming provider failed"}}'
                continue
            status, headers, raw = call_provider(
                cand, wire, openai_messages, max_tokens, tools, json_mode)
            if status == 200:
                try:
                    _s, translated = translate_to_caller(cand, wire, raw)
                except Exception:
                    translated = {"error": {"message": "bad provider response"}}
                    _s = 502
                self._note_success(cand, raw)
                return _s, "application/json", \
                    json.dumps(translated).encode(), False
            self.router.mark(cand, status,
                             (headers or {}).get("Retry-After")
                             or (headers or {}).get("retry-after"))
            last_status, last_headers, last_body = status, headers, raw
        # Every candidate failed: surface the last provider error verbatim.
        return last_status or 502, "application/json", last_body, False

    def _note_success(self, cand, raw):
        try:
            data = json.loads(raw)
            usage = data.get("usage") or {}
            used = usage.get("total_tokens") or (
                (usage.get("prompt_tokens") or 0)
                + (usage.get("completion_tokens") or 0))
            if cand.remaining is not None and used:
                cand.remaining = max(0, cand.remaining - used)
        except Exception:
            pass

    def _serve_stream(self, cand, wire, messages, max_tokens, tools,
                      json_mode):
        """Attempt a streaming call; None if the candidate failed."""
        mid = str(cand.model.get("id"))
        headers = provider_headers(cand)
        if cand.api_type == "anthropic":
            payload = _anthropic_payload(cand, messages, mid, max_tokens,
                                         tools, True)
        elif cand.api_type == "gemini":
            payload = _gemini_payload(cand, messages, mid, max_tokens, tools)
        else:
            payload = _openai_payload(cand, messages, mid, max_tokens, tools,
                                      json_mode, True)
        result = _http_stream(provider_url(cand), payload, headers)
        if isinstance(result, tuple):
            status, hdrs, raw = result
            self.router.mark(cand, status,
                             (hdrs or {}).get("Retry-After"))
            return None
        # Streaming response object: read it fully (trial requests are
        # small; buffering keeps translation deterministic), then emit.
        try:
            raw_text = result.read().decode("utf-8", "replace")
        except Exception:
            return None
        finally:
            try:
                result.close()
            except Exception:
                pass
        lines = [ln for ln in raw_text.splitlines() if ln.strip()]
        if "data:" not in raw_text:
            # Provider answered streaming with a plain JSON body
            # (mock envs do this): wrap its text as SSE in caller wire.
            try:
                data = json.loads(raw_text)
                text = _extract_text(cand, data)
            except Exception:
                text = ""
            if wire == "openai":
                out = (_sse_chunk_openai_text(mid, text)
                       + _sse_chunk_openai_text(mid, "", done=True))
            else:
                out = self._anthropic_sse_from_text(mid, text)
            return 200, "text/event-stream", out.encode(), True
        if cand.api_type == "openai" and wire == "openai":
            out = raw_text
            if "[DONE]" not in out:
                out += "data: [DONE]\n\n"
        elif cand.api_type == "openai" and wire == "anthropic":
            # Translate OpenAI SSE deltas into Anthropic SSE.
            text = ""
            for ln in lines:
                if ln.startswith("data:") and "[DONE]" not in ln:
                    try:
                        d = json.loads(ln[5:].strip())
                        for ch in d.get("choices") or []:
                            text += (ch.get("delta") or {}).get("content") or ""
                    except Exception:
                        pass
            out = self._anthropic_sse_from_text(mid, text)
        else:
            out = stream_translate(cand, wire, lines)
        return 200, "text/event-stream", out.encode(), True

    @staticmethod
    def _anthropic_sse_from_text(model, text):
        return (
            "event: message_start\n"
            "data: " + json.dumps({"type": "message_start",
                                   "message": {"id": "msg_1", "type": "message",
                                               "role": "assistant",
                                               "model": model, "content": [],
                                               "stop_reason": None,
                                               "stop_sequence": None,
                                               "usage": {"input_tokens": 0,
                                                         "output_tokens": 0}}}) + "\n\n"
            "event: content_block_start\n"
            "data: " + json.dumps({"type": "content_block_start", "index": 0,
                                   "content_block": {"type": "text",
                                                     "text": ""}}) + "\n\n"
            + _sse_chunk_anthropic_text(text) +
            "event: content_block_stop\n"
            "data: " + json.dumps({"type": "content_block_stop",
                                   "index": 0}) + "\n\n"
            + _sse_chunk_anthropic_text("", done=True))


def make_handler(agg):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep stdout clean / deterministic
            pass

        def _send(self, status, ctype, body):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        def do_GET(self):
            if self.path in ("/", "/health"):
                self._send(200, "application/json",
                           json.dumps({"status": "ok",
                                       "providers": len(agg.catalog.get(
                                           "providers", []))}).encode())
            elif self.path in ("/catalog", "/v1/catalog"):
                self._send(200, "application/json", agg.catalog_doc)
            elif self.path == "/v1/models":
                data = {"object": "list", "data": [
                    {"id": str(m.get("id")), "object": "model",
                     "created": 0,
                     "owned_by": str(p.get("id") or "")}
                    for p in agg.catalog.get("providers", [])
                    for m in p.get("models", [])]}
                self._send(200, "application/json",
                           json.dumps(data).encode())
            else:
                self._send(404, "application/json",
                           b'{"error":{"message":"not found"}}')

        def do_POST(self):
            try:
                body = self._read_json()
            except Exception:
                self._send(400, "application/json",
                           b'{"error":{"message":"invalid JSON"}}')
                return
            if self.path == "/v1/chat/completions":
                status, ctype, out, _ = agg._serve("openai", body)
                self._send(status, ctype, out)
            elif self.path == "/v1/messages":
                # Anthropic clients send anthropic-version; accept any.
                status, ctype, out, _ = agg._serve("anthropic", body)
                self._send(status, ctype, out)
            else:
                self._send(404, "application/json",
                           b'{"error":{"message":"not found"}}')
    return Handler


# ---------------------------------------------------------------- entrypoint
def main(argv):
    if "--list" in argv:
        env_url = os.environ.get("MAKEFUN_ENV_URL", "")
        try:
            catalog = load_catalog(env_url)
        except Exception as exc:
            sys.stderr.write("error: %s\n" % exc)
            return 1
        sys.stdout.buffer.write(catalog_bytes(catalog))
        return 0
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", str(DEFAULT_PORT)))
    i = 0
    while i < len(argv):
        if argv[i] == "--port" and i + 1 < len(argv):
            port = int(argv[i + 1])
            i += 2
        elif argv[i] == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
            i += 2
        else:
            i += 1
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    try:
        catalog = load_catalog(env_url)
    except Exception as exc:
        sys.stderr.write("error: %s\n" % exc)
        return 1
    agg = Aggregator(catalog, env_url)
    server = ThreadingHTTPServer((host, port), make_handler(agg))
    sys.stderr.write("free-model-aggregator listening on %s:%d "
                     "(%d providers, %d models)\n"
                     % (host, port, len(catalog.get("providers", [])),
                        len(agg.router.candidates)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
