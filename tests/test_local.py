#!/usr/bin/env python3
"""Local smoke test: mock env + aggregator, no make.fun kit needed.

Starts tests/mock_env.py on 18081 and main.py on 18080, then:
 1. OpenAI plain request   -> expect pulse-flash (nova 402, orbit 429 first)
 2. Same request again     -> expect orbit-sonnet (nova 402, orbit now 200)
 3. Anthropic request      -> expect orbit-sonnet translated to Anthropic wire
 4. OpenAI streaming req   -> expect SSE with [DONE]
 5. Tools request          -> must NOT land on pulse (no tools)
Prints PASS/FAIL per check and exits non-zero on any failure.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_PORT, AGG_PORT = 18081, 18080
ENV = dict(os.environ, MAKEFUN_ENV_URL="http://127.0.0.1:%d" % MOCK_PORT)


def post(port, path, obj, headers=None):
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (port, path),
        data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def wait(port, path="/health", tries=50):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:%d%s" % (port, path), timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.1)
    return False


def main():
    mock = subprocess.Popen([sys.executable, "tests/mock_env.py",
                             "--port", str(MOCK_PORT)], cwd=ROOT)
    agg = subprocess.Popen([sys.executable, "main.py", "--port",
                            str(AGG_PORT)], cwd=ROOT, env=ENV)
    checks = []
    try:
        assert wait(MOCK_PORT, "/catalog"), "mock env did not start"
        assert wait(AGG_PORT), "aggregator did not start"

        s, b = post(AGG_PORT, "/v1/chat/completions",
                    {"model": "auto", "messages": [
                        {"role": "user", "content": "Hello"}]})
        d = json.loads(b)
        got = d.get("choices", [{}])[0].get("message", {}).get("model") \
            or d.get("model")
        checks.append(("req1 failover nova402+orbit429 -> pulse-flash",
                       s == 200 and d.get("model") == "pulse-flash"
                       and "pulse-flash" in str(d)))
        print("req1:", s, b[:200])

        s, b = post(AGG_PORT, "/v1/chat/completions",
                    {"model": "auto", "messages": [
                        {"role": "user", "content": "Hello again"}]})
        d = json.loads(b)
        checks.append(("req2 failover nova402 -> orbit-sonnet",
                       s == 200 and d.get("model") == "orbit-sonnet"))
        print("req2:", s, b[:200])

        s, b = post(AGG_PORT, "/v1/messages",
                    {"model": "auto", "max_tokens": 64, "messages": [
                        {"role": "user", "content": "Hello"}]},
                    {"anthropic-version": "2023-06-01"})
        d = json.loads(b)
        checks.append(("anthropic wire -> orbit-sonnet, type=message",
                       s == 200 and d.get("type") == "message"
                       and d.get("model") == "orbit-sonnet"))
        print("anthropic:", s, b[:200])

        s, b = post(AGG_PORT, "/v1/chat/completions",
                    {"model": "auto", "stream": True, "messages": [
                        {"role": "user", "content": "Hello"}]})
        checks.append(("streaming SSE ends with [DONE]",
                       s == 200 and "[DONE]" in b and "data:" in b))
        print("stream:", s, b[:300])

        s, b = post(AGG_PORT, "/v1/chat/completions",
                    {"model": "auto", "messages": [
                        {"role": "user", "content": "Call a tool"}],
                     "tools": [{"type": "function",
                                "function": {"name": "get_weather",
                                             "parameters": {"type": "object",
                                                            "properties": {}}}}]})
        d = json.loads(b)
        checks.append(("tools request skips pulse (no tools)",
                       s == 200 and d.get("model") in
                       ("orbit-sonnet", "alpha-mini", "nova-pro")))
        print("tools:", s, b[:200])
    finally:
        agg.terminate()
        mock.terminate()
    failed = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(("PASS " if ok else "FAIL ") + n)
    print("ALL PASS" if not failed else "FAILURES: %s" % failed)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
