"""The runtime-mode switch must authenticate against token-protected nodes."""

import io
import json

import scripts.set_runtime_mode as runtime_mode


def test_verify_endpoint_sends_the_node_token(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["key"] = request.get_header("X-api-key")
        return io.BytesIO(json.dumps({"runtime_mode": "process"}).encode("utf-8"))

    monkeypatch.setenv("STATUS_API_TOKEN", "secret")
    monkeypatch.setattr(runtime_mode.urllib.request, "urlopen", fake_urlopen)
    result = runtime_mode.verify_endpoint("http://node:8000", "process", timeout_seconds=1)
    assert result["status"] == "ok"
    assert seen["key"] == "secret"
