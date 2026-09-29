import io
import json
import urllib.error

import pytest

from kgqa.llm import make_llm
from kgqa.llm.openrouter import OpenRouterError, OpenRouterLLM


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok(text="```sparql\nASK { ?s ?p ?o }\n```"):
    return _Resp(json.dumps({"model": "m", "choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}).encode())


def test_request_shape_and_usage(monkeypatch):
    seen = {}

    def fake(req, timeout):
        seen["auth"] = req.headers["Authorization"]
        seen["body"] = json.loads(req.data)
        return _ok()

    monkeypatch.setattr("urllib.request.urlopen", fake)
    llm = OpenRouterLLM(model="a:free", fallback_models=["b:free"], api_key="k")
    r = llm.complete("sys", "prompt")
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["models"] == ["a:free", "b:free"] and seen["body"]["messages"][0]["role"] == "system"
    assert (r.input_tokens, r.output_tokens) == (11, 7) and "ASK" in r.text


def test_retries_rate_limit(monkeypatch):
    calls = []

    def fake(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "rate limited", {"Retry-After": "0"}, io.BytesIO(b"slow down"))
        return _ok()

    monkeypatch.setattr("urllib.request.urlopen", fake)
    assert OpenRouterLLM(api_key="k").complete("s", "p").text
    assert len(calls) == 2


def test_empty_completion_and_missing_key(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _ok(text=""))
    with pytest.raises(OpenRouterError):
        OpenRouterLLM(api_key="k").complete("s", "p")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenRouterError):
        make_llm("openrouter")


def test_caching_llm_replays(tmp_path):
    from kgqa.llm import CachingLLM, ScriptedLLM

    inner = ScriptedLLM(["one"])
    assert CachingLLM(inner, tmp_path / "c.jsonl").complete("s", "p").text == "one"
    assert CachingLLM(inner, tmp_path / "c.jsonl").complete("s", "p").text == "one"  # no second call
    assert len(inner.prompts) == 1


class _Trickle(io.BytesIO):
    """A response that keeps sending keep-alive whitespace and never finishes."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read1(self, n):
        import time

        time.sleep(0.05)
        return b" "


def test_wall_clock_deadline_beats_keepalive(monkeypatch):
    import time

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: _Trickle())
    llm = OpenRouterLLM(api_key="k", timeout=0.5)
    t0 = time.monotonic()
    with pytest.raises(OpenRouterError, match="no response within"):
        llm.complete("s", "p")
    assert time.monotonic() - t0 < 2.0


def test_reasoning_effort_sent(monkeypatch):
    seen = {}

    def fake(req, timeout):
        seen["body"] = json.loads(req.data)
        return _ok()

    monkeypatch.setattr("urllib.request.urlopen", fake)
    OpenRouterLLM(api_key="k").complete("s", "p")
    assert seen["body"]["reasoning"] == {"effort": "low"}


def test_dotenv_loader_does_not_override(tmp_path, monkeypatch):
    import os

    from kgqa.cli import load_dotenv

    (tmp_path / ".env").write_text("# comment\nexport OPENROUTER_API_KEY='sk-or-file'\nOTHER_KEY=x\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OTHER_KEY", "from-shell")
    load_dotenv(tmp_path / ".env")
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-file" and os.environ["OTHER_KEY"] == "from-shell"
