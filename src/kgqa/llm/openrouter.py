"""OpenRouter adapter (OpenAI-compatible chat completions). Standard library only.

The API key is read from OPENROUTER_API_KEY, never from kgqa.toml. Free models (ids ending in
":free") are rate limited; 429s and 5xx responses are retried with backoff, and `fallback_models`
lets OpenRouter route to another model when the first is unavailable.

`timeout` is a wall-clock deadline for the whole call, retries included. A per-socket timeout is
not enough: while a model is queued or reasoning, OpenRouter keeps the connection alive by sending
whitespace, which resets socket timeouts indefinitely.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

from kgqa.llm import LLMResponse

log = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.8-27b:free"
# DEFAULT_MODEL = "inclusionai/ling-3.0-flash-sante:free"

class OpenRouterError(RuntimeError):
    pass


class OpenRouterLLM:
    name = "openrouter"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        fallback_models: list[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        timeout: float = 120.0,
        max_retries: int = 4,
        reasoning_effort: str | None = "low",
        api_key: str | None = None,
        base_url: str = API_URL,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise OpenRouterError("Set the OPENROUTER_API_KEY environment variable")
        self.model = model
        self.fallback_models = list(fallback_models or [])
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        self.name = f"openrouter:{model}"

    def _body(self, system: str, prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.fallback_models:
            body["models"] = [self.model, *self.fallback_models]
        if self.reasoning_effort:
            # one SPARQL query does not need long deliberation; ignored by non-reasoning models
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    def _read(self, resp: Any, deadline: float) -> bytes:
        chunks = []
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError
            chunk = resp.read1(8192) if hasattr(resp, "read1") else resp.read(8192)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "X-Title": "kgqa"},
        )
        start = time.monotonic()
        deadline = start + self.timeout
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                # the socket may block until the deadline, never past it; keep-alive bytes cannot extend it
                with urllib.request.urlopen(req, timeout=remaining) as resp:
                    return json.loads(self._read(resp, deadline))
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if (e.code == 429 or e.code >= 500) and attempt < self.max_retries:
                    wait = min(float(e.headers.get("Retry-After") or delay), 60.0, max(0.0, deadline - time.monotonic()))
                    log.warning("openrouter %s: HTTP %s, retrying in %.0fs (%s)", self.model, e.code, wait, detail[:120])
                    time.sleep(wait)
                    delay *= 2
                    continue
                raise OpenRouterError(f"HTTP {e.code}: {detail}") from e
            except TimeoutError:
                break  # re-sending would repeat the model's work and spend quota again
            except (urllib.error.URLError, OSError) as e:
                if isinstance(getattr(e, "reason", None), TimeoutError) or time.monotonic() >= deadline:
                    break
                log.warning("openrouter %s: %s, retrying", self.model, e or type(e).__name__)
                time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                delay *= 2
        raise OpenRouterError(f"{self.model}: no response within {self.timeout:.0f}s (raise [llm] timeout, or try another model)")

    def complete(self, system: str, prompt: str) -> LLMResponse:
        log.info("openrouter %s: calling (deadline %.0fs)", self.model, self.timeout)
        t0 = time.monotonic()
        data = self._post(self._body(system, prompt))
        log.info("openrouter %s: answered in %.1fs", data.get("model", self.model), time.monotonic() - t0)
        if "error" in data:
            raise OpenRouterError(str(data["error"]))
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        if not text.strip():
            # reasoning models can spend the whole budget thinking; raise max_tokens if this happens
            raise OpenRouterError(f"empty completion from {data.get('model', self.model)} (finish_reason={choice.get('finish_reason')})")
        usage = data.get("usage") or {}
        return LLMResponse(text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


def make(**kwargs: Any) -> OpenRouterLLM:
    return OpenRouterLLM(**kwargs)
