"""LLM interface used by the constrained generator and the fallback path.

No provider is wired in. To hook one up, write a factory and point the config at it:

    # my_llm.py
    from kgqa.llm import LLMResponse
    class MyLLM:
        name = "my-llm"
        def complete(self, system: str, prompt: str) -> LLMResponse:
            text, usage = call_your_api(system, prompt)
            return LLMResponse(text, usage.input_tokens, usage.output_tokens)
    def make(**kwargs):
        return MyLLM()

    # kgqa.toml
    [llm]
    backend = "python:my_llm:make"
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from kgqa.trace import LLMLog, current_trace


class LLMNotConfigured(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> LLMResponse: ...


class NullLLM:
    """No LLM configured: generation and fallback degrade to templates and the review queue."""

    name = "none"

    def complete(self, system: str, prompt: str) -> LLMResponse:
        raise LLMNotConfigured("No LLM backend configured (set [llm].backend in kgqa.toml)")


class ScriptedLLM:
    """Returns canned responses in order, or computes them with a function. For tests and demos."""

    name = "scripted"

    def __init__(self, responses: Sequence[str] | Callable[[str, str], str]) -> None:
        self._fn = responses if callable(responses) else None
        self._queue = list(responses) if not callable(responses) else []
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> LLMResponse:
        self.prompts.append((system, prompt))
        if self._fn is not None:
            return LLMResponse(self._fn(system, prompt))
        if not self._queue:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return LLMResponse(self._queue.pop(0))


def complete_traced(llm: LLMClient, system: str, prompt: str) -> LLMResponse:
    start = time.perf_counter()
    resp = llm.complete(system, prompt)
    trace = current_trace()
    if trace is not None:
        trace.llm.append(
            LLMLog(
                stage=trace.current_stage,
                prompt_chars=len(system) + len(prompt),
                latency_ms=(time.perf_counter() - start) * 1000,
                input_tokens=resp.input_tokens or (len(system) + len(prompt)) // 4,
                output_tokens=resp.output_tokens or len(resp.text) // 4,
            )
        )
    return resp


def make_llm(backend: str = "none", **kwargs: Any) -> LLMClient:
    if backend in ("", "none"):
        return NullLLM()
    if backend.startswith("python:"):
        _, module, attr = backend.split(":", 2)
        return getattr(importlib.import_module(module), attr)(**kwargs)
    raise ValueError(f"Unknown LLM backend {backend!r}; use 'none' or 'python:module:factory'")
