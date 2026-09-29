"""Per-question trace: stage timings, every jev call, every query, every gate decision.

The eval harness reads traces to compute per-stage latency, cost, calibration and failure
attribution, so anything a metric needs must be recorded here.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

_current: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("kgqa_trace", default=None)


def current_trace() -> Trace | None:
    return _current.get()


@dataclass
class Gate:
    name: str
    choice: str
    confidence: float
    threshold: float
    passed: bool
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class QueryLog:
    stage: str
    sparql: str
    graph: str | None
    latency_ms: float
    rows: int
    error: str | None = None
    timeout: bool = False
    truncated: bool = False


@dataclass
class JevLog:
    stage: str
    state: Any
    questions: dict[str, Any]
    answers: dict[str, Any]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cached: bool


@dataclass
class LLMLog:
    stage: str
    prompt_chars: int
    latency_ms: float
    input_tokens: int
    output_tokens: int


@dataclass
class Trace:
    question: str
    stages: dict[str, float] = field(default_factory=dict)
    jev: list[JevLog] = field(default_factory=list)
    queries: list[QueryLog] = field(default_factory=list)
    llm: list[LLMLog] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    _stage: str = ""

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        prev, self._stage = self._stage, name
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start) * 1000
            self._stage = prev

    @property
    def current_stage(self) -> str:
        return self._stage

    def record_jev(self, stage: str, state: Any, questions: dict[str, Any], resp: Any) -> None:
        from kgqa.jev.recording import answer_to_json

        self.jev.append(
            JevLog(
                stage=stage or self._stage,
                state=state,
                questions=questions,
                answers={k: answer_to_json(a) for k, a in resp.answers.items()},
                latency_ms=resp.latency_ms,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cached=resp.cached,
            )
        )

    def gate(self, name: str, choice: str, confidence: float, threshold: float, probabilities: dict[str, float] | None = None, *, passed: bool | None = None) -> bool:
        ok = confidence >= threshold if passed is None else passed
        self.gates.append(Gate(name, choice, confidence, threshold, ok, dict(probabilities or {})))
        return ok

    def event(self, kind: str, **data: Any) -> None:
        self.events.append({"kind": kind, "stage": self._stage, **data})

    @property
    def total_ms(self) -> float:
        return sum(v for k, v in self.stages.items() if "/" not in k)

    @property
    def jev_tokens(self) -> int:
        return sum(j.input_tokens + j.output_tokens for j in self.jev)

    @property
    def llm_tokens(self) -> int:
        return sum(l.input_tokens + l.output_tokens for l in self.llm)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_stage", None)
        d["total_ms"] = self.total_ms
        return d


@contextmanager
def tracing(question: str) -> Iterator[Trace]:
    trace = Trace(question)
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)
