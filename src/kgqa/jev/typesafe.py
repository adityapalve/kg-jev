"""TypeSafe `system_one` backend (the real jev controller).

Requires `pip install typesafe-sdk` and TYPESAFE_API_KEY. One client is kept open for the
controller's lifetime so connection reuse keeps per-call latency down.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from kgqa.jev.questions import (
    Choice,
    ChoiceResult,
    JevResponse,
    Noul,
    NoulResult,
    Question,
    Score,
    ScoreResult,
)


OPENROUTER_BASE_URL = "https://openrouter.ai/api"


def to_sdk_question(q: Question) -> Any:
    import typesafe_sdk as ts

    if isinstance(q, Choice):
        return ts.Choice(criteria=dict(q.criteria), instructions=q.instructions)
    if isinstance(q, Noul):
        criteria = dict(q.criteria) if q.criteria else None
        return ts.Noul(instructions=q.instructions, criteria=criteria)
    if isinstance(q, Score):
        return ts.Score(criteria=list(q.criteria), instructions=q.instructions)
    raise TypeError(type(q).__name__)


def from_sdk_answer(a: Any) -> Any:
    kind = getattr(a, "type", None)
    if kind == "choice":
        return ChoiceResult(a.choice, dict(a.probabilities), float(a.confidence))
    if kind == "noul":
        return NoulResult(float(a.noul))
    if kind == "score":
        return ScoreResult(float(a.score), {int(k): float(v) for k, v in a.probabilities.items()}, float(a.confidence))
    raise TypeError(f"Unknown answer type {kind!r}")


class TypeSafeController:
    name = "typesafe"

    def __init__(self, model: str | None = None, timeout: float | None = None, client: Any = None, **client_kwargs: Any) -> None:
        if client is None:
            from typesafe_sdk import TypeSafeClient

            client = TypeSafeClient(model=model, timeout=timeout, **client_kwargs)
        self.client = client
        self.model = model

    def ask(self, state: Any, questions: Mapping[str, Question]) -> JevResponse:
        start = time.perf_counter()
        resp = self.client.system_one(state=state, questions={k: to_sdk_question(q) for k, q in questions.items()})
        latency = (time.perf_counter() - start) * 1000
        answers = {name: from_sdk_answer(a) for name, a in resp.answers.items()}
        missing = set(questions) - set(answers)
        if missing:
            raise RuntimeError(f"jev returned no answer for: {sorted(missing)}")
        usage = resp.usage
        return JevResponse(
            answers=answers,
            model=resp.model,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            latency_ms=latency,
        )

    def close(self) -> None:
        self.client.close()
