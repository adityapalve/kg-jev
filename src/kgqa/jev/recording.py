"""Wrappers that make controller calls observable and replayable.

RecordingController logs every call into the active trace (for calibration and failure
attribution). CachingController memoizes on disk so eval reruns do not pay for identical calls.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kgqa.jev.controller import Controller
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
from kgqa.trace import current_trace


def question_to_json(q: Question) -> dict[str, Any]:
    d = {k: v for k, v in dataclasses.asdict(q).items() if k != "prior" and v is not None}
    d["type"] = {Choice: "choice", Noul: "noul", Score: "score"}[type(q)]
    return d


def answer_to_json(a: Any) -> dict[str, Any]:
    if isinstance(a, ChoiceResult):
        return {"type": "choice", "choice": a.choice, "probabilities": a.probabilities, "confidence": a.confidence}
    if isinstance(a, NoulResult):
        return {"type": "noul", "noul": a.noul}
    if isinstance(a, ScoreResult):
        return {"type": "score", "score": a.score, "probabilities": {str(k): v for k, v in a.probabilities.items()}, "confidence": a.confidence}
    raise TypeError(type(a).__name__)


def answer_from_json(d: dict[str, Any]) -> Any:
    if d["type"] == "choice":
        return ChoiceResult(d["choice"], d["probabilities"], d["confidence"])
    if d["type"] == "noul":
        return NoulResult(d["noul"])
    return ScoreResult(d["score"], {int(k): v for k, v in d["probabilities"].items()}, d["confidence"])


class RecordingController:
    def __init__(self, inner: Controller) -> None:
        self.inner = inner
        self.name = inner.name

    def ask(self, state: Any, questions: Mapping[str, Question], *, stage: str = "") -> JevResponse:
        resp = self.inner.ask(state, questions)
        trace = current_trace()
        if trace is not None:
            trace.record_jev(stage, state, {k: question_to_json(q) for k, q in questions.items()}, resp)
        return resp


class CachingController:
    def __init__(self, inner: Controller, path: str | Path) -> None:
        self.inner = inner
        self.name = inner.name
        self.path = Path(path)
        self._lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row

    def _key(self, state: Any, questions: Mapping[str, Question]) -> str:
        payload = json.dumps(
            {"backend": self.inner.name, "state": state, "questions": {k: question_to_json(q) for k, q in sorted(questions.items())}},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def ask(self, state: Any, questions: Mapping[str, Question]) -> JevResponse:
        key = self._key(state, questions)
        hit = self._cache.get(key)
        if hit is not None:
            return JevResponse(
                answers={k: answer_from_json(v) for k, v in hit["answers"].items()},
                model=hit["model"],
                cached=True,
            )
        resp = self.inner.ask(state, questions)
        row = {"key": key, "model": resp.model, "answers": {k: answer_to_json(a) for k, a in resp.answers.items()}}
        with self._lock:
            self._cache[key] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return resp
