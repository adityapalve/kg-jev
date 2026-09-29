"""Offline stand-in for jev.

Scores Choice options by idf-weighted lexical overlap between the question and each option's
label and description, then softmaxes. Noul and Score fall back to the code-supplied `prior`.
It is deliberately simple: it lets the whole pipeline, tests and eval harness run without an API
key, and gives a floor that the real controller should beat. Its confidences are not calibrated.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Mapping
from typing import Any

from kgqa import text
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

# Options that mean "none of the above" get a flat score instead of lexical evidence.
ABSTAIN_KEYS = frozenset({"other", "none", "none_of_the_above", "unknown"})


def _state_text(state: Any) -> str:
    if isinstance(state, dict):
        parts = [str(state.get("question", ""))]
        focus = state.get("focus")
        if focus:
            parts.append(text.flatten_json(focus))
        return " ".join(parts)
    return text.flatten_json(state)


class HeuristicController:
    name = "heuristic"

    def __init__(self, temperature: float = 0.6, abstain_score: float = 0.35, prior_weight: float = 2.0) -> None:
        self.temperature = temperature
        self.abstain_score = abstain_score
        self.prior_weight = prior_weight

    def ask(self, state: Any, questions: Mapping[str, Question]) -> JevResponse:
        start = time.perf_counter()
        q_text = _state_text(state)
        q_stems = text.stems(q_text)
        q_bigrams = set(text.bigrams(q_stems))
        answers: dict[str, Any] = {}
        for name, question in questions.items():
            if isinstance(question, Choice):
                answers[name] = self._choice(question, q_stems, q_bigrams)
            elif isinstance(question, Noul):
                answers[name] = NoulResult(question.prior if question.prior is not None else 0.5)
            elif isinstance(question, Score):
                answers[name] = self._score(question)
            else:
                raise TypeError(f"Unsupported question type for {name}: {type(question).__name__}")
        return JevResponse(answers=answers, model=self.name, latency_ms=(time.perf_counter() - start) * 1000)

    def _choice(self, q: Choice, q_stems: list[str], q_bigrams: set[tuple[str, str]]) -> ChoiceResult:
        keys = list(q.criteria)
        docs = [text.stems(f"{k} {text.flatten_json(v)}") for k, v in q.criteria.items()]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        q_set = set(q_stems)
        scores = []
        for key, doc in zip(keys, docs):
            if key.lower() in ABSTAIN_KEYS:
                scores.append(self.abstain_score)
                continue
            doc_set = set(doc)
            s = sum(math.log((n + 1) / (df[t] + 0.5)) for t in q_set & doc_set)
            s += 0.75 * len(q_bigrams & set(text.bigrams(doc)))
            # Mild length normalization so long descriptions do not win by volume.
            s /= math.sqrt(1 + 0.05 * len(doc_set))
            scores.append(s)
        evidence = [s for k, s in zip(keys, scores) if k.lower() not in ABSTAIN_KEYS]
        if evidence and max(evidence) < 0.05:  # nothing matched: prefer an abstain option if one exists
            scores = [2.0 if k.lower() in ABSTAIN_KEYS else s for k, s in zip(keys, scores)]
        if q.prior:
            scores = [s + self.prior_weight * math.log(max(q.prior.get(k, 1e-3), 1e-3)) for k, s in zip(keys, scores)]
        probs = text.softmax(scores, self.temperature)
        pmap = dict(zip(keys, probs))
        best = max(pmap, key=pmap.__getitem__)
        return ChoiceResult(best, pmap, pmap[best])

    def _score(self, q: Score) -> ScoreResult:
        levels = len(q.criteria)
        value = q.prior if q.prior is not None else (levels - 1) / 2
        probs = {i: 1.0 if i == round(value) else 0.0 for i in range(levels)}
        return ScoreResult(float(value), probs, 0.5)
