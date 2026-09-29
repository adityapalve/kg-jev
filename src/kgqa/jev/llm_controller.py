"""A chat LLM standing in for jev: same questions, same answer types.

Used to benchmark the controller itself: the pipeline, templates and schema stay the same and only
the decision-maker changes. An LLM has no native probabilities, so it is asked for a choice plus a
stated confidence; the rest of the probability mass is spread over the other options. Stated
confidences are known to be poorly calibrated, which the eval's calibration table will show.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from typing import Any

from kgqa.jev.questions import Choice, ChoiceResult, JevResponse, Noul, NoulResult, Question, Score, ScoreResult
from kgqa.llm import LLMClient, complete_traced

SYSTEM = """You are the decision component of a question-answering system over a knowledge graph.
You receive a state (the user's question and context) and several named questions.
For a "choice" question, pick exactly one option key from its criteria, and say how confident you are (0 to 1).
For a "yes/no" question, give the probability that the answer is yes (0 to 1).
Answer every question. Return only JSON, no prose, in this form:
{"question_name": {"choice": "<option key>", "confidence": 0.9}, "other_name": {"yes": 0.8}}"""


def _render(questions: Mapping[str, Question]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, q in questions.items():
        if isinstance(q, Choice):
            out[name] = {"type": "choice", "instructions": q.instructions, "options": dict(q.criteria)}
        elif isinstance(q, Noul):
            out[name] = {"type": "yes/no", "instructions": q.instructions, "criteria": q.criteria}
        elif isinstance(q, Score):
            out[name] = {"type": "choice", "instructions": q.instructions, "options": {str(i): c for i, c in enumerate(q.criteria)}}
    return out


def _parse(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _clamp(x: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(x)))
    except (TypeError, ValueError):
        return default


class LLMController:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.name = f"llm:{llm.name}"

    def ask(self, state: Any, questions: Mapping[str, Question]) -> JevResponse:
        start = time.perf_counter()
        prompt = json.dumps({"state": state, "questions": _render(questions)}, ensure_ascii=False, default=str)
        resp = complete_traced(self.llm, SYSTEM, prompt)
        raw = _parse(resp.text)
        answers: dict[str, Any] = {}
        for name, q in questions.items():
            got = raw.get(name) if isinstance(raw.get(name), dict) else {}
            if isinstance(q, Noul):
                answers[name] = NoulResult(_clamp(got.get("yes"), 0.5))
                continue
            keys = [str(i) for i in range(len(q.criteria))] if isinstance(q, Score) else list(q.criteria)
            choice = str(got.get("choice", ""))
            if choice not in keys:
                # an unusable answer counts as no confidence, so the gates send it to fallback
                probs = {k: 1.0 / len(keys) for k in keys}
                result = ChoiceResult(keys[0], probs, 0.0)
            else:
                conf = _clamp(got.get("confidence"), 0.5)
                rest = (1 - conf) / (len(keys) - 1) if len(keys) > 1 else 0.0
                probs = {k: (conf if k == choice else rest) for k in keys}
                result = ChoiceResult(choice, probs, conf)
            if isinstance(q, Score):
                answers[name] = ScoreResult(float(result.choice), {int(k): v for k, v in result.probabilities.items()}, result.confidence)
            else:
                answers[name] = result
        return JevResponse(
            answers=answers,
            model=self.name,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
