"""Question and answer types for the System One controller.

These mirror the TypeSafe `system_one` primitives (Choice, Noul, Score) so pipeline code does not
depend on the SDK. `prior` is never sent to the API: it carries a code-computed hint that only the
offline heuristic backend uses, so the pipeline still runs end to end without an API key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

MAX_CHOICE_OPTIONS = 255


@dataclass
class Choice:
    criteria: Mapping[str, Any]
    instructions: Any = None
    prior: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("Choice needs at least one option")
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"Choice has {len(self.criteria)} options; the limit is {MAX_CHOICE_OPTIONS}. Use jev.large.choose_large.")


@dataclass
class Noul:
    instructions: Any = None
    criteria: Mapping[str, Any] | None = None  # {"true": ..., "false": ...}
    prior: float | None = None


@dataclass
class Score:
    criteria: Sequence[Any]
    instructions: Any = None
    prior: float | None = None


Question = Choice | Noul | Score


@dataclass(frozen=True)
class ChoiceResult:
    choice: str
    probabilities: dict[str, float]
    confidence: float

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])

    def runners_up(self, min_prob: float) -> list[str]:
        return [k for k, p in self.ranked() if k != self.choice and p >= min_prob]


@dataclass(frozen=True)
class NoulResult:
    noul: float

    @property
    def yes(self) -> bool:
        return self.noul >= 0.5

    @property
    def confidence(self) -> float:
        return max(self.noul, 1 - self.noul)


@dataclass(frozen=True)
class ScoreResult:
    score: float
    probabilities: dict[int, float]
    confidence: float


Answer = ChoiceResult | NoulResult | ScoreResult


@dataclass
class JevResponse:
    answers: dict[str, Answer]
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def choice(self, name: str) -> ChoiceResult:
        a = self.answers[name]
        assert isinstance(a, ChoiceResult), name
        return a

    def noul(self, name: str) -> NoulResult:
        a = self.answers[name]
        assert isinstance(a, NoulResult), name
        return a


def certain(option: str) -> ChoiceResult:
    """A ChoiceResult for a decision code made without asking (a single candidate)."""
    return ChoiceResult(option, {option: 1.0}, 1.0)
