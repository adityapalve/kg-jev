"""Controller protocol: one call, a state plus named typed questions, typed answers back."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from kgqa.jev.questions import JevResponse, Question


@runtime_checkable
class Controller(Protocol):
    name: str

    def ask(self, state: Any, questions: Mapping[str, Question]) -> JevResponse:
        """Evaluate every question against `state` in a single call."""
        ...
