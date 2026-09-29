"""System One (jev) controller interface, backends and helpers."""

from __future__ import annotations

import importlib
from typing import Any

from kgqa.jev.controller import Controller
from kgqa.jev.heuristic import HeuristicController
from kgqa.jev.questions import (
    Choice,
    ChoiceResult,
    JevResponse,
    Noul,
    NoulResult,
    Score,
    ScoreResult,
    certain,
)
from kgqa.jev.recording import CachingController, RecordingController

__all__ = [
    "CachingController",
    "Choice",
    "ChoiceResult",
    "Controller",
    "HeuristicController",
    "JevResponse",
    "Noul",
    "NoulResult",
    "RecordingController",
    "Score",
    "ScoreResult",
    "certain",
    "make_controller",
]


def make_controller(backend: str = "heuristic", *, model: str | None = None, timeout: float | None = None, cache: str | None = None, **kwargs: Any) -> RecordingController:
    """Build the configured controller, wrapped for caching (optional) and trace recording.

    backend: "heuristic" (offline), "typesafe" (real jev), or "python:module:factory" for a custom
    factory returning an object with `name` and `ask(state, questions) -> JevResponse`.
    """
    inner: Controller
    if backend == "heuristic":
        inner = HeuristicController(**kwargs)
    elif backend == "typesafe":
        from kgqa.jev.typesafe import TypeSafeController

        inner = TypeSafeController(model=model, timeout=timeout)
    elif backend.startswith("python:"):
        _, module, attr = backend.split(":", 2)
        inner = getattr(importlib.import_module(module), attr)(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown controller backend {backend!r}")
    if cache:
        inner = CachingController(inner, cache)
    return RecordingController(inner)
