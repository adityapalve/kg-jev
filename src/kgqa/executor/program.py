"""A program is a short sequence of small SPARQL queries.

Each step may reference earlier steps' outputs with `{{name}}` placeholders, which the executor
fills with a bounded, chunked VALUES list. `post` says how the last step becomes an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryStep:
    name: str
    sparql: str
    kind: str = "select"  # "select" | "ask"
    var: str | None = None  # output column carried forward
    inputs: list[str] = field(default_factory=list)
    chunkable: bool = False  # safe to split inputs across several queries and union the outputs
    module: str | None = None
    purpose: str = ""


@dataclass
class Program:
    steps: list[QueryStep]
    bindings: dict[str, list[Any]] = field(default_factory=dict)  # initial bindings (anchor IRIs)
    post: str = "column"  # "column" | "scalar" | "ask" | "extreme" | "compare_max" | "compare_min" | "table"
    answer_var: str = "answer"
    measure_var: str = "m"
    source: str = "template"  # "template" | "llm" | "fallback"

    def sparql(self) -> list[str]:
        return [s.sparql for s in self.steps]
