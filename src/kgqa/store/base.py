from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class QueryError(RuntimeError):
    pass


class QueryTimeout(QueryError):
    pass


@dataclass
class SelectResult:
    variables: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)

    def column(self, var: str | None = None) -> list[Any]:
        var = var or (self.variables[0] if self.variables else None)
        return [r[var] for r in self.rows if r.get(var) is not None]


@runtime_checkable
class Store(Protocol):
    """A SPARQL 1.1 endpoint with named graphs. Values come back as Python values; IRIs as rdf.IRI."""

    def select(self, query: str, timeout: float | None = None) -> SelectResult: ...

    def ask(self, query: str, timeout: float | None = None) -> bool: ...

    def construct(self, query: str, timeout: float | None = None) -> str:
        """Return N-Triples."""
        ...
