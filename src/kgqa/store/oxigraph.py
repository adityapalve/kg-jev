"""Embedded Oxigraph store (pyoxigraph). In-memory or on-disk; named graph per module."""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path
from typing import Any

import pyoxigraph as ox

from kgqa.rdf import IRI, literal_to_python
from kgqa.store.base import QueryError, QueryTimeout, SelectResult

_FORMATS = {
    ".ttl": ox.RdfFormat.TURTLE,
    ".nt": ox.RdfFormat.N_TRIPLES,
    ".nq": ox.RdfFormat.N_QUADS,
    ".trig": ox.RdfFormat.TRIG,
    ".rdf": ox.RdfFormat.RDF_XML,
    ".owl": ox.RdfFormat.RDF_XML,
}

# Oxigraph has no per-query cancellation; a timed-out query is abandoned on a worker thread.
_POOL = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="oxigraph")


def term_to_python(term: Any) -> Any:
    if term is None:
        return None
    if isinstance(term, ox.NamedNode):
        return IRI(term.value)
    if isinstance(term, ox.Literal):
        return literal_to_python(term.value, term.datatype.value if term.datatype else None, term.language)
    if isinstance(term, ox.BlankNode):
        return f"_:{term.value}"
    return str(term)


class OxigraphStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
        self.store = ox.Store(str(self.path)) if self.path else ox.Store()

    def load_file(self, file: str | Path, graph: str | None) -> None:
        file = Path(file)
        fmt = _FORMATS.get(file.suffix.lower())
        if fmt is None:
            raise ValueError(f"Unsupported RDF file type: {file}")
        to_graph = ox.NamedNode(graph) if graph else ox.DefaultGraph()
        if fmt in (ox.RdfFormat.N_QUADS, ox.RdfFormat.TRIG):
            self.store.bulk_load(path=str(file), format=fmt)
        else:
            self.store.bulk_load(path=str(file), format=fmt, to_graph=to_graph)

    def clear_graph(self, graph: str) -> None:
        self.store.remove_graph(ox.NamedNode(graph))

    def _run(self, fn: Any, timeout: float | None) -> Any:
        # Results are materialized on the worker: pyoxigraph result iterators are thread-bound.
        fut = _POOL.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except cf.TimeoutError as e:
            raise QueryTimeout(f"Query exceeded {timeout}s") from e
        except SyntaxError as e:
            raise QueryError(f"SPARQL syntax error: {e}") from e
        except OSError as e:
            raise QueryError(str(e)) from e

    def select(self, query: str, timeout: float | None = None) -> SelectResult:
        def run() -> SelectResult:
            res = self.store.query(query)
            if not isinstance(res, ox.QuerySolutions):
                raise QueryError("Expected a SELECT query")
            variables = [v.value for v in res.variables]
            return SelectResult(variables, [{v: term_to_python(sol[v]) for v in variables} for sol in res])

        return self._run(run, timeout)

    def ask(self, query: str, timeout: float | None = None) -> bool:
        def run() -> bool:
            res = self.store.query(query)
            if not isinstance(res, ox.QueryBoolean):
                raise QueryError("Expected an ASK query")
            return bool(res)

        return self._run(run, timeout)

    def construct(self, query: str, timeout: float | None = None) -> str:
        return self._run(lambda: ox.serialize(self.store.query(query), format=ox.RdfFormat.N_TRIPLES).decode(), timeout)


def check_syntax(query: str) -> str | None:
    """Return an error message if the query does not parse, else None (runs on an empty store)."""
    try:
        ox.Store().query(query)
    except SyntaxError as e:
        return str(e)
    except Exception as e:  # evaluation errors on an empty store are not syntax errors
        return None if "parse" not in str(e).lower() else str(e)
    return None
