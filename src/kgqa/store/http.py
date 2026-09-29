"""SPARQL 1.1 Protocol client for remote endpoints (QLever, Oxigraph server, qEndpoint, WDQS)."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from kgqa.rdf import IRI, literal_to_python
from kgqa.store.base import QueryError, QueryTimeout, SelectResult


def binding_to_python(b: dict[str, Any] | None) -> Any:
    if b is None:
        return None
    kind = b.get("type")
    if kind == "uri":
        return IRI(b["value"])
    if kind in ("literal", "typed-literal"):
        return literal_to_python(b["value"], b.get("datatype"), b.get("xml:lang"))
    if kind == "bnode":
        return f"_:{b['value']}"
    return b.get("value")


class HttpSparqlStore:
    def __init__(self, endpoint: str, *, user_agent: str = "kgqa/0.1 (research prototype)", default_timeout: float = 60.0, headers: dict[str, str] | None = None) -> None:
        self.endpoint = endpoint
        self.default_timeout = default_timeout
        self.headers = {"User-Agent": user_agent, **(headers or {})}

    def _post(self, query: str, accept: str, timeout: float | None) -> bytes:
        data = urllib.parse.urlencode({"query": query}).encode()
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={**self.headers, "Accept": accept, "Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.default_timeout) as resp:
                return resp.read()
        except (socket.timeout, TimeoutError) as e:
            raise QueryTimeout(str(e)) from e
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if e.code in (408, 504) or "timeout" in body.lower():
                raise QueryTimeout(body) from e
            raise QueryError(f"HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise QueryError(str(e)) from e

    def select(self, query: str, timeout: float | None = None) -> SelectResult:
        raw = json.loads(self._post(query, "application/sparql-results+json", timeout))
        variables = raw.get("head", {}).get("vars", [])
        rows = [{v: binding_to_python(b.get(v)) for v in variables} for b in raw.get("results", {}).get("bindings", [])]
        return SelectResult(variables, rows)

    def ask(self, query: str, timeout: float | None = None) -> bool:
        raw = json.loads(self._post(query, "application/sparql-results+json", timeout))
        return bool(raw.get("boolean"))

    def construct(self, query: str, timeout: float | None = None) -> str:
        return self._post(query, "application/n-triples", timeout).decode()
