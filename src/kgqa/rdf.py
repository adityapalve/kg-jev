"""RDF term helpers: IRIs as a distinct string type, literal conversion, SPARQL serialization."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

XSD = "http://www.w3.org/2001/XMLSchema#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
SH = "http://www.w3.org/ns/shacl#"
SKOS = "http://www.w3.org/2004/02/skos/core#"
SCHEMA = "http://schema.org/"
KGQA = "https://kgqa.dev/ns#"

RDF_TYPE = RDF + "type"
RDFS_LABEL = RDFS + "label"

STANDARD_PREFIXES = {
    "rdf": RDF,
    "rdfs": RDFS,
    "xsd": XSD,
    "owl": OWL,
    "sh": SH,
    "skos": SKOS,
    "schema": SCHEMA,
    "kgqa": KGQA,
}

NUMERIC_TYPES = {XSD + t for t in ("integer", "int", "long", "short", "decimal", "double", "float", "nonNegativeInteger", "positiveInteger")}
DATE_TYPES = {XSD + "date", XSD + "dateTime", XSD + "gYear", XSD + "gYearMonth"}


class IRI(str):
    """A string that is an IRI rather than a plain literal."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"IRI({str.__repr__(self)})"


def literal_to_python(value: str, datatype: str | None, lang: str | None = None) -> Any:
    if datatype is None or lang:
        return value
    try:
        if datatype in (XSD + "integer", XSD + "int", XSD + "long", XSD + "short", XSD + "nonNegativeInteger", XSD + "positiveInteger", XSD + "gYear"):
            return int(value)
        if datatype in (XSD + "decimal", XSD + "double", XSD + "float"):
            return float(value)
        if datatype == XSD + "boolean":
            return value in ("true", "1")
        if datatype == XSD + "date":
            return dt.date.fromisoformat(value[:10])
        if datatype == XSD + "dateTime":
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return value


def sparql_term(value: Any) -> str:
    """Serialize a Python value as a SPARQL term."""
    if isinstance(value, IRI):
        return f"<{value}>"
    if isinstance(value, bool):
        return f'"{str(value).lower()}"^^<{XSD}boolean>'
    if isinstance(value, int):
        return f'"{value}"^^<{XSD}integer>'
    if isinstance(value, (float, Decimal)):
        return f'"{value}"^^<{XSD}decimal>'
    if isinstance(value, dt.datetime):
        return f'"{value.isoformat()}"^^<{XSD}dateTime>'
    if isinstance(value, dt.date):
        return f'"{value.isoformat()}"^^<{XSD}date>'
    s = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{s}"'


def local_name(iri: str) -> str:
    for sep in ("#", "/", ":"):
        if sep in iri:
            tail = iri.rsplit(sep, 1)[1]
            if tail:
                return tail
    return iri


class Prefixes:
    def __init__(self, extra: dict[str, str] | None = None) -> None:
        self.map = {**STANDARD_PREFIXES, **(extra or {})}

    def compact(self, iri: str) -> str:
        best = None
        for p, ns in self.map.items():
            if iri.startswith(ns) and (best is None or len(ns) > len(self.map[best])):
                best = p
        if best is None:
            return f"<{iri}>"
        return f"{best}:{iri[len(self.map[best]):]}"

    def expand(self, curie: str) -> str:
        if curie.startswith("<") and curie.endswith(">"):
            return curie[1:-1]
        p, _, rest = curie.partition(":")
        return self.map[p] + rest if p in self.map else curie

    def header(self) -> str:
        return "\n".join(f"PREFIX {p}: <{ns}>" for p, ns in self.map.items())


def is_numeric_type(datatype: str | None) -> bool:
    return datatype in NUMERIC_TYPES


def is_date_type(datatype: str | None) -> bool:
    return datatype in DATE_TYPES
