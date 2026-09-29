"""Label/alias index over graph entities, with optional embedding lookup.

In-memory and fine for slices up to a few million labels. For full Wikidata, implement the same
`lookup_exact` / `lookup_fuzzy` methods on top of a search service and pass that to EntityLinker.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from kgqa.config import ModuleConfig
from kgqa.rdf import IRI
from kgqa.store.base import Store
from kgqa.text import normalize, trigram_similarity, trigrams


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class EntityRecord:
    iri: str
    label: str
    labels: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    module: str = ""
    value_of: str | None = None  # set for literal values (e.g. a city name); iri is then synthetic


class EntityIndex:
    def __init__(self, records: dict[str, EntityRecord] | None = None, embedder: Embedder | None = None) -> None:
        self.records: dict[str, EntityRecord] = records or {}
        self.embedder = embedder
        self._by_norm: dict[str, list[tuple[str, bool]]] = defaultdict(list)  # norm label -> [(iri, is_primary)]
        self._by_trigram: dict[str, set[str]] = defaultdict(set)
        self._vectors: list[list[float]] | None = None
        self._vector_keys: list[tuple[str, str]] = []
        self._reindex()

    def _reindex(self) -> None:
        self._by_norm.clear()
        self._by_trigram.clear()
        for r in self.records.values():
            for lbl in dict.fromkeys([r.label, *r.labels]):
                n = normalize(lbl)
                if not n:
                    continue
                self._by_norm[n].append((r.iri, lbl == r.label))
                for t in trigrams(lbl):
                    self._by_trigram[t].add(n)
        if self.embedder is not None:
            self._vector_keys = [(n, iri) for n, entries in self._by_norm.items() for iri, _ in entries]
            self._vectors = self.embedder.embed([n for n, _ in self._vector_keys]) if self._vector_keys else []

    @property
    def max_label_tokens(self) -> int:
        return max((len(n.split()) for n in self._by_norm), default=1)

    def lookup_exact(self, norm: str) -> list[tuple[EntityRecord, float]]:
        return [(self.records[iri], 1.0 if primary else 0.92) for iri, primary in self._by_norm.get(norm, [])]

    def lookup_fuzzy(self, text: str, min_sim: float = 0.72, k: int = 5) -> list[tuple[EntityRecord, float]]:
        cands: dict[str, int] = defaultdict(int)
        for t in trigrams(text):
            for n in self._by_trigram.get(t, ()):
                cands[n] += 1
        top = sorted(cands, key=lambda n: -cands[n])[:50]
        scored = sorted(((n, trigram_similarity(text, n)) for n in top), key=lambda x: -x[1])
        out: list[tuple[EntityRecord, float]] = []
        for n, sim in scored:
            if sim < min_sim:
                break
            for iri, _ in self._by_norm[n]:
                out.append((self.records[iri], sim * 0.9))
        return out[:k]

    def lookup_vector(self, text: str, k: int = 5, min_cos: float = 0.8) -> list[tuple[EntityRecord, float]]:
        if self.embedder is None or not self._vectors:
            return []
        (q,) = self.embedder.embed([text])
        qn = math.sqrt(sum(x * x for x in q)) or 1.0
        scored = []
        for (n, iri), v in zip(self._vector_keys, self._vectors):
            vn = math.sqrt(sum(x * x for x in v)) or 1.0
            scored.append((sum(a * b for a, b in zip(q, v)) / (qn * vn), iri))
        scored.sort(reverse=True)
        return [(self.records[iri], cos * 0.85) for cos, iri in scored[:k] if cos >= min_cos]

    # ---- build / persist ----
    @classmethod
    def build(cls, store: Store, modules: list[ModuleConfig], embedder: Embedder | None = None, max_values: int = 200) -> EntityIndex:
        """Index entity labels, plus the distinct string values of properties with at most
        `max_values` of them (cities, countries, roles, status codes), which become equality filters."""
        records: dict[str, EntityRecord] = {}
        for m in modules:
            q = f"""SELECT ?p (COUNT(DISTINCT ?o) AS ?n) WHERE {{ GRAPH <{m.graph}> {{ ?s ?p ?o
                    FILTER(isLiteral(?o) && (datatype(?o) = <http://www.w3.org/2001/XMLSchema#string> || lang(?o) != "")) }} }} GROUP BY ?p"""
            for row in store.select(q).rows:
                prop, n = row["p"], row["n"]
                if prop in m.label_predicates or n > max_values:
                    continue
                for v in store.select(f"SELECT DISTINCT ?o WHERE {{ GRAPH <{m.graph}> {{ ?s <{prop}> ?o }} }}").column("o"):
                    key = f"value:{prop}#{v}"
                    records[key] = EntityRecord(iri=key, label=str(v), module=m.name, value_of=prop)
        for m in modules:
            for lp in m.label_predicates:
                q = f"""SELECT ?s ?l ?t WHERE {{ GRAPH <{m.graph}> {{ ?s <{lp}> ?l . OPTIONAL {{ ?s <{m.type_predicate}> ?t }} }}
                        FILTER(isIRI(?s) && (lang(?l) = "" || langMatches(lang(?l), "en"))) }}"""
                for row in store.select(q).rows:
                    s, label, t = row["s"], str(row["l"]), row.get("t")
                    rec = records.get(s)
                    if rec is None:  # label predicates are in priority order, so the first label seen is primary
                        rec = records[s] = EntityRecord(iri=s, label=label, module=m.name)
                    elif label != rec.label and label not in rec.labels:
                        rec.labels.append(label)
                    if t and t not in rec.types:
                        rec.types.append(t)
        return cls(records, embedder)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(r) for r in self.records.values()]))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> EntityIndex:
        rows = json.loads(Path(path).read_text())
        records = {}
        for r in rows:
            rec = EntityRecord(**r)
            rec.iri = IRI(rec.iri)
            records[rec.iri] = rec
        return cls(records, embedder)
