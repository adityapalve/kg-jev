"""Label/alias index over graph entities, with optional embedding lookup.

In-memory and fine for slices up to a few million labels. For full Wikidata, implement the same
`lookup_exact` / `lookup_fuzzy` methods on top of a search service and pass that to EntityLinker.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from kgqa.config import ModuleConfig
from kgqa.rdf import IRI, local_name
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
    popularity: int = 0  # how many triples point at this entity; ranks otherwise-equal candidates


# Alias kinds and how much a match on each is trusted. Weaker aliases make the linker ask jev.
PRIMARY, ALT, STRIPPED, DEMONYM, SURNAME = 1.0, 0.92, 0.85, 0.8, 0.6
PERSON_TYPES = {"Person", "Human", "Employee"}
DEMONYMS = {
    "Afghanistan": "Afghan", "Argentina": "Argentine Argentinian", "Australia": "Australian", "Austria": "Austrian",
    "Belgium": "Belgian", "Brazil": "Brazilian", "Canada": "Canadian", "Chile": "Chilean", "China": "Chinese",
    "Colombia": "Colombian", "Cuba": "Cuban", "Czech Republic": "Czech", "Czechoslovakia": "Czechoslovak",
    "Denmark": "Danish", "Egypt": "Egyptian", "Finland": "Finnish", "France": "French", "Germany": "German",
    "Greece": "Greek", "Hong Kong": "Hong Kong", "Hungary": "Hungarian", "Iceland": "Icelandic", "India": "Indian",
    "Indonesia": "Indonesian", "Iran": "Iranian", "Ireland": "Irish", "Israel": "Israeli", "Italy": "Italian",
    "Japan": "Japanese", "Mexico": "Mexican", "Netherlands": "Dutch", "New Zealand": "New Zealand",
    "Nigeria": "Nigerian", "Norway": "Norwegian", "Pakistan": "Pakistani", "Peru": "Peruvian",
    "Philippines": "Filipino Philippine", "Poland": "Polish", "Portugal": "Portuguese", "Romania": "Romanian",
    "Russia": "Russian", "Soviet Union": "Soviet", "South Africa": "South African", "South Korea": "South Korean Korean",
    "Spain": "Spanish", "Sweden": "Swedish", "Switzerland": "Swiss", "Taiwan": "Taiwanese", "Thailand": "Thai",
    "Turkey": "Turkish", "Ukraine": "Ukrainian", "United Kingdom": "British English Scottish Welsh",
    "United States": "American US", "Venezuela": "Venezuelan", "Vietnam": "Vietnamese",
}
_DEMONYM_ALIASES = {normalize(country): [normalize(d) for d in ds.split()] for country, ds in DEMONYMS.items()}
_STRIP = re.compile(r"\s*\([^)]*\)\s*$")


class EntityIndex:
    def __init__(self, records: dict[str, EntityRecord] | None = None, embedder: Embedder | None = None) -> None:
        self.records: dict[str, EntityRecord] = records or {}
        self.embedder = embedder
        self._by_norm: dict[str, list[tuple[str, float]]] = defaultdict(list)  # norm alias -> [(iri, alias score)]
        self._by_trigram: dict[str, set[str]] = defaultdict(set)
        self._vectors: list[list[float]] | None = None
        self._vector_keys: list[tuple[str, str]] = []
        self._reindex()

    def _aliases(self, r: EntityRecord) -> list[tuple[str, float]]:
        out = [(r.label, PRIMARY), *((l, ALT) for l in r.labels)]
        out += [(base, STRIPPED) for n, _ in list(out) if (base := _STRIP.sub("", n)) != n and base]  # "Titanic (1997 film)"
        if r.value_of is None:
            for n, _ in list(out):
                out += [(d, DEMONYM) for d in _DEMONYM_ALIASES.get(normalize(n), [])]  # "Danish" -> Denmark
            if any(local_name(t) in PERSON_TYPES for t in r.types):
                words = _STRIP.sub("", r.label).split()
                if 2 <= len(words) <= 4 and words[-1][:1].isupper() and words[-1].isalpha() and len(words[-1]) >= 4:
                    out.append((words[-1], SURNAME))  # "Kurosawa" -> Akira Kurosawa
        return out

    def _reindex(self) -> None:
        self._by_norm.clear()
        self._by_trigram.clear()
        for r in self.records.values():
            best: dict[str, float] = {}
            for name, score in self._aliases(r):
                n = normalize(name)
                if n and score > best.get(n, 0.0):
                    best[n] = score
            for n, score in best.items():
                self._by_norm[n].append((r.iri, score))
                if score >= STRIPPED:  # fuzzy matching only over real names, not surnames/demonyms
                    for t in trigrams(n):
                        self._by_trigram[t].add(n)
        if self.embedder is not None:
            self._vector_keys = [(n, iri) for n, entries in self._by_norm.items() for iri, _ in entries]
            self._vectors = self.embedder.embed([n for n, _ in self._vector_keys]) if self._vector_keys else []

    def _rank(self, rec: EntityRecord, score: float) -> float:
        # popularity breaks ties between otherwise equal matches (many people are called Taylor)
        return score + 0.04 * math.log10(1 + rec.popularity)

    @property
    def max_label_tokens(self) -> int:
        return max((len(n.split()) for n in self._by_norm), default=1)

    def lookup_exact(self, norm: str) -> list[tuple[EntityRecord, float]]:
        hits = [(self.records[iri], self._rank(self.records[iri], score)) for iri, score in self._by_norm.get(norm, [])]
        return sorted(hits, key=lambda h: -h[1])

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
        graphs = " ".join(f"<{m.graph}>" for m in modules)
        q = f"SELECT ?o (COUNT(*) AS ?n) WHERE {{ VALUES ?g {{ {graphs} }} GRAPH ?g {{ ?s ?p ?o FILTER(isIRI(?o)) }} }} GROUP BY ?o"
        for row in store.select(q).rows:
            if row["o"] in records:
                records[row["o"]].popularity = row["n"]
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
