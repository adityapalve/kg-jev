"""Schema-level paths between classes.

Code enumerates the *valid* ways to get from one class to another (or to an attribute) using the
schema cards; jev only chooses among them. A path is a sequence of edges; an edge is one or more
primitive steps, each scoped to the module (named graph) that holds its triples. Shared-identifier
joins are edges with two steps in two different modules, which is what makes cross-graph questions
executable as a sequence of small queries.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from kgqa.rdf import local_name
from kgqa.schema.cards import PropertyCard, SchemaCatalog


@dataclass(frozen=True)
class Step:
    prop: str
    inverse: bool
    module: str


@dataclass(frozen=True)
class Edge:
    steps: tuple[Step, ...]
    src: str
    dst: str | None  # class IRI, or None when the edge ends in a literal
    kind: str  # "object" | "inverse" | "join" | "attribute"
    prop: str  # the property that names the edge
    datatype: str | None = None

    def key(self) -> str:
        name = local_name(self.prop)
        if self.kind == "inverse":
            return "^" + name
        if self.kind == "join":
            return f"{name}={local_name(self.steps[-1].prop)}"
        return name

    def describe(self, cat: SchemaCatalog) -> str:
        p = cat.properties.get(self.prop)
        label = p.label if p else local_name(self.prop)
        src = cat.label(self.src)
        if self.kind == "inverse":
            return f"{src} ← {label} of {cat.label(self.dst or '')}"
        if self.kind == "join":
            other = cat.properties.get(self.steps[-1].prop)
            return f"{src} → {label} (matched on shared {other.label if other else ''} identifier) → {cat.label(self.dst or '')}"
        target = cat.label(self.dst) if self.dst else local_name(self.datatype or "value")
        return f"{src} → {label} → {target}"


@dataclass(frozen=True)
class SchemaPath:
    edges: tuple[Edge, ...]

    @property
    def steps(self) -> tuple[Step, ...]:
        return tuple(s for e in self.edges for s in e.steps)

    @property
    def dst(self) -> str | None:
        return self.edges[-1].dst

    @property
    def datatype(self) -> str | None:
        return self.edges[-1].datatype

    @property
    def modules(self) -> list[str]:
        out: list[str] = []
        for s in self.steps:
            if s.module not in out:
                out.append(s.module)
        return out

    @property
    def cross_module(self) -> bool:
        return len(self.modules) > 1

    def key(self) -> str:
        return ">".join(e.key() for e in self.edges)

    def describe(self, cat: SchemaCatalog) -> str:
        parts = []
        for e in self.edges:
            p = cat.properties.get(e.prop)
            label = p.label if p else local_name(e.prop)
            if e.kind == "inverse":
                parts.append(f"(is {label} of) {cat.label(e.dst or '')}")
            elif e.kind == "join":
                parts.append(f"{label} → {cat.label(e.dst or '')}")
            else:
                parts.append(f"{label}" + (f" → {cat.label(e.dst)}" if e.dst else ""))
        return f"{cat.label(self.edges[0].src)}: " + " / ".join(parts)

    def label(self, cat: SchemaCatalog) -> str:
        """Short readable option key, e.g. "engine → horsepower" or "sold vehicle model (matched on model code)"."""
        parts = []
        for e in self.edges:
            p = cat.properties.get(e.prop)
            name = p.label if p else local_name(e.prop)
            if e.kind == "inverse":
                parts.append(f"{cat.label(e.dst or '')} whose {name} it is")
            elif e.kind == "join":
                other = cat.properties.get(e.steps[-1].prop)
                parts.append(f"{cat.label(e.dst or '')} with matching {other.label if other else 'identifier'}")
            else:
                parts.append(name)
        return " → ".join(parts)

    def explain(self, cat: SchemaCatalog) -> str:
        """A plain-language description of what this path reaches, for the controller to read."""
        src = cat.label(self.edges[0].src)
        clauses = []
        for e in self.edges:
            p = cat.properties.get(e.prop)
            name = p.label if p else local_name(e.prop)
            if e.kind == "inverse":
                clauses.append(f"the {cat.label(e.dst or '')} whose {name} it is")
            elif e.kind == "join":
                other = cat.properties.get(e.steps[-1].prop)
                clauses.append(f"the {cat.label(e.dst or '')} whose {other.label if other else 'identifier'} equals its {name} (a link across the {' and '.join(self.modules)} graphs)")
            elif e.kind == "attribute":
                clauses.append(f"its {name} ({_a(local_name(e.datatype or 'value'))} value)")
            else:
                clauses.append(f"its {name}, {_a(cat.label(e.dst or ''))}")
        text = f"From {_a(src)}, follow " + ", then ".join(clauses) + "."
        details = []
        for e in self.edges:
            p = cat.properties.get(e.prop)
            if p and p.description:
                alts = f" Also called: {', '.join(p.alt_labels)}." if p.alt_labels else ""
                details.append(f"{p.label[:1].upper()}{p.label[1:]}: {p.description}{alts}")
        return " ".join([text, *details])

    def lexical_text(self, cat: SchemaCatalog) -> str:
        """Text used for shortlisting and by the offline controller: labels, aliases, descriptions."""
        bits = [self.describe(cat)]
        for e in self.edges:
            for iri in (e.prop, *(s.prop for s in e.steps)):
                p = cat.properties.get(iri)
                if p:
                    bits += [p.label, *p.alt_labels, p.description]
            if e.dst and e.dst in cat.classes:
                c = cat.classes[e.dst]
                bits += [c.label, *c.alt_labels]
        return " ".join(dict.fromkeys(b for b in bits if b))


def _a(noun: str) -> str:
    return f"{'an' if noun[:1].lower() in 'aeiou' else 'a'} {noun}"


class SchemaGraph:
    def __init__(self, cat: SchemaCatalog) -> None:
        self.cat = cat

    def _prop_edges(self, cls: str, p: PropertyCard) -> list[Edge]:
        step = Step(p.iri, False, self.cat.module_for(p.iri, cls))
        if p.kind == "object":
            return [Edge((step,), cls, r, "object", p.iri) for r in (p.ranges() or [None])]
        edges = [Edge((step,), cls, None, "attribute", p.iri, p.datatype)]
        for other in p.joins_with:
            q = self.cat.properties.get(other)
            if q is None:
                continue
            for dom in q.domains or [None]:
                edges.append(Edge((step, Step(q.iri, True, self.cat.module_for(q.iri, dom))), cls, dom, "join", p.iri))
        return edges

    def out_edges(self, cls: str) -> list[Edge]:
        edges = []
        for p in self.cat.properties_of(cls):
            edges += self._prop_edges(cls, p)
        return edges

    def in_edges(self, cls: str) -> list[Edge]:
        edges = []
        for p in self.cat.incoming_of(cls):
            for dom in p.domains or [None]:
                edges.append(Edge((Step(p.iri, True, self.cat.module_for(p.iri, dom)),), cls, dom, "inverse", p.iri))
        return edges

    def neighbours(self, cls: str) -> list[Edge]:
        return [e for e in self.out_edges(cls) if e.kind != "attribute"] + self.in_edges(cls)

    def _compatible(self, end: str | None, target: str) -> bool:
        return end is not None and (self.cat.is_a(target, end) or self.cat.is_a(end, target))

    def paths_between(self, src: str, dst: str, max_edges: int = 3) -> list[SchemaPath]:
        """All acyclic class-to-class paths from `src` to a class compatible with `dst`."""
        out: list[SchemaPath] = []
        queue: deque[tuple[tuple[Edge, ...], frozenset[str]]] = deque([((), frozenset({src}))])
        while queue:
            path, seen = queue.popleft()
            here = path[-1].dst if path else src
            if here is None or len(path) >= max_edges:
                continue
            for e in self.neighbours(here):
                if path and e.prop == path[-1].prop and e.kind != path[-1].kind and e.dst == path[-1].src:
                    continue  # immediately walking back over the same property
                new = (*path, e)
                if self._compatible(e.dst, dst):
                    out.append(SchemaPath(new))
                    continue
                if e.dst and e.dst not in seen:
                    queue.append((new, seen | {e.dst}))
        out.sort(key=lambda p: len(p.edges))
        return out

    def value_paths(self, src: str, max_edges: int = 2, *, attributes_only: bool = False, numeric_only: bool = False, temporal_only: bool = False) -> list[SchemaPath]:
        """Paths from `src` ending in an attribute (or, unless attributes_only, in an entity)."""
        from kgqa.rdf import is_date_type, is_numeric_type

        def keep(e: Edge) -> bool:
            if e.kind == "attribute":
                if numeric_only:
                    return is_numeric_type(e.datatype)
                if temporal_only:
                    return is_date_type(e.datatype) or (e.datatype or "").endswith("gYear") or "year" in e.prop.lower()
                return True
            return not (attributes_only or numeric_only or temporal_only)

        out: list[SchemaPath] = []
        frontier: list[tuple[tuple[Edge, ...], frozenset[str]]] = [((), frozenset({src}))]
        for _ in range(max_edges):
            nxt = []
            for path, seen in frontier:
                here = path[-1].dst if path else src
                if here is None:
                    continue
                for e in self.out_edges(here) + self.in_edges(here):
                    if path and e.prop == path[-1].prop:
                        continue
                    new = (*path, e)
                    if keep(e):
                        out.append(SchemaPath(new))
                    if e.dst and e.dst not in seen and e.kind != "attribute":
                        nxt.append((new, seen | {e.dst}))
            frontier = nxt
        return out
