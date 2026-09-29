"""Schema slicer: a small, fixed-budget text slice of the schema around the selected classes.

The slice is what an LLM generator sees, and its IRIs are the allow-list the generator's output is
checked against. Budget is enforced by dropping the least relevant material first: neighbour
classes, then rarely used properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kgqa.rdf import Prefixes, local_name
from kgqa.schema.cards import ClassCard, PropertyCard, SchemaCatalog
from kgqa.schema.paths import SchemaGraph
from kgqa.text import approx_tokens


@dataclass
class SchemaSlice:
    focus: list[str]
    classes: list[str]
    properties: list[str]
    modules: list[str]
    text: str
    allowed_iris: set[str] = field(default_factory=set)

    @property
    def tokens(self) -> int:
        return approx_tokens(self.text)


def _card_line(cat: SchemaCatalog, p: PropertyCard, px: Prefixes) -> str:
    card = ""
    if p.min_count is not None or p.max_count is not None:
        card = f" [{p.min_count or 0}..{p.max_count if p.max_count is not None else '*'}]"
    target = cat.label(p.range_class) if p.kind == "object" and p.range_class else local_name(p.datatype or "literal")
    arrow = "→" if p.kind == "object" else ":"
    alts = f" (aka {', '.join(p.alt_labels[:4])})" if p.alt_labels else ""
    samples = f" e.g. {', '.join(p.samples[:3])}" if p.samples else ""
    join = f" JOINS {', '.join(px.compact(j) for j in p.joins_with)} (shared identifier, other graph)" if p.joins_with else ""
    graph = f" @{p.module}"
    return f"  - {px.compact(p.iri)} {arrow} {target}{card}{graph}: {p.label}{alts}. {p.description}{samples}{join}".rstrip()


def _class_block(cat: SchemaCatalog, c: ClassCard, px: Prefixes, max_props: int | None) -> list[str]:
    subs = cat.descendants(c.iri)
    lines = [f"## {px.compact(c.iri)} \"{c.label}\" @{c.module} ({cat.instance_count(c.iri)} instances){': ' + c.description if c.description else ''}"]
    if c.parents:
        lines.append(f"  subclass of: {', '.join(px.compact(p) for p in c.parents)}")
    if subs:
        lines.append(f"  subclasses: {', '.join(px.compact(s) for s in subs)}")
    props = sorted(cat.properties_of(c.iri), key=lambda p: -p.usage)
    for p in props[:max_props] if max_props is not None else props:
        lines.append(_card_line(cat, p, px))
    return lines


class Slicer:
    def __init__(self, cat: SchemaCatalog, prefixes: Prefixes | None = None, token_budget: int = 1500, hops: int = 1) -> None:
        self.cat = cat
        self.px = prefixes or Prefixes()
        self.budget = token_budget
        self.hops = hops
        self.graph = SchemaGraph(cat)

    def neighbourhood(self, focus: list[str]) -> list[str]:
        out = [c for c in focus if c in self.cat.classes]
        frontier = list(out)
        for _ in range(self.hops):
            nxt = []
            for c in frontier:
                for e in self.graph.neighbours(c):
                    if e.dst and e.dst in self.cat.classes and e.dst not in out:
                        out.append(e.dst)
                        nxt.append(e.dst)
            frontier = nxt
        return out

    def slice(self, focus: list[str], extra_classes: list[str] = ()) -> SchemaSlice:
        focus = [c for c in dict.fromkeys([*focus, *extra_classes]) if c in self.cat.classes]
        classes = self.neighbourhood(focus)
        for max_props, n_classes in ((None, len(classes)), (None, len(focus)), (8, len(focus)), (4, len(focus))):
            chosen = classes[:n_classes]
            text, props, modules = self._render(chosen, max_props)
            if approx_tokens(text) <= self.budget:
                break
        allowed = {*chosen, *props, *(self.cat.modules[m].graph for m in modules)}
        for c in chosen:
            allowed.update(self.cat.descendants(c))
        return SchemaSlice(focus=focus, classes=chosen, properties=props, modules=modules, text=text, allowed_iris=allowed)

    def _render(self, classes: list[str], max_props: int | None) -> tuple[str, list[str], list[str]]:
        lines: list[str] = []
        props: list[str] = []
        modules: list[str] = []
        for c in classes:
            card = self.cat.classes[c]
            if card.module not in modules:
                modules.append(card.module)
            block = _class_block(self.cat, card, self.px, max_props)
            lines += block
            shown = sorted(self.cat.properties_of(c), key=lambda p: -p.usage)
            for p in shown[:max_props] if max_props is not None else shown:
                if p.iri not in props:
                    props.append(p.iri)
                    if p.module not in modules:
                        modules.append(p.module)
                for j in p.joins_with:
                    if j not in props:
                        props.append(j)
        used = sorted({pfx for line in lines for pfx, ns in self.px.map.items() if f"{pfx}:" in line})
        header = ["# Prefixes (declare these in the query before using them)"] + [f"PREFIX {pfx}: <{self.px.map[pfx]}>" for pfx in used]
        header += ["# Graphs (use GRAPH <iri> { ... } for each module's triples)"]
        for m in modules:
            mc = self.cat.modules[m]
            header.append(f"- {m}: <{mc.graph}> type predicate {self.px.compact(mc.type_predicate)} — {mc.description}")
        return "\n".join(header + lines), props, modules

    def module_overview(self, modules: list[str] | None = None) -> str:
        lines = []
        for name, mc in self.cat.modules.items():
            if modules and name not in modules:
                continue
            lines.append(f"- {name} <{mc.graph}>: {mc.description}. Classes: {', '.join(self.cat.label(c) for c in mc.classes)}")
        return "\n".join(lines)
