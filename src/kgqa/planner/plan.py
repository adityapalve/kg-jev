"""Query plan IR: what the controller decided, in a form templates compile deterministically."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kgqa.jev import ChoiceResult
from kgqa.linking.literals import LiteralMention
from kgqa.schema.cards import SchemaCatalog
from kgqa.schema.paths import SchemaPath

SHAPES: dict[str, str] = {
    "lookup": "Look up a value or related entity of one named entity: what is the X of Y, who makes Y, where is Y, when was Y, which X does Y have",
    "path": "Follow a chain of relations from one named entity to another thing: the X of the Y of Z, which region is the dealer of Z in",
    "count": "Count how many entities match: how many, number of, count of",
    "aggregate": "Compute a total, sum, average, mean, minimum or maximum of a numeric value across many entities: total, sum, average, overall, combined",
    "list": "List every entity that matches some conditions: which, list, show all, what are the, name all",
    "superlative": "Find the single entity with the highest or lowest value: most, least, highest, lowest, cheapest, most expensive, biggest, newest, oldest, top",
    "compare": "Compare two or more named entities on one value: which is more, greater, higher, cheaper, faster, than, or, versus",
    "boolean": "A yes/no question about whether a fact holds: is, does, did, was, are, has",
    "unsupported": "Needs something none of the above can express: a breakdown per group (per region, by dealer, for each), "
    "negation or exclusion (not, except, without, never), the top or bottom N with N more than one, "
    "arithmetic between values (difference, ratio, percentage), or a combination of several of these",
}
UNSUPPORTED_SHAPE = "unsupported"
ANCHOR_SHAPES = {"lookup", "path", "compare", "boolean"}

OPS: dict[str, str] = {
    "eq": "equal to, exactly, in, on, during, of, for, from that period",
    "gt": "greater than, more than, over, above, after, later than, exceeds, newer than",
    "ge": "at least, or more, no less than, since, from onwards",
    "lt": "less than, under, below, before, earlier than, older than, cheaper than",
    "le": "at most, or less, up to, no more than, until, by",
}

AGGS: dict[str, str] = {
    "sum": "total, sum, overall, combined, altogether, in total",
    "avg": "average, mean, typical",
    "min": "minimum, lowest, smallest, least",
    "max": "maximum, highest, largest, biggest, most",
}

ORDERS: dict[str, str] = {
    "desc": "highest, most, largest, biggest, maximum, top, most expensive, newest, latest, last, more, greater, higher, faster, more powerful",
    "asc": "lowest, least, smallest, minimum, cheapest, oldest, earliest, first, fewest, less, lower, cheaper, slower",
}


@dataclass(frozen=True)
class Anchor:
    iri: str
    label: str
    types: tuple[str, ...]
    text: str = ""


@dataclass
class Link:
    """Target ?x reaches `anchor` along `path` (path oriented from ?x to the anchor)."""

    anchor: Anchor
    path: SchemaPath


@dataclass
class Filter:
    """Target ?x reaches a literal along `path`; the literal must satisfy `op`."""

    path: SchemaPath
    op: str
    literal: LiteralMention


@dataclass
class QueryPlan:
    shape: str
    module: str
    target_class: str | None
    subject: Anchor | None = None
    compare: list[Anchor] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    answer_path: SchemaPath | None = None  # lookup/path answer, or the measure for aggregate/superlative/compare
    agg: str | None = None
    order: str | None = None
    limit: int | None = None
    decisions: dict[str, ChoiceResult] = field(default_factory=dict)
    overrides: dict[str, str] = field(default_factory=dict)  # decision -> option forced by repair
    notes: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)  # parts of the question the plan could not place
    context: Any = field(default=None, repr=False, compare=False)  # planner state for repair

    @property
    def confidence(self) -> float:
        if self.unresolved:
            return 0.0
        confs = [d.confidence for k, d in self.decisions.items() if k not in self.overrides]
        return min(confs) if confs else 1.0

    @property
    def weakest_decision(self) -> str | None:
        cands = {k: d for k, d in self.decisions.items() if len(d.probabilities) > 1}
        return min(cands, key=lambda k: cands[k].confidence) if cands else None

    @property
    def cross_module(self) -> bool:
        paths = [l.path for l in self.links] + [f.path for f in self.filters] + ([self.answer_path] if self.answer_path else [])
        mods = {m for p in paths for m in p.modules} | {self.module}
        return len(mods) > 1

    @property
    def hops(self) -> int:
        paths = [l.path for l in self.links] + [f.path for f in self.filters] + ([self.answer_path] if self.answer_path else [])
        return max((len(p.steps) for p in paths), default=0)

    def describe(self, cat: SchemaCatalog) -> str:
        parts = [f"shape={self.shape}"]
        if self.target_class:
            parts.append(f"target={cat.label(self.target_class)} in {self.module}")
        if self.subject:
            parts.append(f"subject={self.subject.label}")
        if self.compare:
            parts.append("compare=" + " vs ".join(a.label for a in self.compare))
        for l in self.links:
            parts.append(f"linked to {l.anchor.label} via {l.path.describe(cat)}")
        for f in self.filters:
            parts.append(f"filter {f.path.describe(cat)} {f.op} {f.literal.text}")
        if self.answer_path:
            parts.append(("measure " if self.shape in ("aggregate", "superlative", "compare") else "answer ") + self.answer_path.describe(cat))
        if self.agg:
            parts.append(f"agg={self.agg}")
        if self.order:
            parts.append(f"order={self.order}")
        return "; ".join(parts)

    def explain(self, cat: SchemaCatalog) -> str:
        """The plan as one plain sentence, for the controller to judge against the question."""
        ops = {"eq": "in/equal to", "gt": "greater than", "ge": "at least", "lt": "less than", "le": "at most"}
        cls = cat.label(self.target_class) if self.target_class else "thing"
        cond = [f"linked to {l.anchor.label} by {l.path.label(cat)}" for l in self.links]
        cond += [f"whose {f.path.label(cat)} is {ops.get(f.op, f.op)} {f.literal.value if f.literal.kind == 'string' else f.literal.text}" for f in self.filters]
        where = f" {', and '.join(cond)}" if cond else ""
        measure = self.answer_path.label(cat) if self.answer_path else ""
        if self.shape in ("lookup", "path") and self.subject and self.answer_path:
            edges = self.answer_path.edges
            if len(edges) == 1 and edges[0].kind == "inverse":  # "the movies whose film director is Kubrick"
                prop = cat.properties.get(edges[0].prop)
                return f"Find the {cat.label(edges[0].dst or '')} records whose {prop.label if prop else 'relation'} is {self.subject.label}."
            return f"Look up the {measure} of {self.subject.label}."
        if self.shape == "count":
            return f"Count the {cls} records{where}."
        if self.shape == "list":
            return f"List the {cls} records{where}."
        if self.shape == "aggregate":
            return f"Compute the {self.agg} of the {measure} over {cls} records{where}."
        if self.shape == "superlative":
            return f"Find the {cls}{where} with the {'highest' if self.order == 'desc' else 'lowest'} {measure}."
        if self.shape == "compare":
            names = " and ".join(a.label for a in self.compare)
            return f"Compare {names} by {measure} and pick the one with the {'higher' if self.order == 'desc' else 'lower'} value."
        if self.shape == "boolean" and self.subject:
            return f"Check whether {self.subject.label} is a {cls}{where}."
        return self.describe(cat)

    def to_json(self, cat: SchemaCatalog) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "module": self.module,
            "target_class": self.target_class,
            "summary": self.describe(cat),
            "confidence": self.confidence,
            "cross_module": self.cross_module,
            "hops": self.hops,
            "decisions": {k: {"choice": d.choice, "confidence": d.confidence} for k, d in self.decisions.items()},
            "properties": sorted({s.prop for p in self.paths() for s in p.steps}),
        }

    def paths(self) -> list[SchemaPath]:
        return [l.path for l in self.links] + [f.path for f in self.filters] + ([self.answer_path] if self.answer_path else [])
