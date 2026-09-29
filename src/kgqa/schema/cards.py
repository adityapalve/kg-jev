"""Schema cards: compact, per-module / per-class / per-property summaries of the graph schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kgqa.rdf import is_date_type, is_numeric_type, local_name
from kgqa.text import split_camel


@dataclass
class PropertyCard:
    iri: str
    label: str
    module: str
    kind: str  # "object" | "datatype"
    description: str = ""
    alt_labels: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    range_class: str | None = None
    extra_ranges: list[str] = field(default_factory=list)  # observed target classes the declared range misses
    datatype: str | None = None
    min_count: int | None = None
    max_count: int | None = None
    usage: int = 0
    module_usage: dict[str, int] = field(default_factory=dict)  # a property can appear in several graphs
    samples: list[str] = field(default_factory=list)
    joins_with: list[str] = field(default_factory=list)  # shared-identifier properties in other modules
    identifier_scheme: str | None = None

    def ranges(self) -> list[str]:
        return [r for r in [self.range_class, *self.extra_ranges] if r]

    @property
    def numeric(self) -> bool:
        return is_numeric_type(self.datatype)

    @property
    def temporal(self) -> bool:
        return is_date_type(self.datatype)

    def value_type(self) -> str:
        if self.kind == "object":
            return local_name(self.range_class or "resource")
        return local_name(self.datatype or "literal")


@dataclass
class ClassCard:
    iri: str
    label: str
    module: str
    description: str = ""
    alt_labels: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    instance_count: int = 0  # direct instances
    properties: list[str] = field(default_factory=list)  # properties with this class as domain


@dataclass
class JoinLink:
    """Two properties in different modules whose values share an identifier scheme."""

    left: str
    right: str
    left_module: str
    right_module: str
    overlap: int = 0  # distinct shared values observed


@dataclass
class ModuleCard:
    name: str
    graph: str
    description: str
    type_predicate: str
    classes: list[str] = field(default_factory=list)
    top_classes: list[str] = field(default_factory=list)
    triple_count: int = 0
    linked_modules: list[str] = field(default_factory=list)


@dataclass
class SchemaCatalog:
    modules: dict[str, ModuleCard] = field(default_factory=dict)
    classes: dict[str, ClassCard] = field(default_factory=dict)
    properties: dict[str, PropertyCard] = field(default_factory=dict)
    joins: list[JoinLink] = field(default_factory=list)

    # ---- lookups ----
    def label(self, iri: str) -> str:
        if iri in self.classes:
            return self.classes[iri].label
        if iri in self.properties:
            return self.properties[iri].label
        return split_camel(local_name(iri))

    def ancestors(self, cls: str) -> list[str]:
        out, stack = [], list(self.classes[cls].parents) if cls in self.classes else []
        while stack:
            c = stack.pop()
            if c not in out:
                out.append(c)
                if c in self.classes:
                    stack.extend(self.classes[c].parents)
        return out

    def descendants(self, cls: str) -> list[str]:
        out, stack = [], list(self.classes[cls].children) if cls in self.classes else []
        while stack:
            c = stack.pop()
            if c not in out:
                out.append(c)
                if c in self.classes:
                    stack.extend(self.classes[c].children)
        return out

    def is_a(self, cls: str, other: str) -> bool:
        return cls == other or other in self.ancestors(cls)

    def properties_of(self, cls: str) -> list[PropertyCard]:
        """Properties applicable to instances of `cls`, including inherited ones."""
        seen: dict[str, PropertyCard] = {}
        for c in [cls, *self.ancestors(cls)]:
            for p in self.classes[c].properties if c in self.classes else []:
                if p in self.properties:
                    seen.setdefault(p, self.properties[p])
        return list(seen.values())

    def incoming_of(self, cls: str) -> list[PropertyCard]:
        targets = {cls, *self.ancestors(cls)}
        return [p for p in self.properties.values() if p.kind == "object" and targets & set(p.ranges())]

    def instance_count(self, cls: str) -> int:
        return sum(self.classes[c].instance_count for c in [cls, *self.descendants(cls)] if c in self.classes)

    def module_for(self, prop: str, subject_class: str | None) -> str:
        """The graph holding `prop` triples whose subject is a `subject_class` instance."""
        card = self.properties[prop]
        if subject_class in self.classes and self.classes[subject_class].module in card.module_usage:
            return self.classes[subject_class].module
        return card.module

    def module_of(self, iri: str) -> str | None:
        if iri in self.classes:
            return self.classes[iri].module
        if iri in self.properties:
            return self.properties[iri].module
        return None

    def all_iris(self) -> set[str]:
        return {*self.classes, *self.properties, *(m.graph for m in self.modules.values())}

    # ---- persistence ----
    def to_json(self) -> dict[str, Any]:
        return {
            "modules": {k: asdict(v) for k, v in self.modules.items()},
            "classes": {k: asdict(v) for k, v in self.classes.items()},
            "properties": {k: asdict(v) for k, v in self.properties.items()},
            "joins": [asdict(j) for j in self.joins],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> SchemaCatalog:
        return cls(
            modules={k: ModuleCard(**v) for k, v in d["modules"].items()},
            classes={k: ClassCard(**v) for k, v in d["classes"].items()},
            properties={k: PropertyCard(**v) for k, v in d["properties"].items()},
            joins=[JoinLink(**j) for j in d.get("joins", [])],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> SchemaCatalog:
        return cls.from_json(json.loads(Path(path).read_text()))
