"""Configuration, loaded from TOML (see kgqa.toml at the repo root)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kgqa.rdf import RDF_TYPE, RDFS_LABEL, SCHEMA, SKOS


@dataclass
class ModuleConfig:
    name: str
    graph: str
    description: str = ""
    files: list[str] = field(default_factory=list)
    type_predicate: str = RDF_TYPE
    label_predicates: list[str] = field(default_factory=lambda: [RDFS_LABEL, SKOS + "altLabel", SCHEMA + "name"])


@dataclass
class Gates:
    """Confidence thresholds. Starting points; tune on the dev set with `kgqa eval`."""

    module: float = 0.45
    klass: float = 0.35
    entity: float = 0.5
    plan: float = 0.35
    verify: float = 0.5


@dataclass
class Limits:
    query_limit: int = 1000
    query_timeout_s: float = 20.0
    values_chunk: int = 500
    max_intermediate: int = 10_000
    max_repairs: int = 2
    max_path_edges: int = 3
    slice_token_budget: int = 1500


@dataclass
class Config:
    root: Path
    modules: list[ModuleConfig]
    store_kind: str = "oxigraph"
    store_path: str | None = None
    store_endpoint: str | None = None
    ontology_files: list[str] = field(default_factory=list)
    shapes_files: list[str] = field(default_factory=list)
    catalog_path: str = ".kgqa/schema.json"
    entity_index_path: str = ".kgqa/entities.json"
    controller_backend: str = "heuristic"
    controller_model: str | None = None
    controller_timeout: float | None = None
    controller_cache: str | None = None
    llm_backend: str = "none"
    llm_options: dict[str, Any] = field(default_factory=dict)
    llm_cache: str | None = ".kgqa/llm-cache.jsonl"
    generation: str = "auto"  # template | llm | auto
    beam_width: int = 2
    beam_min_prob: float = 0.15
    tie_policy: str = "try_next"  # try_next | fallback
    hop_mode: str = "app"  # app (small queries joined in code at module boundaries) | fused (one query)
    review_queue: str = ".kgqa/review_queue.jsonl"
    prefixes: dict[str, str] = field(default_factory=dict)
    gates: Gates = field(default_factory=Gates)
    limits: Limits = field(default_factory=Limits)

    def path(self, p: str | None) -> Path | None:
        if p is None:
            return None
        pp = Path(p)
        return pp if pp.is_absolute() else self.root / pp

    def module(self, name: str) -> ModuleConfig:
        for m in self.modules:
            if m.name == name:
                return m
        raise KeyError(name)


def load_config(path: str | Path = "kgqa.toml") -> Config:
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    store = raw.get("store", {})
    schema = raw.get("schema", {})
    ctrl = raw.get("controller", {})
    llm = raw.get("llm", {})
    pipe = raw.get("pipeline", {})
    modules = [ModuleConfig(**m) for m in raw.get("modules", [])]
    gates = Gates(**{("klass" if k == "class" else k): v for k, v in raw.get("gates", {}).items()})
    limits = Limits(**raw.get("limits", {}))
    return Config(
        root=path.parent,
        modules=modules,
        store_kind=store.get("kind", "oxigraph"),
        store_path=store.get("path"),
        store_endpoint=store.get("endpoint"),
        ontology_files=schema.get("ontology", []),
        shapes_files=schema.get("shapes", []),
        catalog_path=schema.get("catalog", ".kgqa/schema.json"),
        entity_index_path=schema.get("entity_index", ".kgqa/entities.json"),
        controller_backend=ctrl.get("backend", "heuristic"),
        controller_model=ctrl.get("model"),
        controller_timeout=ctrl.get("timeout"),
        controller_cache=ctrl.get("cache"),
        llm_backend=llm.get("backend", "none"),
        llm_options={k: v for k, v in llm.items() if k not in ("backend", "cache")},
        llm_cache=llm.get("cache", ".kgqa/llm-cache.jsonl"),
        generation=pipe.get("generation", "auto"),
        beam_width=pipe.get("beam_width", 2),
        beam_min_prob=pipe.get("beam_min_prob", 0.15),
        tie_policy=pipe.get("tie_policy", "try_next"),
        hop_mode=pipe.get("hop_mode", "app"),
        review_queue=pipe.get("review_queue", ".kgqa/review_queue.jsonl"),
        prefixes=raw.get("prefixes", {}),
        gates=gates,
        limits=limits,
    )
