"""Offline build steps: load data into the store, build schema cards and the entity index."""

from __future__ import annotations

import logging
from pathlib import Path

from kgqa.config import Config
from kgqa.linking import EntityIndex
from kgqa.schema import SchemaBuilder, SchemaCatalog
from kgqa.store import HttpSparqlStore, OxigraphStore, Store

log = logging.getLogger(__name__)


def open_store(cfg: Config, *, in_memory: bool = False) -> Store:
    if cfg.store_kind == "http":
        if not cfg.store_endpoint:
            raise ValueError("store.kind = 'http' needs store.endpoint")
        return HttpSparqlStore(cfg.store_endpoint, default_timeout=cfg.limits.query_timeout_s)
    return OxigraphStore(None if in_memory else cfg.path(cfg.store_path))


def load_data(cfg: Config, store: Store, *, replace: bool = True) -> dict[str, int]:
    """Load each module's files into its named graph (embedded store only)."""
    if not isinstance(store, OxigraphStore):
        raise TypeError("Loading files is only supported for the embedded Oxigraph store; load remote endpoints with their own tools.")
    counts = {}
    for m in cfg.modules:
        if replace:
            store.clear_graph(m.graph)
        for f in m.files:
            store.load_file(cfg.path(f), m.graph)
        counts[m.name] = store.select(f"SELECT (COUNT(*) AS ?n) WHERE {{ GRAPH <{m.graph}> {{ ?s ?p ?o }} }}").column("n")[0]
    return counts


def store_is_empty(store: Store) -> bool:
    return not store.ask("ASK { GRAPH ?g { ?s ?p ?o } }")


def build_catalog(cfg: Config, store: Store) -> SchemaCatalog:
    builder = SchemaBuilder(store, cfg.modules, [cfg.path(f) for f in cfg.ontology_files], [cfg.path(f) for f in cfg.shapes_files], prune_unused=cfg.prune_unused)
    return builder.build()


def build_all(cfg: Config, store: Store, *, load: bool = True, save: bool = True) -> tuple[SchemaCatalog, EntityIndex]:
    if load and isinstance(store, OxigraphStore):
        load_data(cfg, store)
    catalog = build_catalog(cfg, store)
    index = EntityIndex.build(store, cfg.modules)
    if save:
        catalog.save(cfg.path(cfg.catalog_path))
        index.save(cfg.path(cfg.entity_index_path))
    return catalog, index


def load_or_build(cfg: Config, store: Store) -> tuple[SchemaCatalog, EntityIndex]:
    cat_path, idx_path = cfg.path(cfg.catalog_path), cfg.path(cfg.entity_index_path)
    empty = isinstance(store, OxigraphStore) and store_is_empty(store)
    if empty or not (Path(cat_path).exists() and Path(idx_path).exists()):
        log.info("building store/catalog/index (empty store or missing artifacts)")
        return build_all(cfg, store, load=empty)
    return SchemaCatalog.load(cat_path), EntityIndex.load(idx_path)
