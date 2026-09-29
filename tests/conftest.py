from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from kgqa.build import build_all, open_store
from kgqa.config import load_config
from kgqa.jev import make_controller
from kgqa.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
O = "http://example.org/onto#"
D = "http://example.org/id/"


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("kgqa")
    base = load_config(ROOT / "kgqa.toml")
    return dataclasses.replace(
        base,
        store_path=None,
        catalog_path=str(tmp / "schema.json"),
        entity_index_path=str(tmp / "entities.json"),
        review_queue=str(tmp / "review.jsonl"),
        controller_cache=None,
    )


@pytest.fixture(scope="session")
def built(cfg):
    store = open_store(cfg, in_memory=True)
    catalog, index = build_all(cfg, store)
    return store, catalog, index


@pytest.fixture(scope="session")
def store(built):
    return built[0]


@pytest.fixture(scope="session")
def catalog(built):
    return built[1]


@pytest.fixture(scope="session")
def index(built):
    return built[2]


@pytest.fixture
def pipeline(cfg, built):
    store, catalog, index = built
    return Pipeline(cfg, store, catalog, index, make_controller("heuristic"))
