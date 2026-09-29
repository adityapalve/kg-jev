"""Build a QALD-9 subset the DBpedia slice can answer: data/dbpedia/benchmark.json.

QALD-9 was written against DBpedia 2016-10; the slice is current DBpedia, and only part of it.
So a question is kept only when
  - its gold SPARQL runs on the slice (over the union of the module graphs) and returns an answer,
  - it touches the slice's schema (a property or class the slice holds), and
  - every dbr: resource it names exists in the slice.
Gold answers are recomputed on the slice at eval time; each item is tagged "matches-qald" when that
agrees with QALD's official answer and "drift" when it does not.

Run after `kgqa --config dbpedia.toml build`:
  .venv/bin/python data/dbpedia/build_benchmark.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kgqa.config import load_config  # noqa: E402
from kgqa.eval.benchmark import _qald, gold_answers  # noqa: E402
from kgqa.eval.metrics import answer_sets  # noqa: E402
from kgqa.schema import SchemaCatalog  # noqa: E402
from kgqa.store import OxigraphStore  # noqa: E402

HERE = Path(__file__).parent
SOURCES = {
    "train": "https://raw.githubusercontent.com/ag-sc/QALD/master/9/data/qald-9-train-multilingual.json",
    "test": "https://raw.githubusercontent.com/ag-sc/QALD/master/9/data/qald-9-test-multilingual.json",
}
PREFIX_DECL = re.compile(r"PREFIX\s+([\w-]*):\s*<([^>]+)>", re.I)
STANDARD = {
    "dbo": "http://dbpedia.org/ontology/",
    "dbr": "http://dbpedia.org/resource/",
    "res": "http://dbpedia.org/resource/",
    "dbp": "http://dbpedia.org/property/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "yago": "http://dbpedia.org/class/yago/",
}


def iris_in(sparql: str) -> set[str]:
    decls = {**STANDARD, **dict(PREFIX_DECL.findall(sparql))}
    body = PREFIX_DECL.sub("", sparql)
    body = re.sub(r'"(?:[^"\\]|\\.)*"', '""', body)
    out = set(re.findall(r"<([^<>\s]+)>", body))
    for p, local in re.findall(r"(?<![\w?$<])([A-Za-z][\w-]*):([A-Za-z0-9_][\w.()%,'-]*)", re.sub(r"<[^>]*>", "", body)):
        if p in decls:
            out.add(decls[p] + local.rstrip(".,"))
    return out


def with_prefixes(sparql: str) -> str:
    declared = {p for p, _ in PREFIX_DECL.findall(sparql)}
    return "".join(f"PREFIX {p}: <{ns}>\n" for p, ns in STANDARD.items() if p not in declared) + sparql


def main() -> None:
    cfg = load_config(ROOT / "dbpedia.toml")
    store = OxigraphStore(cfg.path(cfg.store_path))
    cat = SchemaCatalog.load(cfg.path(cfg.catalog_path))
    schema_iris = set(cat.classes) | set(cat.properties)

    items = []
    for split, url in SOURCES.items():
        path = HERE / f"qald-9-{split}.json"
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        for it in _qald(json.loads(path.read_text())):
            it.id = f"{split}-{it.id}"
            items.append(it)

    kept, reasons = [], {"no sparql": 0, "off-schema": 0, "missing entity": 0, "no answer on slice": 0}
    for it in items:
        if not it.gold_sparql:
            reasons["no sparql"] += 1
            continue
        sparql = with_prefixes(it.gold_sparql)
        used = iris_in(sparql)
        if not used & schema_iris:
            reasons["off-schema"] += 1
            continue
        resources = [i for i in used if i.startswith("http://dbpedia.org/resource/")]
        if any(not store.ask(f"ASK {{ GRAPH ?g {{ <{r}> ?p ?o }} }}") for r in resources):
            reasons["missing entity"] += 1
            continue
        official = it.gold_answers
        it.gold_answers, it.gold_sparql = None, sparql
        answers = gold_answers(it, store)
        if not answers or answers == [False]:
            reasons["no answer on slice"] += 1
            continue
        pred, gold = answer_sets(answers, official or [])
        tags = ["qald9", "matches-qald" if pred == gold else "drift"]
        kept.append({"id": it.id, "question": it.question, "gold_sparql": sparql, "gold": {"entities": resources}, "tags": tags})

    out = {"name": "qald9-dbpedia-film-slice", "graph": "data/dbpedia", "items": kept}
    (HERE / "benchmark.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(items)} QALD-9 questions -> kept {len(kept)} ({sum('matches-qald' in k['tags'] for k in kept)} match QALD's official answers)")
    print("dropped:", reasons)


if __name__ == "__main__":
    main()
