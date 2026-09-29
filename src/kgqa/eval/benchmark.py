"""Benchmark loaders: native JSON, QALD (9/10 JSON), LC-QuAD 2.0.

Gold answers come from executing the gold SPARQL against the configured store, unless the item
carries explicit answers. QALD and LC-QuAD target public Wikidata/DBpedia; check the graph version
your store holds before trusting their gold answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kgqa.rdf import IRI, literal_to_python
from kgqa.store.base import QueryError, Store


@dataclass
class Item:
    id: str
    question: str
    gold_sparql: str | None = None
    gold_answers: list[Any] | None = None
    gold: dict[str, Any] = field(default_factory=dict)  # module, class, shape, properties, entities
    tags: list[str] = field(default_factory=list)
    expect_unanswered: bool = False


def _native(d: dict[str, Any]) -> list[Item]:
    return [
        Item(
            id=str(x["id"]),
            question=x["question"],
            gold_sparql=x.get("gold_sparql"),
            gold_answers=x.get("gold_answers"),
            gold=x.get("gold") or {},
            tags=x.get("tags", []),
            expect_unanswered=x.get("expect_unanswered", False),
        )
        for x in d["items"]
    ]


def _qald_answers(answers: list[dict[str, Any]]) -> list[Any] | None:
    out: list[Any] = []
    for a in answers or []:
        if "boolean" in a:
            return [bool(a["boolean"])]
        for b in a.get("results", {}).get("bindings", []):
            for v in b.values():
                out.append(IRI(v["value"]) if v.get("type") == "uri" else literal_to_python(v["value"], v.get("datatype")))
    return out or None


def _qald(d: dict[str, Any], lang: str = "en") -> list[Item]:
    items = []
    for q in d["questions"]:
        text = next((x["string"] for x in q.get("question", []) if x.get("language") == lang), None)
        if not text:
            continue
        items.append(Item(id=str(q.get("id")), question=text, gold_sparql=(q.get("query") or {}).get("sparql"), gold_answers=_qald_answers(q.get("answers", [])), tags=["qald"]))
    return items


def _lcquad2(rows: list[dict[str, Any]]) -> list[Item]:
    items = []
    for r in rows:
        text = r.get("question") or r.get("paraphrased_question") or r.get("NNQT_question")
        if not text or text in ("[]", "n/a"):
            continue
        items.append(Item(id=str(r.get("uid")), question=text, gold_sparql=r.get("sparql_wikidata"), tags=["lcquad2"]))
    return items


def load_benchmark(path: str | Path) -> list[Item]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "items" in data:
        return _native(data)
    if isinstance(data, dict) and "questions" in data:
        return _qald(data)
    if isinstance(data, list) and data and "sparql_wikidata" in data[0]:
        return _lcquad2(data)
    raise ValueError(f"Unrecognized benchmark format: {path}")


def gold_answers(item: Item, store: Store, timeout: float = 60.0) -> list[Any] | None:
    if item.gold_answers is not None:
        return item.gold_answers
    if not item.gold_sparql:
        return None
    body = "\n".join(l for l in item.gold_sparql.splitlines() if not l.strip().upper().startswith("PREFIX")).lstrip().upper()
    try:
        if body.startswith("ASK"):
            return [store.ask(item.gold_sparql, timeout=timeout)]
        res = store.select(item.gold_sparql, timeout=timeout)
    except QueryError:
        return None
    return list(dict.fromkeys(v for r in res.rows for v in [r.get(res.variables[0])] if v is not None)) if res.variables else []
