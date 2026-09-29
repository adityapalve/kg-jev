"""Answer matching, latency percentiles, calibration and failure attribution."""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from kgqa.rdf import IRI


def norm_value(v: Any) -> Any:
    if isinstance(v, tuple):
        return ("row", tuple(norm_value(x) for x in v))
    if isinstance(v, IRI):
        return ("iri", str(v))
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", round(float(v), 2))
    if isinstance(v, (dt.date, dt.datetime)):
        return ("date", v.isoformat()[:10])
    s = str(v).strip()
    try:
        return ("num", round(float(s), 2))
    except ValueError:
        return ("str", s.lower())


def answer_sets(pred: Sequence[Any], gold: Sequence[Any]) -> tuple[set, set]:
    p, g = {norm_value(v) for v in pred}, {norm_value(v) for v in gold}
    # "nothing" and a zero aggregate are the same answer
    zero = {("num", 0.0)}
    if not p and g == zero:
        p = zero
    if not g and p == zero:
        g = zero
    return p, g


def exact_match(pred: Sequence[Any], gold: Sequence[Any]) -> bool:
    p, g = answer_sets(pred, gold)
    return p == g


def f1(pred: Sequence[Any], gold: Sequence[Any]) -> float:
    p, g = answer_sets(pred, gold)
    if not p and not g:
        return 1.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def percentile(values: Sequence[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def reliability(pairs: Iterable[tuple[float, bool]], bins: int = 5) -> dict[str, Any]:
    """Reliability curve and expected calibration error for (confidence, correct) pairs."""
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    pairs = list(pairs)
    for c, ok in pairs:
        buckets[min(int(c * bins), bins - 1)].append((c, ok))
    rows, ece = [], 0.0
    for b in range(bins):
        xs = buckets.get(b, [])
        if not xs:
            continue
        conf = sum(c for c, _ in xs) / len(xs)
        acc = sum(ok for _, ok in xs) / len(xs)
        ece += len(xs) / len(pairs) * abs(conf - acc)
        rows.append({"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": len(xs), "confidence": conf, "accuracy": acc})
    return {"n": len(pairs), "ece": ece, "bins": rows}


SHAPE_EQUIV = {"path": "lookup"}


def same_shape(a: str | None, b: str | None) -> bool:
    return SHAPE_EQUIV.get(a or "", a) == SHAPE_EQUIV.get(b or "", b)


def attribute_failure(row: dict[str, Any], gold: dict[str, Any], linked: list[str]) -> str | None:
    """First pipeline stage whose decision disagrees with the gold annotations."""
    if row["correct"]:
        return None
    if row.get("expect_unanswered"):
        return "silent_answer"  # answered something that has no answer in the graph
    if row["status"] == "error" and row.get("timeout"):
        return "execution_timeout"
    if gold.get("entities") and not set(gold["entities"]) <= set(linked):
        return "entity_linking"
    route = row.get("route") or {}
    gates = {g["name"]: g["choice"] for g in reversed(row.get("gates", []))}  # first decision wins
    if row["arm"] != "A":
        # questions gated before a route was chosen still carry the router's decisions in the trace
        module = route.get("module") or gates.get("module")
        shape = gates.get("shape") or row.get("shape")
        if gold.get("module") and module and module != gold["module"]:
            return "routing_module"
        if gold.get("class") and route.get("class") and route["class"] != gold["class"]:
            return "routing_class"
        if gold.get("shape") and shape and not same_shape(shape, gold["shape"]):
            return "query_shape"
        props = set((row.get("plan") or {}).get("properties", []))
        if gold.get("properties") and row.get("plan") and not set(gold["properties"]) <= props:
            return "property_choice"
    if row["status"] == "unanswered":
        return "fallback_unanswered" if row["source"] == "fallback" else "gated_or_generation"
    if row["status"] == "error":
        return "execution_error"
    if row["source"] == "fallback":
        return "fallback_wrong"
    if row["source"] == "llm":
        return "generation"
    return "other"
