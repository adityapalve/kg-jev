"""Runs benchmark arms and aggregates metrics.

Arms (docs/architecture.md section 8):
  A  baseline: entity linking + LLM with retrieved schema, one query (no routing)
  B  router + planner + constrained LLM generation
  C  router + planner + templates (LLM only if no template applies and one is configured)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from kgqa.eval.benchmark import Item, gold_answers
from kgqa.eval.metrics import attribute_failure, exact_match, f1, percentile, reliability, same_shape
from kgqa.llm import NullLLM
from kgqa.pipeline import Answer, Pipeline

ARMS = {"A": "baseline (LLM + retrieved schema)", "B": "router + constrained LLM", "C": "router + templates"}


def run_arm(pipe: Pipeline, arm: str, question: str) -> Answer:
    if arm == "A":
        return pipe.ask_baseline(question)
    if arm == "B":
        return pipe.ask(question, generation="llm")
    return pipe.ask(question, generation="template" if isinstance(pipe.llm, NullLLM) else "auto")


def _row(arm: str, item: Item, ans: Answer, gold: list[Any] | None) -> dict[str, Any]:
    tr = ans.trace
    if item.expect_unanswered:
        correct = ans.status in ("unanswered", "no_data")
        score = 1.0 if correct else 0.0
    elif gold is None:
        correct, score = False, 0.0
    else:
        correct = ans.status in ("answered", "fallback", "no_data") and exact_match(ans.values, gold)
        score = f1(ans.values, gold) if ans.status in ("answered", "fallback", "no_data") else 0.0
    row = {
        "arm": arm,
        "id": item.id,
        "question": item.question,
        "tags": item.tags,
        "expect_unanswered": item.expect_unanswered,
        "status": ans.status,
        "source": ans.source,
        "shape": ans.shape,
        "route": ans.route,
        "plan": ans.plan,
        "pred": ans.labels,
        "pred_values": [str(v) for v in ans.values],
        "gold": [str(v) for v in gold] if gold is not None else None,
        "correct": correct,
        "f1": score,
        "confidence": ans.confidence,
        "reason": ans.reason,
        "latency_ms": tr.total_ms if tr else 0.0,
        "stages": dict(tr.stages) if tr else {},
        "jev_calls": len(tr.jev) if tr else 0,
        "jev_tokens": tr.jev_tokens if tr else 0,
        "llm_calls": len(tr.llm) if tr else 0,
        "llm_tokens": tr.llm_tokens if tr else 0,
        "queries": len(tr.queries) if tr else 0,
        "timeout": any(q.timeout for q in tr.queries) if tr else False,
        "gates": [{"name": g.name, "choice": g.choice, "confidence": g.confidence, "passed": g.passed} for g in (tr.gates if tr else [])],
        "sparql": ans.sparql,
    }
    return row


def _gate_correct(gate: dict[str, Any], row: dict[str, Any], gold: dict[str, Any], cat_label) -> bool | None:
    name = gate["name"].split(":")[0]
    if name == "module" and gold.get("module"):
        return gate["choice"] == gold["module"]
    if name == "class" and gold.get("class"):
        return gate["choice"] == cat_label(gold["class"])
    if name == "shape" and gold.get("shape"):
        return same_shape(gate["choice"], gold["shape"])
    if name in ("plan", "verify"):
        return row["correct"]
    return None


def evaluate(pipe: Pipeline, items: list[Item], arms: list[str], progress=None) -> dict[str, Any]:
    golds = {it.id: gold_answers(it, pipe.store) for it in items}
    rows: list[dict[str, Any]] = []
    for arm in arms:
        if arm in ("A", "B") and isinstance(pipe.llm, NullLLM):
            continue  # needs an LLM; reported as skipped
        for it in items:
            try:
                ans = run_arm(pipe, arm, it.question)
            except Exception as e:  # one broken question (API error, bug) must not end a long run
                ans = Answer(it.question, "error", reason=f"{type(e).__name__}: {e}")
            row = _row(arm, it, ans, golds[it.id])
            row["failure"] = attribute_failure(row, it.gold, ans.entities)
            row["gate_correct"] = [_gate_correct(g, row, it.gold, pipe.cat.label) for g in row["gates"]]
            rows.append(row)
            if progress:
                progress(row)
    return {"arms": arms, "skipped": [a for a in arms if a in ("A", "B") and isinstance(pipe.llm, NullLLM)], "rows": rows, "summary": summarize(rows)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    for arm, rs in by_arm.items():
        n = len(rs)
        lat = [r["latency_ms"] for r in rs]
        stages: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            for s, ms in r["stages"].items():
                stages[s].append(ms)
        fb = [r for r in rs if r["source"] == "fallback" and arm != "A"]
        gate_pairs: dict[str, list[tuple[float, bool]]] = defaultdict(list)
        for r in rs:
            for g, ok in zip(r["gates"], r["gate_correct"]):
                if ok is not None:
                    gate_pairs[g["name"].split(":")[0]].append((g["confidence"], ok))
        answered = [r for r in rs if r["status"] in ("answered", "no_data", "fallback") and not r["expect_unanswered"]]
        out[arm] = {
            "name": ARMS.get(arm, arm),
            "n": n,
            "accuracy": sum(r["correct"] for r in rs) / n if n else 0.0,
            "f1": sum(r["f1"] for r in rs) / n if n else 0.0,
            "answered_rate": len(answered) / max(1, sum(not r["expect_unanswered"] for r in rs)),
            "precision_when_answered": (sum(r["correct"] for r in answered) / len(answered)) if answered else 0.0,
            "latency_p50_ms": percentile(lat, 0.5),
            "latency_p95_ms": percentile(lat, 0.95),
            "stage_p50_ms": {s: percentile(v, 0.5) for s, v in stages.items()},
            "stage_p95_ms": {s: percentile(v, 0.95) for s, v in stages.items()},
            "jev_calls_mean": sum(r["jev_calls"] for r in rs) / n if n else 0.0,
            "jev_tokens_mean": sum(r["jev_tokens"] for r in rs) / n if n else 0.0,
            "llm_tokens_mean": sum(r["llm_tokens"] for r in rs) / n if n else 0.0,
            "queries_mean": sum(r["queries"] for r in rs) / n if n else 0.0,
            "timeout_rate": sum(r["timeout"] for r in rs) / n if n else 0.0,
            "fallback_rate": len(fb) / n if n else 0.0,
            "fallback_accuracy": (sum(r["correct"] for r in fb) / len(fb)) if fb else None,
            "calibration": {g: reliability(p) for g, p in gate_pairs.items()},
            "answer_calibration": reliability([(r["confidence"], r["correct"]) for r in rs if r["status"] == "answered"]),
            "failures": dict(Counter(r["failure"] for r in rs if r["failure"])),
            "by_tag": {t: sum(r["correct"] for r in rs if t in r["tags"]) / max(1, sum(t in r["tags"] for r in rs)) for t in sorted({t for r in rs for t in r["tags"]})},
            "sources": dict(Counter(r["source"] or "none" for r in rs)),
        }
    return out
