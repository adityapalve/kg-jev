"""Verifier: is this result an answer to the question, and if not, what to do next.

Cheap code checks run first (errors, empty hops, value types against the shapes, cardinality).
Then one jev call asks whether the result answers the question and, for empty results, why it is
empty. For empties, code runs probe queries (does the anchor have that relation at all? do any
targets exist without the filters?) and puts the evidence in the state, because the distinction
"wrong property" vs "no such data" cannot be made from the question alone.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kgqa.config import Gates, Limits
from kgqa.executor import ExecResult
from kgqa.executor.program import Program
from kgqa.jev import Choice, Noul, RecordingController
from kgqa.planner.plan import QueryPlan
from kgqa.rdf import IRI, is_date_type, is_numeric_type
from kgqa.schema.cards import SchemaCatalog
from kgqa.store.base import QueryError, Store
from kgqa.trace import current_trace

EMPTY_REASONS = {
    "wrong_property": "The query used the wrong relation or property, so the data it needs was never touched",
    "wrong_graph": "The question is about a different kind of thing or a different data module than the one queried",
    "no_data": "The query is right; the graph simply has no matching data, so the true answer is none / zero",
}


@dataclass
class Verdict:
    action: str  # "accept" | "accept_empty" | "repair" | "next_route" | "fallback"
    reason: str
    answers: float = 0.0  # P(result answers the question)
    suspect: str | None = None  # plan slot to repair
    checks: dict[str, Any] = field(default_factory=dict)
    empty_reason: str | None = None


def _type_ok(values: list[Any], plan: QueryPlan) -> bool | None:
    if not plan.answer_path or plan.shape not in ("lookup", "path") or not values:
        return None
    dt_ = plan.answer_path.datatype
    if plan.answer_path.dst:  # entity-valued
        return all(isinstance(v, IRI) for v in values)
    if is_numeric_type(dt_):
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    if is_date_type(dt_):
        return all(isinstance(v, (dt.date, int)) for v in values)
    return True


class Verifier:
    def __init__(self, cat: SchemaCatalog, controller: RecordingController, store: Store, gates: Gates, limits: Limits) -> None:
        self.cat = cat
        self.controller = controller
        self.store = store
        self.gates = gates
        self.limits = limits

    def _labels(self, values: list[Any], labeler) -> list[str]:
        return [labeler(v) for v in values[:10]]

    def _probe(self, plan: QueryPlan, program: Program, res: ExecResult, relax: Callable[[], int | None] | None) -> dict[str, Any]:
        """Evidence for why a result is empty."""
        ev: dict[str, Any] = {"empty_at_step": res.empty_at}
        step = next((s for s in program.steps if s.name == res.empty_at), None)
        if step:
            ev["empty_step_purpose"] = step.purpose
        try:
            for i, link in enumerate(plan.links):
                first = link.path.steps[-1]  # the step touching the anchor
                g = self.cat.modules[first.module].graph
                pat = f"?o <{first.prop}> <{link.anchor.iri}>" if not first.inverse else f"<{link.anchor.iri}> <{first.prop}> ?o"
                ev[f"anchor_{i}_has_relation"] = self.store.ask(f"ASK {{ GRAPH <{g}> {{ {pat} }} }}", timeout=self.limits.query_timeout_s)
            if plan.subject and plan.answer_path:
                s0 = plan.answer_path.steps[0]
                g = self.cat.modules[s0.module].graph
                pat = f"<{plan.subject.iri}> <{s0.prop}> ?o" if not s0.inverse else f"?o <{s0.prop}> <{plan.subject.iri}>"
                ev["subject_has_relation"] = self.store.ask(f"ASK {{ GRAPH <{g}> {{ {pat} }} }}", timeout=self.limits.query_timeout_s)
            if plan.filters and relax is not None:
                n = relax()
                ev["matches_without_value_filters"] = n
                ev["empty_only_after_filters"] = bool(n)
        except QueryError as e:
            ev["probe_error"] = str(e)
        return ev

    def _empty_prior(self, ev: dict[str, Any]) -> dict[str, float]:
        if ev.get("subject_has_relation") is False or any(v is False for k, v in ev.items() if k.startswith("anchor_")):
            return {"wrong_property": 0.7, "wrong_graph": 0.2, "no_data": 0.1}
        if ev.get("empty_only_after_filters"):
            return {"wrong_property": 0.15, "wrong_graph": 0.05, "no_data": 0.8}
        return {"wrong_property": 0.4, "wrong_graph": 0.2, "no_data": 0.4}

    def verify(
        self,
        question: str,
        plan: QueryPlan | None,
        program: Program,
        res: ExecResult,
        labeler: Callable[[Any], str] = str,
        relax: Callable[[], int | None] | None = None,
    ) -> Verdict:
        """`relax` counts matches with the plan's literal filters dropped (a probe for empties)."""
        trace = current_trace()
        if res.error:
            return Verdict("repair" if not res.timeout else "fallback", f"execution error: {res.error}", suspect=plan.weakest_decision if plan else None)
        checks: dict[str, Any] = {"rows": len(res.rows), "values": len(res.values), "truncated": res.truncated}
        empty = res.empty
        if plan is not None:
            checks["type_ok"] = _type_ok(res.values, plan)
            if plan.answer_path and plan.shape in ("lookup", "path") and len(plan.answer_path.steps) == 1:
                p = self.cat.properties.get(plan.answer_path.steps[0].prop)
                if p and p.max_count == 1 and not plan.answer_path.steps[0].inverse:
                    checks["cardinality_ok"] = len(res.values) <= 1
        prior_ok = 0.85
        if empty:
            prior_ok = 0.25
        if checks.get("type_ok") is False or checks.get("cardinality_ok") is False:
            prior_ok = 0.15
        if res.truncated:
            prior_ok = min(prior_ok, 0.5)

        state: dict[str, Any] = {
            "question": question,
            "plan": plan.describe(self.cat) if plan else "LLM-generated query",
            "result": self._labels(res.values, labeler) if res.values else "(empty)",
            "checks": checks,
        }
        questions: dict[str, Any] = {
            "answers": Noul(
                instructions="Does this result answer the question as asked?",
                criteria={"true": "The result is the kind of thing the question asks for and plausibly correct", "false": "The result is the wrong kind of thing, from the wrong relation, or missing"},
                prior=prior_ok,
            )
        }
        ev: dict[str, Any] = {}
        if empty and plan is not None:
            ev = self._probe(plan, program, res, relax)
            state["evidence"] = ev
            questions["empty_reason"] = Choice(criteria=EMPTY_REASONS, instructions="The result is empty. Why?", prior=self._empty_prior(ev))
        resp = self.controller.ask(state, questions, stage="verify")
        answers = resp.noul("answers").noul
        checks.update(ev)
        if trace:
            trace.gate("verify", "yes" if answers >= self.gates.verify else "no", answers, self.gates.verify, {"yes": answers, "no": 1 - answers})

        if empty and plan is not None:
            why = resp.choice("empty_reason")
            if trace:
                trace.gate("empty_reason", why.choice, why.confidence, self.gates.verify, why.probabilities)
            if why.choice == "no_data" and why.confidence >= self.gates.verify:
                return Verdict("accept_empty", "no matching data", answers, checks=checks, empty_reason=why.choice)
            if why.choice == "wrong_graph":
                return Verdict("next_route", "empty: likely wrong graph/class", answers, checks=checks, empty_reason=why.choice)
            suspect = self._suspect(plan, res)
            return Verdict("repair", "empty: likely wrong property", answers, suspect=suspect, checks=checks, empty_reason=why.choice)
        if answers >= self.gates.verify:
            return Verdict("accept", "verified", answers, checks=checks)
        return Verdict("repair", "result judged not to answer the question", answers, suspect=plan.weakest_decision if plan else None, checks=checks)

    def _suspect(self, plan: QueryPlan, res: ExecResult) -> str | None:
        at = res.empty_at or ""
        if at.startswith("link"):
            return f"link_{at[4:].split('_')[0]}"
        if at.startswith("filter"):
            return f"filter_{at[6:].split('_')[0]}"
        if plan.shape in ("lookup", "path"):
            return "answer"
        return plan.weakest_decision
