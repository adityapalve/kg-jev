"""The pipeline: link -> route -> plan -> generate -> execute -> verify, with gates and fallback."""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from typing import Any

from kgqa.build import load_or_build, open_store
from kgqa.config import Config
from kgqa.executor import ExecResult, Executor
from kgqa.executor.program import Program
from kgqa.fallback import FallbackPath
from kgqa.generator import ConstrainedLLMGenerator, GenerationError, TemplateError, TemplateGenerator
from kgqa.jev import RecordingController, make_controller
from kgqa.linking import EntityIndex, EntityLinker, LinkResult
from kgqa.llm import CachingLLM, LLMClient, LLMNotConfigured, NullLLM, make_llm
from kgqa.planner import Planner, QueryPlan
from kgqa.planner.plan import UNSUPPORTED_SHAPE
from kgqa.rdf import IRI, Prefixes, local_name
from kgqa.router import Router
from kgqa.schema import SchemaCatalog, Slicer
from kgqa.store import Store
from kgqa.trace import Trace, tracing
from kgqa.verifier import Verifier


def _jsonable(v: Any) -> Any:
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    return v if isinstance(v, (int, float, bool)) or v is None else str(v)


@dataclass
class Answer:
    question: str
    status: str  # "answered" | "no_data" | "fallback" | "unanswered" | "error"
    values: list[Any] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    shape: str | None = None
    route: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    sparql: list[str] = field(default_factory=list)
    source: str | None = None  # "template" | "llm" | "fallback"
    confidence: float = 0.0
    reason: str | None = None
    entities: list[str] = field(default_factory=list)  # linked entity IRIs
    trace: Trace | None = None

    @property
    def answered(self) -> bool:
        return self.status in ("answered", "no_data", "fallback")

    def text(self) -> str:
        if self.status == "no_data":
            return "No matching data."
        if not self.answered:
            return f"Unanswered ({self.reason})."
        if len(self.labels) == 1:
            return self.labels[0]
        return ", ".join(self.labels)

    def to_json(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "status": self.status,
            "values": [_jsonable(v) for v in self.values],
            "labels": self.labels,
            "shape": self.shape,
            "route": self.route,
            "plan": self.plan,
            "sparql": self.sparql,
            "source": self.source,
            "confidence": self.confidence,
            "reason": self.reason,
            "entities": self.entities,
        }


class Pipeline:
    def __init__(self, cfg: Config, store: Store, catalog: SchemaCatalog, index: EntityIndex, controller: RecordingController, llm: LLMClient | None = None) -> None:
        self.cfg = cfg
        self.store = store
        self.cat = catalog
        self.index = index
        self.controller = controller
        self.llm = llm or NullLLM()
        self.px = Prefixes(cfg.prefixes)
        self.linker = EntityLinker(index, catalog, controller, cfg.gates)
        self.router = Router(catalog, controller, cfg)
        self.planner = Planner(catalog, controller, cfg)
        self.slicer = Slicer(catalog, self.px, cfg.limits.slice_token_budget)
        self.templates = TemplateGenerator(catalog, cfg.limits.query_limit, cfg.hop_mode)
        self.llm_gen = ConstrainedLLMGenerator(self.llm, catalog, self.px)
        self.executor = Executor(store, cfg.limits)
        self.verifier = Verifier(catalog, controller, store, cfg.gates, cfg.limits)
        self.fallback = FallbackPath(catalog, self.llm, self.executor, self.slicer, cfg.path(cfg.review_queue))

    @classmethod
    def from_config(cls, cfg: Config, *, store: Store | None = None, controller: RecordingController | None = None, llm: LLMClient | None = None) -> Pipeline:
        if str(cfg.root) not in sys.path:  # so "python:module:factory" backends resolve next to kgqa.toml
            sys.path.insert(0, str(cfg.root))
        store = store or open_store(cfg)
        catalog, index = load_or_build(cfg, store)
        controller = controller or make_controller(cfg.controller_backend, model=cfg.controller_model, timeout=cfg.controller_timeout, cache=str(cfg.path(cfg.controller_cache)) if cfg.controller_cache else None)
        if llm is None:
            llm = make_llm(cfg.llm_backend, **cfg.llm_options)
            if cfg.llm_cache and not isinstance(llm, NullLLM):
                llm = CachingLLM(llm, cfg.path(cfg.llm_cache))
        return cls(cfg, store, catalog, index, controller, llm)

    # ------------------------------------------------------------------ helpers
    def label(self, v: Any) -> str:
        if isinstance(v, tuple):  # a table row: "Mountain West: 5,320,393.37"
            parts = [self.label(x) for x in v]
            return f"{parts[0]}: {', '.join(parts[1:])}" if len(parts) > 1 else parts[0] if parts else ""
        if isinstance(v, IRI):
            rec = self.index.records.get(v)
            return rec.label if rec else self.cat.label(v) if (v in self.cat.classes or v in self.cat.properties) else local_name(v)
        if isinstance(v, float):
            return f"{v:,.2f}".rstrip("0").rstrip(".")
        if isinstance(v, bool):
            return "yes" if v else "no"
        return str(v)

    def _generate(self, question: str, link: LinkResult, plan: QueryPlan, mode: str, feedback: list[str]) -> tuple[Program | None, str | None, str | None]:
        """Returns (program, source, failure reason)."""
        use_template = mode == "template" or (mode == "auto" and self.templates.supports(plan) and plan.confidence >= self.cfg.gates.plan)
        if use_template:
            try:
                return self.templates.compile(plan), "template", None
            except TemplateError as e:
                if mode == "template":
                    return None, None, str(e)
        if isinstance(self.llm, NullLLM):
            return None, None, "no template applies and no LLM is configured"
        focus = [c for c in [plan.target_class, *(t for a in ([plan.subject] if plan.subject else []) for t in a.types)] if c]
        for p in plan.paths():
            focus += [e.dst for e in p.edges if e.dst]
        sl = self.slicer.slice(focus or [c for c in self.cat.modules[plan.module].top_classes])
        try:
            return self.llm_gen.generate(question, sl, link, plan, feedback), "llm", None
        except (GenerationError, LLMNotConfigured) as e:
            return None, None, f"generation failed: {e}"

    def _relaxed_count(self, plan: QueryPlan) -> int | None:
        if plan.shape not in ("count", "list", "aggregate", "superlative") or not plan.filters:
            return None
        relaxed = dataclasses.replace(plan, shape="count", filters=[], answer_path=None, agg=None, order=None, limit=None)
        try:
            res = self.executor.run(self.templates.compile(relaxed))
        except TemplateError:
            return None
        return res.values[0] if res.values else 0

    def _answer(self, question: str, status: str, trace: Trace, *, link: LinkResult | None = None, res: ExecResult | None = None, program: Program | None = None, plan: QueryPlan | None = None, shape: str | None = None, route: Any = None, source: str | None = None, confidence: float = 0.0, reason: str | None = None) -> Answer:
        values = res.values if res else []
        return Answer(
            question=question,
            status=status,
            values=values,
            labels=[self.label(v) for v in values],
            shape=plan.shape if plan else shape,
            route={"module": route.module, "class": route.klass, "class_label": self.cat.label(route.klass) if route.klass else None, "prob": route.prob} if route else None,
            plan=plan.to_json(self.cat) if plan else None,
            sparql=program.sparql() if program else [],
            source=source,
            confidence=confidence,
            reason=reason,
            entities=[m.chosen.iri for m in link.mentions if m.chosen] if link else [],
            trace=trace,
        )

    def _fallback(self, question: str, link: LinkResult, reason: str, trace: Trace, shape: str | None = None, *, baseline: bool = False) -> Answer:
        trace.event("fallback", reason=reason)
        # if the LLM itself just failed (outage, rate limit, timeout), asking it again only doubles the wait
        llm_down = "LLM call failed" in reason
        with trace.stage("fallback"):
            fr = self.fallback.answer(question, link, reason, use_llm=not llm_down)
        if fr.status == "answered":
            status = "answered" if baseline else "fallback"
            return self._answer(question, status, trace, link=link, res=fr.result, program=fr.program, shape=shape, source="fallback", reason=reason)
        status = "unanswered" if fr.status == "queued" else "error"
        return self._answer(question, status, trace, link=link, program=fr.program, shape=shape, source="fallback", reason=f"{reason}; {fr.error}")

    # ------------------------------------------------------------------ entry points
    def ask(self, question: str, generation: str | None = None) -> Answer:
        mode = generation or self.cfg.generation
        with tracing(question) as trace:
            with trace.stage("linking"):
                link = self.linker.link(question)
            with trace.stage("route"):
                rr = self.router.route(question, link)
            if rr.fallback_reason:
                return self._fallback(question, link, rr.fallback_reason, trace, rr.shape.choice)
            if rr.shape.choice == UNSUPPORTED_SHAPE:
                return self._direct_llm(question, link, rr, trace)
            routes = rr.routes if self.cfg.tie_policy == "try_next" else rr.routes[:1]
            last_reason = "no route produced an answer"
            for ri, route in enumerate(routes):
                with trace.stage("plan"):
                    plan = self.planner.plan(question, link, route, rr.shape.choice)
                passed = trace.gate("plan", plan.shape, plan.confidence, self.cfg.gates.plan, {k: d.confidence for k, d in plan.decisions.items()})
                if not passed and mode != "llm":
                    last_reason = f"plan gate on route {ri} ({self.cat.label(route.klass or '')}): {'; '.join(plan.unresolved) or f'confidence {plan.confidence:.2f}'}"
                    trace.event("plan_rejected", route=ri, reason=last_reason)
                    continue
                feedback: list[str] = []
                for attempt in range(self.cfg.limits.max_repairs + 1):
                    with trace.stage("generate"):
                        program, source, why = self._generate(question, link, plan, mode, feedback)
                    if program is None:
                        last_reason = why or "generation failed"
                        break
                    with trace.stage("execute"):
                        res = self.executor.run(program)
                    with trace.stage("verify"):
                        verdict = self.verifier.verify(question, plan, program, res, self.label, relax=lambda p=plan: self._relaxed_count(p))
                    trace.event("verdict", route=ri, attempt=attempt, action=verdict.action, reason=verdict.reason, suspect=verdict.suspect, source=source)
                    if verdict.action in ("accept", "accept_empty"):
                        status = "answered" if verdict.action == "accept" else "no_data"
                        # the answer is only as sure as its weakest decision
                        conf = min([route.prob, rr.shape.confidence, plan.confidence, verdict.answers, *(m.confidence for m in link.mentions)])
                        return self._answer(question, status, trace, link=link, res=res, program=program, plan=plan, route=route, source=source, confidence=conf, reason=verdict.reason)
                    if verdict.action == "fallback":
                        return self._fallback(question, link, verdict.reason, trace, plan.shape)
                    if verdict.action == "next_route":
                        last_reason = verdict.reason
                        break
                    if source == "llm":
                        feedback = [f"Query:\n{program.sparql()[-1]}", f"Problem: {verdict.reason}. Evidence: {verdict.checks}"]
                        continue
                    repaired = self.planner.repair(plan, verdict.suspect)
                    if repaired is None:
                        last_reason = f"{verdict.reason}; no repair left"
                        break
                    plan = repaired
                else:
                    last_reason = f"repairs exhausted ({self.cfg.limits.max_repairs})"
            return self._fallback(question, link, last_reason, trace, rr.shape.choice)

    def _direct_llm(self, question: str, link: LinkResult, rr: Any, trace: Trace) -> Answer:
        """No template can express this question: the LLM writes SPARQL over the routed schema slice.

        The routing decisions still narrow the schema; no template plan is passed, since forcing the
        question into a template shape is exactly what this path avoids.
        """
        trace.event("direct_llm", reason="shape needs LLM generation")
        if isinstance(self.llm, NullLLM):
            return self._fallback(question, link, "question needs LLM generation (no template shape fits)", trace, UNSUPPORTED_SHAPE)
        focus = [r.klass for r in rr.routes if r.klass] + [t for m in link.mentions for t in m.types]
        route = rr.routes[0] if rr.routes else None
        feedback: list[str] = []
        seen: set[str] = set()
        for attempt in range(self.cfg.limits.max_repairs + 1):
            with trace.stage("generate"):
                sl = self.slicer.slice(focus or list(self.cat.classes)[:4])
                try:
                    program = self.llm_gen.generate(question, sl, link, None, feedback)
                except (GenerationError, LLMNotConfigured) as e:
                    return self._fallback(question, link, f"generation failed: {e}", trace, UNSUPPORTED_SHAPE)
            key = " ".join(program.sparql()[-1].split())
            if key in seen:  # the LLM repeated itself; another round will not change the verdict
                trace.event("repair_stalled", attempt=attempt)
                break
            seen.add(key)
            with trace.stage("execute"):
                res = self.executor.run(program)
            with trace.stage("verify"):
                verdict = self.verifier.verify(question, None, program, res, self.label)
            trace.event("verdict", route=0, attempt=attempt, action=verdict.action, reason=verdict.reason, source="llm")
            if verdict.action == "accept":
                conf = min([rr.shape.confidence, verdict.answers, *(m.confidence for m in link.mentions)])
                return self._answer(question, "answered", trace, link=link, res=res, program=program, shape=UNSUPPORTED_SHAPE, route=route, source="llm", confidence=conf, reason=verdict.reason)
            feedback = [f"Query:\n{program.sparql()[-1]}", f"Problem: {verdict.reason}. Result: {res.values[:5] or 'empty'}. Error: {res.error or 'none'}"]
        return self._fallback(question, link, "LLM query did not verify", trace, UNSUPPORTED_SHAPE)

    def ask_baseline(self, question: str) -> Answer:
        """Arm A: entity linking + LLM with retrieved schema, one query, no controller routing."""
        with tracing(question) as trace:
            with trace.stage("linking"):
                link = self.linker.link(question)
            return self._fallback(question, link, "baseline", trace, baseline=True)
