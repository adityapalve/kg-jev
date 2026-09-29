"""Query planner: turns (question, linked entities, route, shape) into a QueryPlan.

Code enumerates the candidate schema paths for every slot (how each anchor connects to the
target, which property each literal constrains, which value to measure); jev picks among them.
All slot questions go in one call and are answered independently, then combined here in code.
Repair re-materializes the plan with the next-best option for a suspect slot, reusing the
probabilities already returned, so it costs no extra controller call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kgqa.config import Config
from kgqa.jev import Choice, ChoiceResult, RecordingController, certain
from kgqa.linking import LinkResult
from kgqa.linking.literals import LiteralMention
from kgqa.planner.plan import AGGS, OPS, ORDERS, Anchor, Filter, Link, QueryPlan
from kgqa.rdf import is_date_type, is_numeric_type
from kgqa.schema.cards import SchemaCatalog
from kgqa.schema.paths import SchemaGraph, SchemaPath
from kgqa.text import BM25, stems, tokens
from kgqa.trace import current_trace

if TYPE_CHECKING:
    from kgqa.router.router import Route

NONE = "none"


@dataclass
class Slot:
    name: str
    options: dict[str, Any]  # option key -> value (SchemaPath, op, Anchor, ...)
    criteria: dict[str, str]
    instructions: str
    prior: dict[str, float] | None = None
    allow_none: bool = True
    result: ChoiceResult | None = None
    tried: set[str] = field(default_factory=set)

    none_text: str = "None of these options fits the question"

    def question(self) -> Choice:
        crit = dict(self.criteria)
        prior = dict(self.prior) if self.prior else None
        if self.allow_none:
            crit[NONE] = self.none_text
            if prior is not None:
                prior[NONE] = 0.05
        return Choice(criteria=crit, instructions=self.instructions, prior=prior)


@dataclass
class PlanContext:
    question: str
    shape: str
    route: Route
    anchors: list[Anchor]
    literals: list[LiteralMention]
    slots: dict[str, Slot] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


def _window_prior(question: str, span: tuple[int, int], texts: dict[str, str], width: int = 6) -> dict[str, float]:
    """Offline-only prior: options sharing words with the few words right before a literal."""
    before = stems(question[: span[0]])[-width:]
    after = stems(question[span[1] :])[:2]
    near = set(before + after)
    out = {}
    for k, t in texts.items():
        overlap = len(near & set(stems(t)))
        out[k] = 1.0 + 2.0 * overlap
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def _op_prior(question: str, lit: LiteralMention) -> dict[str, float]:
    """Offline-only prior from comparison words just before the literal."""
    window = tokens(question[: lit.span[0]], keep_stopwords=True)[-5:]
    text = " ".join(window)
    ws = set(window)
    prior = {k: 0.05 for k in OPS}
    if "at least" in text or ws & {"since", "from"}:
        op = "ge"
    elif "at most" in text or "up to" in text or ws & {"until"}:
        op = "le"
    elif ws & {"over", "above", "after", "exceeding", "beyond"} or ("than" in ws and ws & {"more", "greater", "higher", "larger", "bigger", "later", "newer", "expensive", "longer"}):
        op = "gt"
    elif ws & {"under", "below", "before"} or ("than" in ws and ws & {"less", "fewer", "lower", "smaller", "earlier", "older", "cheaper", "shorter"}):
        op = "lt"
    else:
        op = "eq"
    prior[op] = 0.8
    return prior


def _length_prior(paths: dict[str, SchemaPath]) -> dict[str, float]:
    raw = {k: 0.6 ** (len(p.edges) - 1) for k, p in paths.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


class Planner:
    def __init__(self, cat: SchemaCatalog, controller: RecordingController, cfg: Config, max_options: int = 60) -> None:
        self.cat = cat
        self.graph = SchemaGraph(cat)
        self.controller = controller
        self.cfg = cfg
        self.max_options = max_options

    # ------------------------------------------------------------------ candidate slots
    def _path_slot(self, name: str, question: str, paths: list[SchemaPath], instructions: str, prior: dict[str, float] | None = None) -> Slot:
        uniq: dict[str, SchemaPath] = {}
        for p in paths:
            uniq.setdefault(p.key(), p)
        if len(uniq) > self.max_options:
            keys = list(uniq)
            bm = BM25([stems(uniq[k].lexical_text(self.cat)) for k in keys])
            ranked = sorted(range(len(keys)), key=lambda i: (-bm.score(stems(question), i), len(uniq[keys[i]].edges)))
            uniq = {keys[i]: uniq[keys[i]] for i in ranked[: self.max_options]}
        # Readable keys and sentence descriptions: jev reads these. Keys map back to paths here.
        readable: dict[str, SchemaPath] = {}
        for k, p in uniq.items():
            label = p.label(self.cat)
            readable[label if label not in readable else f"{label} [{k}]"] = p
        uniq = readable
        criteria = {k: p.explain(self.cat) for k, p in uniq.items()}
        base = _length_prior(uniq)
        if prior:
            base = {k: base[k] * prior.get(k, 1.0) for k in base}
            total = sum(base.values())
            base = {k: v / total for k, v in base.items()}
        return Slot(name, dict(uniq), criteria, instructions, base)

    def _anchor_class(self, a: Anchor) -> str | None:
        known = [t for t in a.types if t in self.cat.classes]
        # most specific type first
        known.sort(key=lambda t: -len(self.cat.ancestors(t)))
        return known[0] if known else None

    def _literal_paths(self, cls: str, lit: LiteralMention) -> list[SchemaPath]:
        attrs = self.graph.value_paths(cls, 2, attributes_only=True)
        out = []
        for p in attrs:
            dt = p.datatype
            is_year_prop = is_numeric_type(dt) and "year" in (p.edges[-1].prop.lower())
            if lit.kind == "year" and (is_year_prop or is_date_type(dt)):
                out.append(p)
            elif lit.kind in ("date_range", "date") and is_date_type(dt):
                out.append(p)
            elif lit.kind == "number" and is_numeric_type(dt) and not is_year_prop:
                out.append(p)
            elif lit.kind == "string" and p.edges[-1].prop == lit.prop:
                out.append(p)
        return out

    def _slots_for_target(self, ctx: PlanContext, target: str, anchors: list[Anchor]) -> None:
        q = ctx.question
        for i, a in enumerate(anchors):
            a_cls = self._anchor_class(a)
            if a_cls is None:
                ctx.unresolved.append(f"no type for {a.label}")
                continue
            if self.cat.is_a(a_cls, target) and ctx.shape != "boolean":
                ctx.notes.append(f"{a.label} is itself a {self.cat.label(target)}")
            paths = self.graph.paths_between(target, a_cls, self.cfg.limits.max_path_edges)
            if not paths:
                ctx.unresolved.append(f"no schema path from {self.cat.label(target)} to {a.label}")
                continue
            slot = self._path_slot(f"link_{i}", q, paths, f'The question is about {self.cat.label(target)} records connected to "{a.label}". Which connection does it mean?')
            slot.none_text = f'The question does not connect {self.cat.label(target)} records to "{a.label}" in any of these ways'
            ctx.slots[f"link_{i}"] = slot
            ctx.fixed[f"link_{i}"] = a
        for j, lit in enumerate(ctx.literals):
            paths = self._literal_paths(target, lit)
            if not paths:
                ctx.unresolved.append(f"no property of {self.cat.label(target)} fits the value {lit.text}")
                continue
            slot = self._path_slot(f"filter_{j}", q, paths, f'Which property of the {self.cat.label(target)} does the value "{lit.text}" constrain?')
            win = _window_prior(q, lit.span, slot.criteria)
            slot.prior = {k: slot.prior[k] * win[k] for k in slot.prior}
            ctx.slots[f"filter_{j}"] = slot
            ctx.fixed[f"filter_{j}"] = lit
            if lit.kind == "string":
                continue  # a named value is an equality constraint
            ctx.slots[f"op_{j}"] = Slot(f"op_{j}", {k: k for k in OPS}, dict(OPS), f'How does the question compare the value to "{lit.text}"?', _op_prior(q, lit), allow_none=False)

    def _measure_slot(self, ctx: PlanContext, cls: str, temporal: bool = False) -> None:
        paths = self.graph.value_paths(cls, 2, numeric_only=True)
        if temporal:
            paths += self.graph.value_paths(cls, 1, temporal_only=True)
        paths = [p for p in paths if "year" not in p.edges[-1].prop.lower() or temporal]
        if not paths:
            ctx.unresolved.append(f"no numeric value on {self.cat.label(cls)}")
            return
        ctx.slots["measure"] = self._path_slot("measure", ctx.question, paths, f"Which value of the {self.cat.label(cls)} does the question measure or rank by?")

    def build_context(self, question: str, link: LinkResult, route: Route, shape: str) -> PlanContext:
        anchors = [Anchor(m.chosen.iri, m.chosen.label, tuple(m.types), m.text) for m in link.mentions if m.chosen]
        ctx = PlanContext(question, shape, route, anchors, list(link.literals))
        if shape in ("lookup", "path"):
            if not anchors:
                ctx.unresolved.append("lookup without a named entity")
                return ctx
            if len(anchors) > 1:
                ctx.slots["subject"] = Slot("subject", {a.label: a for a in anchors}, {a.label: f"{a.text}: {a.label}" for a in anchors}, "Which named entity is the question asking about?", {a.label: (0.6 if i == 0 else 0.4 / (len(anchors) - 1)) for i, a in enumerate(anchors)}, allow_none=False)
            subject = anchors[0]
            s_cls = self._anchor_class(subject)
            if s_cls is None:
                ctx.unresolved.append(f"no type for {subject.label}")
                return ctx
            paths = self.graph.value_paths(s_cls, 2)
            ctx.slots["answer"] = self._path_slot("answer", question, paths, f'What does the question ask about "{subject.label}"?')
        elif shape in ("count", "list", "aggregate", "superlative"):
            target = route.klass
            if target is None:
                ctx.unresolved.append("no target class")
                return ctx
            self._slots_for_target(ctx, target, anchors)
            if shape == "aggregate":
                self._measure_slot(ctx, target)
                ctx.slots["agg"] = Slot("agg", {k: k for k in AGGS}, dict(AGGS), "Which aggregate does the question ask for?", allow_none=False)
            elif shape == "superlative":
                self._measure_slot(ctx, target, temporal=True)
                ctx.slots["order"] = Slot("order", {k: k for k in ORDERS}, dict(ORDERS), "Does the question want the highest or the lowest value?", allow_none=False)
        elif shape == "compare":
            if len(anchors) < 2:
                ctx.unresolved.append("compare needs two named entities")
                return ctx
            cls = self._anchor_class(anchors[0])
            if cls is None:
                ctx.unresolved.append("no type for compared entities")
                return ctx
            self._measure_slot(ctx, cls, temporal=True)
            ctx.slots["order"] = Slot("order", {k: k for k in ORDERS}, dict(ORDERS), "Which one does the question want: the one with the higher or the lower value?", allow_none=False)
        elif shape == "boolean":
            if not anchors:
                ctx.unresolved.append("yes/no question without a named entity")
                return ctx
            subject = anchors[0]
            s_cls = self._anchor_class(subject)
            if s_cls is None:
                ctx.unresolved.append(f"no type for {subject.label}")
                return ctx
            self._slots_for_target(ctx, s_cls, anchors[1:])
            if not anchors[1:] and not ctx.literals:
                ctx.unresolved.append("yes/no question with nothing to check")
        else:
            ctx.unresolved.append(f"unsupported shape {shape}")
        return ctx

    # ------------------------------------------------------------------ plan
    def plan(self, question: str, link: LinkResult, route: Route, shape: str) -> QueryPlan:
        ctx = self.build_context(question, link, route, shape)
        asked = {name: s for name, s in ctx.slots.items() if len(s.options) > 1 or s.allow_none}
        for name, s in ctx.slots.items():
            if name not in asked:
                only = next(iter(s.options))
                s.result = certain(only)
        if asked:
            state = {
                "question": question,
                "entities": [f"{a.text} = {a.label}" for a in ctx.anchors],
                "values": [l.describe() for l in ctx.literals],
                "target": self.cat.label(route.klass) if route.klass else None,
            }
            resp = self.controller.ask(state, {n: s.question() for n, s in asked.items()}, stage="plan")
            for n, s in asked.items():
                s.result = resp.choice(n)
        return self.materialize(ctx, {})

    def materialize(self, ctx: PlanContext, overrides: dict[str, str]) -> QueryPlan:
        route = ctx.route
        plan = QueryPlan(shape=ctx.shape, module=route.module, target_class=route.klass, overrides=dict(overrides), notes=list(ctx.notes), unresolved=list(ctx.unresolved), context=ctx)

        def pick(name: str) -> Any:
            s = ctx.slots[name]
            key = overrides.get(name, s.result.choice)
            plan.decisions[name] = s.result
            if key == NONE:
                plan.unresolved.append(f"{name}: none of the candidates fit")
                return None
            s.tried.add(key)
            return s.options[key]

        if ctx.shape in ("lookup", "path"):
            if ctx.unresolved:
                return plan
            subject = pick("subject") if "subject" in ctx.slots else ctx.anchors[0]
            plan.subject = subject
            plan.target_class = self._anchor_class(subject)
            plan.module = self.cat.classes[plan.target_class].module if plan.target_class else plan.module
            plan.answer_path = pick("answer")
            return plan

        if ctx.shape == "compare":
            if ctx.unresolved:
                return plan
            cls = self._anchor_class(ctx.anchors[0])
            plan.target_class = cls
            plan.module = self.cat.classes[cls].module if cls else plan.module
            plan.compare = [a for a in ctx.anchors if self._anchor_class(a) and cls and (self.cat.is_a(self._anchor_class(a), cls) or self.cat.is_a(cls, self._anchor_class(a)))]
            plan.answer_path = pick("measure")
            plan.order = pick("order")
            return plan

        if ctx.shape == "boolean" and ctx.anchors:
            plan.subject = ctx.anchors[0]
            plan.target_class = self._anchor_class(ctx.anchors[0])
            if plan.target_class:
                plan.module = self.cat.classes[plan.target_class].module
        for name in sorted(ctx.slots):
            if name.startswith("link_"):
                path = pick(name)
                if path is not None:
                    plan.links.append(Link(ctx.fixed[name], path))
            elif name.startswith("filter_"):
                path = pick(name)
                j = name.split("_")[1]
                op = pick(f"op_{j}") if f"op_{j}" in ctx.slots else "eq"
                if path is not None:
                    plan.filters.append(Filter(path, op, ctx.fixed[name]))
        if ctx.shape == "aggregate" and "measure" in ctx.slots:
            plan.answer_path = pick("measure")
            plan.agg = pick("agg")
        if ctx.shape == "superlative" and "measure" in ctx.slots:
            plan.answer_path = pick("measure")
            plan.order = pick("order")
            plan.limit = 1
        return plan

    # ------------------------------------------------------------------ repair
    def repair(self, plan: QueryPlan, suspect: str | None = None) -> QueryPlan | None:
        """Swap the suspect slot (or the least confident one) for its next-best untried option."""
        ctx: PlanContext = plan.context
        if ctx is None:
            return None
        order = [suspect] if suspect in ctx.slots else []
        order += sorted((n for n in ctx.slots if n not in order), key=lambda n: ctx.slots[n].result.confidence if ctx.slots[n].result else 1.0)
        for name in order:
            s = ctx.slots[name]
            if s.result is None:
                continue
            for key, p in s.result.ranked():
                if key == NONE or key in s.tried or key not in s.options or p < 0.02:
                    continue
                trace = current_trace()
                if trace:
                    trace.event("repair", slot=name, previous=plan.overrides.get(name, s.result.choice), next=key, prob=p)
                return self.materialize(ctx, {**plan.overrides, name: key})
        return None
