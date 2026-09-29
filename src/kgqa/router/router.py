"""Router: which module (named graph) and which class the question is about.

The first call asks three independent questions: module, query shape, and (when the whole class
list fits in one Choice) a flat class choice. Code combines module and flat-class probabilities,
so the two views check each other. When that combined answer is confident the route is final;
otherwise the router descends the class hierarchy module by module (module, class, subclass) with
a beam, which is also the path used when the ontology has more classes than one Choice allows.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextvars
from dataclasses import dataclass, field

from kgqa.config import Config
from kgqa.jev import Choice, ChoiceResult, RecordingController
from kgqa.jev.large import choose_large
from kgqa.jev.questions import MAX_CHOICE_OPTIONS
from kgqa.linking import LinkResult
from kgqa.planner.plan import ANCHOR_SHAPES, SHAPES
from kgqa.schema.cards import SchemaCatalog
from kgqa.text import normalize
from kgqa.trace import current_trace

OTHER = "other"
ANY = "any"
_EPS = 0.05


@dataclass
class Route:
    module: str
    klass: str | None
    prob: float
    path: list[str] = field(default_factory=list)  # class chain, top-down


@dataclass
class RouteResult:
    routes: list[Route]
    shape: ChoiceResult
    module: ChoiceResult  # combined module distribution
    klass: ChoiceResult | None = None  # flat class distribution, when asked
    fallback_reason: str | None = None


def shape_prior(question: str, link: LinkResult) -> dict[str, float]:
    """Offline-only prior from surface cues. Never sent to jev."""
    q = f" {normalize(question)} "
    words = q.split()
    first = words[0] if words else ""
    prior = {k: 1.0 for k in SHAPES}
    n_ent = len(link.mentions)
    if first in {"is", "does", "did", "was", "are", "has", "have", "were", "do", "can"}:
        prior["boolean"] = 8.0
    if "how many" in q or "number of" in q:
        prior["count"] = 8.0
    if any(f" {w} " in q for w in ("total", "sum", "average", "mean", "overall", "combined")):
        prior["aggregate"] = 6.0
    if any(f" {w} " in q for w in ("most", "least", "highest", "lowest", "cheapest", "newest", "oldest", "biggest", "largest", "smallest", "fastest", "earliest", "latest", "first", "last")):
        prior["superlative"] = 4.0
    if any(f" {w} " in q for w in ("per", "each", "except", "not", "without", "never", "breakdown", "difference", "ratio", "percentage", "percent")) or " by each " in q or any(f" top {n} " in q for n in "23456789"):
        prior["unsupported"] = 30.0  # "total revenue per region" is a breakdown, not one aggregate
    if n_ent == 0:
        for k in ("lookup", "path", "compare", "boolean"):
            prior[k] *= 0.1
    if n_ent >= 2 and any(f" {w} " in q for w in ("or", "than", "vs", "versus")):
        prior["compare"] *= 4.0
    if first in {"which", "list", "show", "name"} and max(prior.values()) == 1.0 and n_ent == 0:
        prior["list"] *= 3.0
    if n_ent >= 1 and first in {"what", "who", "where", "when", "which"} and max(prior.values()) == 1.0:
        prior["lookup"] *= 2.5
        prior["path"] *= 1.5
    total = sum(prior.values())
    return {k: v / total for k, v in prior.items()}


def _normalize(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values()) or 1.0
    return {k: v / total for k, v in d.items()}


def _result(probs: dict[str, float]) -> ChoiceResult:
    best = max(probs, key=probs.__getitem__)
    return ChoiceResult(best, probs, probs[best])


class Router:
    def __init__(self, cat: SchemaCatalog, controller: RecordingController, cfg: Config) -> None:
        self.cat = cat
        self.controller = controller
        self.cfg = cfg
        self.gates = cfg.gates
        self._class_keys = self._class_labels(list(cat.classes))

    # ---- option text ----
    def _module_criteria(self) -> dict[str, str]:
        crit = {name: f"{m.description}. Classes: {', '.join(self.cat.label(c) for c in m.classes)}" for name, m in self.cat.modules.items()}
        crit[OTHER] = "None of these data modules holds the answer"
        return crit

    def _class_labels(self, classes: list[str]) -> dict[str, str]:
        """Unique, readable option keys for classes (label, disambiguated by module if needed)."""
        keys: dict[str, str] = {}
        for c in classes:
            card = self.cat.classes[c]
            key = card.label if card.label not in keys.values() else f"{card.label} ({card.module})"
            keys[c] = key
        return keys

    def _class_text(self, c: str) -> str:
        card = self.cat.classes[c]
        alts = f" Also called: {', '.join(card.alt_labels)}." if card.alt_labels else ""
        kinds = self.cat.descendants(c)
        kinds_s = f" Kinds: {', '.join(self.cat.label(k) for k in kinds)}." if kinds else ""
        # own properties only: repeating inherited ones makes every subclass look like its parent
        own = [self.cat.properties[p].label for p in card.properties if p in self.cat.properties][:12]
        has = f" Has: {', '.join(own)}." if own else f" Same properties as {', '.join(self.cat.label(p) for p in card.parents)}." if card.parents else ""
        return f"{card.description}{alts}{kinds_s}{has} In the {card.module} module."

    def _class_prior(self, classes: list[str], link: LinkResult, shape_probs: dict[str, float]) -> dict[str, float]:
        """Offline-only prior: when counting/listing, the target is rarely the anchor's own type."""
        anchor_types = {t for m in link.mentions for t in m.types}
        set_like = sum(shape_probs.get(s, 0) for s in ("count", "list", "aggregate", "superlative"))
        # classes that own a mentioned value ("Denver" is a dealer's city) are likely targets
        value_domains = {d for l in link.literals if l.prop in self.cat.properties for d in self.cat.properties[l.prop].domains}
        prior = {}
        for c in classes:
            w = 1.0 - 0.7 * set_like if c in anchor_types else 1.0
            if c in value_domains:
                w *= 3.0
            prior[self._class_keys[c]] = w
        return prior

    def _module_prior(self, link: LinkResult) -> dict[str, float]:
        """Offline-only prior: the module that owns a mentioned value is a likely home."""
        owners = {self.cat.classes[d].module for l in link.literals if l.prop in self.cat.properties for d in self.cat.properties[l.prop].domains if d in self.cat.classes}
        prior = {m: (3.0 if m in owners else 1.0) for m in self.cat.modules}
        prior[OTHER] = 1.0
        return prior

    def describe_literal(self, lit) -> str:
        if lit.kind == "string" and lit.prop in self.cat.properties:
            p = self.cat.properties[lit.prop]
            owners = ", ".join(self.cat.label(d) for d in p.domains)
            return f'{lit.text} = "{lit.value}", a {p.label} of a {owners}'
        return lit.describe()

    # ---- hierarchical descent ----
    def _pick_class(self, question: str, options: list[str], parent: str | None, module: str, flat: dict[str, float] | None) -> ChoiceResult:
        keys = self._class_labels(options)
        crit = {keys[c]: self._class_text(c) for c in options}
        if parent:
            crit[ANY] = f"Any {self.cat.label(parent)}, not one specific kind"
        instr = f"Which kind of {self.cat.label(parent)} is the question about?" if parent else "Which kind of thing is the question about (the thing being counted, listed, aggregated or described)?"
        res = choose_large(lambda s, q: self.controller.ask(s, q, stage="route/class"), {"question": question, "module": module}, crit, instr, abstain_key=ANY if parent else OTHER)
        back = {v: k for k, v in keys.items()}
        probs = {back.get(k, k): v for k, v in res.probabilities.items()}
        if flat:  # fold in the flat class evidence for each subtree
            probs = _normalize({k: v * (_EPS + sum(flat.get(d, 0.0) for d in ([k, *self.cat.descendants(k)] if k in self.cat.classes else [parent] if parent else []))) for k, v in probs.items()})
        return _result(probs)

    def _descend(self, question: str, module: str, p_module: float, flat: dict[str, float] | None) -> list[Route]:
        top = self.cat.modules[module].top_classes
        if not top:
            return []
        level = self._pick_class(question, top, None, module, flat)
        trace = current_trace()
        if trace:
            trace.gate(f"class:{module}", self.cat.label(level.choice), level.confidence, self.gates.klass, {self.cat.label(k): v for k, v in level.probabilities.items()})
        routes = []
        for cls, p in level.ranked():
            if cls == OTHER or p < self.cfg.beam_min_prob:
                continue
            r = Route(module, cls, p_module * p, [cls])
            while r.klass and self.cat.classes[r.klass].children:
                sub = self._pick_class(question, self.cat.classes[r.klass].children, r.klass, module, flat)
                if sub.choice in (ANY, r.klass) or sub.confidence < self.gates.klass:
                    break
                r = Route(module, sub.choice, r.prob * sub.confidence, [*r.path, sub.choice])
            routes.append(r)
        return routes

    # ---- entry point ----
    def route(self, question: str, link: LinkResult) -> RouteResult:
        entities = [f"{m.text} = {m.chosen.label} ({', '.join(self.cat.label(t) for t in m.types)}, {m.chosen.module} graph)" for m in link.mentions if m.chosen]
        state = {"question": question, "entities": entities, "values": [self.describe_literal(l) for l in link.literals]}
        sp = shape_prior(question, link)
        classes = list(self.cat.classes)
        questions = {
            "module": Choice(criteria=self._module_criteria(), instructions="Which data module holds the answer to the question?", prior=self._module_prior(link)),
            "shape": Choice(criteria=SHAPES, instructions="What kind of query does the question need?", prior=sp),
        }
        flat_asked = len(classes) < MAX_CHOICE_OPTIONS
        if flat_asked:
            crit = {self._class_keys[c]: self._class_text(c) for c in classes}
            crit[OTHER] = "None of these kinds of thing"
            questions["class"] = Choice(criteria=crit, instructions="Which kind of thing is the question about (the thing being counted, listed, aggregated or described)?", prior=self._class_prior(classes, link, sp))
        resp = self.controller.ask(state, questions, stage="route")
        shape = resp.choice("shape")
        trace = current_trace()
        if trace:
            trace.gate("shape", shape.choice, shape.confidence, 0.0, shape.probabilities)

        flat: dict[str, float] | None = None
        flat_res: ChoiceResult | None = None
        if flat_asked:
            back = {v: k for k, v in self._class_keys.items()}
            flat_res = resp.choice("class")
            flat = {back[k]: v for k, v in flat_res.probabilities.items() if k in back}

        # Anchor-centred questions are routed by the anchor's type; no class search needed.
        anchors = [m for m in link.mentions if m.chosen and m.types]
        if shape.choice in ANCHOR_SHAPES and anchors:
            klass = anchors[0].types[0]
            module = self.cat.classes[klass].module if klass in self.cat.classes else anchors[0].chosen.module
            if trace:
                trace.gate("module", module, 1.0, self.gates.module, {module: 1.0})
            return RouteResult([Route(module, klass, 1.0, [klass])], shape, resp.choice("module"), flat_res)

        raw_mod = resp.choice("module")
        if flat:
            mod_probs = {m: p * (_EPS + sum(flat.get(c, 0.0) for c in self.cat.modules[m].classes)) if m != OTHER else p * (_EPS + flat_res.probabilities.get(OTHER, 0.0)) for m, p in raw_mod.probabilities.items()}
            mod = _result(_normalize(mod_probs))
        else:
            mod = raw_mod
        ok = mod.choice != OTHER and mod.confidence >= self.gates.module
        if trace:
            trace.gate("module", mod.choice, mod.confidence, self.gates.module, mod.probabilities, passed=ok)
        if not ok:
            return RouteResult([], shape, mod, flat_res, fallback_reason=f"module gate: {mod.choice} @ {mod.confidence:.2f}")

        # Fast path: the flat class answer agrees with the module and is confident on its own.
        if flat and flat_res.choice != OTHER:
            best = max(flat, key=flat.__getitem__)
            in_module = {c: p for c, p in flat.items() if self.cat.classes[c].module == mod.choice}
            share = flat[best] / (sum(in_module.values()) or 1.0)
            if self.cat.classes[best].module == mod.choice and share >= 0.5 and flat[best] >= self.gates.klass:
                if trace:
                    trace.gate(f"class:{mod.choice}", self.cat.label(best), flat[best], self.gates.klass, {self.cat.label(k): v for k, v in flat.items()})
                routes = [Route(mod.choice, best, mod.confidence * flat[best], [best])]
                for c, p in sorted(in_module.items(), key=lambda kv: -kv[1])[1 : self.cfg.beam_width]:
                    if p >= self.cfg.beam_min_prob:
                        routes.append(Route(mod.choice, c, mod.confidence * p, [c]))
                return RouteResult(routes, shape, mod, flat_res)

        beam = [(m, p) for m, p in mod.ranked() if m != OTHER and p >= self.cfg.beam_min_prob][: self.cfg.beam_width]
        with cf.ThreadPoolExecutor(max_workers=len(beam)) as pool:
            contexts = [contextvars.copy_context() for _ in beam]  # captured here so worker calls land in this trace
            results = list(pool.map(lambda cm: cm[0].run(self._descend, question, cm[1][0], cm[1][1], flat), zip(contexts, beam)))
        routes = sorted((r for rs in results for r in rs), key=lambda r: -r.prob)[: self.cfg.beam_width * 2]
        if not routes:
            return RouteResult([], shape, mod, flat_res, fallback_reason="class gate: no class above beam threshold")
        best_route = routes[0]
        if best_route.prob / max(mod.confidence, 1e-9) < self.gates.klass:
            return RouteResult(routes, shape, mod, flat_res, fallback_reason=f"class gate: {self.cat.label(best_route.klass or '')} @ {best_route.prob:.2f}")
        return RouteResult(routes, shape, mod, flat_res)
