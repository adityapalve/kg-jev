"""Template generator: compiles a QueryPlan into a Program, deterministically.

Constraint paths (links to anchors, literal filters) are split at module boundaries. The segment
next to the target variable ?x runs inside the main query; segments in other modules run first as
their own small queries, walking from the far end (the anchor, or the literal filter) towards ?x,
and hand their results forward as VALUES. With hop_mode="fused" everything runs as one query with
a GRAPH block per module instead, which is the baseline to measure the hop strategy against.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from kgqa.executor.program import Program, QueryStep
from kgqa.linking.literals import LiteralMention
from kgqa.planner.plan import Filter, QueryPlan
from kgqa.rdf import IRI, XSD, is_date_type, sparql_term
from kgqa.schema.cards import SchemaCatalog
from kgqa.schema.paths import Step

TEMPLATE_SHAPES = {"lookup", "path", "count", "list", "aggregate", "superlative", "compare", "boolean"}


class TemplateError(ValueError):
    pass


@dataclass
class _Segment:
    module: str
    steps: list[Step]


def _segments(steps: tuple[Step, ...] | list[Step]) -> list[_Segment]:
    out: list[_Segment] = []
    for s in steps:
        if out and out[-1].module == s.module:
            out[-1].steps.append(s)
        else:
            out.append(_Segment(s.module, [s]))
    return out


def _triple(a: str, s: Step, b: str) -> str:
    return f"{b} <{s.prop}> {a} ." if s.inverse else f"{a} <{s.prop}> {b} ."


class _Vars:
    def __init__(self) -> None:
        self._n = itertools.count()

    def new(self, prefix: str = "v") -> str:
        return f"?{prefix}{next(self._n)}"


def filter_expr(var: str, f: Filter) -> str:
    lit: LiteralMention = f.literal
    dt = f.path.datatype
    op = f.op
    if lit.kind == "string":
        return f"STR({var}) = {sparql_term(str(lit.value))}"
    if lit.kind in ("year", "date_range", "date") and is_date_type(dt):
        lo, hi = lit.low, lit.high
        if dt == XSD + "dateTime":
            lo_t, hi_t = f'"{lo.isoformat()}T00:00:00"^^<{XSD}dateTime>', f'"{hi.isoformat()}T23:59:59"^^<{XSD}dateTime>'
        else:
            lo_t, hi_t = sparql_term(lo), sparql_term(hi)
        return {
            "eq": f"{var} >= {lo_t} && {var} <= {hi_t}",
            "gt": f"{var} > {hi_t}",
            "ge": f"{var} >= {lo_t}",
            "lt": f"{var} < {lo_t}",
            "le": f"{var} <= {hi_t}",
        }[op]
    value = lit.value if lit.value is not None else lit.low
    sym = {"eq": "=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}[op]
    return f"{var} {sym} {sparql_term(value)}"


class TemplateGenerator:
    def __init__(self, cat: SchemaCatalog, query_limit: int = 1000, hop_mode: str = "app") -> None:
        self.cat = cat
        self.limit = query_limit
        self.hop_mode = hop_mode

    def supports(self, plan: QueryPlan) -> bool:
        if plan.shape not in TEMPLATE_SHAPES or plan.unresolved:
            return False
        if plan.shape in ("lookup", "path"):
            return plan.subject is not None and plan.answer_path is not None
        if plan.shape in ("aggregate", "superlative", "compare"):
            return plan.answer_path is not None
        return True

    def graph(self, module: str) -> str:
        return self.cat.modules[module].graph

    def _type_block(self, plan: QueryPlan, x: str) -> str:
        cls = plan.target_class
        mc = self.cat.modules[plan.module]
        classes = [cls, *self.cat.descendants(cls)] if cls else []
        values = " ".join(f"<{c}>" for c in classes)
        return f"GRAPH <{mc.graph}> {{ VALUES ?cls {{ {values} }} {x} <{mc.type_predicate}> ?cls . }}"

    def _chain(self, start: str, steps: list[Step], vars: _Vars, end: str | None = None) -> tuple[str, str]:
        """Triple patterns from `start` along `steps`; returns (patterns, end variable)."""
        pats, cur = [], start
        for i, s in enumerate(steps):
            nxt = end if (end and i == len(steps) - 1) else vars.new()
            pats.append(_triple(cur, s, nxt))
            cur = nxt
        return " ".join(pats), cur

    def _constraint(self, x: str, steps: tuple[Step, ...], terminal: tuple, plan: QueryPlan, vars: _Vars, prog: Program, label: str) -> str:
        """Main-query patterns constraining ?x; earlier walk steps are appended to `prog`.

        terminal: ("values", binding_name) or ("filter", Filter).
        """
        segs = _segments(steps)
        fused_n = len(segs) if self.hop_mode == "fused" else (1 if segs and segs[0].module == plan.module else 0)
        fused, walked = segs[:fused_n], segs[fused_n:]
        term_kind, term = terminal
        carry: str | None = term if term_kind == "values" else None
        # walk far -> near through segments outside the main query
        for k, seg in reversed(list(enumerate(walked))):
            near = vars.new("n")
            far_filter = ""
            if carry is None:  # the far-most segment holds the literal filter
                pats, far = self._chain(near, seg.steps, vars)
                far_filter = f" FILTER({filter_expr(far, term)})"
                values = ""
                inputs: list[str] = []
            else:
                far = vars.new("f")
                pats, _ = self._chain(near, seg.steps, vars, end=far)
                values = f"VALUES {far} {{ {{{{{carry}}}}} }} "
                inputs = [carry]
            name = f"{label}_hop{k}"
            sparql = f"SELECT DISTINCT {near} WHERE {{ GRAPH <{self.graph(seg.module)}> {{ {values}{pats}{far_filter} }} }} LIMIT {self.limit * 10}"
            prog.steps.append(QueryStep(name, sparql, "select", near.lstrip("?"), inputs, chunkable=bool(inputs), module=seg.module, purpose=f"{label}: hop in {seg.module}"))
            carry = name
        blocks = []
        cur = x
        for i, seg in enumerate(fused):
            last = i == len(fused) - 1
            if last and carry is not None and not walked:
                end = vars.new("a")
                pats, _ = self._chain(cur, seg.steps, vars, end=end)
                blocks.append(f"GRAPH <{self.graph(seg.module)}> {{ {pats} VALUES {end} {{ {{{{{carry}}}}} }} }}")
                cur = end
            else:
                pats, cur = self._chain(cur, seg.steps, vars)
                blocks.append(f"GRAPH <{self.graph(seg.module)}> {{ {pats} }}")
        if walked:
            blocks.append(f"VALUES {cur} {{ {{{{{carry}}}}} }}")
        elif term_kind == "filter":
            blocks.append(f"FILTER({filter_expr(cur, term)})")
        return " ".join(blocks)

    def _measure(self, x: str, plan: QueryPlan, vars: _Vars, m: str = "?m") -> str:
        blocks, cur = [], x
        segs = _segments(plan.answer_path.steps)
        for i, seg in enumerate(segs):
            pats, cur = self._chain(cur, seg.steps, vars, end=m if i == len(segs) - 1 else None)
            blocks.append(f"GRAPH <{self.graph(seg.module)}> {{ {pats} }}")
        return " ".join(blocks)

    def _where(self, plan: QueryPlan, prog: Program, vars: _Vars, x: str = "?x") -> str:
        parts = []
        if plan.shape == "boolean":
            prog.bindings["subject"] = [IRI(plan.subject.iri)]
            parts.append(f"VALUES {x} {{ {{{{subject}}}} }}")
        else:
            parts.append(self._type_block(plan, x))
        for i, link in enumerate(plan.links):
            name = f"anchor{i}"
            prog.bindings[name] = [IRI(link.anchor.iri)]
            parts.append(self._constraint(x, link.path.steps, ("values", name), plan, vars, prog, f"link{i}"))
        for j, f in enumerate(plan.filters):
            parts.append(self._constraint(x, f.path.steps, ("filter", f), plan, vars, prog, f"filter{j}"))
        return " ".join(parts)

    def compile(self, plan: QueryPlan) -> Program:
        if not self.supports(plan):
            raise TemplateError(f"no template for plan: {plan.shape} {plan.unresolved}")
        prog = Program(steps=[])
        vars = _Vars()
        shape = plan.shape
        if shape in ("lookup", "path"):
            prog.bindings["subject"] = [IRI(plan.subject.iri)]
            carry = "subject"
            segs = _segments(plan.answer_path.steps)
            for k, seg in enumerate(segs):
                start = vars.new("s")
                pats, end = self._chain(start, seg.steps, vars)
                name = f"hop{k}" if k < len(segs) - 1 else "answer"
                sparql = f"SELECT DISTINCT {end} WHERE {{ GRAPH <{self.graph(seg.module)}> {{ VALUES {start} {{ {{{{{carry}}}}} }} {pats} }} }} LIMIT {self.limit}"
                prog.steps.append(QueryStep(name, sparql, "select", end.lstrip("?"), [carry], chunkable=True, module=seg.module, purpose=f"hop {k + 1} in {seg.module}"))
                carry = name
            prog.post = "column"
            return prog

        if shape == "compare":
            prog.bindings["compared"] = [IRI(a.iri) for a in plan.compare]
            where = f"VALUES ?x {{ {{{{compared}}}} }} " + self._measure("?x", plan, vars)
            prog.steps.append(QueryStep("answer", f"SELECT ?x ?m WHERE {{ {where} }} LIMIT {self.limit}", "select", "x", ["compared"], module=plan.module, purpose="measure compared entities"))
            prog.post = "compare_max" if plan.order == "desc" else "compare_min"
            return prog

        where = self._where(plan, prog, vars)
        inputs = [n for n in prog.bindings] + [s.name for s in prog.steps]
        if shape == "boolean":
            prog.steps.append(QueryStep("answer", f"ASK {{ {where} }}", "ask", None, inputs, module=plan.module, purpose="check fact"))
            prog.post = "ask"
        elif shape == "count":
            prog.steps.append(QueryStep("answer", f"SELECT (COUNT(DISTINCT ?x) AS ?answer) WHERE {{ {where} }}", "select", "answer", inputs, module=plan.module, purpose="count"))
            prog.post = "scalar"
        elif shape == "list":
            prog.steps.append(QueryStep("answer", f"SELECT DISTINCT ?x WHERE {{ {where} }} LIMIT {self.limit}", "select", "x", inputs, module=plan.module, purpose="list"))
            prog.post = "column"
        elif shape == "aggregate":
            fn = {"sum": "SUM", "avg": "AVG", "min": "MIN", "max": "MAX"}[plan.agg or "sum"]
            inner = f"SELECT DISTINCT ?x ?m WHERE {{ {where} {self._measure('?x', plan, vars)} }}"
            prog.steps.append(QueryStep("answer", f"SELECT ({fn}(?m) AS ?answer) (COUNT(?x) AS ?n) WHERE {{ {{ {inner} }} }}", "select", "answer", inputs, module=plan.module, purpose=f"{fn.lower()} of measure"))
            prog.post = "scalar"
        elif shape == "superlative":
            direction = "DESC" if plan.order == "desc" else "ASC"
            sparql = f"SELECT DISTINCT ?x ?m WHERE {{ {where} {self._measure('?x', plan, vars)} }} ORDER BY {direction}(?m) LIMIT 50"
            prog.steps.append(QueryStep("answer", sparql, "select", "x", inputs, module=plan.module, purpose="rank by measure"))
            prog.post = "extreme"
        return prog
