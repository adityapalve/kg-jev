"""Executor: runs a Program step by step with limits, timeouts and bounded VALUES passing."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from kgqa.config import Limits
from kgqa.executor.program import Program, QueryStep
from kgqa.rdf import sparql_term
from kgqa.store.base import QueryError, QueryTimeout, SelectResult, Store
from kgqa.trace import QueryLog, current_trace

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_HAS_LIMIT = re.compile(r"\bLIMIT\s+\d+\s*$", re.I)
_AGGREGATE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|SAMPLE)\s*\(", re.I)


@dataclass
class ExecResult:
    values: list[Any] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    bindings: dict[str, list[Any]] = field(default_factory=dict)
    empty_at: str | None = None  # first step whose output was empty
    error: str | None = None
    timeout: bool = False
    truncated: bool = False
    step_rows: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def empty(self) -> bool:
        return not self.values or (self.values == [0] and self.empty_at is not None)


def ensure_limit(sparql: str, limit: int) -> str:
    s = sparql.strip().rstrip(";")
    if s.upper().startswith("ASK") or re.search(r"^\s*(PREFIX[^\n]*\n\s*)*ASK\b", s, re.I):
        return s
    if _HAS_LIMIT.search(s):
        return s
    return f"{s}\nLIMIT {limit}"


class Executor:
    def __init__(self, store: Store, limits: Limits | None = None) -> None:
        self.store = store
        self.limits = limits or Limits()

    def _fill(self, sparql: str, bindings: dict[str, list[Any]], override: dict[str, list[Any]] | None = None) -> str:
        def sub(m: re.Match[str]) -> str:
            vals = (override or {}).get(m[1], bindings.get(m[1], []))
            return " ".join(sparql_term(v) for v in vals[: self.limits.max_intermediate])

        return _PLACEHOLDER.sub(sub, sparql)

    def _query(self, step: QueryStep, sparql: str) -> SelectResult | bool:
        trace = current_trace()
        start = time.perf_counter()
        log = QueryLog(stage=trace.current_stage if trace else "", sparql=sparql, graph=step.module, latency_ms=0.0, rows=0)
        try:
            if step.kind == "ask":
                out: SelectResult | bool = self.store.ask(sparql, timeout=self.limits.query_timeout_s)
                log.rows = 1
            else:
                out = self.store.select(sparql, timeout=self.limits.query_timeout_s)
                log.rows = len(out.rows)
            return out
        except QueryTimeout as e:
            log.error, log.timeout = str(e), True
            raise
        except QueryError as e:
            log.error = str(e)
            raise
        finally:
            log.latency_ms = (time.perf_counter() - start) * 1000
            if trace:
                trace.queries.append(log)

    def run(self, program: Program) -> ExecResult:
        res = ExecResult(bindings={k: list(v) for k, v in program.bindings.items()})
        last: SelectResult | bool | None = None
        for step in program.steps:
            if any(not res.bindings.get(i) for i in step.inputs if "{{" + i + "}}" in step.sparql):
                # an upstream hop came back empty: nothing downstream can match, so skip the query
                res.empty_at = res.empty_at or step.name
                res.bindings[step.name] = []
                last = SelectResult([step.var or "answer"], []) if step.kind == "select" else False
                continue
            try:
                last = self._run_step(step, res)
            except QueryTimeout as e:
                res.error, res.timeout = f"timeout in {step.name}: {e}", True
                return res
            except QueryError as e:
                res.error = f"{step.name}: {e}"
                return res
            if isinstance(last, SelectResult):
                col = list(dict.fromkeys(r[step.var] for r in last.rows if step.var and r.get(step.var) is not None))
                if len(col) > self.limits.max_intermediate:
                    col, res.truncated = col[: self.limits.max_intermediate], True
                res.bindings[step.name] = col
                res.step_rows[step.name] = len(last.rows)
                if not last.rows and res.empty_at is None:
                    res.empty_at = step.name
        self._post(program, last, res)
        return res

    def _run_step(self, step: QueryStep, res: ExecResult) -> SelectResult | bool:
        limit = self.limits.query_limit
        chunk_on = next((i for i in step.inputs if "{{" + i + "}}" in step.sparql), None) if step.chunkable else None
        values = res.bindings.get(chunk_on, []) if chunk_on else []
        if not chunk_on or len(values) <= self.limits.values_chunk:
            sparql = self._fill(step.sparql, res.bindings)
            if step.kind == "select" and not _AGGREGATE.search(sparql.split("WHERE")[0]):
                sparql = ensure_limit(sparql, limit)
            return self._query(step, sparql)
        # paginate the VALUES list; walk steps are DISTINCT selects, so union is exact
        merged = SelectResult([step.var or ""], [])
        size = self.limits.values_chunk
        for i in range(0, min(len(values), self.limits.max_intermediate), size):
            sparql = self._fill(step.sparql, res.bindings, {chunk_on: values[i : i + size]})
            out = self._query(step, ensure_limit(sparql, limit))
            assert isinstance(out, SelectResult)
            merged.variables = out.variables
            merged.rows.extend(out.rows)
        if len(values) > self.limits.max_intermediate:
            res.truncated = True
        return merged

    def _post(self, program: Program, last: SelectResult | bool | None, res: ExecResult) -> None:
        if last is None:
            return
        if isinstance(last, bool):
            res.values = [last]
            return
        res.rows = last.rows
        post = program.post
        var = program.steps[-1].var if program.steps else program.answer_var
        if post == "ask":
            res.values = [bool(last.rows)]
        elif post == "scalar":
            row = last.rows[0] if last.rows else {}
            v = row.get(program.answer_var, next(iter(row.values()), None))
            if "n" in row:  # aggregate: SUM/AVG over zero rows is not an answer
                res.values = [] if v is None or row["n"] == 0 else [v]
            else:  # count: zero is an answer, but flag it so the verifier asks why
                res.values = [] if v is None else [v]
            if not res.values or res.values == [0]:
                res.empty_at = res.empty_at or program.steps[-1].name
        elif post == "table":  # one tuple per row, in the query's column order
            res.values = [tuple(r.get(v) for v in last.variables) for r in last.rows]
        elif post == "extreme":
            rows = [r for r in last.rows if r.get(program.measure_var) is not None]
            if rows:
                top = rows[0][program.measure_var]
                res.values = list(dict.fromkeys(r[var] for r in rows if r[program.measure_var] == top))
        elif post in ("compare_max", "compare_min"):
            best: dict[Any, Any] = {}
            for r in last.rows:
                x, m = r.get("x"), r.get(program.measure_var)
                if x is None or m is None:
                    continue
                if x not in best or (m > best[x] if post == "compare_max" else m < best[x]):
                    best[x] = m
            if best:
                target = max(best.values()) if post == "compare_max" else min(best.values())
                res.values = [x for x, m in best.items() if m == target]
        else:
            res.values = list(dict.fromkeys(r[var] for r in last.rows if var and r.get(var) is not None)) if var else [next(iter(r.values())) for r in last.rows]
