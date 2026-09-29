"""Constrained LLM generator: the LLM writes SPARQL against a schema slice, code checks it.

Checks before anything runs: it parses, it is a SELECT or ASK, and every IRI it uses is in the
allow-list (slice classes/properties/graphs, linked entity IRIs, standard vocabularies). Failures
go back to the LLM as feedback, up to a retry budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kgqa.executor.program import Program, QueryStep
from kgqa.linking import LinkResult
from kgqa.llm import LLMClient, LLMNotConfigured, complete_traced
from kgqa.planner.plan import QueryPlan
from kgqa.rdf import OWL, RDF, RDFS, SKOS, XSD, Prefixes
from kgqa.schema.cards import SchemaCatalog
from kgqa.schema.slicer import SchemaSlice
from kgqa.store.oxigraph import check_syntax

SAFE_NAMESPACES = (RDF, RDFS, XSD, OWL, SKOS)

SYSTEM = """You translate a question into ONE SPARQL 1.1 query over an RDF knowledge graph split into named graphs.
Rules:
- Use only the classes, properties and graph IRIs in the schema slice, and only the entity IRIs listed. Never invent IRIs.
- Put each module's triple patterns inside GRAPH <graph-iri> { ... } for the graph that holds them (shown as @module).
- Properties marked JOINS link two graphs through a shared identifier value: match the literal values, not the entities.
- Include subclasses when a class has them (VALUES ?cls { ... } ?x a ?cls).
- For yes/no questions write ASK. Otherwise SELECT the answer as ?answer (a single aggregate like (COUNT(DISTINCT ?x) AS ?answer), or the entities/values).
- Add a LIMIT to non-aggregate SELECTs.
Return only the query inside a ```sparql code block."""


class GenerationError(ValueError):
    def __init__(self, message: str, sparql: str | None = None) -> None:
        super().__init__(message)
        self.sparql = sparql


_BLOCK = re.compile(r"```(?:sparql)?\s*(.*?)```", re.S | re.I)
_IRI = re.compile(r"<([^<>\s\"{}|^`\\]+)>")
_PREFIX_DECL = re.compile(r"PREFIX\s+([\w-]*):\s*<([^>]+)>", re.I)
_PNAME = re.compile(r"(?<![\w<?$\"'])([A-Za-z][\w-]*)?:([A-Za-z_][\w.-]*)")


def extract_sparql(text: str) -> str:
    m = _BLOCK.search(text)
    return (m.group(1) if m else text).strip()


def used_iris(sparql: str) -> set[str]:
    decls = {p: ns for p, ns in _PREFIX_DECL.findall(sparql)}
    body = _PREFIX_DECL.sub("", sparql)
    body_no_str = re.sub(r'"(?:[^"\\]|\\.)*"', '""', body)
    iris = set(_IRI.findall(body_no_str))
    for prefix, local in _PNAME.findall(re.sub(r"<[^>]*>", "", body_no_str)):
        if prefix in decls:
            iris.add(decls[prefix] + local)
    return iris


def add_missing_prefixes(sparql: str, prefixes: Prefixes) -> str:
    """Declare prefixes the query uses but forgot to declare, when the config knows them."""
    declared = {p for p, _ in _PREFIX_DECL.findall(sparql)}
    body = re.sub(r"<[^>]*>", "", re.sub(r'"(?:[^"\\]|\\.)*"', '""', _PREFIX_DECL.sub("", sparql)))
    used = {p for p, _ in _PNAME.findall(body) if p}
    missing = sorted(p for p in used - declared if p in prefixes.map)
    return "".join(f"PREFIX {p}: <{prefixes.map[p]}>\n" for p in missing) + sparql if missing else sparql


def _suggest(iri: str, allowed: set[str]) -> str:
    """Closest allowed IRI by local name, e.g. a right name in the wrong namespace."""
    from kgqa.rdf import local_name

    name = local_name(iri).lower()
    hits = [a for a in allowed if local_name(a).lower() == name]
    return f" (did you mean <{hits[0]}>?)" if hits else ""


def validate(sparql: str, allowed: set[str]) -> list[str]:
    errors = []
    head = re.sub(r"(?im)^\s*PREFIX[^\n]*\n", "", sparql).lstrip().upper()
    if not (head.startswith("SELECT") or head.startswith("ASK")):
        errors.append("The query must be a SELECT or ASK query.")
    syntax = check_syntax(sparql)
    if syntax:
        errors.append(f"Syntax error: {syntax}")
    unknown = sorted(i for i in used_iris(sparql) if i not in allowed and not i.startswith(SAFE_NAMESPACES))
    if unknown:
        errors.append("These IRIs are not in the schema slice or entity list: " + ", ".join(f"<{u}>{_suggest(u, allowed)}" for u in unknown[:10]))
    return errors


def program_from_sparql(sparql: str, source: str) -> Program:
    head = re.sub(r"(?im)^\s*PREFIX[^\n]*\n", "", sparql).lstrip()
    if head.upper().startswith("ASK"):
        return Program([QueryStep("answer", sparql, "ask", None, purpose="LLM query")], post="ask", source=source)
    proj = re.search(r"SELECT\s+(?:DISTINCT\s+|REDUCED\s+)?(.*?)\s+(?:WHERE|FROM|\{)", head, re.I | re.S)
    proj_s = proj.group(1) if proj else ""
    aliases = re.findall(r"AS\s+\?(\w+)", proj_s, re.I)
    bare = proj_s
    while re.search(r"\([^()]*\)", bare):  # strip (possibly nested) expressions, innermost first
        bare = re.sub(r"\([^()]*\)", "", bare)
    plain = re.findall(r"\?(\w+)", bare)
    has_agg = bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", proj_s, re.I))
    var = "answer" if "answer" in aliases + plain else (aliases[0] if aliases else plain[0] if plain else "answer")
    if has_agg and not plain:
        return Program([QueryStep("answer", sparql, "select", var, purpose="LLM query")], post="scalar", answer_var=var, source=source)
    if len(set(plain + aliases)) > 1:  # e.g. ?region (SUM(?rev) AS ?answer): keep every column
        return Program([QueryStep("answer", sparql, "select", var, purpose="LLM query")], post="table", answer_var=var, source=source)
    return Program([QueryStep("answer", sparql, "select", var, purpose="LLM query")], post="column", answer_var=var, source=source)


@dataclass
class Attempt:
    sparql: str
    errors: list[str]


class ConstrainedLLMGenerator:
    def __init__(self, llm: LLMClient, cat: SchemaCatalog, prefixes: Prefixes | None = None, max_attempts: int = 3) -> None:
        self.llm = llm
        self.cat = cat
        self.px = prefixes or Prefixes()
        self.max_attempts = max_attempts

    def prompt(self, question: str, sl: SchemaSlice, link: LinkResult, plan: QueryPlan | None, feedback: list[str]) -> str:
        ents = [f"- \"{m.text}\" = <{m.chosen.iri}> {m.chosen.label} (a {', '.join(self.cat.label(t) for t in m.types)}, @{m.chosen.module})" for m in link.mentions if m.chosen]
        vals = [f"- {l.describe()}" for l in link.literals]
        parts = [f"Question: {question}", "", "Entities:", *(ents or ["- (none)"]), "", "Values:", *(vals or ["- (none)"]), "", "Schema slice:", sl.text]
        if plan is not None:
            parts += ["", f"Controller's plan (a strong hint; follow it unless it is clearly wrong): {plan.describe(self.cat)}"]
        if feedback:
            parts += ["", "Your previous attempt failed:", *feedback]
        return "\n".join(parts)

    def generate(self, question: str, sl: SchemaSlice, link: LinkResult, plan: QueryPlan | None = None, feedback: list[str] | None = None, *, source: str = "llm", extra_allowed: set[str] | None = None) -> Program:
        allowed = set(sl.allowed_iris) | {m.chosen.iri for m in link.mentions if m.chosen} | (extra_allowed or set())
        notes = list(feedback or [])
        last: Attempt | None = None
        for _ in range(self.max_attempts):
            try:
                text = complete_traced(self.llm, SYSTEM, self.prompt(question, sl, link, plan, notes)).text
            except LLMNotConfigured:
                raise
            except Exception as e:  # provider errors degrade to fallback / review queue, not a crash
                raise GenerationError(f"LLM call failed: {type(e).__name__}: {e}") from e
            sparql = add_missing_prefixes(extract_sparql(text), self.px)
            errors = validate(sparql, allowed)
            if not errors:
                return program_from_sparql(sparql, source)
            last = Attempt(sparql, errors)
            notes = [f"Query:\n{sparql}", *errors]
        raise GenerationError("; ".join(last.errors) if last else "no attempt", last.sparql if last else None)
