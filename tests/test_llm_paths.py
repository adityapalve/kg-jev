import pytest
from conftest import D, O

from kgqa.generator import ConstrainedLLMGenerator, GenerationError, used_iris, validate
from kgqa.jev import make_controller
from kgqa.llm import ScriptedLLM
from kgqa.pipeline import Pipeline

GOOD = f"""```sparql
PREFIX o: <{O}>
SELECT (COUNT(DISTINCT ?x) AS ?answer) WHERE {{ GRAPH <http://example.org/graph/catalog> {{ ?x a o:SUV }} }}
```"""

INVENTED = f"""```sparql
SELECT ?x WHERE {{ GRAPH <http://example.org/graph/catalog> {{ ?x <{O}isLuxury> true }} }}
```"""


def test_used_iris_expands_prefixes_and_ignores_strings():
    q = f'PREFIX o: <{O}>\nSELECT ?x WHERE {{ ?x o:city "o:notAnIri" ; <{O}region> ?r }}'
    assert used_iris(q) == {O + "city", O + "region"}


def test_validate_rejects_invented_iris_and_non_queries():
    allowed = {O + "SUV", "http://example.org/graph/catalog"}
    assert validate("SELECT ?x WHERE { GRAPH <http://example.org/graph/catalog> { ?x a <http://example.org/onto#SUV> } }", allowed) == []
    assert any("not in the schema" in e for e in validate(f"SELECT ?x WHERE {{ ?x <{O}isLuxury> true }}", allowed))
    assert validate("DELETE WHERE { ?s ?p ?o }", allowed)


def test_generator_retries_with_feedback(catalog, pipeline):
    llm = ScriptedLLM([INVENTED, GOOD])
    gen = ConstrainedLLMGenerator(llm, catalog)
    link = pipeline.linker.link("How many SUVs are there?")
    sl = pipeline.slicer.slice([O + "SUV"])
    prog = gen.generate("How many SUVs are there?", sl, link)
    assert prog.post == "scalar" and len(llm.prompts) == 2
    assert "isLuxury" in llm.prompts[1][1]  # the failure was fed back


def test_generator_gives_up_after_budget(catalog, pipeline):
    gen = ConstrainedLLMGenerator(ScriptedLLM([INVENTED] * 3), catalog)
    with pytest.raises(GenerationError):
        gen.generate("q", pipeline.slicer.slice([O + "SUV"]), pipeline.linker.link("q"))


def test_arm_b_uses_llm_against_the_slice(cfg, built):
    store, catalog, index = built
    llm = ScriptedLLM(lambda system, prompt: GOOD)
    pipe = Pipeline(cfg, store, catalog, index, make_controller("heuristic"), llm)
    ans = pipe.ask("How many SUVs are there?", generation="llm")
    assert ans.source == "llm" and ans.values == [8]
    system, prompt = llm.prompts[0]
    assert "Controller's plan" in prompt and "GRAPH" in system


def test_arm_a_baseline_single_query(cfg, built):
    store, catalog, index = built
    sparql = f"```sparql\nSELECT ?answer WHERE {{ GRAPH <http://example.org/graph/catalog> {{ <{D}model/defender> <{O}basePrice> ?answer }} }}\n```"
    pipe = Pipeline(cfg, store, catalog, index, make_controller("heuristic"), ScriptedLLM(lambda s, p: sparql))
    ans = pipe.ask_baseline("What is the base price of the Defender?")
    assert ans.status == "answered" and ans.values == [56900.0] and ans.source == "fallback"
    assert not [j for j in ans.trace.jev if j.stage == "route"]  # no routing in the baseline


def test_slice_declares_prefixes_and_missing_ones_are_filled(pipeline):
    from kgqa.generator.constrained import add_missing_prefixes

    sl = pipeline.slicer.slice([O + "Order"])
    assert f"PREFIX o: <{O}>" in sl.text  # the model must be told what o: means
    fixed = add_missing_prefixes("SELECT ?x WHERE { ?x o:revenue ?r }", pipeline.px)
    assert fixed.startswith(f"PREFIX o: <{O}>")


def test_wrong_namespace_gets_a_suggestion():
    errs = validate("SELECT ?x WHERE { ?x <http://example.org/ontology/revenue> ?r }", {O + "revenue"})
    assert f"did you mean <{O}revenue>" in errs[0]


def test_unsupported_shape_goes_straight_to_llm(cfg, built):
    store, catalog, index = built
    group_by = f"""```sparql
PREFIX o: <{O}>
SELECT ?region (SUM(?rev) AS ?total) WHERE {{
  GRAPH <http://example.org/graph/sales> {{ ?order o:dealer ?d ; o:revenue ?rev . ?d o:region ?region }}
}} GROUP BY ?region ORDER BY DESC(?total)
```"""
    llm = ScriptedLLM(lambda s, p: group_by)
    pipe = Pipeline(cfg, store, catalog, index, make_controller("heuristic"), llm)
    ans = pipe.ask("What is the total revenue per region?")
    assert ans.shape == "unsupported" and ans.source == "llm" and ans.status == "answered"
    assert "Controller's plan" not in llm.prompts[0][1]  # no template plan forced on the LLM
    assert len(ans.values) == 3  # one row per region


def test_grouped_llm_result_keeps_its_group_column(cfg, built):
    store, catalog, index = built
    group_by = f"""```sparql
PREFIX o: <{O}>
SELECT ?region (SUM(?rev) AS ?answer) WHERE {{
  GRAPH <http://example.org/graph/sales> {{ ?order o:dealer ?d ; o:revenue ?rev . ?d o:region ?region }}
}} GROUP BY ?region
```"""
    pipe = Pipeline(cfg, store, catalog, index, make_controller("heuristic"), ScriptedLLM(lambda s, p: group_by))
    ans = pipe.ask("What is the total revenue per region?")
    assert sorted(l.split(":")[0] for l in ans.labels) == ["Mountain West", "Northeast", "Pacific"]
    assert all(isinstance(v, tuple) and len(v) == 2 for v in ans.values)


def test_identical_retry_stops_early(cfg, built):
    store, catalog, index = built
    same = f"```sparql\nSELECT ?x WHERE {{ GRAPH <http://example.org/graph/catalog> {{ ?x <{O}basePrice> ?p FILTER(?p > 1e12) }} }}\n```"
    llm = ScriptedLLM(lambda s, p: same)
    pipe = Pipeline(cfg, store, catalog, index, make_controller("heuristic"), llm)
    ans = pipe.ask("What is the total revenue per region?")
    assert any(e["kind"] == "repair_stalled" for e in ans.trace.events)
