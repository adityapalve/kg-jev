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
