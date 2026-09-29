import dataclasses

from conftest import D

from kgqa.config import Limits
from kgqa.executor import Executor, ensure_limit
from kgqa.generator import TemplateGenerator


def _plan(pipeline, question):
    link = pipeline.linker.link(question)
    rr = pipeline.router.route(question, link)
    return pipeline.planner.plan(question, link, rr.routes[0], rr.shape.choice)


def test_cross_graph_join_runs_as_two_small_queries(pipeline):
    plan = _plan(pipeline, "How many orders included the Tesla Model Y?")
    prog = pipeline.templates.compile(plan)
    assert [s.module for s in prog.steps] == ["catalog", "sales"]  # hop in catalog, then main query in sales
    res = pipeline.executor.run(prog)
    assert res.values == [12]
    assert res.bindings["link0_hop0"] == ["TMY-20"]  # the shared identifier carried as VALUES


def test_fused_mode_gives_the_same_answer_in_one_query(pipeline, catalog):
    plan = _plan(pipeline, "How many orders included the Tesla Model Y?")
    prog = TemplateGenerator(catalog, hop_mode="fused").compile(plan)
    assert len(prog.steps) == 1
    assert pipeline.executor.run(prog).values == [12]


def test_values_are_chunked_and_unioned(pipeline, store):
    plan = _plan(pipeline, "Which employees work at dealers in the Northeast region?")
    prog = pipeline.templates.compile(plan)
    small = Executor(store, dataclasses.replace(Limits(), values_chunk=1))
    assert sorted(small.run(prog).values) == sorted(pipeline.executor.run(prog).values)


def test_empty_upstream_hop_short_circuits(pipeline):
    plan = _plan(pipeline, "How many orders included the Tesla Model Y?")
    prog = pipeline.templates.compile(plan)
    prog.bindings["anchor0"] = [type(prog.bindings["anchor0"][0])(D + "model/does-not-exist")]
    res = pipeline.executor.run(prog)
    assert res.empty_at == "link0_hop0"
    assert res.empty


def test_every_select_gets_a_limit():
    assert ensure_limit("SELECT ?x WHERE { ?x ?p ?o }", 10).endswith("LIMIT 10")
    assert ensure_limit("SELECT ?x WHERE { ?x ?p ?o } LIMIT 3", 10).endswith("LIMIT 3")
    assert ensure_limit("ASK { ?x ?p ?o }", 10) == "ASK { ?x ?p ?o }"
