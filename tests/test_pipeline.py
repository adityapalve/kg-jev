from conftest import ROOT

from kgqa.eval import evaluate, load_benchmark, render


def test_cross_graph_lookup_with_trace(pipeline):
    ans = pipeline.ask("Which region is the dealer that employs Alice Moreno in?")
    assert ans.status == "answered" and ans.labels == ["Mountain West"]
    tr = ans.trace
    assert {q.graph for q in tr.queries} == {"people", "sales"}  # one hop per graph
    assert {g.name for g in tr.gates} >= {"shape", "module", "plan", "verify"}
    assert ans.confidence > 0


def test_empty_result_is_diagnosed_as_no_data(pipeline):
    ans = pipeline.ask("How many orders were placed in Q1 2019?")
    assert ans.status == "no_data"
    kinds = [e["action"] for e in ans.trace.events if e["kind"] == "verdict"]
    assert kinds == ["accept_empty"]


def test_out_of_scope_question_falls_back_and_is_queued(pipeline, cfg):
    ans = pipeline.ask("What is the weather in Paris?")
    assert ans.status == "unanswered" and ans.source == "fallback"
    assert "module gate" in ans.reason
    assert "weather" in (cfg.path(cfg.review_queue)).read_text()


def test_beam_tries_next_route_when_plan_is_unresolved(pipeline):
    ans = pipeline.ask("How many orders did Mustang place in Q3 2025?")
    assert ans.values == [5]


def test_template_arm_regression_on_sample_benchmark(pipeline):
    items = load_benchmark(ROOT / "data/sample/benchmark.json")
    result = evaluate(pipeline, items, ["C"])
    summary = result["summary"]["C"]
    # A floor for the offline heuristic controller; the real controller should beat it.
    assert summary["accuracy"] >= 0.85
    assert summary["by_tag"]["out-of-scope"] == 1.0
    assert "Arm C" in render(result, controller="heuristic", llm="none")


def test_plans_explain_themselves_to_the_verifier(pipeline):
    ans = pipeline.ask("What was the total revenue from F-150 orders in 2025?")
    (verify,) = [j for j in ans.trace.jev if j.stage == "verify"]
    said = verify.state["what_the_system_did"]
    assert said.startswith("Compute the sum of the revenue over order records linked to Ford F-150")
    assert "fidelity" not in verify.questions["answers"]["instructions"]  # plain words only
    assert "Do not judge whether the stored values match" in verify.questions["answers"]["instructions"]
