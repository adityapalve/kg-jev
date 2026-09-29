import json

import httpx2
import pytest

from kgqa.jev import CachingController, Choice, HeuristicController, Noul, RecordingController
from kgqa.jev.large import choose_large
from kgqa.jev.typesafe import TypeSafeController
from kgqa.trace import tracing


def test_heuristic_choice_and_abstain():
    h = HeuristicController()
    q = {"m": Choice(criteria={"catalog": "vehicle models, prices", "people": "employees, departments", "other": None})}
    assert h.ask({"question": "average price of models"}, q).choice("m").choice == "catalog"
    assert h.ask({"question": "weather in Paris"}, q).choice("m").choice == "other"


def test_heuristic_noul_uses_prior():
    assert HeuristicController().ask("x", {"n": Noul(prior=0.9)}).noul("n").noul == 0.9


def test_recording_controller_writes_to_active_trace():
    ctrl = RecordingController(HeuristicController())
    with tracing("q") as tr:
        ctrl.ask({"question": "q"}, {"n": Noul(prior=0.2)}, stage="verify")
    assert tr.jev[0].stage == "verify" and tr.jev[0].answers["n"]["noul"] == 0.2


def test_caching_controller_replays(tmp_path):
    calls = []

    class Counting(HeuristicController):
        def ask(self, state, questions):
            calls.append(1)
            return super().ask(state, questions)

    path = tmp_path / "cache.jsonl"
    q = {"m": Choice(criteria={"a": "apples", "b": "bananas"})}
    first = CachingController(Counting(), path).ask("apples", q)
    again = CachingController(Counting(), path).ask("apples", q)  # fresh instance reads the file
    assert again.cached and again.choice("m") == first.choice("m") and len(calls) == 1


def test_choose_large_runs_a_tournament():
    calls = []
    h = HeuristicController()

    def ask(state, questions):
        calls.append(len(questions))
        return h.ask(state, questions)

    options = {f"opt{i}": f"filler option number {i}" for i in range(600)}
    options["opt_target"] = "horsepower engine power"
    res = choose_large(ask, {"question": "engine horsepower"}, options)
    assert res.choice == "opt_target"
    assert calls == [3, 1]  # three groups in one call, then the final round


def test_typesafe_controller_wire_format():
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "model": "jev-test",
                "usage": {"input_tokens": 42, "output_tokens": 3},
                "answers": {
                    "module": {"type": "choice", "choice": "sales", "confidence": 0.9, "probabilities": {"sales": 0.9, "other": 0.1}},
                    "ok": {"type": "noul", "noul": 0.8},
                },
            },
        )

    from typesafe_sdk import TypeSafeClient

    client = TypeSafeClient(api_key="test-key", transport=httpx2.MockTransport(handler))
    ctrl = TypeSafeController(client=client)
    resp = ctrl.ask({"question": "q"}, {"module": Choice(criteria={"sales": "orders", "other": None}, prior={"sales": 1.0}), "ok": Noul(instructions="ok?", prior=0.1)})
    assert seen["path"] == "/v1/systemone"
    assert seen["body"]["questions"]["module"] == {"type": "choice", "criteria": {"sales": "orders", "other": None}}  # prior never sent
    assert resp.choice("module").choice == "sales" and resp.noul("ok").noul == 0.8
    assert resp.input_tokens == 42 and resp.model == "jev-test"


def test_choice_option_limit():
    with pytest.raises(ValueError):
        Choice(criteria={str(i): None for i in range(256)})


def test_openrouter_backend_targets_openrouter(monkeypatch):
    from kgqa.jev import make_controller

    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["model"] = json.loads(request.content)["model"]
        return httpx2.Response(200, json={"model": "jev-1.13", "usage": {"input_tokens": 5}, "answers": {"n": {"type": "noul", "noul": 0.7}}, "id": "gen-1", "provider": "TypeSafe"})

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    real = httpx2.Client
    monkeypatch.setattr(httpx2, "Client", lambda **kw: real(**{**kw, "transport": httpx2.MockTransport(handler)}))
    ctrl = make_controller("openrouter")
    assert ctrl.ask({"question": "q"}, {"n": Noul(instructions="?")}).noul("n").noul == 0.7  # OpenRouter's extra fields pass through
    assert seen == {"url": "https://openrouter.ai/api/v1/systemone", "auth": "Bearer sk-or-test", "model": "jev-1.13"}


def test_llm_controller_parses_and_degrades():
    from kgqa.jev.llm_controller import LLMController
    from kgqa.llm import ScriptedLLM

    good = ScriptedLLM(['```json\n{"m": {"choice": "sales", "confidence": 0.8}, "ok": {"yes": 0.3}}\n```'])
    resp = LLMController(good).ask({"question": "q"}, {"m": Choice(criteria={"sales": "orders", "people": "staff", "other": None}), "ok": Noul(instructions="?")})
    m = resp.choice("m")
    assert m.choice == "sales" and m.confidence == 0.8 and abs(sum(m.probabilities.values()) - 1) < 1e-9
    assert resp.noul("ok").noul == 0.3

    bad = ScriptedLLM(['{"m": {"choice": "invented_option", "confidence": 0.99}}'])
    m = LLMController(bad).ask({"question": "q"}, {"m": Choice(criteria={"a": None, "b": None})}).choice("m")
    assert m.confidence == 0.0  # an invalid option is treated as no decision, so the gates catch it
