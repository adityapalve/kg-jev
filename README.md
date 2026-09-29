# kgqa: System One routing layer for knowledge-graph QA

A fast controller (jev, via the TypeSafe `system_one` API) makes typed, confidence-gated decisions:
which graph, which class, which query shape, which schema path, and whether the result answers the
question. Templates or an LLM then write *small* SPARQL queries against a schema slice, and plain
code executes them, carrying bindings across graph boundaries.

- Design: [docs/architecture.md](docs/architecture.md)
- What's built, decisions, next steps: [docs/implementation-plan.md](docs/implementation-plan.md)

## Quickstart (offline, no API keys)

```bash
uv venv && uv pip install -e ".[dev]"
```

```bash
.venv/bin/kgqa build
```

```bash
.venv/bin/kgqa ask "What was the total revenue from F-150 orders in 2025?" --trace
```

```bash
.venv/bin/kgqa eval --arms C
```

```bash
.venv/bin/python -m pytest -q
```

`build` loads the sample graph (`data/sample/`: catalog, sales and people as three named
graphs) into an embedded Oxigraph store and writes schema cards and the entity index to `.kgqa/`.
`eval` writes `reports/latest/report.md`.

Out of the box the controller is `heuristic`, an offline lexical stand-in for jev, so everything
runs without keys. Its numbers are a floor, not a measurement of jev.

## Plugging in the real services

**jev:** `pip install typesafe-sdk`, export `TYPESAFE_API_KEY`, then in `kgqa.toml`:

```toml
[controller]
backend = "typesafe"
```

**LLM** (arms A and B, `auto` generation, the fallback path): fill in `examples/llm_adapter.py` or
write your own factory, then:

```toml
[llm]
backend = "python:examples.llm_adapter:make"
```

## Other commands

```bash
.venv/bin/kgqa schema order dealer
```

```bash
.venv/bin/kgqa ask "How many employees work in the Pacific region?" --json --trace
```

```bash
.venv/bin/kgqa fetch --endpoint https://query.wikidata.org/sparql --query examples/wikidata/automobile_models.rq --out data/wikidata/automobiles.nt
```

## Pointing at your own graph

Copy `kgqa.toml` and edit it. Add one `[[modules]]` block per named graph (graph IRI, description,
files, type predicate), plus the ontology and SHACL files if you have them. Cards fall back to
statistics when you don't. Declare shared-identifier joins in the ontology with
`kgqa:joinsWith` or a shared `kgqa:identifierScheme`. Then run `kgqa build`. To use a running
endpoint (QLever, Oxigraph server) set `[store] kind = "http"` and `endpoint`.

Benchmarks: native JSON (see `data/sample/benchmark.json`), QALD JSON, or LC-QuAD 2.0 JSON, via
`kgqa eval --benchmark path`.
