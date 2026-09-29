# Implementation plan: System One routing layer for KG QA

Companion to [architecture.md](architecture.md). This document records how the design was turned
into code, what is built, the decisions made along the way, and what is left to do.

## 1. Status at a glance

| Milestone (architecture §11) | Status | Where |
|---|---|---|
| 1. Local endpoint + one domain slice + schema card builder | Built | `store/`, `schema/builder.py`, `data/sample/` |
| 2. Entity linking + baseline arm A | Built (arm A needs an LLM adapter) | `linking/`, `fallback/`, `eval/runner.py` |
| 3. Router (module, class) with confidence gating; arm B | Built (arm B needs an LLM adapter) | `router/`, `generator/constrained.py` |
| 4. Query shape + property planner with templates; arm C | Built, runs offline | `planner/`, `generator/templates.py` |
| 5. Verifier + repair loop; fallback rate and calibration | Built | `verifier/`, `pipeline.py`, `eval/metrics.py` |
| 6. Second graph + cross-graph questions | Built: 3 graphs, IRI links and a shared-identifier join | `schema/paths.py`, `executor/` |
| 7. Write-up; decide what transfers to OBDA | Not started: needs real jev + LLM numbers | – |

Everything runs today with **no API keys**: an offline heuristic controller stands in for jev, and
arm C (templates) needs no LLM. Two things are left to plug in:

1. **jev**: set `TYPESAFE_API_KEY`, then set `[controller] backend = "typesafe"`. The adapter is written and
   tested against the SDK's wire format (`src/kgqa/jev/typesafe.py`, `tests/test_jev.py`).
2. **An LLM**: implement `complete(system, prompt)` (see `examples/llm_adapter.py`) and set
   `[llm] backend = "python:module:factory"`. That enables arms A and B, the LLM path of `auto`
   generation, and the fallback path.

## 2. Pipeline, as built

```
question
  └─ linking     label/alias index → candidates; jev Choice for ambiguous mentions (+ "none")
                 typed literals (quarters, years, dates, numbers); low-cardinality string values
  └─ route       ONE jev call: module + shape + flat class (when ≤254 classes)
                 code combines module × class evidence → gate → fast path, or hierarchical
                 descent (module → class → subclass) with a beam, in parallel per module
  └─ plan        code enumerates valid schema paths per slot; ONE jev call picks among them:
                 link_i (how target reaches anchor i), filter_j + op_j (literal j), measure, agg, order
  └─ generate    template compile (deterministic)  |  constrained LLM on the schema slice
  └─ execute     small queries; hops at graph boundaries carry VALUES; LIMIT, timeout, chunking
  └─ verify      code checks + probes → ONE jev call: answers? / why empty?
                 accept | accept_empty (no data) | repair (next-best option, no extra call)
                 | next route in beam | fallback
  └─ fallback    LLM + BM25-retrieved schema, one query (= arm A), else human review queue
```

Typical cost on the sample graph: 3 to 4 controller calls and 1 to 2 SPARQL queries per question.

## 3. Code map

| Path | Role |
|---|---|
| `src/kgqa/jev/` | Question/answer types mirroring TypeSafe `system_one`; `TypeSafeController` (real jev); `HeuristicController` (offline stand-in); `RecordingController` (trace), `CachingController` (disk replay); `large.choose_large` (tournament for >255 options) |
| `src/kgqa/schema/` | `SchemaBuilder` (ontology + SHACL + statistics → cards), `SchemaCatalog`, `SchemaGraph` (path enumeration incl. joins), `Slicer` (token-budgeted slice + IRI allow-list) |
| `src/kgqa/linking/` | `EntityIndex` (labels, aliases, string values, optional embeddings), `EntityLinker` (detection, disambiguation), `extract_literals` |
| `src/kgqa/router/` | `Router`: combined first call, gates, beam, hierarchical descent |
| `src/kgqa/planner/` | `QueryPlan` IR, `Planner` (slots, one call, `repair`) |
| `src/kgqa/generator/` | `TemplateGenerator` (plan → Program), `ConstrainedLLMGenerator` (prompt, validate IRIs/syntax, retry with feedback) |
| `src/kgqa/executor/` | `Program`/`QueryStep`, `Executor` (placeholder VALUES, chunking, caps, post-processing) |
| `src/kgqa/verifier/` | `Verifier` (checks, empty-result probes, verdicts) |
| `src/kgqa/fallback/` | `FallbackPath` (arm A / safety net, review queue) |
| `src/kgqa/pipeline.py` | Orchestration, gates, repair loop, `Answer` |
| `src/kgqa/eval/` | Benchmark loaders (native, QALD, LC-QuAD 2.0), arms A/B/C, metrics, markdown report |
| `src/kgqa/trace.py` | Per-question trace: stage timings, every jev call, query, LLM call, gate |
| `data/sample/` | Generated 3-graph enterprise sample + ontology + SHACL + 36-question benchmark |

## 4. Design decisions (and where they depart from architecture.md)

1. **Code enumerates, jev chooses.** "Property choice" is really *schema-path* choice: for every
   slot the planner enumerates the valid paths through the schema graph (up to 3 edges, including
   inverse edges and shared-identifier joins) and jev picks one. jev never sees an option that
   cannot be executed, and a plan is always well-formed.
2. **"Needs a second hop / second graph" is derived, not asked.** Architecture §5.5 lists these as
   jev questions; here they follow from the chosen path (`plan.hops`, `plan.cross_module`), so
   there is one fewer thing to get wrong.
3. **Flat class question alongside the module question.** When the ontology has ≤254 classes the
   first call also asks for the class directly, and code multiplies module and class evidence.
   A confident, consistent answer skips the hierarchical descent (saves 1 to 2 calls). The
   hierarchical beam (§5.3) is still used when the combined answer is weak or the ontology is
   too large for one Choice.
4. **Shapes.** Added `list` and `superlative` to the six in §5.5 because they need different
   templates. `path` and `lookup` compile the same way (a path is a lookup with more hops).
5. **Hops at graph boundaries.** Constraint paths are split into per-graph segments. The segment
   next to the target runs in the main query; other segments run first as small `SELECT DISTINCT`
   queries from the far end (anchor or literal filter), and their results are passed on as chunked
   `VALUES`. `hop_mode = "fused"` compiles the same plan into one query with a `GRAPH` block per module,
   which is the control for the §9 risk that the engine's planner already wins.
6. **Shared-identifier joins are edges.** `kgqa:joinsWith` or a shared `kgqa:identifierScheme` in
   the ontology creates a two-step edge (`soldModelCode = modelCode`) that the path enumerator
   treats like any other relation. The builder records the observed value overlap on each join.
7. **String values are entities too.** Distinct values of low-cardinality string properties
   (city, country, role; ≤200 values) are indexed and become equality filters. Enterprise graphs are
   full of these (status codes, categories), and the LLM should not have to guess them.
8. **Repair costs no controller call.** Each slot keeps jev's full distribution; repair swaps the
   suspect slot (the hop that came back empty, else the least confident slot) for its next-best
   untried option. In LLM mode, repair means regenerating with the failure attached (§5.8).
9. **Empty results get evidence.** Before asking "why is this empty?", code runs probes (does the
   anchor have that relation at all? does the query match once literal filters are dropped?) and
   puts the answers in the state. This is aimed at open question 5 in §12.
10. **Confidence is the minimum over decisions.** The answer's confidence is the minimum of entity,
    shape, route, plan and verify confidences, and it is what the "final answer" calibration row in
    the report measures.
11. **Offline stand-in via `prior`.** Question objects carry an optional `prior` that is never
    sent to the API; only the heuristic backend uses it. Priors are code-computed cues (question
    word, comparison words near a literal, value ownership). This keeps the pipeline, tests and
    eval runnable anywhere, and the heuristic is the floor jev has to beat.

## 5. Evaluation harness

`kgqa eval --arms A,B,C` writes `reports/latest/report.md` and `results.json` with:

- execution accuracy (exact answer-set match), mean F1, answered rate, precision when answered
- latency p50/p95 end to end and per stage (`linking`, `route`, `route/class`, `plan`, `generate`,
  `execute`, `verify`, `fallback`)
- cost per question: jev calls and tokens, LLM tokens, SPARQL queries
- timeout rate, fallback rate, and accuracy on the fallback subset
- per-gate reliability curves and ECE (`module`, `class`, `shape`, `plan`, `verify`, `entity`),
  plus final-answer calibration
- failure attribution: first disagreeing stage among entity linking → module → class → shape →
  property → generation → execution → fallback, with `silent_answer` for out-of-scope questions
  that got an answer
- accuracy by tag (cross-graph, join, multi-hop, ambiguous-entity, no-data, out-of-scope, …)

Current offline result (heuristic controller, arm C, sample benchmark): **34/36 correct**, both
out-of-scope questions abstain, 3.2 controller calls per question. **Treat this as a smoke test,
not a result:** the benchmark was written while the heuristic was being tuned, so it is not
held out, and the heuristic's confidences mean nothing. The two failures are instructive anyway.
One is a shape misread that the plan gate caught (sent to fallback). The other is a silent
misroute at 0.21 confidence, which is the case calibration is meant to catch.

## 6. Next steps, in order

1. **Plug in jev** and rerun `kgqa eval --arms C`. Compare with the heuristic floor. Look at per-gate
   calibration before touching thresholds. (Resolves open question 1 for this graph size.)
2. **Plug in an LLM** and run all three arms. This gives the number arm A sets and whether B and C beat
   it on latency, cost and calibrated fallback (the §8 hypothesis).
3. **Write a held-out benchmark.** Have someone who did not write the templates write 100+
   questions for the sample graph, then add a Wikidata slice (`examples/wikidata/`) and a
   QALD/LC-QuAD subset (loaders exist; check graph versions).
4. **Tune gates** on a dev split using the reliability tables; then set `tie_policy` from data
   (open question 3).
5. **Ablations the harness already supports:** `hop_mode` app vs fused (§9 "engine planner may
   already win"); `generation` template vs llm vs auto (open question 2: template coverage);
   ontology with and without SHACL shapes (open question 4: rebuild with `shapes = []`).
6. **Scale-out work** once a real graph is in: an external entity index (search service or
   vector DB behind `EntityIndex.lookup_*`), a store per module to make app-side hops real
   federation, and `SERVICE` compilation where the endpoint supports it.

## 7. DBpedia film slice (milestone 6, real data)

`dbpedia.toml` runs the pipeline on a materialized slice of current DBpedia in three named graphs:

- **films**: every `dbo:Film` with a director, a cast and a US-dollar gross (~20k films): credits,
  runtime (seconds), gross and budget (USD, stored as `xsd:double`), country, release date
- **people**: everyone credited on those films: DBpedia types, birth/death dates and places
- **places**: those birth/death places and film countries, plus their countries: types, country,
  population

`data/dbpedia/fetch_dbpedia.py` downloads it from the public endpoint (resumable, batched,
throttled). `data/dbpedia/build_benchmark.py` keeps the QALD-9 questions the slice can answer and
recomputes their gold answers on it. Items are tagged `matches-qald` or `drift` against QALD's
2016 answers. The ontology is DBpedia's own (~800 classes, ~3k properties), with no SHACL, pruned to
what the slice uses.

Changes real data forced:
- a property can live in several graphs (`dbo:country` on films and on places); schema steps
  now pick the graph from the subject's class (`SchemaCatalog.module_for`)
- observed subject classes are added to declared domains when the data disagrees with the ontology
- literal-typed ranges (`rdf:langString`, `http://dbpedia.org/datatype/...`) are values, not links
- Wikipedia-style disambiguators are aliases: "Titanic (1997 film)" also answers to "Titanic"
- benchmark gold queries run over the union of graphs (QALD queries do not name graphs)
- leniency: files load with relaxed IRI validation
- DBpedia's labels are not the everyday words (`dbo:Film` is "movie", `dbo:writer` is "auteur"), so
  IRI local names and extra labels become aliases
- entity linking needed demonyms (Danish → Denmark), surname aliases (Kurosawa) ranked by
  popularity (in-degree), and weaker scores for disambiguator-stripped names so words like
  "actors" are not linked to the film *Actors*; on the QALD subset this cut linking failures 11 → 1
- Virtuoso returns `\uXXXX`-escaped IRIs in N-Triples that it then rejects inside queries; the
  fetcher decodes them and bisects any batch that still fails

Findings so far:
- the slice holds ~695k triples (films 308k, people 317k, places 71k), 222 classes and 17
  properties in use, 71,844 labelled entities; build takes ~11 s
- current DBpedia types people only as Person/Animal/Eukaryote/Species (no Actor or Director), so
  routing leans on properties rather than fine-grained classes
- data quality is a real factor: the top "US-dollar" grosses are mislabelled currencies
  (*Ask This of Rikyu* at $664 billion); Christopher Nolan has no birthplace
- the QALD-9 filter keeps 45 of 558 questions; 7 still match QALD's 2016 answers, the rest drift
- the offline stand-in scores 5/45 (11%) and falls back on 71%; it is not meaningful at this scale,
  so the real measurement is `kgqa --config dbpedia.toml eval --benchmark data/dbpedia/benchmark.json
  --arms C --controller openrouter`

## 8. Known limitations

- Templates do not cover: GROUP BY ("revenue per region"), negation ("models not made by Ford"),
  top-k with k>1, ordering on list questions, multiple subjects in a lookup, or arithmetic over two
  measures. These go to constrained LLM generation (when configured) or the fallback.
- Aggregate main queries are not chunked; their `VALUES` are capped at `max_intermediate`.
- Oxigraph has no query cancellation; a timed-out query is abandoned on its worker thread.
- All modules live in one store (named graphs). Cross-graph hops already run as separate queries,
  so moving to one endpoint per module is a small change (a store per `QueryStep.module`).
- Wikidata: property labels sit on `wd:P…` while triples use `wdt:P…`. The example CONSTRUCT copies
  labels across; a general fix belongs in the schema builder. The Wikidata example is untested
  against live endpoints.
- The heuristic controller's accuracy and confidences are not evidence about jev.
