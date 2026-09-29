"""Command line: build, ask, eval, schema, fetch."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from kgqa.config import load_config


def load_dotenv(path: Path) -> None:
    """Read KEY=value lines from a .env file into the environment (never overriding what is set)."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _pipeline(args: argparse.Namespace):
    from kgqa.pipeline import Pipeline

    cfg = load_config(args.config)
    if getattr(args, "controller", None):
        cfg.controller_backend = args.controller
    if getattr(args, "controller_model", None):
        cfg.controller_model = args.controller_model
    if getattr(args, "no_cache", False):  # timing runs must not replay cached answers
        cfg.controller_cache = None
        cfg.llm_cache = None
    if getattr(args, "llm", None):
        cfg.llm_backend = args.llm
    if getattr(args, "model", None):
        cfg.llm_options["model"] = args.model
    return Pipeline.from_config(cfg)


def cmd_build(args: argparse.Namespace) -> None:
    from kgqa.build import build_all, open_store

    cfg = load_config(args.config)
    store = open_store(cfg)
    cat, idx = build_all(cfg, store, load=not args.no_load)
    print(f"modules: {', '.join(f'{m.name} ({m.triple_count} triples)' for m in cat.modules.values())}")
    print(f"classes: {len(cat.classes)}  properties: {len(cat.properties)}  joins: {len(cat.joins)}  indexed labels/values: {len(idx.records)}")
    print(f"wrote {cfg.catalog_path} and {cfg.entity_index_path}")


def cmd_ask(args: argparse.Namespace) -> None:
    pipe = _pipeline(args)
    ans = pipe.ask(args.question, generation=args.generation)
    if args.json:
        out = ans.to_json()
        if args.trace:
            out["trace"] = ans.trace.to_json()
        print(json.dumps(out, indent=2, default=str))
        return
    print(ans.text())
    print(f"\n  status: {ans.status}   source: {ans.source}   confidence: {ans.confidence:.2f}")
    if ans.route:
        print(f"  route: {ans.route['module']} / {ans.route['class_label']}")
    if ans.plan:
        print(f"  plan: {ans.plan['summary']}")
    if ans.reason and ans.status != "answered":
        print(f"  reason: {ans.reason}")
    tr = ans.trace
    print(f"  cost: {len(tr.jev)} jev calls, {len(tr.queries)} queries, {len(tr.llm)} LLM calls, {tr.total_ms:.0f} ms")
    if args.trace:
        print("\n  stages (ms): " + ", ".join(f"{k}={v:.1f}" for k, v in tr.stages.items()))
        for g in tr.gates:
            print(f"  gate {g.name}: {g.choice} @ {g.confidence:.2f} (threshold {g.threshold:.2f}) {'pass' if g.passed else 'FAIL'}")
        for e in tr.events:
            print(f"  event: {e}")
        for q in tr.queries:
            print(f"\n  -- {q.graph or ''} {q.latency_ms:.1f} ms, {q.rows} rows{' ERROR ' + q.error if q.error else ''}\n  {q.sparql}")


def cmd_eval(args: argparse.Namespace) -> None:
    from kgqa.eval import evaluate, load_benchmark, render

    pipe = _pipeline(args)
    items = load_benchmark(args.benchmark)
    if args.limit:
        items = items[: args.limit]
    arms = [a.strip().upper() for a in args.arms.split(",")]

    def progress(row: dict) -> None:
        mark = "✓" if row["correct"] else "✗"
        print(f"  [{row['arm']}] {mark} {row['id']} {row['question'][:60]:<60} {row['status']:<10} {', '.join(row['pred'])[:40]}", file=sys.stderr)

    result = evaluate(pipe, items, arms, progress=progress)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=1, default=str))
    report = render(result, controller=pipe.controller.name, llm=pipe.llm.name)
    (out / "report.md").write_text(report)
    for arm, m in result["summary"].items():
        print(f"arm {arm} ({m['name']}): accuracy {m['accuracy']:.0%}, F1 {m['f1']:.2f}, p50 {m['latency_p50_ms']:.1f} ms, fallback {m['fallback_rate']:.0%}, jev calls/q {m['jev_calls_mean']:.1f}")
    if result["skipped"]:
        print(f"skipped arms {result['skipped']}: configure [llm].backend to run them")
    print(f"report: {out / 'report.md'}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Side-by-side summary of several eval runs (each a directory with results.json)."""
    from kgqa.eval.report import compare

    runs = {Path(d).name: json.loads((Path(d) / "results.json").read_text()) for d in args.runs}
    text = compare(runs)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


def cmd_schema(args: argparse.Namespace) -> None:
    from kgqa.rdf import Prefixes
    from kgqa.schema import SchemaCatalog, Slicer

    cfg = load_config(args.config)
    cat = SchemaCatalog.load(cfg.path(cfg.catalog_path))
    px = Prefixes(cfg.prefixes)
    slicer = Slicer(cat, px, cfg.limits.slice_token_budget)
    if not args.classes:
        print(slicer.module_overview())
        return
    focus = [px.expand(c) if ":" in c else next((k for k, v in cat.classes.items() if v.label.lower() == c.lower()), c) for c in args.classes]
    sl = slicer.slice(focus)
    print(sl.text)
    print(f"\n~{sl.tokens} tokens, {len(sl.allowed_iris)} allowed IRIs")


def cmd_fetch(args: argparse.Namespace) -> None:
    """Materialize a slice of a remote graph (e.g. Wikidata) with a CONSTRUCT query."""
    from kgqa.store import HttpSparqlStore

    store = HttpSparqlStore(args.endpoint, default_timeout=args.timeout)
    nt = store.construct(Path(args.query).read_text(), timeout=args.timeout)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(nt)
    print(f"wrote {nt.count(chr(10))} triples to {args.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="kgqa", description="System One routing layer for knowledge graph QA")
    ap.add_argument("--config", default="kgqa.toml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="load data, build schema cards and entity index")
    b.add_argument("--no-load", action="store_true", help="do not (re)load module files into the store")
    b.set_defaults(fn=cmd_build)

    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    a.add_argument("--generation", choices=["template", "llm", "auto"])
    a.add_argument("--controller", help="override [controller].backend: heuristic | typesafe | openrouter | openrouter-llm")
    a.add_argument("--controller-model", help="model for the controller, e.g. jev-1.13, or an LLM id with openrouter-llm")
    a.add_argument("--no-cache", action="store_true", help="do not replay cached controller/LLM answers (for timing)")
    a.add_argument("--llm", help="override [llm].backend, e.g. openrouter")
    a.add_argument("--model", help="override [llm].model, e.g. qwen/qwen3.8-27b:free")
    a.add_argument("--trace", action="store_true")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_ask)

    e = sub.add_parser("eval", help="run benchmark arms and write a report")
    e.add_argument("--benchmark", default="data/sample/benchmark.json")
    e.add_argument("--arms", default="A,B,C")
    e.add_argument("--controller", help="override [controller].backend: heuristic | typesafe | openrouter | openrouter-llm")
    e.add_argument("--controller-model", help="model for the controller, e.g. jev-1.13, or an LLM id with openrouter-llm")
    e.add_argument("--no-cache", action="store_true", help="do not replay cached controller/LLM answers (for timing)")
    e.add_argument("--llm", help="override [llm].backend, e.g. openrouter")
    e.add_argument("--model", help="override [llm].model, e.g. qwen/qwen3.8-27b:free")
    e.add_argument("--limit", type=int)
    e.add_argument("--out", default="reports/latest")
    e.set_defaults(fn=cmd_eval)

    c = sub.add_parser("compare", help="compare eval runs side by side")
    c.add_argument("runs", nargs="+", help="eval output directories")
    c.add_argument("--out", help="write the markdown table here")
    c.set_defaults(fn=cmd_compare)

    s = sub.add_parser("schema", help="print the module overview or a schema slice")
    s.add_argument("classes", nargs="*", help="class labels or prefixed IRIs to slice around")
    s.set_defaults(fn=cmd_schema)

    f = sub.add_parser("fetch", help="CONSTRUCT a slice from a remote endpoint into N-Triples")
    f.add_argument("--endpoint", required=True)
    f.add_argument("--query", required=True, help="file with a CONSTRUCT query")
    f.add_argument("--out", required=True)
    f.add_argument("--timeout", type=float, default=60.0)
    f.set_defaults(fn=cmd_fetch)

    args = ap.parse_args(argv)
    load_dotenv(Path(args.config).resolve().parent / ".env")  # API keys live here, gitignored
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("kgqa.llm").setLevel(logging.INFO)  # LLM calls are slow; always show progress
    args.fn(args)


if __name__ == "__main__":
    main()
