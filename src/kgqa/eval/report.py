"""Markdown report for an eval run."""

from __future__ import annotations

from typing import Any


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _ms(x: float) -> str:
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


def render(result: dict[str, Any], *, controller: str, llm: str) -> str:
    s = result["summary"]
    lines = [
        "# kgqa evaluation",
        "",
        f"Controller: `{controller}` · LLM: `{llm}`",
        "",
    ]
    if result["skipped"]:
        lines += [f"Skipped arms (need an LLM): {', '.join(result['skipped'])}", ""]
    if controller == "heuristic":
        lines += ["> The heuristic controller is an offline lexical stand-in for jev. Its numbers are a floor for the pipeline, not a measurement of jev; its confidences are not calibrated.", ""]
    arms = list(s)
    lines += ["## Summary", "", "| metric | " + " | ".join(f"{a}: {s[a]['name']}" for a in arms) + " |", "|---|" + "---|" * len(arms)]
    metrics = [
        ("questions", lambda m: str(m["n"])),
        ("execution accuracy", lambda m: _pct(m["accuracy"])),
        ("mean F1", lambda m: f"{m['f1']:.2f}"),
        ("answered rate", lambda m: _pct(m["answered_rate"])),
        ("precision when answered", lambda m: _pct(m["precision_when_answered"])),
        ("latency p50 / p95 (ms)", lambda m: f"{_ms(m['latency_p50_ms'])} / {_ms(m['latency_p95_ms'])}"),
        ("jev calls / question", lambda m: f"{m['jev_calls_mean']:.1f}"),
        ("jev tokens / question", lambda m: f"{m['jev_tokens_mean']:.0f}"),
        ("LLM tokens / question", lambda m: f"{m['llm_tokens_mean']:.0f}"),
        ("SPARQL queries / question", lambda m: f"{m['queries_mean']:.1f}"),
        ("timeout rate", lambda m: _pct(m["timeout_rate"])),
        ("fallback rate", lambda m: _pct(m["fallback_rate"])),
        ("accuracy on fallback subset", lambda m: _pct(m["fallback_accuracy"])),
    ]
    for name, fn in metrics:
        lines.append(f"| {name} | " + " | ".join(fn(s[a]) for a in arms) + " |")

    for a in arms:
        m = s[a]
        lines += ["", f"## Arm {a}: {m['name']}", "", "### Latency by stage (ms)", "", "| stage | p50 | p95 |", "|---|---|---|"]
        for st in m["stage_p50_ms"]:
            lines.append(f"| {st} | {_ms(m['stage_p50_ms'][st])} | {_ms(m['stage_p95_ms'][st])} |")
        lines += ["", "### Failure attribution", "", "| stage | count |", "|---|---|"]
        for k, v in sorted(m["failures"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v} |")
        if not m["failures"]:
            lines.append("| (none) | 0 |")
        lines += ["", "### Accuracy by tag", "", "| tag | accuracy |", "|---|---|"]
        for t, acc in m["by_tag"].items():
            lines.append(f"| {t} | {_pct(acc)} |")
        lines += ["", "### Calibration (reliability per gate)", "", "| gate | n | ECE | bins (confidence → accuracy, n) |", "|---|---|---|---|"]
        for g, cal in sorted(m["calibration"].items()):
            bins = "; ".join(f"{b['confidence']:.2f}→{b['accuracy']:.2f} ({b['n']})" for b in cal["bins"])
            lines.append(f"| {g} | {cal['n']} | {cal['ece']:.2f} | {bins} |")
        ac = m["answer_calibration"]
        if ac["n"]:
            bins = "; ".join(f"{b['confidence']:.2f}→{b['accuracy']:.2f} ({b['n']})" for b in ac["bins"])
            lines.append(f"| final answer | {ac['n']} | {ac['ece']:.2f} | {bins} |")

    lines += ["", "## Per question", "", "| arm | id | question | status | source | correct | predicted | gold | failure |", "|---|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        pred = ", ".join(r["pred"])[:60]
        gold = "(abstain)" if r["expect_unanswered"] else ", ".join((r["gold"] or [])[:4])[:60]
        q = r["question"].replace("|", "/")
        lines.append(f"| {r['arm']} | {r['id']} | {q} | {r['status']} | {r['source'] or ''} | {'✓' if r['correct'] else '✗'} | {pred} | {gold} | {r['failure'] or ''} |")
    return "\n".join(lines) + "\n"


def compare(runs: dict[str, dict[str, Any]]) -> str:
    """Markdown table comparing eval runs (each run's first arm)."""
    names = list(runs)
    arms = {n: next(iter(r["summary"].values())) for n, r in runs.items()}
    rows = [
        ("accuracy", lambda m: _pct(m["accuracy"])),
        ("precision when answered", lambda m: _pct(m["precision_when_answered"])),
        ("answered rate", lambda m: _pct(m["answered_rate"])),
        ("latency p50 (ms)", lambda m: _ms(m["latency_p50_ms"])),
        ("latency p95 (ms)", lambda m: _ms(m["latency_p95_ms"])),
        ("controller calls / q", lambda m: f"{m['jev_calls_mean']:.1f}"),
        ("LLM tokens / q", lambda m: f"{m['llm_tokens_mean']:.0f}"),
        ("fallback rate", lambda m: _pct(m["fallback_rate"])),
    ]
    stages = sorted({st for m in arms.values() for st in m["stage_p50_ms"]})
    rows += [(f"{st} p50 (ms)", lambda m, st=st: _ms(m["stage_p50_ms"].get(st, 0.0))) for st in stages]
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(arms[n]) for n in names) + " |")
    cal = [f"| {n} | " + ", ".join(f"{g} ECE {c['ece']:.2f}" for g, c in sorted(arms[n]["calibration"].items()) if g in ("module", "plan", "verify")) + " |" for n in names]
    return "\n".join(lines + ["", "| run | calibration |", "|---|---|", *cal]) + "\n"
