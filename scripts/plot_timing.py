"""Render the timing benchmark as a self-contained HTML page (no re-run; reads saved results).

  .venv/bin/python scripts/plot_timing.py [run_dir ...] [--out reports/timing-plot.html]

Defaults to the three runs written by scripts/timing_benchmark.sh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    "jev + templates": "reports/timing-jev-templates",
    "LLM + templates": "reports/timing-llm-templates",
    "LLM only": "reports/timing-llm-only",
}
STAGES = ["linking", "route", "plan", "verify", "execute", "fallback"]


def load(runs: dict[str, str]) -> dict:
    arms = []
    for name, d in runs.items():
        res = json.loads((ROOT / d / "results.json").read_text())
        rows = res["rows"]
        summary = next(iter(res["summary"].values()))
        points = [
            {
                "q": r["question"],
                "s": r["latency_ms"] / 1000,
                "ok": bool(r["correct"]),
                "status": r["status"],
                "source": r["source"] or "",
            }
            for r in rows
        ]
        n = len(rows) or 1
        stage_mean = {st: sum(r["stages"].get(st, 0.0) for r in rows) / n / 1000 for st in STAGES}
        arms.append(
            {
                "name": name,
                "points": points,
                "median": summary["latency_p50_ms"] / 1000,
                "p95": summary["latency_p95_ms"] / 1000,
                "accuracy": summary["accuracy"],
                "correct": sum(p["ok"] for p in points),
                "n": len(points),
                "stages": stage_mean,
                "total_mean": sum(stage_mean.values()),
            }
        )
    return {"arms": arms, "stages": STAGES}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Timing Benchmark</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --surface-2: #f3f2ef; --grid: #e4e3df;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #242423; --grid: #34342f;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #94938a;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --surface-2: #242423; --grid: #34342f;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #94938a;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300;
}
html, body { margin: 0; }
body { background: var(--surface-1); }
.viz-root { min-height: 100vh; box-sizing: border-box; background: var(--surface-1); color: var(--text-primary); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 24px 16px 48px; }
.wrap { max-width: 920px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 15px; margin: 32px 0 2px; font-weight: 600; }
.sub { color: var(--text-secondary); margin: 0 0 12px; }
.tiles { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }
.tile { background: var(--surface-2); border-radius: 8px; padding: 12px 14px; }
.tile .k { color: var(--text-secondary); font-size: 12px; display: flex; gap: 6px; align-items: center; }
.tile .v { font-size: 24px; font-weight: 600; margin-top: 2px; }
.tile .d { color: var(--text-muted); font-size: 12px; }
.sw { width: 10px; height: 10px; border-radius: 3px; display: inline-block; flex: none; }
.note { color: var(--text-secondary); font-size: 13px; margin: 8px 0 0; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--text-secondary); font-size: 12px; margin: 8px 0 4px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--text-secondary); font-size: 12px; }
svg text.halo { paint-order: stroke; stroke: var(--surface-1); stroke-width: 4px; stroke-linejoin: round; font-weight: 600; }
.tip { position: fixed; pointer-events: none; background: var(--surface-2); color: var(--text-primary); border: 1px solid var(--grid);
  border-radius: 6px; padding: 8px 10px; font-size: 12px; max-width: 320px; box-shadow: 0 4px 16px rgba(0,0,0,.12); opacity: 0; transition: opacity .08s; z-index: 10; }
.tip b { font-weight: 600; }
details { margin-top: 28px; color: var(--text-secondary); }
summary { cursor: pointer; }
table { border-collapse: collapse; margin-top: 10px; font-size: 13px; width: 100%; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--grid); color: var(--text-primary); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 500; }
@media (max-width: 560px) { .tiles { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="viz-root"><div class="wrap">
  <h1>jev + templates vs LLM: latency on 45 DBpedia questions</h1>
  <p class="sub">Same pipeline and data, no caching. The LLM is openai/gpt-6-luna in every LLM role; jev is jev-1.13. One run.</p>

  <div class="tiles" id="tiles"></div>
  <p class="note">Accuracy differs by one to two questions out of 45 (each arm got exactly one question the others missed), so this run cannot rank the arms on accuracy. Latency is the clear difference.</p>

  <h2>Latency per question</h2>
  <p class="sub">Each dot is one question (log scale). Filled = correct, ring = wrong. The line marks the median.</p>
  <div class="legend" id="legend1"></div>
  <svg id="dots" role="img" aria-label="Per-question latency by arm"></svg>

  <h2>Where the time goes</h2>
  <p class="sub">Mean seconds per question by pipeline stage. For LLM only, the fallback stage is its single LLM query.</p>
  <div class="legend" id="legend2"></div>
  <svg id="bars" role="img" aria-label="Mean time per question by stage"></svg>

  <details><summary>Table view</summary><div id="table"></div></details>
</div></div>
<div class="tip" id="tip"></div>
<script>
// ?theme=light|dark pins the theme; ?static hides interactive-only parts (for screenshots)
const params = new URLSearchParams(location.search);
if (params.get("theme")) document.documentElement.dataset.theme = params.get("theme");
if (params.has("static")) document.addEventListener("DOMContentLoaded", () => document.querySelector("details").remove());
const DATA = __DATA__;
const arms = DATA.arms, stages = DATA.stages;
const armColor = ["var(--s1)", "var(--s2)", "var(--s3)"];
const stageColor = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)"];
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const fmt = s => s < 10 ? s.toFixed(2) + " s" : s.toFixed(1) + " s";
const tip = document.getElementById("tip");
function showTip(ev, html) { tip.innerHTML = html; tip.style.opacity = 1; const x = Math.min(ev.clientX + 14, innerWidth - 330); tip.style.left = x + "px"; tip.style.top = (ev.clientY + 14) + "px"; }
function hideTip() { tip.style.opacity = 0; }
const esc = s => s.replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c]));

// tiles
document.getElementById("tiles").innerHTML = arms.map((a, i) => `
  <div class="tile"><div class="k"><span class="sw" style="background:${armColor[i]}"></span>${a.name}</div>
  <div class="v">${fmt(a.median)}</div>
  <div class="d">median latency · ${a.correct}/${a.n} correct (${Math.round(a.accuracy * 100)}%)</div></div>`).join("");

// legends
document.getElementById("legend1").innerHTML = arms.map((a, i) => `<span><span class="sw" style="background:${armColor[i]}"></span>${a.name}</span>`).join("");
document.getElementById("legend2").innerHTML = stages.map((s, i) => `<span><span class="sw" style="background:${stageColor[i]}"></span>${s}</span>`).join("");

// chart 1: dot strips, log x
(function () {
  const svg = document.getElementById("dots");
  const W = 900, rowH = 70, left = 128, right = 24, top = 8, H = top + rowH * arms.length + 34;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const all = arms.flatMap(a => a.points.map(p => p.s));
  const lo = Math.log10(Math.min(0.2, ...all)), hi = Math.log10(Math.max(40, ...all));
  const x = s => left + (Math.log10(s) - lo) / (hi - lo) * (W - left - right);
  for (const t of [0.3, 1, 3, 10, 30]) {
    el("line", {x1: x(t), x2: x(t), y1: top, y2: top + rowH * arms.length, stroke: "var(--grid)", "stroke-width": 1}, svg);
    const lab = el("text", {x: x(t), y: top + rowH * arms.length + 20, "text-anchor": "middle"}, svg); lab.textContent = t + " s";
  }
  arms.forEach((a, i) => {
    const cy = top + rowH * i + rowH / 2;
    const name = el("text", {x: 0, y: cy + 4, fill: "var(--text-primary)"}, svg); name.textContent = a.name;
    // deterministic jitter so the run renders identically every time
    a.points.forEach((p, j) => {
      const jit = ((j * 37) % 23) / 22 - 0.5;
      const cyj = cy + jit * (rowH - 30);
      const c = el("circle", {cx: x(p.s), cy: cyj, r: 4.5, fill: p.ok ? armColor[i] : "var(--surface-1)", stroke: armColor[i], "stroke-width": 2}, svg);
      const hit = el("circle", {cx: x(p.s), cy: cyj, r: 9, fill: "transparent"}, svg);
      hit.addEventListener("mousemove", ev => showTip(ev, `<b>${esc(p.q)}</b><br>${fmt(p.s)} · ${p.ok ? "correct" : "wrong"} · ${p.status}${p.source ? " via " + p.source : ""}`));
      hit.addEventListener("mouseleave", hideTip);
    });
    const mx = x(a.median);
    el("line", {x1: mx, x2: mx, y1: cy - rowH / 2 + 6, y2: cy + rowH / 2 - 6, stroke: "var(--text-primary)", "stroke-width": 2}, svg);
    const ml = el("text", {x: mx + 6, y: cy - rowH / 2 + 16, fill: "var(--text-primary)", class: "halo"}, svg); ml.textContent = "median " + fmt(a.median);
  });
})();

// chart 2: stacked horizontal bars, mean seconds per stage
(function () {
  const svg = document.getElementById("bars");
  const W = 900, barH = 26, gap = 22, left = 128, right = 80, top = 8, H = top + (barH + gap) * arms.length + 22;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const max = Math.max(...arms.map(a => a.total_mean));
  const x = s => left + s / max * (W - left - right);
  for (let t = 0; t <= max; t += 2) {
    el("line", {x1: x(t), x2: x(t), y1: top, y2: H - 22, stroke: "var(--grid)", "stroke-width": 1}, svg);
    const lab = el("text", {x: x(t), y: H - 6, "text-anchor": "middle"}, svg); lab.textContent = t + " s";
  }
  arms.forEach((a, i) => {
    const y = top + (barH + gap) * i;
    const name = el("text", {x: 0, y: y + barH / 2 + 4, fill: "var(--text-primary)"}, svg); name.textContent = a.name;
    let acc = 0;
    const segs = stages.map((s, k) => ({s, k, v: a.stages[s] || 0})).filter(d => d.v > 0.005);
    segs.forEach((d, idx) => {
      const x0 = x(acc), x1 = x(acc + d.v); acc += d.v;
      const last = idx === segs.length - 1;
      const w = Math.max(1, x1 - x0 - (last ? 0 : 2));  // 2px surface gap between segments
      const r = el("rect", {x: x0, y, width: w, height: barH, fill: stageColor[d.k], rx: last ? 4 : 0}, svg);
      if (last && w > 8) el("rect", {x: x0, y, width: Math.min(w, 6), height: barH, fill: stageColor[d.k]}, svg);  // square the start
      r.addEventListener("mousemove", ev => showTip(ev, `<b>${a.name}</b><br>${d.s}: ${fmt(d.v)} per question on average`));
      r.addEventListener("mouseleave", hideTip);
    });
    const tl = el("text", {x: x(a.total_mean) + 8, y: y + barH / 2 + 4, fill: "var(--text-primary)"}, svg); tl.textContent = fmt(a.total_mean);
  });
})();

// table view
document.getElementById("table").innerHTML = `<table><tr><th>arm</th><th>correct</th><th>p50</th><th>p95</th>${stages.map(s => `<th>${s} (mean)</th>`).join("")}<th>mean total</th></tr>` +
  arms.map(a => `<tr><td>${a.name}</td><td>${a.correct}/${a.n}</td><td>${fmt(a.median)}</td><td>${fmt(a.p95)}</td>${stages.map(s => `<td>${fmt(a.stages[s] || 0)}</td>`).join("")}<td>${fmt(a.total_mean)}</td></tr>`).join("") + "</table>";
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="name=dir pairs; defaults to the three timing runs")
    ap.add_argument("--out", default="reports/timing-plot.html")
    args = ap.parse_args()
    runs = dict(r.split("=", 1) for r in args.runs) if args.runs else DEFAULT_RUNS
    out = ROOT / args.out
    out.write_text(PAGE.replace("__DATA__", json.dumps(load(runs))))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
