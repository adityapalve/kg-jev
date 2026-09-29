#!/usr/bin/env bash
# Three-way timing benchmark on the DBpedia QALD subset, with caching off so every call is real.
#   jev + templates   : jev decides, templates write SPARQL, the LLM only for what templates cannot do
#   llm + templates   : the same pipeline, with the LLM answering jev's questions
#   llm only (arm A)  : the LLM writes one query per question from retrieved schema
# Usage: LLM_MODEL=openai/gpt-6-luna scripts/timing_benchmark.sh [--limit N]
# Use a synchronous model: ":batch" variants go through OpenRouter's async Batch API (minutes per call).
set -euo pipefail
cd "$(dirname "$0")/.."
model="${LLM_MODEL:-openai/gpt-6-luna}"
common=(--config dbpedia.toml eval --benchmark data/dbpedia/benchmark.json --llm openrouter --model "$model" --no-cache "$@")

echo "== jev + templates"
.venv/bin/kgqa "${common[@]}" --arms C --controller openrouter --out reports/timing-jev-templates
echo "== llm + templates"
.venv/bin/kgqa "${common[@]}" --arms C --controller openrouter-llm --controller-model "$model" --out reports/timing-llm-templates
echo "== llm only (arm A)"
.venv/bin/kgqa "${common[@]}" --arms A --controller openrouter --out reports/timing-llm-only

.venv/bin/kgqa compare reports/timing-jev-templates reports/timing-llm-templates reports/timing-llm-only --out reports/timing-comparison.md
echo "== all done"
