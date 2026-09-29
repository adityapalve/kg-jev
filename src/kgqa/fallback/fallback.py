"""Fallback: the broad, slow path. Also serves as evaluation arm A (the single-query baseline).

LLM + retrieved schema -> one SPARQL query. Retrieval is BM25 over class and property cards, so
it needs no controller. With no LLM configured, or when the LLM fails, the question goes to a
human-review queue (JSONL) and comes back unanswered.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from kgqa.executor import ExecResult, Executor
from kgqa.executor.program import Program
from kgqa.generator.constrained import ConstrainedLLMGenerator, GenerationError
from kgqa.linking import LinkResult
from kgqa.llm import LLMClient, LLMNotConfigured, NullLLM
from kgqa.schema.cards import SchemaCatalog
from kgqa.schema.slicer import Slicer
from kgqa.text import BM25, stems


@dataclass
class FallbackResult:
    status: str  # "answered" | "queued" | "error"
    program: Program | None = None
    result: ExecResult | None = None
    error: str | None = None


class FallbackPath:
    def __init__(self, cat: SchemaCatalog, llm: LLMClient, executor: Executor, slicer: Slicer, review_queue: str | Path | None, top_classes: int = 4) -> None:
        self.cat = cat
        self.llm = llm
        self.executor = executor
        self.slicer = slicer
        self.queue = Path(review_queue) if review_queue else None
        self.top_classes = top_classes
        self.generator = ConstrainedLLMGenerator(llm, cat, slicer.px)
        self._class_ids = list(cat.classes)
        docs = []
        for c in self._class_ids:
            card = cat.classes[c]
            words = [card.label, *card.alt_labels, card.description]
            for p in cat.properties_of(c):
                words += [p.label, *p.alt_labels]
            docs.append(stems(" ".join(words)))
        self._bm25 = BM25(docs)

    def retrieve(self, question: str, link: LinkResult) -> list[str]:
        hits = [self._class_ids[i] for i, _ in self._bm25.top(stems(question), self.top_classes)]
        for m in link.mentions:
            for t in m.types:
                if t in self.cat.classes and t not in hits:
                    hits.append(t)
        return hits or self._class_ids[: self.top_classes]

    def enqueue(self, question: str, reason: str, link: LinkResult | None) -> None:
        if self.queue is None:
            return
        self.queue.parent.mkdir(parents=True, exist_ok=True)
        row = {"time": dt.datetime.now().isoformat(timespec="seconds"), "question": question, "reason": reason, "entities": [m.chosen.iri for m in (link.mentions if link else []) if m.chosen]}
        with self.queue.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def answer(self, question: str, link: LinkResult, reason: str, *, use_llm: bool = True) -> FallbackResult:
        if isinstance(self.llm, NullLLM) or not use_llm:
            why = "no LLM configured" if isinstance(self.llm, NullLLM) else "LLM unavailable"
            self.enqueue(question, f"{reason}; {why}", link)
            return FallbackResult("queued", error=why)
        classes = self.retrieve(question, link)
        big = Slicer(self.cat, self.slicer.px, token_budget=self.slicer.budget * 3, hops=1)
        sl = big.slice(classes)
        try:
            program = self.generator.generate(question, sl, link, source="fallback", extra_allowed=self.cat.all_iris())
        except (GenerationError, LLMNotConfigured) as e:
            self.enqueue(question, f"{reason}; generation failed: {e}", link)
            return FallbackResult("queued", error=str(e))
        result = self.executor.run(program)
        if result.error:
            self.enqueue(question, f"{reason}; execution failed: {result.error}", link)
            return FallbackResult("error", program, result, result.error)
        return FallbackResult("answered", program, result)
