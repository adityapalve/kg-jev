"""Entity resolver: mention detection, candidate lookup, jev disambiguation, literal extraction.

Identifiers come only from the index, never from a model. Ambiguous mentions (several candidate
IRIs, a mention that is also a schema word like "sales", or a fuzzy match) become one jev Choice
each, all in a single call, with a "none" option so generic words can be rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from kgqa.config import Gates
from kgqa.jev import Choice, ChoiceResult, RecordingController, certain
from kgqa.linking.index import EntityIndex, EntityRecord
from kgqa.linking.literals import LiteralMention, extract_literals
from kgqa.schema.cards import SchemaCatalog
from kgqa.text import STOPWORDS, normalize
from kgqa.trace import current_trace

_WORD = re.compile(r"[^\W_]+(?:['’]s\b)?")
NONE_KEY = "none"


@dataclass
class Candidate:
    iri: str
    label: str
    types: list[str]
    module: str
    score: float
    value_of: str | None = None


@dataclass
class Mention:
    text: str
    span: tuple[int, int]
    candidates: list[Candidate]
    capitalized: bool = False
    schema_overlap: bool = False
    chosen: Candidate | None = None
    confidence: float = 0.0
    decision: ChoiceResult | None = None

    @property
    def iri(self) -> str | None:
        return self.chosen.iri if self.chosen else None

    @property
    def types(self) -> list[str]:
        return self.chosen.types if self.chosen else []


@dataclass
class LinkResult:
    mentions: list[Mention] = field(default_factory=list)  # resolved
    literals: list[LiteralMention] = field(default_factory=list)
    rejected: list[Mention] = field(default_factory=list)  # judged "none" or below the gate

    def summary(self, cat: SchemaCatalog) -> list[str]:
        out = [f'"{m.text}" -> {m.chosen.label} ({", ".join(cat.label(t) for t in m.types)})' for m in self.mentions if m.chosen]
        out += [f'"{l.text}" -> {l.kind}' for l in self.literals]
        return out


class EntityLinker:
    def __init__(self, index: EntityIndex, catalog: SchemaCatalog, controller: RecordingController, gates: Gates | None = None, max_candidates: int = 8) -> None:
        self.index = index
        self.cat = catalog
        self.controller = controller
        self.gates = gates or Gates()
        self.max_candidates = max_candidates
        self.schema_terms = set()
        for c in catalog.classes.values():
            self.schema_terms.update(normalize(x) for x in [c.label, *c.alt_labels])
        for p in catalog.properties.values():
            self.schema_terms.update(normalize(x) for x in [p.label, *p.alt_labels])

    # ---- detection ----
    def detect(self, question: str) -> list[Mention]:
        words = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(question)]
        norms = [normalize(w) for w, _, _ in words]
        used = [False] * len(words)
        mentions: list[Mention] = []
        max_n = min(self.index.max_label_tokens, 8)
        for n in range(max_n, 0, -1):
            for i in range(len(words) - n + 1):
                if any(used[i : i + n]):
                    continue
                span_norm = " ".join(norms[i : i + n]).strip()
                if not span_norm or (n == 1 and (span_norm in STOPWORDS or span_norm.isdigit())):
                    continue
                hits = self.index.lookup_exact(span_norm)
                if not hits and span_norm.endswith("s") and len(span_norm) > 4:
                    hits = self.index.lookup_exact(span_norm[:-1])  # "Sales Managers" -> "Sales Manager"
                if not hits:
                    continue
                for k in range(i, i + n):
                    used[k] = True
                raw = question[words[i][1] : words[i + n - 1][2]]
                mentions.append(self._mention(raw, (words[i][1], words[i + n - 1][2]), hits, words[i : i + n], i == 0, span_norm))
        # fuzzy pass over capitalized runs the exact pass missed (typos, missing punctuation)
        i = 0
        while i < len(words):
            if used[i] or not words[i][0][:1].isupper() or i == 0:
                i += 1
                continue
            j = i
            while j < len(words) and not used[j] and (words[j][0][:1].isupper() or any(c.isdigit() for c in words[j][0])):
                j += 1
            raw = question[words[i][1] : words[j - 1][2]]
            hits = self.index.lookup_fuzzy(raw) or self.index.lookup_vector(raw)
            if hits:
                for k in range(i, j):
                    used[k] = True
                mentions.append(self._mention(raw, (words[i][1], words[j - 1][2]), hits, words[i:j], False, normalize(raw)))
            i = max(j, i + 1)
        mentions.sort(key=lambda m: m.span)
        return mentions

    def _mention(self, raw: str, span: tuple[int, int], hits: list[tuple[EntityRecord, float]], words: list[tuple[str, int, int]], sentence_start: bool, norm: str) -> Mention:
        best: dict[str, Candidate] = {}
        for rec, score in hits:
            if rec.iri not in best or best[rec.iri].score < score:
                best[rec.iri] = Candidate(rec.iri, rec.label, rec.types, rec.module, score, rec.value_of)
        cands = sorted(best.values(), key=lambda c: -c.score)[: self.max_candidates]
        capitalized = any((w[:1].isupper() and not (sentence_start and k == 0)) or any(ch.isdigit() for ch in w) or (w.isupper() and len(w) > 1) for k, (w, _, _) in enumerate(words))
        return Mention(raw, span, cands, capitalized=capitalized, schema_overlap=norm in self.schema_terms)

    # ---- disambiguation ----
    def _needs_choice(self, m: Mention) -> bool:
        return len(m.candidates) > 1 or m.schema_overlap or m.candidates[0].score < 0.9

    def _describe(self, c: Candidate) -> str:
        if c.value_of:
            return f'The value "{c.label}" of the {self.cat.label(c.value_of)} property (in the {c.module} graph)'
        types = [self.cat.classes[t] for t in c.types if t in self.cat.classes]
        kind = "; ".join(f"a {t.label}: {t.description}" for t in types) or "an entity"
        return f"{c.label}, {kind} (in the {c.module} graph)"

    def _options(self, m: Mention) -> tuple[dict[str, str], dict[str, Candidate], dict[str, float]]:
        criteria: dict[str, str] = {}
        keys: dict[str, Candidate] = {}
        prior: dict[str, float] = {}
        for c in m.candidates:
            type_label = self.cat.label(c.value_of) if c.value_of else self.cat.label(c.types[0]) if c.types else "entity"
            key = f"{c.label} [{type_label}]"
            n = 2
            while key in criteria:
                key, n = f"{c.label} [{type_label} {n}]", n + 1
            criteria[key] = self._describe(c)
            keys[key] = c
            prior[key] = c.score
        criteria[NONE_KEY] = f'"{m.text}" is used as a general word or category here, not as the name of one specific thing'
        # Offline-only prior: a lowercase word that is also a schema term is usually not a name.
        none_prior = 0.15 if m.capitalized else (0.8 if m.schema_overlap else 0.3)
        prior = {k: p * (1 - none_prior) for k, p in prior.items()}
        prior[NONE_KEY] = none_prior
        return criteria, keys, prior

    def link(self, question: str) -> LinkResult:
        mentions = self.detect(question)
        result = LinkResult()
        ambiguous = [m for m in mentions if self._needs_choice(m)]
        options = {f"mention_{i}": self._options(m) for i, m in enumerate(ambiguous)}
        if ambiguous:
            questions = {
                name: Choice(criteria=crit, prior=prior, instructions=f'In this question, what does "{m.text}" refer to?')
                for (name, (crit, _, prior)), m in zip(options.items(), ambiguous)
            }
            state = {"question": question, "focus": " ".join(f'"{m.text}"' for m in ambiguous)}
            resp = self.controller.ask(state, questions, stage="linking")
        trace = current_trace()
        for m in mentions:
            if m in ambiguous:
                name = f"mention_{ambiguous.index(m)}"
                dec = resp.choice(name)
                _, keys, _ = options[name]
                m.decision = dec
                m.confidence = dec.confidence
                p_none = dec.probabilities.get(NONE_KEY, 0.0)
                best = max((k for k in dec.probabilities if k != NONE_KEY), key=dec.probabilities.__getitem__)
                is_entity = dec.choice != NONE_KEY or p_none < 0.5
                passed = is_entity and dec.probabilities[best] >= self.gates.entity
                if trace:
                    trace.gate(f"entity:{m.text}", dec.choice, dec.confidence, self.gates.entity, dec.probabilities, passed=passed)
                if is_entity:
                    # An entity we cannot disambiguate confidently is kept (best guess) with its low
                    # confidence, which caps the answer's confidence; dropping it would silently
                    # answer a different question.
                    m.chosen = keys[best]
                    m.confidence = dec.probabilities[best]
                    result.mentions.append(m)
                else:
                    result.rejected.append(m)
            else:
                m.chosen, m.confidence, m.decision = m.candidates[0], m.candidates[0].score, certain(m.candidates[0].label)
                result.mentions.append(m)
        # mentions resolved to a literal value (e.g. a city) become string literals, not anchors
        values = [m for m in result.mentions if m.chosen and m.chosen.value_of]
        result.mentions = [m for m in result.mentions if m not in values]
        strings = [LiteralMention("string", m.text, m.span, value=m.chosen.label, prop=m.chosen.value_of) for m in values]
        found = extract_literals(question, [m.span for m in result.mentions] + [m.span for m in values])
        result.literals = sorted(found + strings, key=lambda l: l.span)
        return result
