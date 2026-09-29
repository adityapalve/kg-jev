"""Small text utilities shared by linking, retrieval and the offline controller.

Nothing here is clever: tokenization, a light stemmer, BM25 and character trigram similarity.
They exist so the pipeline runs end to end without an embedding service.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

STOPWORDS = frozenset(
    """a an the of in on at to for from by with and or is are was were be been being do does did
    has have had it its this that these those there their them they he she his her i me my we our
    you your as into than then so such can could would should will shall may might must
    what which who whom whose when where why how please tell show give list""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def normalize(text: str) -> str:
    """Lowercase, strip accents, turn possessives and punctuation into spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"['’]s\b", "", text)
    return " ".join(_TOKEN.findall(text.lower()))


def split_camel(text: str) -> str:
    return _CAMEL.sub(" ", text).replace("_", " ")


def stem(token: str) -> str:
    """Tiny suffix stripper. Both sides of every comparison go through it, so it only has to be
    consistent, not linguistically right (hire/hired -> hir, price/prices -> pric)."""
    if len(token) <= 3 or token.isdigit():
        return token
    for suffix, repl, min_len in (("ies", "y", 5), ("sses", "ss", 5), ("ing", "", 6), ("est", "", 7), ("ed", "", 5), ("es", "e", 5), ("s", "", 4)):
        if token.endswith(suffix) and len(token) >= min_len and not token.endswith("ss"):
            token = token[: -len(suffix)] + repl
            break
    if token.endswith("e") and len(token) > 3:
        token = token[:-1]
    return token


def tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    toks = normalize(split_camel(text)).split()
    return toks if keep_stopwords else [t for t in toks if t not in STOPWORDS]


def stems(text: str, *, keep_stopwords: bool = False) -> list[str]:
    return [stem(t) for t in tokens(text, keep_stopwords=keep_stopwords)]


def bigrams(seq: Sequence[str]) -> list[tuple[str, str]]:
    return list(zip(seq, seq[1:]))


def trigrams(text: str) -> set[str]:
    s = f"  {normalize(text)} "
    return {s[i : i + 3] for i in range(len(s) - 2)}


def trigram_similarity(a: str, b: str) -> float:
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def flatten_json(value: object) -> str:
    """Render the text content of a JSON-ish value (criteria descriptions can be objects)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{k} {flatten_json(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(flatten_json(v) for v in value)
    return str(value)


class BM25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, docs: Iterable[Sequence[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.docs = [Counter(d) for d in docs]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def score(self, query: Sequence[str], index: int) -> float:
        doc, length = self.docs[index], self.lengths[index]
        total = 0.0
        for t in set(query):
            f = doc.get(t)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            total += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
        return total

    def top(self, query: Sequence[str], k: int) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: -s[1])
        return scored[:k]


def approx_tokens(text: str) -> int:
    """Rough LLM token estimate (about four characters per token)."""
    return max(1, len(text) // 4)
