"""Typed literal mentions: quarters, months, dates, years, numbers and amounts."""

from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m} | {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
_ORD = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "last": 4}


@dataclass
class LiteralMention:
    kind: str  # "date_range" | "date" | "year" | "number" | "string"
    text: str
    span: tuple[int, int]
    value: Any = None
    low: Any = None
    high: Any = None
    prop: str | None = None  # for "string": the property whose value this is

    def describe(self) -> str:
        if self.kind == "string":
            return f'{self.text} = "{self.value}"'
        if self.kind == "date_range":
            return f"{self.text} = the period {self.low.isoformat()} to {self.high.isoformat()}"
        if self.kind == "year":
            return f"{self.text} = the year {self.value}"
        if self.kind == "date":
            return f"{self.text} = the date {self.value.isoformat()}"
        return f"{self.text} = the number {self.value}"


def _quarter(year: int, q: int) -> tuple[dt.date, dt.date]:
    start = dt.date(year, 3 * (q - 1) + 1, 1)
    end_month = 3 * q
    return start, dt.date(year, end_month, calendar.monthrange(year, end_month)[1])


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("quarter", re.compile(r"\bQ([1-4])\s*(?:of\s+)?((?:19|20)\d{2})\b", re.I)),
    ("quarter_word", re.compile(r"\b(first|second|third|fourth|last|1st|2nd|3rd|4th)\s+quarter\s+(?:of\s+)?((?:19|20)\d{2})\b", re.I)),
    ("iso_date", re.compile(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b")),
    ("month_year", re.compile(r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\s+((?:19|20)\d{2})\b", re.I)),
    ("year", re.compile(r"(?<![\d$.,-])((?:19|20)\d{2})(?![\d,.]\d|-\d)")),
    ("number", re.compile(r"(?<![\w.])\$?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k|thousand|million|m\b)?", re.I)),
]


def extract_literals(text: str, blocked: list[tuple[int, int]] = ()) -> list[LiteralMention]:
    """Find literal mentions outside the `blocked` spans (entity mentions), longest pattern first."""
    taken = list(blocked)
    out: list[LiteralMention] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            a, b = m.span()
            if not free(a, b):
                continue
            lit = None
            if kind == "quarter":
                lo, hi = _quarter(int(m[2]), int(m[1]))
                lit = LiteralMention("date_range", m[0], (a, b), low=lo, high=hi)
            elif kind == "quarter_word":
                lo, hi = _quarter(int(m[2]), _ORD[m[1].lower()])
                lit = LiteralMention("date_range", m[0], (a, b), low=lo, high=hi)
            elif kind == "iso_date":
                try:
                    d = dt.date(int(m[1]), int(m[2]), int(m[3]))
                except ValueError:
                    continue
                lit = LiteralMention("date", m[0], (a, b), value=d, low=d, high=d)
            elif kind == "month_year":
                y, mo = int(m[2]), _MONTHS[m[1].lower()]
                lit = LiteralMention("date_range", m[0], (a, b), low=dt.date(y, mo, 1), high=dt.date(y, mo, calendar.monthrange(y, mo)[1]))
            elif kind == "year":
                y = int(m[1])
                lit = LiteralMention("year", m[0], (a, b), value=y, low=dt.date(y, 1, 1), high=dt.date(y, 12, 31))
            elif kind == "number":
                raw = m[1].replace(",", "")
                val: float = float(raw)
                mult = (m[2] or "").lower()
                val *= {"k": 1e3, "thousand": 1e3, "million": 1e6, "m": 1e6}.get(mult, 1)
                lit = LiteralMention("number", m[0].strip(), (a, b), value=int(val) if val.is_integer() else val)
            if lit:
                out.append(lit)
                taken.append((a, b))
    out.sort(key=lambda l: l.span)
    return out
