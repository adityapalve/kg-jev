import datetime as dt

from kgqa.linking.literals import extract_literals
from kgqa.text import BM25, normalize, stem, stems


def test_stemmer_is_consistent_across_inflections():
    assert stem("hired") == stem("hire")
    assert stem("prices") == stem("price")
    assert stem("cheapest") == stem("cheap")
    assert stems("Vehicle models") == stems("vehicle model")


def test_normalize_handles_possessive_and_camel_case():
    assert normalize("Defender's engine") == "defender engine"


def test_bm25_ranks_matching_doc_first():
    bm = BM25([["engine", "horsepow"], ["base", "pric"], ["region"]])
    assert bm.top(["pric"], 1)[0][0] == 1


def test_quarter_and_year_literals():
    lits = extract_literals("How many orders in Q3 2025 versus 2024?")
    q3, y = lits
    assert q3.kind == "date_range" and q3.low == dt.date(2025, 7, 1) and q3.high == dt.date(2025, 9, 30)
    assert y.kind == "year" and y.value == 2024


def test_numbers_with_currency_and_multiplier():
    (a,) = extract_literals("cost more than $50,000")
    (b,) = extract_literals("under 45k")
    assert a.value == 50000 and b.value == 45000


def test_blocked_spans_are_skipped():
    q = "Is the Model 3 made by Tesla?"
    start = q.index("Model 3")
    assert extract_literals(q, [(start, start + len("Model 3"))]) == []
