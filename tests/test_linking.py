import pytest
from conftest import D, O

from kgqa.jev import make_controller
from kgqa.linking import EntityLinker


@pytest.fixture
def linker(index, catalog, cfg):
    return EntityLinker(index, catalog, make_controller("heuristic"), cfg.gates)


def test_alias_links_to_entity(linker):
    r = linker.link("What is the base price of the Defender?")
    assert [m.iri for m in r.mentions] == [D + "model/defender"]


def test_ambiguous_mention_is_disambiguated_by_context(linker):
    r = linker.link("How many orders did Mustang place in Q3 2025?")
    assert r.mentions[0].iri == D + "dealer/mustang"
    assert r.mentions[0].decision is not None and len(r.mentions[0].decision.probabilities) == 3  # two candidates + none
    assert r.literals[0].kind == "date_range"


def test_lowercase_schema_word_is_not_an_entity(linker):
    r = linker.link("What was the total sales revenue for the F-150 in 2025?")
    assert [m.iri for m in r.mentions] == [D + "model/f150"]
    assert [m.text for m in r.rejected] == ["sales"]


def test_literal_values_become_string_filters(linker):
    r = linker.link("Which dealers are in Denver?")
    assert r.mentions == []
    (lit,) = r.literals
    assert lit.kind == "string" and lit.value == "Denver" and lit.prop == O + "city"


def test_plural_value_mention(linker):
    (lit,) = linker.link("How many Sales Managers are there?").literals
    assert lit.value == "Sales Manager"


def test_demonym_surname_and_popularity_aliases():
    from kgqa.linking.index import DEMONYM, SURNAME, EntityIndex, EntityRecord

    P = "http://ex.org/Person"
    idx = EntityIndex({
        "dk": EntityRecord("dk", "Denmark"),
        "ak": EntityRecord("ak", "Akira Kurosawa", types=[P], popularity=40),
        "rt": EntityRecord("rt", "Robert Taylor (actor)", types=[P], popularity=2),
        "et": EntityRecord("et", "Elizabeth Taylor", types=[P], popularity=90),
        "ti": EntityRecord("ti", "Titanic (1997 film)"),
    })
    assert idx.lookup_exact("danish")[0][0].iri == "dk" and idx.lookup_exact("danish")[0][1] >= DEMONYM
    assert idx.lookup_exact("kurosawa")[0][0].iri == "ak"
    assert [r.iri for r, _ in idx.lookup_exact("taylor")] == ["et", "rt"]  # the more popular Taylor first
    assert all(score < 0.9 for _, score in idx.lookup_exact("taylor"))  # weak: the linker will ask jev
    assert idx.lookup_exact("titanic")[0][0].iri == "ti"
    assert SURNAME < DEMONYM
