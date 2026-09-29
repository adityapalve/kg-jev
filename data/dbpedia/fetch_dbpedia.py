"""Materialize a films / people / places slice of DBpedia, plus the DBpedia ontology.

Films: every dbo:Film with a director, a cast and a US-dollar box-office gross (~20k).
People: everyone credited on those films (director, cast, writer, producer, composer, ...).
Places: those people's birth/death places and the films' countries, plus the countries of
those places.

Output (N-Triples, one file per module) goes to data/dbpedia/. Each stage is skipped if its file
already exists, so an interrupted run resumes. Polite to the public endpoint: POST, small batches,
a pause between requests, retries with backoff.

Run:  .venv/bin/python data/dbpedia/fetch_dbpedia.py
"""

from __future__ import annotations

import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://dbpedia.org/sparql"
OUT = Path(__file__).parent
DBO = "http://dbpedia.org/ontology/"
PAUSE = 0.4
UA = "kgqa/0.1 (research prototype; slice download)"

PREFIXES = """PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""
USD = "<http://dbpedia.org/datatype/usDollar>"


def post(query: str, accept: str, retries: int = 5) -> str:
    data = urllib.parse.urlencode({"query": PREFIXES + query}).encode()
    delay = 5.0
    for attempt in range(retries):
        req = urllib.request.Request(ENDPOINT, data=data, headers={"Accept": accept, "User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read().decode()
            time.sleep(PAUSE)
            return body
        except (urllib.error.URLError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 400:
                raise  # a malformed query does not get better with retries
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1} after {e}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def select_column(query: str) -> list[str]:
    import json

    rows = json.loads(post(query, "application/sparql-results+json"))["results"]["bindings"]
    return [next(iter(r.values()))["value"] for r in rows]


def construct(query: str) -> str:
    return post(query, "application/n-triples")


def values(iris: list[str]) -> str:
    return " ".join(f"<{i}>" for i in iris if not re.search(r'[<>"{}|^`\\\s]', i))


def construct_split(template: str, iris: list[str]) -> str:
    """Run one batch; on HTTP 400 (a malformed IRI) bisect so one bad IRI only loses itself."""
    try:
        return construct(template.replace("{VALUES}", values(iris)))
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        if len(iris) == 1:
            print(f"    skipping unqueryable IRI {iris[0]}", file=sys.stderr)
            return ""
        mid = len(iris) // 2
        return construct_split(template, iris[:mid]) + "\n" + construct_split(template, iris[mid:])


def batched_construct(name: str, iris: list[str], template: str, batch: int) -> str:
    out = []
    for i in range(0, len(iris), batch):
        chunk = iris[i : i + batch]
        out.append(construct_split(template, chunk))
        if (i // batch) % 10 == 0:
            print(f"  {name}: {min(i + batch, len(iris))}/{len(iris)}", file=sys.stderr)
    return "\n".join(out)


def normalize(nt: str) -> str:
    """US-dollar amounts become xsd:double so they compare as numbers; drop duplicate lines."""
    nt = nt.replace(f"^^{USD}", "^^<http://www.w3.org/2001/XMLSchema#double>")
    lines = [l for l in dict.fromkeys(l.strip() for l in nt.splitlines()) if l and not l.startswith("#")]
    return "\n".join(lines) + "\n"


_UCHAR = re.compile(r"\\u([0-9A-Fa-f]{4})|\\U([0-9A-Fa-f]{8})")


def unescape(iri: str) -> str:
    """N-Triples writes non-ASCII as \\uXXXX; Virtuoso rejects those escapes inside a query."""
    return _UCHAR.sub(lambda m: chr(int(m.group(1) or m.group(2), 16)), iri)


def objects(nt: str, props: list[str]) -> list[str]:
    pat = re.compile(r"^<[^>]+>\s+<(" + "|".join(re.escape(p) for p in props) + r")>\s+<([^>]+)>")
    return sorted({unescape(m.group(2)) for l in nt.splitlines() if (m := pat.match(l))})


def stage(path: Path, build) -> str:
    if path.exists():
        print(f"{path.name}: exists, skipping", file=sys.stderr)
        return path.read_text()
    print(f"{path.name}: fetching", file=sys.stderr)
    text = normalize(build())
    path.write_text(text)
    print(f"{path.name}: {text.count(chr(10))} triples", file=sys.stderr)
    return text


FILM = """CONSTRUCT {
  ?f a dbo:Film ; rdfs:label ?l ; dbo:director ?dir ; dbo:starring ?st ; dbo:writer ?w ; dbo:producer ?pr ;
     dbo:musicComposer ?mc ; dbo:cinematography ?ci ; dbo:editing ?ed ; dbo:runtime ?rt ; dbo:gross ?g ;
     dbo:budget ?b ; dbo:country ?co ; dbo:releaseDate ?rd .
} WHERE {
  VALUES ?f { {VALUES} }
  { ?f rdfs:label ?l FILTER(LANG(?l) = "en") } UNION { ?f dbo:director ?dir } UNION { ?f dbo:starring ?st }
  UNION { ?f dbo:writer ?w } UNION { ?f dbo:producer ?pr } UNION { ?f dbo:musicComposer ?mc }
  UNION { ?f dbo:cinematography ?ci } UNION { ?f dbo:editing ?ed } UNION { ?f dbo:runtime ?rt }
  UNION { ?f dbo:gross ?g FILTER(DATATYPE(?g) = <http://dbpedia.org/datatype/usDollar>) }
  UNION { ?f dbo:budget ?b FILTER(DATATYPE(?b) = <http://dbpedia.org/datatype/usDollar>) }
  UNION { ?f dbo:country ?co } UNION { ?f dbo:releaseDate ?rd }
}"""

PERSON = """CONSTRUCT {
  ?p a ?t ; rdfs:label ?l ; dbo:birthDate ?bd ; dbo:deathDate ?dd ; dbo:birthPlace ?bp ; dbo:deathPlace ?dp .
} WHERE {
  VALUES ?p { {VALUES} }
  { ?p a ?t FILTER(STRSTARTS(STR(?t), "http://dbpedia.org/ontology/")) }
  UNION { ?p rdfs:label ?l FILTER(LANG(?l) = "en") }
  UNION { ?p dbo:birthDate ?bd FILTER(DATATYPE(?bd) = xsd:date) } UNION { ?p dbo:deathDate ?dd FILTER(DATATYPE(?dd) = xsd:date) }
  UNION { ?p dbo:birthPlace ?bp } UNION { ?p dbo:deathPlace ?dp }
}"""

PLACE = """CONSTRUCT {
  ?x a ?t ; rdfs:label ?l ; dbo:country ?c ; dbo:populationTotal ?pop .
} WHERE {
  VALUES ?x { {VALUES} }
  { ?x a ?t FILTER(STRSTARTS(STR(?t), "http://dbpedia.org/ontology/")) }
  UNION { ?x rdfs:label ?l FILTER(LANG(?l) = "en") }
  UNION { ?x dbo:country ?c } UNION { ?x dbo:populationTotal ?pop }
}"""


ONTOLOGY = """CONSTRUCT { ?x a ?t ; rdfs:label ?l ; rdfs:comment ?c ; rdfs:subClassOf ?sup ; rdfs:domain ?d ; rdfs:range ?r . }
WHERE {
  VALUES ?x { {VALUES} }
  { ?x a ?t FILTER(?t IN (owl:Class, owl:ObjectProperty, owl:DatatypeProperty)) }
  UNION { ?x rdfs:label ?l FILTER(LANG(?l) = "en") } UNION { ?x rdfs:comment ?c FILTER(LANG(?c) = "en") }
  UNION { ?x rdfs:subClassOf ?sup FILTER(STRSTARTS(STR(?sup), "http://dbpedia.org/ontology/")) }
  UNION { ?x rdfs:domain ?d } UNION { ?x rdfs:range ?r }
}"""


def ontology() -> str:
    # Page IRIs with plain SELECTs: Virtuoso ignores LIMIT/OFFSET in a sub-select inside CONSTRUCT.
    iris: list[str] = []
    for types in ("owl:Class", "owl:ObjectProperty owl:DatatypeProperty"):
        offset = 0
        while True:
            page = select_column(
                f"""SELECT DISTINCT ?x WHERE {{ VALUES ?tt {{ {types} }} ?x a ?tt FILTER(STRSTARTS(STR(?x), "{DBO}")) }}
                ORDER BY ?x LIMIT 10000 OFFSET {offset}"""
            )
            iris += page
            if len(page) < 10000:
                break
            offset += 10000
    print(f"  ontology: {len(iris)} classes and properties", file=sys.stderr)
    return batched_construct("ontology", iris, ONTOLOGY, 200)


def main() -> None:
    stage(OUT / "ontology.nt", ontology)

    def films() -> str:
        iris: list[str] = []
        for offset in range(0, 40000, 10000):
            page = select_column(
                f"""SELECT DISTINCT ?f WHERE {{ ?f a dbo:Film ; dbo:director ?d ; dbo:starring ?s ; dbo:gross ?g
                FILTER(DATATYPE(?g) = {USD}) }} ORDER BY ?f LIMIT 10000 OFFSET {offset}"""
            )
            iris += page
            if len(page) < 10000:
                break
        print(f"  films: {len(iris)} IRIs", file=sys.stderr)
        return batched_construct("films", iris, FILM, 60)

    films_nt = stage(OUT / "films.nt", films)
    credits = [DBO + p for p in ("director", "starring", "writer", "producer", "musicComposer", "cinematography", "editing")]
    people_nt = stage(OUT / "people.nt", lambda: batched_construct("people", objects(films_nt, credits), PERSON, 400))

    def places() -> str:
        first = sorted(set(objects(people_nt, [DBO + "birthPlace", DBO + "deathPlace"])) | set(objects(films_nt, [DBO + "country"])))
        nt = batched_construct("places", first, PLACE, 400)
        more = sorted(set(objects(nt, [DBO + "country"])) - set(first))
        if more:
            nt += "\n" + batched_construct("countries", more, PLACE, 400)
        return nt

    stage(OUT / "places.nt", places)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
