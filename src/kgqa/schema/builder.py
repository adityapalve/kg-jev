"""Build schema cards from ontology + SHACL shapes + data statistics.

Precedence for each fact: SHACL shape > ontology axiom > observed data. Graphs without an
ontology (e.g. a Wikidata slice) still get cards from statistics alone, with labels looked up in
the store where available.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pyoxigraph as ox

from kgqa.config import ModuleConfig
from kgqa.rdf import KGQA, OWL, RDFS, SH, SKOS, XSD, local_name
from kgqa.schema.cards import ClassCard, JoinLink, ModuleCard, PropertyCard, SchemaCatalog
from kgqa.store.base import Store
from kgqa.store.oxigraph import OxigraphStore
from kgqa.text import split_camel

log = logging.getLogger(__name__)

_SKIP_PROPS = {RDFS + "label", SKOS + "altLabel", SKOS + "prefLabel", RDFS + "comment", "http://schema.org/name", OWL + "sameAs"}


_LITERAL_RANGES = {RDFS + "Literal", "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString", "http://www.w3.org/1999/02/22-rdf-syntax-ns#PlainLiteral"}


def _is_literal_range(rng: str) -> bool:
    # XSD types, rdf:langString, and custom datatypes such as DBpedia's <http://dbpedia.org/datatype/usDollar>
    return rng.startswith(XSD) or rng in _LITERAL_RANGES or "/datatype/" in rng


def _vals(rows: Iterable[dict], *keys: str) -> list[tuple]:
    return [tuple(r.get(k) for k in keys) for r in rows]


class SchemaBuilder:
    def __init__(self, store: Store, modules: list[ModuleConfig], ontology_files: list[str | Path] = (), shapes_files: list[str | Path] = (), sample_values: int = 3, prune_unused: bool = False) -> None:
        self.store = store
        self.modules = modules
        self.sample_values = sample_values
        self.prune_unused = prune_unused
        self.schema = OxigraphStore()
        for f in [*ontology_files, *shapes_files]:
            self.schema.load_file(f, None)

    # ------------------------------------------------------------------ ontology
    def _ontology_classes(self, cat: SchemaCatalog) -> None:
        q = f"""SELECT ?c ?l ?cm WHERE {{
            VALUES ?t {{ <{OWL}Class> <{RDFS}Class> }} ?c a ?t .
            OPTIONAL {{ ?c <{RDFS}label> ?l }} OPTIONAL {{ ?c <{RDFS}comment> ?cm }} FILTER(isIRI(?c)) }}"""
        for c, label, comment in _vals(self.schema.select(q).rows, "c", "l", "cm"):
            cat.classes.setdefault(c, ClassCard(iri=c, label=label or split_camel(local_name(c)), module="", description=comment or ""))
        for c, parent in _vals(self.schema.select(f"SELECT ?c ?p WHERE {{ ?c <{RDFS}subClassOf> ?p FILTER(isIRI(?p)) }}").rows, "c", "p"):
            if c in cat.classes:
                cat.classes[c].parents.append(parent)
                cat.classes.setdefault(parent, ClassCard(iri=parent, label=split_camel(local_name(parent)), module=""))
                cat.classes[parent].children.append(c)
        for c, alt in _vals(self.schema.select(f"SELECT ?c ?a WHERE {{ ?c <{SKOS}altLabel> ?a }}").rows, "c", "a"):
            if c in cat.classes:
                cat.classes[c].alt_labels.append(alt)

    def _ontology_properties(self, cat: SchemaCatalog) -> None:
        q = f"""SELECT ?p ?t ?l ?cm ?d ?r ?scheme WHERE {{
            VALUES ?t {{ <{OWL}ObjectProperty> <{OWL}DatatypeProperty> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> }}
            ?p a ?t .
            OPTIONAL {{ ?p <{RDFS}label> ?l }} OPTIONAL {{ ?p <{RDFS}comment> ?cm }}
            OPTIONAL {{ ?p <{RDFS}domain> ?d }} OPTIONAL {{ ?p <{RDFS}range> ?r }}
            OPTIONAL {{ ?p <{KGQA}identifierScheme> ?scheme }} }}"""
        for p, t, label, comment, dom, rng, scheme in _vals(self.schema.select(q).rows, "p", "t", "l", "cm", "d", "r", "scheme"):
            card = cat.properties.get(p)
            if card is None:
                kind = "object" if t == OWL + "ObjectProperty" else "datatype"
                card = cat.properties[p] = PropertyCard(iri=p, label=label or split_camel(local_name(p)), module="", kind=kind, description=comment or "")
            if dom and dom not in card.domains:
                card.domains.append(dom)
            if rng:
                if _is_literal_range(rng):
                    card.kind, card.datatype = "datatype", rng
                else:
                    card.kind, card.range_class = "object", rng
            if scheme:
                card.identifier_scheme = scheme
        for p, alt in _vals(self.schema.select(f"SELECT ?p ?a WHERE {{ ?p <{SKOS}altLabel> ?a }}").rows, "p", "a"):
            if p in cat.properties:
                cat.properties[p].alt_labels.append(alt)
        for a, b in _vals(self.schema.select(f"SELECT ?a ?b WHERE {{ ?a <{KGQA}joinsWith> ?b }}").rows, "a", "b"):
            for x, y in ((a, b), (b, a)):
                if x in cat.properties and y not in cat.properties[x].joins_with:
                    cat.properties[x].joins_with.append(y)

    def _shapes(self, cat: SchemaCatalog) -> None:
        q = f"""SELECT ?tc ?path ?dt ?cls ?min ?max ?name ?desc WHERE {{
            ?s <{SH}targetClass> ?tc ; <{SH}property> ?ps . ?ps <{SH}path> ?path .
            OPTIONAL {{ ?ps <{SH}datatype> ?dt }} OPTIONAL {{ ?ps <{SH}class> ?cls }}
            OPTIONAL {{ ?ps <{SH}minCount> ?min }} OPTIONAL {{ ?ps <{SH}maxCount> ?max }}
            OPTIONAL {{ ?ps <{SH}name> ?name }} OPTIONAL {{ ?ps <{SH}description> ?desc }}
            FILTER(isIRI(?path)) }}"""
        for tc, path, dt, cls, mn, mx, name, desc in _vals(self.schema.select(q).rows, "tc", "path", "dt", "cls", "min", "max", "name", "desc"):
            card = cat.properties.get(path)
            if card is None:
                card = cat.properties[path] = PropertyCard(iri=path, label=name or split_camel(local_name(path)), module="", kind="object" if cls else "datatype", description=desc or "")
            if tc not in card.domains:
                card.domains.insert(0, tc)
            if dt:
                card.kind, card.datatype = "datatype", dt
            if cls:
                card.kind, card.range_class = "object", cls
            card.min_count = mn if mn is not None else card.min_count
            card.max_count = mx if mx is not None else card.max_count
            cat.classes.setdefault(tc, ClassCard(iri=tc, label=split_camel(local_name(tc)), module=""))

    def _name_aliases(self, cat: SchemaCatalog) -> None:
        """The IRI's own name is often the everyday word (dbo:Film is labelled "movie", dbo:writer
        "auteur"), so keep it, and any extra labels, as aliases."""
        extra: dict[str, list[str]] = {}
        for x, label in _vals(self.schema.select(f"SELECT ?x ?l WHERE {{ ?x <{RDFS}label> ?l }}").rows, "x", "l"):
            extra.setdefault(x, []).append(str(label))
        for card in [*cat.classes.values(), *cat.properties.values()]:
            names = [*extra.get(card.iri, []), split_camel(local_name(card.iri)).lower()]
            for n in names:
                if n and n.lower() != card.label.lower() and n not in card.alt_labels:
                    card.alt_labels.append(n)

    # ------------------------------------------------------------------ statistics
    def _stats(self, cat: SchemaCatalog, m: ModuleConfig) -> tuple[dict[str, int], dict[str, int]]:
        g, tp = m.graph, m.type_predicate
        class_counts = {c: n for c, n in _vals(self.store.select(f"SELECT ?c (COUNT(DISTINCT ?s) AS ?n) WHERE {{ GRAPH <{g}> {{ ?s <{tp}> ?c }} }} GROUP BY ?c").rows, "c", "n")}
        prop_counts = {p: n for p, n in _vals(self.store.select(f"SELECT ?p (COUNT(*) AS ?n) WHERE {{ GRAPH <{g}> {{ ?s ?p ?o }} }} GROUP BY ?p").rows, "p", "n")}
        prop_counts.pop(tp, None)
        for c, n in class_counts.items():
            card = cat.classes.setdefault(c, ClassCard(iri=c, label=split_camel(local_name(c)), module=m.name))
            card.instance_count += n
            if not card.module:
                card.module = m.name
        # observed domains, object ranges and literal datatypes, for properties the schema did not describe
        dom_rows = self.store.select(f"SELECT ?c ?p (COUNT(DISTINCT ?s) AS ?n) WHERE {{ GRAPH <{g}> {{ ?s <{tp}> ?c ; ?p ?o }} }} GROUP BY ?c ?p").rows
        rng_rows = self.store.select(f"SELECT ?p ?rc (COUNT(*) AS ?n) WHERE {{ GRAPH <{g}> {{ ?s ?p ?o FILTER(isIRI(?o)) }} GRAPH ?g2 {{ ?o <{tp}> ?rc }} }} GROUP BY ?p ?rc").rows
        dt_rows = self.store.select(f"SELECT ?p (SAMPLE(DATATYPE(?o)) AS ?dt) WHERE {{ GRAPH <{g}> {{ ?s ?p ?o FILTER(isLiteral(?o)) }} }} GROUP BY ?p").rows
        observed_dom: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for c, p, n in _vals(dom_rows, "c", "p", "n"):
            observed_dom[p].append((c, n))
        observed_rng: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for p, rc, n in _vals(rng_rows, "p", "rc", "n"):
            observed_rng[p].append((rc, n))
        observed_dt = {p: dt for p, dt in _vals(dt_rows, "p", "dt")}
        for p, n in prop_counts.items():
            if p in _SKIP_PROPS:
                continue
            card = cat.properties.get(p)
            if card is None:
                kind = "object" if p in observed_rng else "datatype"
                card = cat.properties[p] = PropertyCard(iri=p, label=split_camel(local_name(p)), module=m.name, kind=kind)
            card.module_usage[m.name] = n
            if n > card.usage:
                card.usage, card.module = n, m.name
            if observed_dom.get(p):
                # Data beats the ontology when they disagree: add subject classes the declared domains
                # do not cover (e.g. dbo:country is used on films and on places).
                total = {c: class_counts.get(c, 0) for c, _ in observed_dom[p]}
                seen = [c for c, k in sorted(observed_dom[p], key=lambda x: -x[1]) if k >= 0.1 * max(total[c], 1)]
                for c in seen:
                    if not any(cat.is_a(c, d) for d in card.domains):
                        card.domains.append(c)
            if card.kind == "object" and observed_rng.get(p):
                # Data beats the ontology here too: dbo:producer is declared to point at Agent, but in
                # current DBpedia a Person is an Animal, not an Agent. Add the most specific observed
                # target classes the declared range does not cover.
                top = max(n for _, n in observed_rng[p])
                frequent = sorted((rc for rc, n in observed_rng[p] if n >= 0.1 * top), key=lambda c: -len(cat.ancestors(c)))
                for rc in frequent:
                    covered = [card.range_class, *card.extra_ranges]
                    if any(c and (cat.is_a(rc, c) or cat.is_a(c, rc)) for c in covered):
                        continue
                    if card.range_class is None:
                        card.range_class = rc
                    else:
                        card.extra_ranges.append(rc)
            observed = observed_dt.get(p)
            if card.kind == "datatype" and (not card.datatype or (observed and observed.startswith(XSD) and not card.datatype.startswith(XSD))):
                card.datatype = observed  # what the data actually holds wins over a custom declared range
            if card.kind == "object" and p not in observed_rng and observed:
                card.kind, card.range_class, card.datatype = "datatype", None, observed  # declared object, holds literals
            samples = self.store.select(
                f"SELECT ?o (SAMPLE(?lbl) AS ?l) WHERE {{ GRAPH <{g}> {{ ?s <{p}> ?o }} OPTIONAL {{ GRAPH ?g2 {{ ?o <{RDFS}label> ?lbl }} }} }} GROUP BY ?o LIMIT {self.sample_values}"
            ).rows
            card.samples = [str(r["l"] if r.get("l") is not None else r["o"]) for r in samples]
        return class_counts, prop_counts

    def _store_labels(self, cat: SchemaCatalog) -> None:
        """Label classes and properties the ontology did not cover, using labels found in the data."""
        unlabeled = [iri for iri, c in [*cat.classes.items(), *cat.properties.items()] if c.label == split_camel(local_name(iri))]
        for i in range(0, len(unlabeled), 200):
            chunk = " ".join(f"<{x}>" for x in unlabeled[i : i + 200])
            q = f"""SELECT ?x ?l WHERE {{ VALUES ?x {{ {chunk} }} GRAPH ?g {{ ?x <{RDFS}label> ?l }} FILTER(lang(?l) = "" || langMatches(lang(?l), "en")) }}"""
            for x, label in _vals(self.store.select(q).rows, "x", "l"):
                card = cat.classes.get(x) or cat.properties.get(x)
                if card is not None:
                    card.label = label

    def _joins(self, cat: SchemaCatalog) -> None:
        by_scheme: dict[str, list[str]] = defaultdict(list)
        for p in cat.properties.values():
            if p.identifier_scheme:
                by_scheme[p.identifier_scheme].append(p.iri)
        for props in by_scheme.values():
            for a in props:
                for b in props:
                    if a != b and cat.properties[a].module != cat.properties[b].module and b not in cat.properties[a].joins_with:
                        cat.properties[a].joins_with.append(b)
        seen = set()
        for p in cat.properties.values():
            for other in p.joins_with:
                key = tuple(sorted((p.iri, other)))
                if key in seen or other not in cat.properties:
                    continue
                seen.add(key)
                q = cat.properties[other]
                ga, gb = cat.modules[p.module].graph, cat.modules[q.module].graph
                overlap = self.store.select(f"SELECT (COUNT(DISTINCT ?v) AS ?n) WHERE {{ GRAPH <{ga}> {{ ?a <{p.iri}> ?v }} GRAPH <{gb}> {{ ?b <{q.iri}> ?v }} }}").column("n")
                cat.joins.append(JoinLink(p.iri, other, p.module, q.module, overlap[0] if overlap else 0))

    # ------------------------------------------------------------------ assemble
    def build(self) -> SchemaCatalog:
        cat = SchemaCatalog()
        self._ontology_classes(cat)
        self._ontology_properties(cat)
        self._shapes(cat)
        self._name_aliases(cat)
        for m in self.modules:
            cat.modules[m.name] = ModuleCard(name=m.name, graph=m.graph, description=m.description, type_predicate=m.type_predicate)
        triple_counts = {}
        for m in self.modules:
            _, prop_counts = self._stats(cat, m)
            triple_counts[m.name] = sum(prop_counts.values())
        if self.prune_unused:
            self._prune(cat)
        self._store_labels(cat)
        self._assign_modules(cat)
        for p in cat.properties.values():
            for d in p.domains:
                if d in cat.classes and p.iri not in cat.classes[d].properties:
                    cat.classes[d].properties.append(p.iri)
        self._joins(cat)
        for name, mc in cat.modules.items():
            mc.triple_count = triple_counts.get(name, 0)
            mc.classes = sorted(c for c, card in cat.classes.items() if card.module == name)
            mc.top_classes = [c for c in mc.classes if not any(cat.classes.get(p) and cat.classes[p].module == name for p in cat.classes[c].parents)]
            linked = set()
            for c in mc.classes:
                for p in cat.properties_of(c):
                    if p.module != name:
                        linked.add(p.module)
                    if p.range_class and p.range_class in cat.classes and cat.classes[p.range_class].module != name:
                        linked.add(cat.classes[p.range_class].module)
            for j in cat.joins:
                if name in (j.left_module, j.right_module):
                    linked.add(j.right_module if j.left_module == name else j.left_module)
            mc.linked_modules = sorted(linked - {name, ""})
        return cat

    def _prune(self, cat: SchemaCatalog) -> None:
        """Drop ontology classes and properties the data never uses (large ontologies like DBpedia's
        declare thousands). A class stays if it or a descendant has instances; a property stays if used."""
        keep = {c for c, card in cat.classes.items() if card.instance_count > 0}
        for c in list(keep):
            keep.update(cat.ancestors(c))
        for c in list(cat.classes):
            if c not in keep:
                del cat.classes[c]
        for card in cat.classes.values():
            card.parents = [p for p in card.parents if p in keep]
            card.children = [ch for ch in card.children if ch in keep]
        for p in [p for p, card in cat.properties.items() if card.usage == 0]:
            del cat.properties[p]
        for card in cat.properties.values():
            card.domains = [d for d in card.domains if d in keep]
            card.extra_ranges = [r for r in card.extra_ranges if r in keep]
            if card.range_class and card.range_class not in keep:
                card.range_class = card.extra_ranges.pop(0) if card.extra_ranges else None

    def _assign_modules(self, cat: SchemaCatalog) -> None:
        # Classes without direct instances (abstract parents, value classes) inherit a module from
        # their descendants, then from properties that use them as domain.
        default = self.modules[0].name if self.modules else ""
        for c, card in cat.classes.items():
            if card.module:
                continue
            for d in cat.descendants(c):
                if cat.classes[d].module:
                    card.module = cat.classes[d].module
                    break
        for c, card in cat.classes.items():
            if not card.module:
                mods = [p.module for p in cat.properties.values() if c in p.domains and p.module]
                card.module = mods[0] if mods else default
        for p in cat.properties.values():
            if not p.module:
                p.module = next((cat.classes[d].module for d in p.domains if d in cat.classes), default)
