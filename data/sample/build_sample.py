"""Generate the sample multi-module enterprise graph (deterministic).

Three modules, each its own named graph:
  catalog  manufacturers, vehicle models (SUV/Sedan/Coupe/Pickup/Hatchback), engines, fuel types
  sales    dealers, regions, orders. Orders reference models by *model code* (a shared
           identifier literal), not by IRI, so catalog <-> sales questions need a join hop.
  people   employees and departments; employees link to dealers in the sales graph by IRI.

Run:  python data/sample/build_sample.py
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
O = "http://example.org/onto#"
D = "http://example.org/id/"

PREFIXES = f"""@prefix o: <{O}> .
@prefix d: <{D}> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix kgqa: <https://kgqa.dev/ns#> .
"""

MANUFACTURERS = {
    "landrover": ("Land Rover", ["LR"], "United Kingdom", 1948),
    "ford": ("Ford", ["Ford Motor Company"], "United States", 1903),
    "tesla": ("Tesla", [], "United States", 2003),
    "toyota": ("Toyota", [], "Japan", 1937),
    "volvo": ("Volvo", ["Volvo Cars"], "Sweden", 1927),
    "jaguar": ("Jaguar", [], "United Kingdom", 1922),
}

FUELS = {"electric": ("Electric", ["EV", "battery electric"]), "petrol": ("Petrol", ["gasoline", "gas"]), "diesel": ("Diesel", []), "hybrid": ("Hybrid", ["hybrid electric"])}

# key: (label, alt labels, class, manufacturer, code, launch year, base price, seats, engine key)
MODELS = {
    "defender": ("Land Rover Defender", ["Defender"], "SUV", "landrover", "DEF-20", 2020, 56900, 5, "p400"),
    "discovery": ("Land Rover Discovery", ["Discovery"], "SUV", "landrover", "DSC-17", 2017, 59000, 7, "d300"),
    "evoque": ("Range Rover Evoque", ["Evoque"], "SUV", "landrover", "EVQ-19", 2019, 47000, 5, "p250"),
    "mustang": ("Ford Mustang", ["Mustang"], "Coupe", "ford", "MUS-15", 2015, 32000, 4, "coyote"),
    "f150": ("Ford F-150", ["F-150", "F150"], "Pickup", "ford", "F15-21", 2021, 36000, 5, "ecoboost35"),
    "lightning": ("Ford F-150 Lightning", ["F-150 Lightning", "Lightning"], "Pickup", "ford", "F1L-22", 2022, 52000, 5, "dualmotor580"),
    "ranger": ("Ford Ranger", ["Ranger"], "Pickup", "ford", "RNG-19", 2019, 28000, 5, "ecoboost23"),
    "model3": ("Tesla Model 3", ["Model 3"], "Sedan", "tesla", "TM3-17", 2017, 40000, 5, "rwd283"),
    "modely": ("Tesla Model Y", ["Model Y"], "SUV", "tesla", "TMY-20", 2020, 45000, 7, "dualmotor384"),
    "corolla": ("Toyota Corolla", ["Corolla"], "Sedan", "toyota", "COR-19", 2019, 22000, 5, "m20a"),
    "rav4": ("Toyota RAV4", ["RAV4"], "SUV", "toyota", "RAV-19", 2019, 29000, 5, "a25hybrid"),
    "prius": ("Toyota Prius", ["Prius"], "Hatchback", "toyota", "PRI-16", 2016, 28000, 5, "2zrhybrid"),
    "xc90": ("Volvo XC90", ["XC90"], "SUV", "volvo", "XC9-15", 2015, 57000, 7, "b6"),
    "ex30": ("Volvo EX30", ["EX30"], "SUV", "volvo", "EX3-24", 2024, 35000, 5, "single268"),
    "ipace": ("Jaguar I-Pace", ["I-Pace"], "SUV", "jaguar", "IPC-18", 2018, 72000, 5, "dual394"),
    "ftype": ("Jaguar F-Type", ["F-Type"], "Coupe", "jaguar", "FTP-13", 2013, 70000, 2, "v8575"),
}

# key: (label, fuel, horsepower)
ENGINES = {
    "p400": ("P400 Ingenium straight-six", "petrol", 395),
    "d300": ("D300 Ingenium diesel", "diesel", 296),
    "p250": ("P250 Ingenium four-cylinder", "petrol", 246),
    "coyote": ("Coyote 5.0 V8", "petrol", 480),
    "ecoboost35": ("3.5 EcoBoost V6", "petrol", 400),
    "dualmotor580": ("Extended range dual motor", "electric", 580),
    "ecoboost23": ("2.3 EcoBoost", "petrol", 270),
    "rwd283": ("Rear-wheel drive motor", "electric", 283),
    "dualmotor384": ("Long range dual motor", "electric", 384),
    "m20a": ("M20A Dynamic Force 2.0", "petrol", 169),
    "a25hybrid": ("A25A hybrid system", "hybrid", 219),
    "2zrhybrid": ("2ZR hybrid system", "hybrid", 121),
    "b6": ("B6 mild hybrid", "hybrid", 295),
    "single268": ("Single motor extended range", "electric", 268),
    "dual394": ("EV400 dual motor", "electric", 394),
    "v8575": ("5.0 supercharged V8", "petrol", 575),
}

REGIONS = {"mountain": "Mountain West", "pacific": "Pacific", "northeast": "Northeast"}

# key: (label, alt labels, city, region)
DEALERS = {
    "downtown": ("Downtown Motors", ["Downtown dealer"], "Denver", "mountain"),
    "bayside": ("Bayside Autos", [], "San Francisco", "pacific"),
    "harbor": ("Harbor Vehicles", [], "Seattle", "pacific"),
    "liberty": ("Liberty Cars", [], "Boston", "northeast"),
    "mustang": ("Mustang Auto Group", ["Mustang"], "Cheyenne", "mountain"),
    "empire": ("Empire Motors", [], "New York", "northeast"),
}

DEPARTMENTS = {"sales": "Sales", "service": "Service", "finance": "Finance"}

# key: (label, dealer, department, role, hire date)
EMPLOYEES = {
    "amoreno": ("Alice Moreno", "downtown", "sales", "Sales Manager", "2019-04-01"),
    "bokafor": ("Ben Okafor", "downtown", "service", "Service Technician", "2021-09-15"),
    "cliang": ("Chen Liang", "bayside", "sales", "Sales Associate", "2022-02-01"),
    "dsilva": ("Diana Silva", "bayside", "finance", "Finance Officer", "2018-06-11"),
    "ehansen": ("Erik Hansen", "harbor", "sales", "Sales Manager", "2017-01-09"),
    "fnoor": ("Farah Noor", "harbor", "service", "Service Advisor", "2023-03-20"),
    "gtanaka": ("Gen Tanaka", "liberty", "sales", "Sales Associate", "2020-10-05"),
    "hbrooks": ("Hannah Brooks", "liberty", "finance", "Finance Manager", "2016-07-18"),
    "ikowalski": ("Igor Kowalski", "mustang", "sales", "Sales Manager", "2015-05-25"),
    "jreyes": ("Julia Reyes", "mustang", "service", "Service Technician", "2024-01-08"),
    "kadams": ("Kofi Adams", "empire", "sales", "Sales Associate", "2021-11-29"),
    "lpetrov": ("Lena Petrov", "empire", "sales", "Sales Manager", "2019-08-12"),
}


def lit(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def alts(labels: list[str]) -> str:
    return "".join(f" ;\n    skos:altLabel {lit(a)}" for a in labels)


def ontology() -> str:
    def cls(name: str, label: str, comment: str, alt: list[str] = (), parent: str | None = None) -> str:
        sub = f" ;\n    rdfs:subClassOf o:{parent}" if parent else ""
        return f"o:{name} a owl:Class ;\n    rdfs:label {lit(label)} ;\n    rdfs:comment {lit(comment)}{alts(list(alt))}{sub} .\n"

    def prop(name: str, kind: str, label: str, comment: str, domain: str, rng: str, alt: list[str] = (), extra: str = "") -> str:
        t = "owl:ObjectProperty" if kind == "object" else "owl:DatatypeProperty"
        return f"o:{name} a {t} ;\n    rdfs:label {lit(label)} ;\n    rdfs:comment {lit(comment)}{alts(list(alt))} ;\n    rdfs:domain o:{domain} ;\n    rdfs:range {rng}{extra} .\n"

    parts = [PREFIXES, "\n# ---- catalog ----\n"]
    parts += [
        cls("Manufacturer", "manufacturer", "A company that makes vehicles (car maker, brand).", ["maker", "brand", "company", "automaker"]),
        cls("VehicleModel", "vehicle model", "A vehicle model offered in the product catalog.", ["model", "vehicle", "car", "product"]),
        cls("SUV", "SUV", "Sport utility vehicle model.", ["sport utility vehicle", "suvs"], "VehicleModel"),
        cls("Sedan", "sedan", "Sedan (saloon) car model.", ["saloon"], "VehicleModel"),
        cls("Coupe", "coupe", "Two-door coupe or sports car model.", ["sports car"], "VehicleModel"),
        cls("Pickup", "pickup truck", "Pickup truck model.", ["truck", "pickup"], "VehicleModel"),
        cls("Hatchback", "hatchback", "Hatchback car model.", [], "VehicleModel"),
        cls("Engine", "engine", "A powertrain (combustion engine or electric motor) fitted to a vehicle model.", ["powertrain", "motor"]),
        cls("FuelType", "fuel type", "Kind of energy an engine uses: electric, petrol, diesel or hybrid.", ["fuel", "energy source"]),
        prop("manufacturer", "object", "manufacturer", "The company that makes the vehicle model.", "VehicleModel", "o:Manufacturer", ["made by", "maker", "built by", "brand", "makes", "manufactures", "produced by"]),
        prop("modelCode", "datatype", "model code", "Product identifier shared with the sales system.", "VehicleModel", "xsd:string", ["product code", "sku"], " ;\n    kgqa:identifierScheme \"model-code\""),
        prop("launchYear", "datatype", "launch year", "Year the model was launched.", "VehicleModel", "xsd:integer", ["launched", "introduced", "release year", "came out"]),
        prop("basePrice", "datatype", "base price", "Base list price in US dollars.", "VehicleModel", "xsd:decimal", ["price", "cost", "msrp", "expensive", "cheap", "costs"]),
        prop("seats", "datatype", "seats", "Number of seats.", "VehicleModel", "xsd:integer", ["seating capacity", "passengers"]),
        prop("engine", "object", "engine", "The engine or motor fitted to the model.", "VehicleModel", "o:Engine", ["powertrain", "motor"]),
        prop("fuelType", "object", "fuel type", "Fuel or energy the engine uses.", "Engine", "o:FuelType", ["fuel", "powered by", "runs on"]),
        prop("horsepower", "datatype", "horsepower", "Peak power output in horsepower.", "Engine", "xsd:integer", ["power", "hp", "powerful", "output"]),
        prop("country", "datatype", "country", "Country where the manufacturer is headquartered.", "Manufacturer", "xsd:string", ["headquarters", "based", "nationality"]),
        prop("foundedYear", "datatype", "founded year", "Year the manufacturer was founded.", "Manufacturer", "xsd:integer", ["founded", "established"]),
    ]
    parts += ["\n# ---- sales ----\n"]
    parts += [
        cls("Dealer", "dealer", "A dealership that sells vehicles and places orders.", ["dealership", "retailer", "store"]),
        cls("Region", "sales region", "A geographic sales region grouping dealers.", ["region", "territory", "area"]),
        cls("Order", "order", "A vehicle sales order placed by a dealer.", ["sale", "sales", "purchase", "transaction"]),
        prop("soldModelCode", "datatype", "sold model code", "Model code of the vehicle model sold in this order.", "Order", "xsd:string", ["model sold", "vehicle sold", "sold"],
             " ;\n    kgqa:joinsWith o:modelCode"),
        prop("dealer", "object", "dealer", "Dealer that placed the order.", "Order", "o:Dealer", ["dealership", "sold by", "placed by"]),
        prop("orderDate", "datatype", "order date", "Date the order was placed.", "Order", "xsd:date", ["date", "ordered", "when"]),
        prop("quantity", "datatype", "quantity", "Number of vehicles in the order.", "Order", "xsd:integer", ["units", "vehicles sold", "number of vehicles", "volume"]),
        prop("revenue", "datatype", "revenue", "Order revenue in US dollars.", "Order", "xsd:decimal", ["sales amount", "income", "earned", "turnover", "value"]),
        prop("region", "object", "region", "Sales region the dealer belongs to.", "Dealer", "o:Region", ["territory", "area", "located in"]),
        prop("city", "datatype", "city", "City where the dealer is located.", "Dealer", "xsd:string", ["town", "location", "located"]),
    ]
    parts += ["\n# ---- people ----\n"]
    parts += [
        cls("Employee", "employee", "A person employed at a dealer.", ["staff", "person", "worker", "people"]),
        cls("Department", "department", "Department an employee belongs to.", ["team", "division"]),
        prop("worksAt", "object", "works at", "Dealer where the employee works.", "Employee", "o:Dealer", ["employer", "employed by", "employs", "works for", "staff of"]),
        prop("department", "object", "department", "Department of the employee.", "Employee", "o:Department", ["team", "division"]),
        prop("role", "datatype", "role", "Job title of the employee.", "Employee", "xsd:string", ["job title", "position", "job"]),
        prop("hireDate", "datatype", "hire date", "Date the employee was hired.", "Employee", "xsd:date", ["hired", "joined", "start date"]),
    ]
    return "\n".join(parts)


def shapes() -> str:
    def shape(cls: str, props: list[tuple[str, str, int, int | None]]) -> str:
        blocks = []
        for path, rng, mn, mx in props:
            kind = "sh:datatype" if rng.startswith("xsd:") else "sh:class"
            mx_s = f" ; sh:maxCount {mx}" if mx is not None else ""
            blocks.append(f"[ sh:path o:{path} ; {kind} {rng} ; sh:minCount {mn}{mx_s} ]")
        props_s = " ,\n        ".join(blocks)
        return f"o:{cls}Shape a sh:NodeShape ;\n    sh:targetClass o:{cls} ;\n    sh:property\n        {props_s} .\n"

    return "\n".join([
        PREFIXES,
        shape("VehicleModel", [("manufacturer", "o:Manufacturer", 1, 1), ("modelCode", "xsd:string", 1, 1), ("launchYear", "xsd:integer", 1, 1), ("basePrice", "xsd:decimal", 1, 1), ("seats", "xsd:integer", 0, 1), ("engine", "o:Engine", 1, None)]),
        shape("Engine", [("fuelType", "o:FuelType", 1, 1), ("horsepower", "xsd:integer", 1, 1)]),
        shape("Manufacturer", [("country", "xsd:string", 1, 1), ("foundedYear", "xsd:integer", 0, 1)]),
        shape("Dealer", [("region", "o:Region", 1, 1), ("city", "xsd:string", 1, 1)]),
        shape("Order", [("soldModelCode", "xsd:string", 1, 1), ("dealer", "o:Dealer", 1, 1), ("orderDate", "xsd:date", 1, 1), ("quantity", "xsd:integer", 1, 1), ("revenue", "xsd:decimal", 1, 1)]),
        shape("Employee", [("worksAt", "o:Dealer", 1, 1), ("department", "o:Department", 1, 1), ("role", "xsd:string", 1, 1), ("hireDate", "xsd:date", 0, 1)]),
    ])


def catalog() -> str:
    out = [PREFIXES]
    for k, (label, alt, country, founded) in MANUFACTURERS.items():
        out.append(f"d:mfr/{k} a o:Manufacturer ; rdfs:label {lit(label)}{alts(alt)} ;\n    o:country {lit(country)} ; o:foundedYear {founded} .")
    for k, (label, alt) in FUELS.items():
        out.append(f"d:fuel/{k} a o:FuelType ; rdfs:label {lit(label)}{alts(alt)} .")
    for k, (label, fuel, hp) in ENGINES.items():
        out.append(f"d:engine/{k} a o:Engine ; rdfs:label {lit(label)} ;\n    o:fuelType d:fuel/{fuel} ; o:horsepower {hp} .")
    for k, (label, alt, cls, mfr, code, year, price, seats, engine) in MODELS.items():
        out.append(
            f"d:model/{k} a o:{cls} ; rdfs:label {lit(label)}{alts(alt)} ;\n"
            f"    o:manufacturer d:mfr/{mfr} ; o:modelCode {lit(code)} ; o:launchYear {year} ;\n"
            f'    o:basePrice "{price}.00"^^xsd:decimal ; o:seats {seats} ; o:engine d:engine/{engine} .'
        )
    return "\n".join(out) + "\n"


def sales() -> str:
    rng = random.Random(42)
    out = [PREFIXES]
    for k, label in REGIONS.items():
        out.append(f"d:region/{k} a o:Region ; rdfs:label {lit(label)} .")
    for k, (label, alt, city, region) in DEALERS.items():
        out.append(f"d:dealer/{k} a o:Dealer ; rdfs:label {lit(label)}{alts(alt)} ;\n    o:city {lit(city)} ; o:region d:region/{region} .")
    start = dt.date(2024, 1, 1)
    model_keys = sorted(MODELS)
    dealer_keys = sorted(DEALERS)
    for i in range(1, 121):
        m = rng.choice(model_keys)
        dealer = rng.choice(dealer_keys)
        date = start + dt.timedelta(days=rng.randrange(0, 731))
        qty = rng.randint(1, 5)
        price = MODELS[m][6]
        revenue = round(qty * price * rng.uniform(0.94, 1.02), 2)
        out.append(
            f"d:order/{i:04d} a o:Order ; rdfs:label {lit(f'Order {i:04d}')} ;\n"
            f"    o:soldModelCode {lit(MODELS[m][4])} ; o:dealer d:dealer/{dealer} ;\n"
            f'    o:orderDate "{date.isoformat()}"^^xsd:date ; o:quantity {qty} ; o:revenue "{revenue:.2f}"^^xsd:decimal .'
        )
    return "\n".join(out) + "\n"


def people() -> str:
    out = [PREFIXES]
    for k, label in DEPARTMENTS.items():
        out.append(f"d:dept/{k} a o:Department ; rdfs:label {lit(label)} .")
    for k, (label, dealer, dept, role, hired) in EMPLOYEES.items():
        out.append(
            f"d:employee/{k} a o:Employee ; rdfs:label {lit(label)} ;\n"
            f'    o:worksAt d:dealer/{dealer} ; o:department d:dept/{dept} ; o:role {lit(role)} ; o:hireDate "{hired}"^^xsd:date .'
        )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- benchmark
# Gold SPARQL is executed at eval time, so gold answers always match the generated data.
# gold.module/class/shape/properties/entities drive calibration and failure attribution.
G = "http://example.org/graph/"
PFX = f"PREFIX o: <{O}>\nPREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"


def e(kind: str, key: str) -> str:
    return f"<{D}{kind}/{key}>"


def item(i, question, shape, module, cls, sparql, props=(), entities=(), tags=(), expect_unanswered=False):
    return {
        "id": f"q{i:02d}",
        "question": question,
        "gold_sparql": PFX + sparql if sparql else None,
        "gold": {"module": module, "class": f"{O}{cls}" if cls else None, "shape": shape, "properties": [f"{O}{p}" for p in props], "entities": [x.strip("<>") for x in entities]},
        "tags": list(tags),
        "expect_unanswered": expect_unanswered,
    }


def benchmark() -> dict:
    cat, sal, ppl = f"<{G}catalog>", f"<{G}sales>", f"<{G}people>"
    defender, mustang, prius, modely, model3 = e("model", "defender"), e("model", "mustang"), e("model", "prius"), e("model", "modely"), e("model", "model3")
    items = [
        item(1, "What is the base price of the Defender?", "lookup", "catalog", "SUV", f"SELECT ?v WHERE {{ GRAPH {cat} {{ {defender} o:basePrice ?v }} }}", ["basePrice"], [defender], ["lookup"]),
        item(2, "Who makes the Mustang?", "lookup", "catalog", "Coupe", f"SELECT ?v WHERE {{ GRAPH {cat} {{ {mustang} o:manufacturer ?v }} }}", ["manufacturer"], [mustang], ["lookup", "ambiguous-entity"]),
        item(3, "When was the Toyota Prius launched?", "lookup", "catalog", "Hatchback", f"SELECT ?v WHERE {{ GRAPH {cat} {{ {prius} o:launchYear ?v }} }}", ["launchYear"], [prius], ["lookup"]),
        item(4, "Which city is Harbor Vehicles located in?", "lookup", "sales", "Dealer", f"SELECT ?v WHERE {{ GRAPH {sal} {{ {e('dealer', 'harbor')} o:city ?v }} }}", ["city"], [e("dealer", "harbor")], ["lookup"]),
        item(5, "What is the horsepower of the Defender's engine?", "path", "catalog", "SUV", f"SELECT ?v WHERE {{ GRAPH {cat} {{ {defender} o:engine/o:horsepower ?v }} }}", ["engine", "horsepower"], [defender], ["multi-hop"]),
        item(6, "Which region is the dealer that employs Alice Moreno in?", "path", "people", "Employee", f"SELECT ?v WHERE {{ GRAPH {ppl} {{ {e('employee', 'amoreno')} o:worksAt ?d }} GRAPH {sal} {{ ?d o:region ?v }} }}", ["worksAt", "region"], [e("employee", "amoreno")], ["multi-hop", "cross-graph"]),
        item(7, "What fuel type does the Jaguar I-Pace use?", "path", "catalog", "SUV", f"SELECT ?v WHERE {{ GRAPH {cat} {{ {e('model', 'ipace')} o:engine/o:fuelType ?v }} }}", ["engine", "fuelType"], [e("model", "ipace")], ["multi-hop"]),
        item(8, "How many SUVs are there?", "count", "catalog", "SUV", f"SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE {{ GRAPH {cat} {{ ?x a o:SUV }} }}", [], [], ["count", "subclass"]),
        item(9, "How many vehicle models does Ford make?", "count", "catalog", "VehicleModel", f"SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE {{ GRAPH {cat} {{ ?x o:manufacturer {e('mfr', 'ford')} }} }}", ["manufacturer"], [e("mfr", "ford")], ["count"]),
        item(10, "How many orders did Mustang place in Q3 2025?", "count", "sales", "Order", f'SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE {{ GRAPH {sal} {{ ?x o:dealer {e("dealer", "mustang")} ; o:orderDate ?d FILTER(?d >= "2025-07-01"^^xsd:date && ?d <= "2025-09-30"^^xsd:date) }} }}', ["dealer", "orderDate"], [e("dealer", "mustang")], ["count", "ambiguous-entity", "date-range"]),
        item(11, "How many employees work at Downtown Motors?", "count", "people", "Employee", f"SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE {{ GRAPH {ppl} {{ ?x o:worksAt {e('dealer', 'downtown')} }} }}", ["worksAt"], [e("dealer", "downtown")], ["count", "cross-graph"]),
        item(12, "How many employees work in the Pacific region?", "count", "people", "Employee", f"SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE {{ GRAPH {ppl} {{ ?x o:worksAt ?d }} GRAPH {sal} {{ ?d o:region {e('region', 'pacific')} }} }}", ["worksAt", "region"], [e("region", "pacific")], ["count", "cross-graph", "multi-hop"]),
        item(13, "How many Sales Managers are there?", "count", "people", "Employee", f'SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE {{ GRAPH {ppl} {{ ?x o:role "Sales Manager" }} }}', ["role"], [], ["count", "value"]),
        item(14, "How many employees were hired after 2020?", "count", "people", "Employee", f'SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE {{ GRAPH {ppl} {{ ?x o:hireDate ?d FILTER(?d > "2020-12-31"^^xsd:date) }} }}', ["hireDate"], [], ["count", "filter"]),
        item(15, "How many orders were placed in Q1 2019?", "count", "sales", "Order", f'SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE {{ GRAPH {sal} {{ ?x o:orderDate ?d FILTER(?d >= "2019-01-01"^^xsd:date && ?d <= "2019-03-31"^^xsd:date) }} }}', ["orderDate"], [], ["count", "no-data"]),
        item(16, "Which models launched after 2019 cost more than $50,000?", "list", "catalog", "VehicleModel", f"SELECT ?x WHERE {{ GRAPH {cat} {{ ?x o:launchYear ?y ; o:basePrice ?p FILTER(?y > 2019 && ?p > 50000) }} }}", ["launchYear", "basePrice"], [], ["list", "filter"]),
        item(17, "Which dealers are in Denver?", "list", "sales", "Dealer", f'SELECT ?x WHERE {{ GRAPH {sal} {{ ?x o:city "Denver" }} }}', ["city"], [], ["list", "value"]),
        item(18, "Which manufacturers are based in the United Kingdom?", "list", "catalog", "Manufacturer", f'SELECT ?x WHERE {{ GRAPH {cat} {{ ?x o:country "United Kingdom" }} }}', ["country"], [], ["list", "value"]),
        item(19, "Which vehicle models are made by Toyota?", "list", "catalog", "VehicleModel", f"SELECT ?x WHERE {{ GRAPH {cat} {{ ?x o:manufacturer {e('mfr', 'toyota')} }} }}", ["manufacturer"], [e("mfr", "toyota")], ["list"]),
        item(20, "Which employees work at dealers in the Northeast region?", "list", "people", "Employee", f"SELECT ?x WHERE {{ GRAPH {ppl} {{ ?x o:worksAt ?d }} GRAPH {sal} {{ ?d o:region {e('region', 'northeast')} }} }}", ["worksAt", "region"], [e("region", "northeast")], ["list", "cross-graph", "multi-hop"]),
        item(21, "What is the average base price of electric vehicles?", "aggregate", "catalog", "VehicleModel", f"SELECT (AVG(?p) AS ?v) WHERE {{ GRAPH {cat} {{ ?x o:engine/o:fuelType {e('fuel', 'electric')} ; o:basePrice ?p }} }}", ["engine", "fuelType", "basePrice"], [e("fuel", "electric")], ["aggregate", "multi-hop"]),
        item(22, "What was the total revenue from F-150 orders in 2025?", "aggregate", "sales", "Order", f'SELECT (SUM(?r) AS ?v) WHERE {{ GRAPH {cat} {{ {e("model", "f150")} o:modelCode ?c }} GRAPH {sal} {{ ?x o:soldModelCode ?c ; o:revenue ?r ; o:orderDate ?d FILTER(YEAR(?d) = 2025) }} }}', ["soldModelCode", "modelCode", "revenue", "orderDate"], [e("model", "f150")], ["aggregate", "cross-graph", "join"]),
        item(23, "What is the total quantity of vehicles ordered by Bayside Autos?", "aggregate", "sales", "Order", f"SELECT (SUM(?q) AS ?v) WHERE {{ GRAPH {sal} {{ ?x o:dealer {e('dealer', 'bayside')} ; o:quantity ?q }} }}", ["dealer", "quantity"], [e("dealer", "bayside")], ["aggregate"]),
        item(24, "What is the maximum horsepower of Land Rover models?", "aggregate", "catalog", "VehicleModel", f"SELECT (MAX(?h) AS ?v) WHERE {{ GRAPH {cat} {{ ?x o:manufacturer {e('mfr', 'landrover')} ; o:engine/o:horsepower ?h }} }}", ["manufacturer", "engine", "horsepower"], [e("mfr", "landrover")], ["aggregate", "multi-hop"]),
        item(25, "What is the cheapest SUV?", "superlative", "catalog", "SUV", f"SELECT ?x WHERE {{ GRAPH {cat} {{ ?x a o:SUV ; o:basePrice ?p }} }} ORDER BY ?p LIMIT 1", ["basePrice"], [], ["superlative"]),
        item(26, "Which manufacturer was founded first?", "superlative", "catalog", "Manufacturer", f"SELECT ?x WHERE {{ GRAPH {cat} {{ ?x o:foundedYear ?y }} }} ORDER BY ?y LIMIT 1", ["foundedYear"], [], ["superlative"]),
        item(27, "What is the most powerful vehicle model?", "superlative", "catalog", "VehicleModel", f"SELECT ?x WHERE {{ GRAPH {cat} {{ ?x o:engine/o:horsepower ?h }} }} ORDER BY DESC(?h) LIMIT 1", ["engine", "horsepower"], [], ["superlative", "multi-hop"]),
        item(28, "Which is more expensive, the Defender or the Model Y?", "compare", "catalog", "SUV", f"SELECT ?x WHERE {{ VALUES ?x {{ {defender} {modely} }} GRAPH {cat} {{ ?x o:basePrice ?p }} }} ORDER BY DESC(?p) LIMIT 1", ["basePrice"], [defender, modely], ["compare"]),
        item(29, "Which has more horsepower, the Ford Mustang or the Jaguar F-Type?", "compare", "catalog", "Coupe", f"SELECT ?x WHERE {{ VALUES ?x {{ {mustang} {e('model', 'ftype')} }} GRAPH {cat} {{ ?x o:engine/o:horsepower ?h }} }} ORDER BY DESC(?h) LIMIT 1", ["engine", "horsepower"], [mustang, e("model", "ftype")], ["compare", "multi-hop"]),
        item(30, "Is the Model 3 made by Tesla?", "boolean", "catalog", "Sedan", f"ASK {{ GRAPH {cat} {{ {model3} o:manufacturer {e('mfr', 'tesla')} }} }}", ["manufacturer"], [model3, e("mfr", "tesla")], ["boolean"]),
        item(31, "Is the Defender more expensive than 60000 dollars?", "boolean", "catalog", "SUV", f"ASK {{ GRAPH {cat} {{ {defender} o:basePrice ?p FILTER(?p > 60000) }} }}", ["basePrice"], [defender], ["boolean", "filter"]),
        item(32, "Does Alice Moreno work at Bayside Autos?", "boolean", "people", "Employee", f"ASK {{ GRAPH {ppl} {{ {e('employee', 'amoreno')} o:worksAt {e('dealer', 'bayside')} }} }}", ["worksAt"], [e("employee", "amoreno"), e("dealer", "bayside")], ["boolean", "cross-graph"]),
        item(33, "Which dealer does Alice Moreno work at?", "lookup", "people", "Employee", f"SELECT ?v WHERE {{ GRAPH {ppl} {{ {e('employee', 'amoreno')} o:worksAt ?v }} }}", ["worksAt"], [e("employee", "amoreno")], ["lookup"]),
        item(34, "How many orders included the Tesla Model Y?", "count", "sales", "Order", f"SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE {{ GRAPH {cat} {{ {modely} o:modelCode ?c }} GRAPH {sal} {{ ?x o:soldModelCode ?c }} }}", ["soldModelCode", "modelCode"], [modely], ["count", "cross-graph", "join"]),
        item(35, "What is the weather in Paris?", None, None, None, None, tags=["out-of-scope"], expect_unanswered=True),
        item(36, "Who is the CEO of Tesla?", None, None, None, None, entities=[e("mfr", "tesla")], tags=["out-of-scope", "missing-property"], expect_unanswered=True),
    ]
    return {"name": "sample-automotive", "graph": "data/sample", "items": items}


def main() -> None:
    files = {"ontology.ttl": ontology(), "shapes.ttl": shapes(), "catalog.ttl": catalog(), "sales.ttl": sales(), "people.ttl": people()}
    for name, content in files.items():
        # Turtle prefixed names cannot contain '/', so expand d:kind/key to full IRIs.
        content = re.sub(r"\bd:(\w+)/([\w-]+)", lambda m: f"<{D}{m[1]}/{m[2]}>", content)
        (HERE / name).write_text(content)
        print(f"wrote {name} ({content.count(chr(10))} lines)")
    (HERE / "benchmark.json").write_text(json.dumps(benchmark(), indent=1) + "\n")
    print("wrote benchmark.json")


if __name__ == "__main__":
    main()
