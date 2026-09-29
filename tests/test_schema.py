from conftest import O

from kgqa.schema import SchemaCatalog, SchemaGraph, Slicer


def test_modules_classes_and_counts(catalog):
    assert set(catalog.modules) == {"catalog", "sales", "people"}
    assert catalog.classes[O + "VehicleModel"].module == "catalog"  # abstract parent, module from children
    assert catalog.instance_count(O + "VehicleModel") == 16
    assert O + "SUV" in catalog.descendants(O + "VehicleModel")


def test_shapes_supply_cardinality_and_types(catalog):
    price = catalog.properties[O + "basePrice"]
    assert price.max_count == 1 and price.numeric
    assert catalog.properties[O + "engine"].range_class == O + "Engine"


def test_shared_identifier_join_is_detected(catalog):
    (join,) = catalog.joins
    assert {join.left, join.right} == {O + "soldModelCode", O + "modelCode"}
    assert join.overlap == 16
    assert "catalog" in catalog.modules["sales"].linked_modules


def test_paths_include_cross_graph_join(catalog):
    g = SchemaGraph(catalog)
    keys = [p.key() for p in g.paths_between(O + "Order", O + "VehicleModel")]
    assert "soldModelCode=modelCode" in keys
    (p,) = [p for p in g.paths_between(O + "Order", O + "VehicleModel") if p.key() == "soldModelCode=modelCode"]
    assert p.modules == ["sales", "catalog"] and p.cross_module


def test_two_hop_path_and_numeric_value_paths(catalog):
    g = SchemaGraph(catalog)
    assert [p.key() for p in g.paths_between(O + "VehicleModel", O + "FuelType")] == ["engine>fuelType"]
    numeric = {p.key() for p in g.value_paths(O + "VehicleModel", 2, numeric_only=True)}
    assert {"basePrice", "engine>horsepower"} <= numeric


def test_slice_respects_budget_and_lists_allowed_iris(catalog):
    sl = Slicer(catalog, token_budget=400).slice([O + "Order"])
    assert sl.tokens <= 400
    assert O + "Order" in sl.allowed_iris and O + "revenue" in sl.allowed_iris
    assert "http://example.org/graph/sales" in sl.allowed_iris


def test_catalog_round_trips(catalog, tmp_path):
    catalog.save(tmp_path / "c.json")
    again = SchemaCatalog.load(tmp_path / "c.json")
    assert again.properties.keys() == catalog.properties.keys()
    assert again.joins == catalog.joins
