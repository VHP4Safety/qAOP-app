"""Anti-drift tests between the served routes and openapi.yaml.

Runs without numpy, scipy, pandas or plotly: api_v1 takes its heavy
dependencies by injection, and the legacy routes are read out of app.py as
text rather than imported. That is what keeps CI fast.

What it pins:
  * every /api/v1 route is documented, and every documented path is served
  * every error code api_v1 can raise is in the spec's enum, and vice versa
  * request/response examples validate against their own schemas
  * the spec's numeric bounds equal validation.py's, so the spec cannot become
    yet another drifting copy of the ranges
  * the response-field paths an existing monitoring consumer depends on
"""

import os
import re
import sys

import pytest
import yaml
from flask import Flask
from jsonschema import Draft202012Validator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import api_v1  # noqa: E402
import calibration  # noqa: E402
import validation  # noqa: E402


@pytest.fixture(scope="module")
def spec():
    with open(os.path.join(REPO, "openapi.yaml"), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def v1_routes():
    """Routes the blueprint actually serves, from a throwaway app."""
    app = Flask(__name__)
    app.register_blueprint(api_v1.bp)
    return {
        rule.rule: {m for m in rule.methods if m in {"GET", "POST", "PUT", "DELETE", "PATCH"}}
        for rule in app.url_map.iter_rules()
        if rule.rule.startswith("/api/v1")
    }


# app.py uses double quotes throughout, so a single-quote-only pattern would
# match nothing and silently compare two empty sets.
ROUTE_RE = re.compile(r"""@app\.route\(\s*['"]([^'"]+)['"]""")


@pytest.fixture(scope="module")
def app_py_routes():
    with open(os.path.join(REPO, "app.py"), encoding="utf-8") as handle:
        found = set(ROUTE_RE.findall(handle.read()))
    # Tripwire: if the regex stops matching, fail loudly instead of passing
    # vacuously.
    assert found, "route regex matched nothing in app.py"
    assert {"/api/health", "/api/models", "/api/ranges", "/api/predict"} <= found, found
    return found


# Routes intentionally absent from the spec, with the reason. Pinned so the
# omission is a decision rather than an oversight.
UNDOCUMENTED_BY_DESIGN = {
    "/": "web UI, not a JSON API",
    "/download-report": "session-coupled HTML download, not JSON",
}


def test_every_v1_route_is_documented(spec, v1_routes):
    for rule in v1_routes:
        assert rule in spec["paths"], f"{rule} is served but not in openapi.yaml"


def test_every_documented_v1_path_is_served(spec, v1_routes):
    for path in spec["paths"]:
        if path.startswith("/api/v1"):
            assert path in v1_routes, f"{path} is documented but not served"


def test_v1_methods_match(spec, v1_routes):
    for rule, methods in v1_routes.items():
        documented = {m.upper() for m in spec["paths"][rule] if m.lower() in
                      {"get", "post", "put", "delete", "patch"}}
        assert documented == methods, f"{rule}: spec {documented} != served {methods}"


def test_legacy_routes_are_documented_or_deliberately_not(spec, app_py_routes):
    for rule in app_py_routes:
        if rule in UNDOCUMENTED_BY_DESIGN or rule.startswith("/api/v1"):
            continue
        assert rule in spec["paths"], (
            f"{rule} is served but neither documented nor listed in "
            f"UNDOCUMENTED_BY_DESIGN"
        )


def test_deliberate_omissions_are_still_real_routes(app_py_routes):
    """Stops UNDOCUMENTED_BY_DESIGN rotting into a list of dead paths."""
    for rule in UNDOCUMENTED_BY_DESIGN:
        assert rule in app_py_routes, f"{rule} no longer exists; drop it from the list"


def test_legacy_paths_are_marked_deprecated(spec):
    for path in ("/api/predict", "/api/models", "/api/ranges", "/api/health"):
        for method, operation in spec["paths"][path].items():
            if method.lower() in {"get", "post"}:
                assert operation.get("deprecated") is True, f"{path} not deprecated"


def test_error_codes_match_the_spec_enum(spec):
    documented = set(spec["components"]["schemas"]["ErrorCode"]["enum"])
    implemented = set(api_v1.ERROR_CODES)
    assert implemented == documented, (
        f"only in code: {implemented - documented}; "
        f"only in spec: {documented - implemented}"
    )


def test_warning_codes_match_the_spec_enum(spec):
    documented = set(spec["components"]["schemas"]["Warning"]["properties"]["code"]["enum"])
    # Every code calibration can emit, gathered by driving it over inputs that
    # trigger each branch.
    emitted = set()
    for args in (("invitro", 10.0, 24.0, 250), ("invitro", 0.001, 200.0, 1),
                 ("invivo", 10.0, 2160.0, 1), ("invivo", 5.0, 672.0, 250),
                 ("invivo", 2.5, 100.0, 10), ("invitro", 75.0, 72.0, 250)):
        emitted |= {w["code"] for w in calibration.extrapolation_warnings(*args)}
    assert emitted, "no warning codes exercised"
    assert emitted <= documented, f"undocumented warning codes: {emitted - documented}"


def test_error_code_statuses_are_documented(spec):
    """Each code's HTTP status must appear on at least one operation."""
    statuses = set()
    for operations in spec["paths"].values():
        for operation in operations.values():
            if isinstance(operation, dict) and "responses" in operation:
                statuses |= set(operation["responses"])
    for code, status in api_v1.ERROR_CODES.items():
        if code in {"not_found", "method_not_allowed", "service_unavailable"}:
            continue  # app-level handlers, not per-operation responses
        assert str(status) in statuses, f"{code} -> {status} documented nowhere"


def _resolve(spec, schema):
    """Inline $refs so a component can be validated standalone."""
    document = {"$defs": spec["components"]["schemas"], **schema}
    import json
    return json.loads(json.dumps(document).replace(
        "#/components/schemas/", "#/$defs/"))


def test_examples_validate_against_their_schemas(spec):
    checked = 0
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            if not isinstance(operation, dict):
                continue
            body = operation.get("requestBody", {})
            for media in body.get("content", {}).values():
                if "schema" not in media:
                    continue
                validator = Draft202012Validator(_resolve(spec, media["schema"]))
                for name, example in media.get("examples", {}).items():
                    validator.validate(example["value"])
                    checked += 1
            for response in operation.get("responses", {}).values():
                for media in response.get("content", {}).values():
                    if "schema" not in media:
                        continue
                    validator = Draft202012Validator(_resolve(spec, media["schema"]))
                    for name, example in media.get("examples", {}).items():
                        validator.validate(example["value"])
                        checked += 1
    assert checked >= 3, f"only {checked} examples checked; the loop may be wrong"


def test_spec_bounds_equal_the_validator(spec):
    """The spec must not become another drifting copy of the ranges."""
    request_schema = spec["components"]["schemas"]["PredictRequest"]
    limits = validation.SIMULATION_LIMITS["simulation_count"]
    simulation = request_schema["properties"]["simulation_count"]
    assert simulation["minimum"] == limits["min"]
    assert simulation["maximum"] == limits["max"]
    assert simulation["default"] == limits["default"]


def test_model_enum_matches_the_validator(spec):
    assert set(spec["components"]["schemas"]["ModelType"]["enum"]) == set(
        validation.PARAMETER_RANGES)


def test_endpoint_enum_matches_calibration(spec):
    documented = set(spec["components"]["schemas"]["EndpointName"]["enum"])
    implemented = set(calibration.endpoints_for("invivo")) | set(
        calibration.endpoints_for("invitro"))
    assert documented == implemented


def test_final_value_and_time_series_differ_on_min_max(spec):
    """They cannot share a component: only final_value has min/max."""
    final = set(spec["components"]["schemas"]["FinalValue"]["required"])
    series = set(spec["components"]["schemas"]["TimeSeries"]["required"])
    assert {"min", "max"} <= final
    assert not {"min", "max"} & series


# Exact response paths an existing monitoring consumer asserts on. Enforced
# here so a change is caught before deploy rather than by a production alert.
MONITORED_PATHS = [
    ("LegacyPredictResponse", ["endpoints", "dna_damage", "final_value", "median"]),
    ("LegacyPredictResponse", ["endpoints", "necrosis", "final_value", "median"]),
    ("LegacyPredictResponse", ["endpoints", "dna_damage", "time_series", "median"]),
    ("LegacyPredictResponse", ["endpoints", "necrosis", "time_series", "median"]),
]


@pytest.mark.parametrize("schema_name,path", MONITORED_PATHS)
def test_monitored_response_paths_still_exist(spec, schema_name, path):
    schemas = spec["components"]["schemas"]

    def walk(node, remaining):
        while "$ref" in node:
            node = schemas[node["$ref"].rsplit("/", 1)[-1]]
        if not remaining:
            return node
        key = remaining[0]
        properties = node.get("properties", {})
        assert key in properties, f"{key} missing (rest: {remaining})"
        return walk(properties[key], remaining[1:])

    leaf = walk(schemas[schema_name], path)
    while "$ref" in leaf:
        leaf = schemas[leaf["$ref"].rsplit("/", 1)[-1]]
    assert leaf, f"{path} resolved to nothing"


def test_spec_is_openapi_31(spec):
    assert spec["openapi"].startswith("3.1")


def test_spec_passes_openapi_validation(spec):
    openapi_spec_validator = pytest.importorskip("openapi_spec_validator")
    openapi_spec_validator.validate(spec)


def test_info_description_carries_the_interpretation_caveats(spec):
    """The description is what a tool-calling model actually reads, so the
    caveats that stop it over-claiming must be in there, not only in llms.txt."""
    description = re.sub(r"\s+", " ", spec["info"]["description"]).lower()
    for phrase in ("fibrosis", "not a prediction interval", "5 mg/kg",
                   "two significant figures", "not acute kidney injury"):
        assert phrase in description, f"missing from info.description: {phrase}"


def test_llms_txt_is_served_and_documented(spec):
    assert "/llms.txt" in spec["paths"]
    with open(os.path.join(REPO, "llms.txt"), encoding="utf-8") as handle:
        text = re.sub(r"\s+", " ", handle.read().replace("**", "")).lower()
    for phrase in ("interstitial fibrosis", "not a prediction interval",
                   "do not compare or average", "5 mg/kg",
                   "not established by this service"):
        assert phrase in text, f"missing from llms.txt: {phrase}"
