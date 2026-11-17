"""Validate real responses against the published OpenAPI schemas.

tests/test_openapi_contract.py checks the spec's own *examples*. This checks
the actual responses the app produces, which is the thing a client depends on.
It needs the scientific stack (it imports app.py and runs the solver), so it
lives in the full CI job rather than the fast contract job.

Iteration counts are kept small deliberately: the point is response shape, not
statistics.
"""

import json
import os
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# This module imports app.py and runs the real solver, so it needs the
# scientific stack. Skipping rather than failing lets `pytest tests/` work in
# the fast contract environment too, where that stack is deliberately absent.
pytest.importorskip("numpy", reason="needs the scientific stack")
pytest.importorskip("scipy", reason="needs the scientific stack")
jsonschema = pytest.importorskip("jsonschema")
if not hasattr(jsonschema, "Draft202012Validator"):
    pytest.skip("needs jsonschema>=4 for the 2020-12 dialect",
                allow_module_level=True)


@pytest.fixture(scope="module")
def spec():
    with open(os.path.join(REPO, "openapi.yaml"), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def client():
    import app
    return app.app.test_client()


def _validator(spec, schema):
    """Inline $refs so a component validates standalone."""
    document = {"$defs": spec["components"]["schemas"], **schema}
    document = json.loads(json.dumps(document).replace(
        "#/components/schemas/", "#/$defs/"))
    return jsonschema.Draft202012Validator(document)


# (name, method, path, body, expected status)
CASES = [
    ("models", "get", "/api/v1/models", None, 200),
    ("ranges_all", "get", "/api/v1/ranges", None, 200),
    ("ranges_one", "get", "/api/v1/ranges?model_type=invivo", None, 200),
    ("health", "get", "/api/v1/health", None, 200),
    ("predict_invitro", "post", "/api/v1/predict",
     {"model_type": "invitro", "dose": 50, "time": 72, "simulation_count": 5}, 200),
    ("predict_invivo", "post", "/api/v1/predict",
     {"model_type": "invivo", "dose": 5, "time": 672, "simulation_count": 5}, 200),
    ("predict_extrapolated", "post", "/api/v1/predict",
     {"model_type": "invivo", "dose": 10, "time": 2000, "simulation_count": 1}, 200),
    ("error_out_of_range", "post", "/api/v1/predict",
     {"model_type": "invivo", "dose": 99, "time": 24}, 400),
    ("error_unknown_model", "post", "/api/v1/predict",
     {"model_type": "zzz", "dose": 5, "time": 24}, 400),
    # The frozen legacy route must keep matching its documented schema.
    ("legacy_predict", "post", "/api/predict",
     {"model_type": "invitro", "dose": 10, "time": 24, "simulation_count": 5}, 200),
]


@pytest.mark.parametrize("name,method,path,body,expected", CASES)
def test_response_conforms_to_published_schema(
        spec, client, name, method, path, body, expected):
    response = client.open(path, method=method.upper(), json=body) if body \
        else client.open(path, method=method.upper())
    assert response.status_code == expected, (
        f"{name}: got {response.status_code}, body {response.get_data()[:200]}")

    documented = spec["paths"][path.split("?")[0]][method]["responses"]
    assert str(expected) in documented, f"{name}: status {expected} undocumented"
    content = documented[str(expected)]["content"]
    schema = list(content.values())[0]["schema"]

    _validator(spec, schema).validate(response.get_json())


def test_calibrated_request_emits_no_warnings(client):
    """5 mg/kg at 672 h is the calibrated in vivo point. If this starts
    emitting warnings, either the calibration record or the model changed."""
    # Uses the production default: a low simulation_count legitimately raises
    # monte_carlo_noise, which would mask what this test is checking.
    response = client.post("/api/v1/predict", json={
        "model_type": "invivo", "dose": 5, "time": 672})
    assert response.status_code == 200
    assert response.get_json()["warnings"] == []


def test_extrapolated_request_warns_about_histopathology(client):
    """The single most important warning: the in vivo histopathology endpoints
    were scored at one dose only."""
    response = client.post("/api/v1/predict", json={
        "model_type": "invivo", "dose": 2.5, "time": 100, "simulation_count": 5})
    warnings = response.get_json()["warnings"]
    codes = {w["code"] for w in warnings}
    assert "histopathology_extrapolated" in codes, codes
    affected = next(w for w in warnings
                    if w["code"] == "histopathology_extrapolated")["affects"]
    assert set(affected) == {"necrosis", "kidney_failure"}


def test_legacy_predict_shape_is_unchanged(client):
    """Field paths an existing monitoring consumer asserts on."""
    response = client.post("/api/predict", json={
        "model_type": "invitro", "dose": 1, "time": 24, "simulation_count": 5})
    body = response.get_json()
    assert response.status_code == 200
    # No wrapper: these are top level.
    for key in ("model_type", "dose", "dose_unit", "time", "time_unit",
                "simulation_count", "endpoints"):
        assert key in body, key
    for endpoint in ("dna_damage", "necrosis"):
        assert isinstance(
            body["endpoints"][endpoint]["final_value"]["median"], float)
        series = body["endpoints"][endpoint]["time_series"]["median"]
        # Asserted against the configured grid, never a literal 100: it is 20
        # under FLASK_ENV=testing, while the deployed value is 100 and two
        # live monitoring checks depend on that. `config` is a module of
        # classes; app.py rebinds the name to the active instance, so read it
        # from there rather than from the module.
        import app as app_module
        assert len(series) == app_module.config.TIME_POINTS


def test_oversized_number_stays_inside_the_error_envelope(client, spec):
    """A JSON integer with more digits than a double can hold used to escape as
    an HTML 500, because float() raises OverflowError rather than ValueError.
    Every /api/v1 failure must be the coded JSON envelope."""
    response = client.post(
        "/api/v1/predict",
        data='{"model_type":"invitro","dose":%s,"time":24}' % ("9" * 400),
        content_type="application/json")

    assert response.status_code == 400, response.get_data()[:200]
    assert response.content_type.startswith("application/json")
    body = response.get_json()
    assert body["error"]["code"] == "parameter_out_of_range"
    _validator(spec, spec["components"]["schemas"]["Error"]).validate(body)


def test_unknown_fields_are_rejected_not_ignored(client, spec):
    """openapi.yaml declares additionalProperties: false, so the server must
    enforce it. A client that sends an invented parameter should be told,
    rather than believing it took effect."""
    response = client.post("/api/v1/predict", json={
        "model_type": "invitro", "dose": 10, "time": 24,
        "smiles": "CCO", "iterations": 100})

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"]["code"] == "invalid_request"
    assert set(body["error"]["details"]["unknown"]) == {"smiles", "iterations"}
    _validator(spec, spec["components"]["schemas"]["Error"]).validate(body)


def test_accepted_fields_still_work(client):
    """The strictness above must not reject the documented fields."""
    response = client.post("/api/v1/predict", json={
        "model_type": "invitro", "dose": 10, "time": 24, "simulation_count": 5})
    assert response.status_code == 200


def test_predict_values_are_json_numbers_not_strings(client):
    response = client.post("/api/v1/predict", json={
        "model_type": "invitro", "dose": 50, "time": 72, "simulation_count": 5})
    final = response.get_json()["endpoints"]["dna_damage"]["final_value"]
    for key, value in final.items():
        assert isinstance(value, (int, float)) and not isinstance(value, bool), \
            f"{key} is {type(value).__name__}"
