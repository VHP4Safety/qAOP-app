"""Versioned JSON API for the qAOP service.

Why this exists beside the unversioned ``/api/*`` routes rather than replacing
them: those routes have a live consumer that pins exact response field paths,
so their 2xx bodies are frozen. ``/api/v1`` is where the contract can be stated
properly -- machine-readable error codes, and explicit reporting of when a
request has left the region the models were calibrated on.

This module deliberately imports only flask plus the project's own dependency-
free modules (``validation``, ``calibration``, ``config``). Everything that
needs numpy, scipy or the ODE models arrives through ``init_api_v1`` by
injection, which is what lets the contract test run without the scientific
stack installed.
"""

import logging

from flask import Blueprint, jsonify, request

import calibration
import validation
from validation import ValidationError

logger = logging.getLogger("qAOP.api_v1")

API_VERSION = "1.0.0"
SERVICE_NAME = "qaop-app"

# Key under which init_api_v1 stashes the injected dependencies on the app.
EXT_KEY = "qaop_api_v1"

bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")

# The only fields POST /api/v1/predict accepts. Mirrors
# PredictRequest.additionalProperties: false in openapi.yaml.
ACCEPTED_PREDICT_FIELDS = ("model_type", "dose", "time", "simulation_count")

# Every error code this surface can return, with its HTTP status. The contract
# test asserts this mapping matches openapi.yaml in both directions, so adding
# a code without documenting it fails CI.
ERROR_CODES = {
    "invalid_request": 400,
    "unknown_model": 400,
    "parameter_out_of_range": 400,
    "malformed_json": 400,
    "unsupported_media_type": 415,
    "not_found": 404,
    "method_not_allowed": 405,
    "prediction_failed": 500,
    "service_unavailable": 503,
}


def _error(code, message, details=None):
    """Build the single error envelope used by every /api/v1 response.

    Shape is ``{"error": {"code", "message", "details"?}}``. The code is the
    part a client should branch on; the message is for humans and may change.
    """
    assert code in ERROR_CODES, f"undeclared error code: {code}"
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return jsonify(body), ERROR_CODES[code]


def _deps():
    from flask import current_app
    return current_app.extensions[EXT_KEY]


def _read_json_body():
    """Parse the request body, mapping transport problems to coded errors.

    The unversioned route called ``request.get_json()`` bare, so a missing
    Content-Type raised Werkzeug's 415 and malformed JSON its 400 -- both of
    which were caught by a broad handler and re-reported as HTTP 500 with the
    Werkzeug text embedded. Returning a coded 415/400 is the whole point of
    doing this explicitly.

    Returns ``(payload, error_response)``; exactly one is None.
    """
    if not request.is_json:
        return None, _error(
            "unsupported_media_type",
            "Request body must be JSON and the Content-Type header must be "
            "'application/json'.",
            {"content_type": request.content_type or None},
        )

    payload = request.get_json(silent=True)
    if payload is None:
        return None, _error(
            "malformed_json",
            "Request body is not valid JSON.",
        )
    if not isinstance(payload, dict):
        return None, _error(
            "invalid_request",
            f"Request body must be a JSON object, got "
            f"{type(payload).__name__}.",
        )
    return payload, None


def _validate_predict_request(payload):
    """Validate a predict request field by field.

    ``validation.validate_prediction_request`` raises one exception type for
    every problem, which is fine for a human-readable message but gives a
    client nothing to branch on. Calling the individual validators lets each
    failure carry its own code and name the offending field.

    Returns ``(validated, error_response)``; exactly one is None.
    """
    missing = [f for f in ("model_type", "dose", "time") if f not in payload]
    if missing:
        return None, _error(
            "invalid_request",
            f"Missing required fields: {', '.join(missing)}.",
            {"missing": missing},
        )

    # openapi.yaml declares additionalProperties: false on PredictRequest, so
    # enforce it rather than letting the spec lie. Silently dropping an
    # unrecognised field is the wrong default for a generated or model-written
    # client: a wrong or invented parameter name should be reported, not
    # ignored while the caller believes it took effect.
    unknown = sorted(set(payload) - set(ACCEPTED_PREDICT_FIELDS))
    if unknown:
        return None, _error(
            "invalid_request",
            f"Unrecognised field(s): {', '.join(unknown)}. This endpoint "
            f"accepts only {', '.join(ACCEPTED_PREDICT_FIELDS)}.",
            {"unknown": unknown, "accepted": list(ACCEPTED_PREDICT_FIELDS)},
        )

    try:
        model_type = validation.validate_model_type(payload["model_type"])
    except ValidationError as exc:
        return None, _error("unknown_model", str(exc), {
            "field": "model_type",
            "supported": sorted(validation.PARAMETER_RANGES),
        })

    validated = {"model_type": model_type}
    for field, validate in (
        ("dose", lambda v: validation.validate_dose(v, model_type)),
        ("time", lambda v: validation.validate_time(v, model_type)),
    ):
        try:
            validated[field] = validate(payload[field])
        except ValidationError as exc:
            return None, _error("parameter_out_of_range", str(exc), {
                "field": field,
                "accepted": validation.PARAMETER_RANGES[model_type][field],
            })

    if "simulation_count" in payload:
        try:
            validated["simulation_count"] = validation.validate_simulation_count(
                payload["simulation_count"])
        except ValidationError as exc:
            return None, _error("parameter_out_of_range", str(exc), {
                "field": "simulation_count",
                "accepted": validation.SIMULATION_LIMITS["simulation_count"],
            })
    else:
        validated["simulation_count"] = (
            validation.SIMULATION_LIMITS["simulation_count"]["default"])

    return validated, None


def _service_block():
    deps = _deps()
    return {
        "name": SERVICE_NAME,
        "version": deps["service_version"],
        "api_version": API_VERSION,
    }


@bp.route("/health", methods=["GET"])
def health_v1():
    """Liveness and data-availability check."""
    try:
        report = _deps()["health_probe"]()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(f"health probe failed: {exc}")
        return _error("service_unavailable", "Health probe failed.")

    body = {
        "status": report["status"],
        "service": _service_block(),
        "checks": report["checks"],
        "configuration": report["configuration"],
        "models": list(report["models"]),
    }
    status = 200 if report["status"] == "healthy" else 503
    return jsonify(body), status


@bp.route("/models", methods=["GET"])
def models_v1():
    """Model catalogue: accepted inputs, calibrated domain, endpoint meanings.

    This is the entry point for a programmatic client. Unlike the unversioned
    ``/api/models`` it reports accepted ranges straight from the validator
    rather than as prose, so the catalogue cannot disagree with what the
    service will accept.
    """
    deps = _deps()
    models = {}
    for model_type, ranges in validation.PARAMETER_RANGES.items():
        envelope = calibration.CALIBRATED[model_type]
        endpoints = calibration.endpoints_for(model_type)
        models[model_type] = {
            "system": envelope["system"],
            "accepted": {
                "dose": ranges["dose"],
                "time": ranges["time"],
                "simulation_count": validation.SIMULATION_LIMITS["simulation_count"],
            },
            "calibrated": {"dose": envelope["dose"], "time": envelope["time"]},
            "endpoints": list(endpoints),
            "endpoint_metadata": {
                name: calibration.ENDPOINT_MEANING[name] for name in endpoints
            },
            "endpoint_calibration": calibration.ENDPOINT_CALIBRATION.get(model_type, {}),
            "time_points": deps["time_points"],
        }

    return jsonify({
        "service": _service_block(),
        "source": calibration.SOURCE,
        "models": models,
        "uncertainty": calibration.UNCERTAINTY,
        "cannot_answer": list(calibration.CANNOT_ANSWER),
        "interpretation_guide": "/llms.txt",
    })


@bp.route("/ranges", methods=["GET"])
def ranges_v1():
    """Accepted and calibrated parameter ranges.

    One fixed response shape, unlike the unversioned ``/api/ranges`` whose body
    changes depending on whether ``model_type`` is supplied.
    """
    model_type = request.args.get("model_type")
    if model_type is not None:
        try:
            model_type = validation.validate_model_type(model_type)
        except ValidationError as exc:
            return _error("unknown_model", str(exc), {
                "field": "model_type",
                "supported": sorted(validation.PARAMETER_RANGES),
            })

    wanted = [model_type] if model_type else sorted(validation.PARAMETER_RANGES)
    return jsonify({
        "service": _service_block(),
        "simulation_count": validation.SIMULATION_LIMITS["simulation_count"],
        "models": {
            name: {
                "accepted": {
                    "dose": validation.PARAMETER_RANGES[name]["dose"],
                    "time": validation.PARAMETER_RANGES[name]["time"],
                },
                "calibrated": {
                    "dose": calibration.CALIBRATED[name]["dose"],
                    "time": calibration.CALIBRATED[name]["time"],
                },
            }
            for name in wanted
        },
    })


@bp.route("/predict", methods=["POST"])
def predict_v1():
    """Run the Monte Carlo ensemble and return summarised endpoint statistics."""
    payload, error = _read_json_body()
    if error:
        return error

    validated, error = _validate_predict_request(payload)
    if error:
        return error

    model_type = validated["model_type"]
    dose = validated["dose"]
    time_value = validated["time"]
    simulation_count = validated["simulation_count"]

    try:
        simulation = _deps()["run_prediction"](
            model_type, dose, time_value, simulation_count)
    except Exception as exc:
        logger.error(f"prediction failed: {exc}")
        # Deliberately does not echo the exception text: the unversioned route
        # did, which is how Werkzeug's internal messages reached clients.
        return _error(
            "prediction_failed",
            "The simulation could not be completed. This is a server-side "
            "problem, not a problem with the request.",
        )

    warnings = calibration.extrapolation_warnings(
        model_type, dose, time_value, simulation_count)

    return jsonify({
        "service": _service_block(),
        "request": {
            "model_type": model_type,
            "dose": dose,
            "dose_unit": simulation["dose_unit"],
            "time": time_value,
            "time_unit": "hours",
            "simulation_count": simulation_count,
            "simulations_succeeded": simulation["successful_simulations"],
        },
        "endpoints": simulation["endpoints"],
        "uncertainty": {
            "kind": calibration.UNCERTAINTY["kind"],
            "is_prediction_interval": False,
            "note": calibration.UNCERTAINTY["note"],
            "reproducibility": calibration.UNCERTAINTY["reproducibility_note"],
        },
        "warnings": warnings,
        "interpretation_guide": "/llms.txt",
    })


def init_api_v1(app, *, run_prediction, health_probe, service_version, time_points):
    """Register the blueprint and record its injected dependencies.

    Args:
        app: the Flask application.
        run_prediction: callable(model_type, dose, time, simulation_count) ->
            {"endpoints", "successful_simulations", "dose_unit"}. The only
            dependency that needs numpy and the ODE models.
        health_probe: callable() -> {"status", "checks", "configuration", "models"}.
        service_version: version string reported in every response.
        time_points: length of the time grid each series carries.
    """
    app.extensions[EXT_KEY] = {
        "run_prediction": run_prediction,
        "health_probe": health_probe,
        "service_version": service_version,
        "time_points": time_points,
    }
    app.register_blueprint(bp)
    return bp
