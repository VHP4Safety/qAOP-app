"""Calibrated domain of the qAOP models, and what each endpoint actually is.

``validation.PARAMETER_RANGES`` says what the service *accepts*. This module
says what it was *calibrated on*, which is a good deal narrower, and the two
must not be confused: both ODE subsystems are linear in dose, with dose
entering only through the initial conditions, so an out-of-range dose produces
output that looks exactly as well-behaved as a calibrated one. The model cannot
fail loudly, which is why /api/v1 reports extrapolation explicitly instead of
leaving the caller to infer it.

Source: Toxicological Sciences, doi:10.1093/toxsci/kfaf177 (PMID 41514178),
with posterior draws in data/draws_cisDDNEC2024-08-05_11-21-18.csv (in vitro)
and data/draws_cisvivo2024-08-07_10-34-00.csv (in vivo).

Entries marked UNCONFIRMED could not be established from this repository and
are deliberately phrased so as not to assert more than is known.
"""

SOURCE = {
    "doi": "10.1093/toxsci/kfaf177",
    "pmid": "41514178",
    "chemical": "cisplatin",
    "note": (
        "This service models cisplatin only. There is no chemical input of any "
        "kind, so nothing it returns applies to another platinum drug or to any "
        "other substance."
    ),
}

# The region the models were actually fitted in. Compare against
# validation.PARAMETER_RANGES, which is what the service accepts.
CALIBRATED = {
    "invitro": {
        "system": "RPTEC/TERT1 renal proximal tubule epithelial cells in culture",
        "dose": {"min": 0.1, "max": 50.0, "unit": "μM"},
        "time": {"min": 0.0, "max": 72.0, "unit": "hours"},
    },
    "invivo": {
        "system": "rat kidney, single intravenous dose",
        # Transcriptomic and kinetic data span this dose-response range.
        "dose": {"min": 1.0, "max": 7.5, "unit": "mg/kg"},
        "time": {"min": 0.0, "max": 672.0, "unit": "hours"},
    },
}

# Narrower still: endpoints whose calibration is thinner than the model's own
# dose-response range. Each entry says when a value stops being data-backed.
ENDPOINT_CALIBRATION = {
    "invivo": {
        # Histopathology was scored at ONE dose only.
        "necrosis": {
            "histopathology_dose_mg_kg": 5.0,
            "caveat": (
                "Expert-graded H&E necrosis severity was scored at a single "
                "5 mg/kg dose (12 time points to 672 h, 3 rats per point). Any "
                "value at another dose is extrapolation with no supporting "
                "histopathology."
            ),
        },
        "kidney_failure": {
            "histopathology_dose_mg_kg": 5.0,
            "caveat": (
                "Expert-graded interstitial fibrosis severity was scored at a "
                "single 5 mg/kg dose. Any value at another dose is "
                "extrapolation with no supporting histopathology."
            ),
        },
    },
    "invitro": {
        "necrosis": {
            "detection_threshold_um": 50.0,
            "caveat": (
                "Propidium-iodide-positive cells were absent below 50 μM and "
                "early (<20 h) data were discarded, and the fitted Hill "
                "exponent is 17.8-37.7, making DNA damage to necrosis a switch. "
                "Below 50 μM the value is a switch output, not a "
                "measurement-backed prediction."
            ),
        },
    },
}

# What each endpoint key physically means. The keys are kept as they are for
# backward compatibility (an existing monitoring consumer pins them), so the
# honest definition has to travel in the metadata instead of in the name.
ENDPOINT_MEANING = {
    "dna_damage": {
        "label": "DNA damage (transcriptomic surrogate)",
        "unit": "arbitrary units (eigengene score)",
        "definition": (
            "Eigengene score of TXG-MAPr WGCNA module rKID:160 (p53 target "
            "genes), computed from log2 fold-change relative to control. It is "
            "a transcriptomic surrogate, not a measured DNA lesion level, not a "
            "concentration and not a cell count."
        ),
        "temporal_behaviour": "peaks and then decays",
        "cross_model_comparable": False,
        "cross_model_note": (
            "UNCONFIRMED whether the in vitro and in vivo values are on the "
            "same scale. Both use module rKID:160, but an eigengene score "
            "depends on the Z-scored log2FC within the uploaded dataset and the "
            "two models were calibrated separately. Do not compare them "
            "numerically."
        ),
    },
    "necrosis": {
        "label": "Necrosis / cell death",
        "unit": "in vitro: fraction of cells; in vivo: 0-1 severity score",
        "definition": (
            "This key means two different measurements. In vitro it is the "
            "fraction (not percentage) of propidium-iodide-positive cells, "
            "capped at the sampled parameter p (posterior range 0.700-0.998, "
            "median 0.767), so the in vitro model structurally cannot predict "
            "total cell death. In vivo it is expert-graded H&E necrosis "
            "severity on a 0-1 ordinal scale."
        ),
        "temporal_behaviour": (
            "in vitro: rises monotonically toward the sampled cap p; "
            "in vivo: peaks and then decays"
        ),
        "cross_model_comparable": False,
        "cross_model_note": (
            "The in vitro fraction and the in vivo severity score are different "
            "quantities. They must not be compared or averaged."
        ),
    },
    "inflammation": {
        "label": "Inflammation (transcriptomic surrogate)",
        "unit": "arbitrary units (eigengene score)",
        "definition": (
            "Eigengene score of TXG-MAPr WGCNA module rKID:7m (NF-κB target "
            "genes), roughly 0-8 on the calibration figure's scale. A "
            "transcriptomic surrogate, not a cell count or cytokine level."
        ),
        "temporal_behaviour": "peaks and then decays",
        "cross_model_comparable": False,
    },
    "kidney_failure": {
        "label": "Interstitial fibrosis severity",
        "unit": "0-1 severity score",
        "definition": (
            "Expert-graded interstitial fibrosis severity, mapped onto 0-1 in "
            "increasing order of severity. The source figure for this state "
            "variable is titled 'Fibrosis' with the y-axis in arbitrary units, "
            "while the AOP diagram box reads 'Kidney Failure'. It is NOT acute "
            "kidney injury, NOT eGFR or creatinine, NOT loss of kidney "
            "function and NOT a probability of renal failure."
        ),
        "temporal_behaviour": (
            "cumulative and non-decreasing, so a longer simulated time "
            "mechanically increases it; its asymptote is dose-dependent"
        ),
        "cross_model_comparable": False,
        "severity_mapping_note": (
            "UNCONFIRMED exact mapping of the five histopathology categories "
            "(Absent/Minimal/Mild/Moderate/Marked) onto 0-1. The source states "
            "only 'a range of values between 0 and 1, in increasing order of "
            "severity'."
        ),
    },
}

# What the percentile band is, and what it leaves out.
UNCERTAINTY = {
    "kind": "parameter uncertainty propagated by Monte Carlo sampling of posterior draws",
    "is_prediction_interval": False,
    "note": (
        "percentile_2_5 and percentile_97_5 are percentiles of the Monte Carlo "
        "ensemble, i.e. propagated parameter uncertainty. This is NOT a "
        "prediction interval: it excludes observation and residual error, "
        "between-animal and between-replicate variability, model-structure "
        "uncertainty, and all in vivo pharmacokinetic uncertainty (the six PK "
        "rate constants and the scale factor are hard-coded, and their omitted "
        "uncertainty is larger than the band shows). The band is also pointwise "
        "over the time grid, not a simultaneous band over trajectories, and in "
        "the in vivo fit it is narrower than the spread of the data it was "
        "fitted to. A narrow band does not mean the prediction is accurate."
    ),
    "reproducibility_note": (
        "Results are stochastic and there is no seed. Repeated identical "
        "250-iteration requests vary by a few percent. Do not quote more than "
        "two significant figures, and do not read a small run-to-run difference "
        "as a signal. min and max are sample extrema, not bounds."
    ),
}

CANNOT_ANSWER = (
    "human or patient risk",
    "any chemical other than cisplatin",
    "repeated, chronic or multi-cycle dosing (exposure is single-dose only; "
    "dose sets the initial conditions and never appears in the derivatives)",
    "a benchmark dose, BMD, BMDL, NOAEL or any point of departure",
    "whether a given dose is safe",
    "kidney function, creatinine or eGFR",
    "in vitro to in vivo extrapolation (nothing converts μM into mg/kg)",
    "any species other than RPTEC/TERT1 cells and rat",
)


def endpoints_for(model_type):
    """Endpoint keys a model produces, in response order."""
    if model_type == "invivo":
        return ("dna_damage", "necrosis", "inflammation", "kidney_failure")
    return ("dna_damage", "necrosis")


def extrapolation_warnings(model_type, dose, time_value, simulation_count):
    """Warnings describing how far a request sits outside the calibrated domain.

    Returns a list of ``{code, message, affects}`` dicts. ``affects`` names the
    endpoints a warning applies to, or is absent when it applies to the whole
    response. Empty list means the request is inside the calibrated domain.
    """
    warnings = []
    envelope = CALIBRATED[model_type]

    if dose < envelope["dose"]["min"] or dose > envelope["dose"]["max"]:
        warnings.append({
            "code": "dose_outside_calibration",
            "message": (
                f"Dose {dose} {envelope['dose']['unit']} is outside the "
                f"calibrated range {envelope['dose']['min']}-"
                f"{envelope['dose']['max']} {envelope['dose']['unit']}. The "
                "model is linear in dose and will not fail loudly; treat the "
                "result as extrapolation."
            ),
        })

    if time_value > envelope["time"]["max"]:
        warnings.append({
            "code": "time_outside_calibration",
            "message": (
                f"Time {time_value} hours is beyond the calibrated window of "
                f"{envelope['time']['max']} hours. Endpoint trajectories "
                "diverge here: kidney failure is cumulative and increases with "
                "simulated time, while the others peak and decay."
            ),
        })

    # The two in vivo histopathology endpoints share one cause, so they share
    # one warning rather than emitting the same code twice.
    histopathology = sorted(
        endpoint for endpoint, info in ENDPOINT_CALIBRATION.get(model_type, {}).items()
        if "histopathology_dose_mg_kg" in info
        and dose != info["histopathology_dose_mg_kg"]
    )
    if histopathology:
        scored_at = ENDPOINT_CALIBRATION[model_type][histopathology[0]][
            "histopathology_dose_mg_kg"]
        warnings.append({
            "code": "histopathology_extrapolated",
            "message": (
                f"Expert-graded histopathology was scored at a single "
                f"{scored_at} mg/kg dose (12 time points to 672 h, 3 rats per "
                f"point). At {dose} mg/kg these endpoints are extrapolation "
                f"with no supporting histopathology."
            ),
            "affects": histopathology,
        })

    for endpoint, info in ENDPOINT_CALIBRATION.get(model_type, {}).items():
        if "detection_threshold_um" in info and dose < info["detection_threshold_um"]:
            warnings.append({
                "code": "below_detection_threshold",
                "message": info["caveat"],
                "affects": [endpoint],
            })

    if simulation_count == 1:
        warnings.append({
            "code": "uncertainty_band_degenerate",
            "message": (
                "With simulation_count=1 the percentile fields are all equal to "
                "the single sampled trajectory, so the band has zero width "
                "while still being shaped like a real one. Use the default of "
                "250 or more for any quantitative reading."
            ),
        })
    elif simulation_count < 50:
        warnings.append({
            "code": "monte_carlo_noise",
            "message": (
                f"simulation_count={simulation_count} is low; the percentile "
                "band endpoints rest on very few order statistics and carry "
                "several times the Monte Carlo error of the mean."
            ),
        })

    return warnings
