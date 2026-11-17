"""Equivalence test for the stats extracted out of /api/predict.

``/api/predict`` is frozen for an existing consumer, so extracting its four
inlined numpy blocks into ``simulation_stats`` had to be a pure refactor. This
test pins that: ``REFERENCE_*`` below are the original expressions copied
verbatim from the pre-refactor app.py, and the extracted function must agree
with them exactly -- not approximately.
"""

import json
import os
import sys

import pytest

# Needs numpy; the fast contract job deliberately has no scientific stack, so
# skip rather than error when `pytest tests/` runs there.
np = pytest.importorskip("numpy", reason="needs the scientific stack")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation_stats import ENDPOINTS_BY_MODEL, summarize_all, summarize_endpoint


def reference_final_value(vals, idx):
    """The original app.py final_value block, copied verbatim."""
    return {
        "mean": float(np.mean(vals[:, idx])),
        "std": float(np.std(vals[:, idx])),
        "median": float(np.median(vals[:, idx])),
        "min": float(np.min(vals[:, idx])),
        "max": float(np.max(vals[:, idx])),
        "percentile_2_5": float(np.percentile(vals[:, idx], 2.5)),
        "percentile_25": float(np.percentile(vals[:, idx], 25)),
        "percentile_75": float(np.percentile(vals[:, idx], 75)),
        "percentile_97_5": float(np.percentile(vals[:, idx], 97.5)),
    }


def reference_time_series(vals, t_vals):
    """The original app.py time_series block, copied verbatim."""
    return {
        "time_points": t_vals.tolist(),
        "mean": np.mean(vals, axis=0).tolist(),
        "std": np.std(vals, axis=0).tolist(),
        "median": np.median(vals, axis=0).tolist(),
        "percentile_2_5": np.percentile(vals, 2.5, axis=0).tolist(),
        "percentile_25": np.percentile(vals, 25, axis=0).tolist(),
        "percentile_75": np.percentile(vals, 75, axis=0).tolist(),
        "percentile_97_5": np.percentile(vals, 97.5, axis=0).tolist(),
    }


def make_series(n_sims, n_points, seed):
    """Deterministic synthetic trajectories. The real service has no seed, so
    equivalence is established on fixed arrays rather than on live runs."""
    rng = np.random.default_rng(seed)
    base = np.linspace(0.0, 1.0, n_points)
    return np.abs(rng.normal(1.0, 0.3, size=(n_sims, 1)) * base
                  + rng.normal(0.0, 0.01, size=(n_sims, n_points)))


# Shapes worth covering: the production default, a single simulation (the
# accepted minimum, where std is 0 and the percentile band collapses), the
# accepted maximum, and the testing-config time grid.
SHAPES = [(250, 100), (1, 100), (1000, 100), (250, 20), (2, 100), (3, 5)]


@pytest.mark.parametrize("n_sims,n_points", SHAPES)
def test_matches_original_inline_expressions(n_sims, n_points):
    vals = make_series(n_sims, n_points, seed=n_sims * 1000 + n_points)
    t_vals = np.linspace(0, 24.0, n_points)
    idx = int(np.argmin(np.abs(t_vals - 24.0)))

    got = summarize_endpoint(vals, t_vals, idx)
    expected = {
        "final_value": reference_final_value(vals, idx),
        "time_series": reference_time_series(vals, t_vals),
    }

    # Exact equality, including key order, via canonical JSON.
    assert json.dumps(got, sort_keys=False) == json.dumps(expected, sort_keys=False)


def test_key_order_is_unchanged():
    """A consumer reading the JSON by path does not care about order, but a
    byte-identical response is the cheapest proof the refactor changed nothing."""
    vals = make_series(10, 100, seed=1)
    t_vals = np.linspace(0, 24.0, 100)
    got = summarize_endpoint(vals, t_vals, 99)

    assert list(got) == ["final_value", "time_series"]
    assert list(got["final_value"]) == [
        "mean", "std", "median", "min", "max",
        "percentile_2_5", "percentile_25", "percentile_75", "percentile_97_5",
    ]
    assert list(got["time_series"]) == [
        "time_points", "mean", "std", "median",
        "percentile_2_5", "percentile_25", "percentile_75", "percentile_97_5",
    ]


def test_time_series_has_no_min_max():
    """final_value carries min/max and time_series does not. The two therefore
    cannot share one OpenAPI component -- the spec must keep them separate."""
    got = summarize_endpoint(make_series(5, 10, seed=2), np.linspace(0, 1, 10), 9)
    assert {"min", "max"} <= set(got["final_value"])
    assert not {"min", "max"} & set(got["time_series"])


def test_series_length_follows_the_time_grid():
    """Asserted against the grid, never a literal 100: TIME_POINTS is 20 under
    FLASK_ENV=testing, while two live tgx-monitor checks expect the deployed 100."""
    for n_points in (20, 100):
        t_vals = np.linspace(0, 24.0, n_points)
        got = summarize_endpoint(make_series(4, n_points, seed=3), t_vals, n_points - 1)
        for key, series in got["time_series"].items():
            assert len(series) == n_points, key


def test_summarize_all_preserves_endpoint_order():
    t_vals = np.linspace(0, 24.0, 10)
    series = {name: make_series(5, 10, seed=i)
              for i, name in enumerate(ENDPOINTS_BY_MODEL["invivo"])}
    got = summarize_all(series, t_vals, 9)
    assert list(got) == list(ENDPOINTS_BY_MODEL["invivo"])


def test_single_simulation_collapses_the_band():
    """simulation_count=1 is accepted, and then the '95% interval' has zero
    width while still being shaped like a real one. Pinned because it is the
    reason /api/v1 warns about it."""
    vals = make_series(1, 50, seed=4)
    got = summarize_endpoint(vals, np.linspace(0, 1, 50), 49)["final_value"]
    assert got["std"] == 0.0
    assert got["percentile_2_5"] == got["percentile_97_5"] == got["mean"]
