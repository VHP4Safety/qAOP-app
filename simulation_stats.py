"""Statistical summarisation of Monte Carlo endpoint trajectories.

Extracted verbatim from the four near-identical blocks that were inlined in
``app.py``'s ``/api/predict`` handler. Two reasons:

1. The API layer (``api_v1.py``) can then be imported and tested without
   numpy, scipy, pandas or plotly, which keeps the contract test fast.
2. The four blocks were identical apart from the array they summarised, so a
   change to one statistic had to be made in four places.

The key order below reproduces the original response exactly -- ``/api/predict``
is frozen for an existing consumer (tgx-monitor pins field paths inside
``endpoints.<name>.final_value`` and ``.time_series``), so this module must be a
pure refactor with no observable change.

``final_value`` carries ``min``/``max``; ``time_series`` deliberately does not,
which is why the two cannot share one schema component.
"""

import numpy as np

# Percentiles reported for every endpoint, in response order.
PERCENTILES = (("percentile_2_5", 2.5), ("percentile_25", 25),
               ("percentile_75", 75), ("percentile_97_5", 97.5))

# The endpoints each model produces, in response order. invitro yields the
# first two only; invivo yields all four.
ENDPOINTS_BY_MODEL = {
    "invitro": ("dna_damage", "necrosis"),
    "invivo": ("dna_damage", "necrosis", "inflammation", "kidney_failure"),
}


def summarize_endpoint(values, t_vals, idx):
    """Summarise one endpoint's Monte Carlo trajectories.

    Args:
        values: 2-D array-like, shape (n_simulations, n_time_points).
        t_vals: the time grid, length n_time_points.
        idx: index into the time grid whose column becomes ``final_value``.

    Returns:
        dict with ``final_value`` (9 scalars) and ``time_series`` (8 arrays),
        all plain Python floats and lists, ready for ``jsonify``.
    """
    values = np.asarray(values)
    column = values[:, idx]

    final_value = {
        "mean": float(np.mean(column)),
        "std": float(np.std(column)),
        "median": float(np.median(column)),
        "min": float(np.min(column)),
        "max": float(np.max(column)),
    }
    for key, q in PERCENTILES:
        final_value[key] = float(np.percentile(column, q))

    time_series = {
        "time_points": np.asarray(t_vals).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
    }
    for key, q in PERCENTILES:
        time_series[key] = np.percentile(values, q, axis=0).tolist()

    return {"final_value": final_value, "time_series": time_series}


def summarize_all(series_by_endpoint, t_vals, idx):
    """Summarise every endpoint in ``series_by_endpoint`` (a dict name -> 2-D array)."""
    return {
        name: summarize_endpoint(values, t_vals, idx)
        for name, values in series_by_endpoint.items()
    }
