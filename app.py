# === Import required packages ===
from flask import Flask, render_template, request, jsonify, session, send_file
from datetime import datetime
import os
import uuid
import numpy as np
from scipy.integrate import odeint
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import re
import time as time_module
import logging
from functools import wraps
from config import get_config
from logger import setup_logging, log_request, log_simulation_start, log_simulation_complete, log_error, log_performance_metric
from validation import validate_prediction_request, get_parameter_ranges, ValidationError

# === Initialize the Flask app with configuration ===
config = get_config()
app = Flask(__name__)
app.config.from_object(config)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# === Initialize logging ===
setup_logging()
logger = logging.getLogger("qAOP.app")

# === In-memory storage for report data (avoids cookie size limit) ===
_report_storage = {}

# === Load pre-calibrated parameter samples from file ===
try:
    draws_invitro_df = pd.read_csv(config.INVITRO_PARAMS_FILE)
    logger.info(f"Loaded invitro parameters: {len(draws_invitro_df)} samples")
except Exception as e:
    logger.error(f"Failed to load invitro parameter file: {config.INVITRO_PARAMS_FILE}")
    log_error(e, {"file": str(config.INVITRO_PARAMS_FILE), "type": "data_loading"})
    raise

try:
    draws_invivo_df = pd.read_csv(config.INVIVO_PARAMS_FILE)
    logger.info(f"Loaded invivo parameters: {len(draws_invivo_df)} samples")
except Exception as e:
    logger.error(f"Failed to load invivo parameter file: {config.INVIVO_PARAMS_FILE}")
    log_error(e, {"file": str(config.INVIVO_PARAMS_FILE), "type": "data_loading"})
    raise

try:
    pk_pars = (
        pd.read_csv(config.PK_SUMMARY_FILE)  # no index_col here
          .set_index('variable')               # now you can .loc['Plasma0']
    )
    logger.info(f"Loaded PK parameters: {len(pk_pars)} parameters")
except Exception as e:
    logger.error(f"Failed to load PK summary file: {config.PK_SUMMARY_FILE}")
    log_error(e, {"file": str(config.PK_SUMMARY_FILE), "type": "data_loading"})
    raise

# === Decorators for error handling and performance monitoring ===
def monitor_performance(func):
    """Decorator to monitor function performance and log errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time_module.time()
        func_logger = logging.getLogger(f"qAOP.{func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            duration = time_module.time() - start_time
            log_performance_metric(
                f"{func.__name__}_duration",
                duration,
                "seconds",
                {"args_count": len(args), "kwargs_count": len(kwargs)}
            )
            func_logger.debug(f"{func.__name__} completed successfully in {duration:.4f}s")
            return result
        except Exception as e:
            duration = time_module.time() - start_time
            func_logger.error(f"{func.__name__} failed after {duration:.4f}s: {str(e)}")
            log_error(e, {
                "function": func.__name__,
                "args_count": len(args),
                "kwargs_count": len(kwargs),
                "duration": duration
            })
            raise
    return wrapper

# === Randomly sample one parameter set from posterior draws ===
@monitor_performance
def sample_random_iteration(model_type="invitro"):
    try:
        if model_type == "invitro":
            if draws_invitro_df.empty:
                raise ValueError("Invitro parameter data is empty")
            sampled_row = draws_invitro_df.sample(1).iloc[0]
            return [
                sampled_row['k_cDD'], sampled_row['d_DD'], sampled_row['k_hillDD'],
                sampled_row['k_inter'], sampled_row['maxdeath'], sampled_row['k_hillnec'],
                sampled_row['p'], sampled_row['h']
            ]
        elif model_type == "invivo":
            if draws_invivo_df.empty:
                raise ValueError("Invivo parameter data is empty")
            sampled_row = draws_invivo_df.sample(1).iloc[0]
            return [
                sampled_row['k_kidDD'], sampled_row['hillDD'], sampled_row['d_DD'], sampled_row['maxdeath'], 
                sampled_row['h1'], sampled_row['k_hillnec'], sampled_row['d_CD'], sampled_row['k_CDINF'], 
                sampled_row['d_INF'], sampled_row['k_INFKF'], sampled_row['hillKF'], sampled_row['INF0']   
            ]
        else:
            raise ValueError(f"Unknown model type '{model_type}'. Choose 'invitro' or 'invivo'.")
    except KeyError as e:
        raise ValueError(f"Missing required parameter '{e.args[0]}' in {model_type} data") from e
    except Exception as e:
        raise RuntimeError(f"Failed to sample parameters for {model_type} model: {str(e)}") from e


# === ODE model defining the in vitro qAOP system ===
def invitro_model(y, t, dose, params):
    Q_api, Q_bas, Q_cell, Q_inter, DD, NEC = y
    N_cell, V_cell, F_outa, F_outb, F_inb, F_ina, V_api, V_bas, K_met = params[:9]
    k_cDD, d_DD, k_hillDD, k_inter, maxdeath, k_hillnec, p, h = params[9:]

    dQ_api_dt = N_cell * ((F_outa * Q_cell / (N_cell * V_cell)) - F_ina * Q_api / V_api - K_met * Q_cell / (N_cell * V_cell))
    dQ_bas_dt = N_cell * ((F_outb * Q_cell / (N_cell * V_cell)) - F_inb * Q_bas / V_bas)
    dQ_cell_dt = N_cell * ((F_ina * Q_api / V_api + F_inb * Q_bas / V_bas) - (F_outa + F_outb) * Q_cell / (N_cell * V_cell))
    dQ_inter_dt = k_inter * (Q_cell - Q_inter)
    dDD_dt = -d_DD * DD + k_cDD * Q_inter / (k_hillDD + Q_inter)
    dNEC_dt = maxdeath * (DD**h / (k_hillnec**h + DD**h)) * (p - NEC)
    
    return [dQ_api_dt, dQ_bas_dt, dQ_cell_dt, dQ_inter_dt, dDD_dt, dNEC_dt]

# === ODE model defining the in vitro qAOP system ===
def invivo_model(y, t, dose, params):
    Plasma, KidneyPt, AccuPt, DD, CD, INF, KF, l = y
    k_kidplas, k_plaskid, k_eplas, k_accukid, k_kidaccu, k_ekid, scale, d_l = params[:8]
    k_kidDD, k_hillDD, d_DD, maxdeath, h, k_hillNEC, d_CD, k_CDINF, d_INF, k_INFKF, k_hillKF = params[8:]

    dPlasma_dt = KidneyPt * k_kidplas - Plasma * (k_plaskid + k_eplas)
    dKidneyPt_dt = AccuPt * k_accukid + Plasma * k_plaskid - KidneyPt * (k_kidplas + k_kidaccu + k_ekid)
    dAccuPt_dt = KidneyPt * k_kidaccu - AccuPt * k_accukid
    dDD_dt = (k_kidDD * scale * (KidneyPt + AccuPt)) / (k_hillDD + scale * (KidneyPt + AccuPt)) - DD * d_DD
    dCD_dt = (maxdeath * (DD**h) / (k_hillNEC**h + DD**h)) * (1 - CD) - d_CD * CD * INF
    dINF_dt = k_CDINF * CD - d_INF * INF
    dKF_dt = (k_INFKF * (INF**(l + 1))) / (k_hillKF**(l + 1) + INF**(l + 1)) * (1 - KF)
    dl_dt = -d_l * l

    
    return [dPlasma_dt, dKidneyPt_dt, dAccuPt_dt, dDD_dt, dCD_dt, dINF_dt, dKF_dt, dl_dt]


# === Fixed model parameters (from experimental setup or calibration) ===
fixed_params_invitro = [2e6, 2005, 1.24e4, 4.518e1, 2.304e3, 2.232e4, 1e12, 2e12, 3.024e4]
fixed_params_invivo = [2.13e-1, 2.19, 6.13e-2, 4.73e-4, 4.7e-3, 4.9e-3, 3.99, 0.005]

# === Define initial conditions based on input dose ===
def initial_conditions_invitro(dose):
    Q_api_0 = dose * 301.1 * 1e-15 * 1e12
    Q_bas_0 = dose * 301.1 * 1e-15 * 2e12
    return [Q_api_0, Q_bas_0, 0, 0, 0, 0]

# === Run the ODE solver for a single simulation ===
@monitor_performance
def run_invitro_model(dose, simulation_time, sampled_params):
    try:
        if dose < 0:
            raise ValueError("Dose must be non-negative")
        if simulation_time <= 0:
            raise ValueError("Simulation time must be positive")
        if not sampled_params:
            raise ValueError("Sampled parameters cannot be empty")
        
        params = fixed_params_invitro + sampled_params
        y0 = initial_conditions_invitro(dose)
        t = np.linspace(0, simulation_time, config.TIME_POINTS)
        solution = odeint(invitro_model, y0, t, args=(dose, params))
        
        # Check for integration issues
        if not np.all(np.isfinite(solution)):
            raise RuntimeError("ODE integration produced non-finite values")
        
        return solution, t
    except Exception as e:
        raise RuntimeError(f"Invitro model simulation failed: {str(e)}") from e

def initial_conditions_invivo(dose, INF0):
    Plasma0_base = pk_pars.loc['Plasma0', 'mean']
    Plasma_0     = Plasma0_base * (dose / 5)  
    KidneyPt_0 = 0
    AccuPt_0 = 0
    DD_0 = 0
    CD_0 = 0
    INF_0 = INF0             # <- Sampled initial inflammation
    KF_0 = 0
    l_0 = 10                 # <- Fixed initial l value
    return [Plasma_0, KidneyPt_0, AccuPt_0, DD_0, CD_0, INF_0, KF_0, l_0]

@monitor_performance
def run_invivo_model(dose, simulation_time, sampled_params):
    try:
        if dose < 0:
            raise ValueError("Dose must be non-negative")
        if simulation_time <= 0:
            raise ValueError("Simulation time must be positive")
        if len(sampled_params) < 12:
            raise ValueError(f"Expected 12 sampled parameters, got {len(sampled_params)}")
        
        params = fixed_params_invivo + sampled_params[:-1]  # last param is INF0
        INF0 = sampled_params[-1]                           # INF0 is last in list
        
        if INF0 < 0:
            raise ValueError("Initial inflammation (INF0) must be non-negative")
        
        y0 = initial_conditions_invivo(dose, INF0)
        t = np.linspace(0, simulation_time, config.TIME_POINTS)
        solution = odeint(invivo_model, y0, t, args=(dose, params))
        
        # Check for integration issues
        if not np.all(np.isfinite(solution)):
            raise RuntimeError("ODE integration produced non-finite values")
        
        return solution, t
    except Exception as e:
        raise RuntimeError(f"Invivo model simulation failed: {str(e)}") from e

# === Helper function for color conversion ===
def color_to_rgb(color):
    """Convert color name to RGB values for RGBA"""
    color_map = {
        'red': '255, 0, 0',
        'blue': '0, 0, 255',
        'green': '0, 128, 0',
        'purple': '128, 0, 128'
    }
    return color_map.get(color, '0, 0, 0')

# === Plotly plotting functions ===
def create_plotly_distribution(values, label, color, time_index, endpoint_stats):
    """Create interactive distribution plot with Plotly"""
    final_values = values[:, time_index]

    # Create figure with histogram and box plot
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=(f'Distribution of {label}', 'Box Plot')
    )

    # Histogram with KDE overlay
    fig.add_trace(
        go.Histogram(
            x=final_values,
            nbinsx=30,
            name='Distribution',
            marker_color=color,
            opacity=0.6,
            histnorm='probability density'
        ),
        row=1, col=1
    )

    # KDE curve
    kde = gaussian_kde(final_values)
    x_vals = np.linspace(final_values.min(), final_values.max(), 300)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=kde(x_vals),
            mode='lines',
            name='KDE',
            line=dict(color='black', width=2)
        ),
        row=1, col=1
    )

    # Add statistical markers (mean, median, CI)
    fig.add_vline(x=endpoint_stats['mean'], line_dash="dash",
                  line_color="blue", annotation_text="Mean", row=1, col=1)
    fig.add_vline(x=endpoint_stats['median'], line_dash="dot",
                  line_color="green", annotation_text="Median", row=1, col=1)
    fig.add_vline(x=endpoint_stats['percentile_2_5'], line_dash="dash",
                  line_color="red", opacity=0.5, annotation_text="95% CI", row=1, col=1)
    fig.add_vline(x=endpoint_stats['percentile_97_5'], line_dash="dash",
                  line_color="red", opacity=0.5, row=1, col=1)

    # Box plot
    fig.add_trace(
        go.Box(
            y=final_values,
            name='Statistics',
            marker_color=color,
            boxmean='sd',  # Show mean and std
            orientation='v'  # Vertical orientation
        ),
        row=2, col=1
    )

    # Update layout for interactivity
    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        height=600
    )

    fig.update_xaxes(title_text='', row=2, col=1)
    fig.update_yaxes(title_text=f'{label} Value', row=2, col=1)
    fig.update_yaxes(title_text='Density', row=1, col=1)

    # Return JSON for frontend rendering
    return fig.to_json()

def create_plotly_timeseries(values, time_points, label, color, dose, unit, endpoint_stats):
    """Create interactive time series plot with confidence bands"""
    fig = go.Figure()

    # Add individual traces (toggleable via legend as a group)
    for i in range(min(50, values.shape[0])):  # Limit to 50 traces for performance
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=values[i],
                mode='lines',
                name='Individual Simulations',
                legendgroup='simulations',
                showlegend=(i == 0),  # Only show first trace in legend
                line=dict(color=color, width=0.5),
                opacity=0.2,
                hoverinfo='skip'
            )
        )

    # 95% Confidence Interval (shaded band)
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=endpoint_stats['time_series']['percentile_97_5'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=endpoint_stats['time_series']['percentile_2_5'],
            mode='lines',
            line=dict(width=0),
            fillcolor=f'rgba({color_to_rgb(color)}, 0.2)',
            fill='tonexty',
            name='95% CI',
            hovertemplate='95% CI<extra></extra>'
        )
    )

    # IQR band (darker shading)
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=endpoint_stats['time_series']['percentile_75'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=endpoint_stats['time_series']['percentile_25'],
            mode='lines',
            line=dict(width=0),
            fillcolor=f'rgba({color_to_rgb(color)}, 0.4)',
            fill='tonexty',
            name='IQR (25-75%)',
        )
    )

    # Median line (dashed)
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=endpoint_stats['time_series']['median'],
            mode='lines',
            name='Median',
            line=dict(color='green', width=2, dash='dot'),
            hovertemplate='Median: %{y:.2f}<extra></extra>'
        )
    )

    # Mean line (bold, primary)
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=endpoint_stats['time_series']['mean'],
            mode='lines',
            name='Mean',
            line=dict(color='black', width=3),
            hovertemplate='Mean: %{y:.2f}<br>Time: %{x:.1f} hrs<extra></extra>'
        )
    )

    # Layout
    fig.update_layout(
        title=f'{label} Over Time (Dose = {dose:.1f} {unit})',
        xaxis_title='Time (hours)',
        yaxis_title=f'{label} Level',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )

    # Return JSON for frontend rendering
    return fig.to_json()

def create_plotly_dose_response(dose_data, endpoint_name, model_type):
    """Create dose-response curve plot (web UI only, no API endpoint)"""
    doses = dose_data['doses']
    means = dose_data['means']
    ci_lower = dose_data['ci_lower']
    ci_upper = dose_data['ci_upper']

    fig = go.Figure()

    # Error bars for 95% CI
    fig.add_trace(
        go.Scatter(
            x=doses,
            y=means,
            mode='lines+markers',
            name=endpoint_name.replace('_', ' ').title(),
            error_y=dict(
                type='data',
                symmetric=False,
                array=[u - m for u, m in zip(ci_upper, means)],
                arrayminus=[m - l for m, l in zip(means, ci_lower)],
                color='rgba(0,0,0,0.3)'
            ),
            marker=dict(size=10),
            line=dict(width=2),
            hovertemplate='Dose: %{x}<br>Mean: %{y:.2f}<br>95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<extra></extra>',
            customdata=list(zip(ci_lower, ci_upper))
        )
    )

    # Layout
    unit = 'μM' if model_type == 'invitro' else 'mg/kg'
    fig.update_layout(
        title=f'Dose-Response Curve: {endpoint_name.replace("_", " ").title()}',
        xaxis_title=f'Dose ({unit})',
        yaxis_title='Response Value',
        xaxis_type='log',  # Logarithmic dose axis
        template='plotly_white',
        height=500,
        showlegend=True
    )

    return fig.to_json()

# === API Routes ===

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint for API monitoring."""
    try:
        # Check data availability
        data_status = {
            "invitro_params": len(draws_invitro_df) > 0,
            "invivo_params": len(draws_invivo_df) > 0,
            "pk_params": len(pk_pars) > 0
        }
        
        # Test basic model functionality
        try:
            test_params_invitro = sample_random_iteration("invitro")
            test_params_invivo = sample_random_iteration("invivo")
            model_status = {
                "invitro_sampling": len(test_params_invitro) == 8,
                "invivo_sampling": len(test_params_invivo) == 12
            }
        except Exception as e:
            logger.warning(f"Model health check failed: {str(e)}")
            model_status = {
                "invitro_sampling": False,
                "invivo_sampling": False
            }
        
        overall_healthy = all(data_status.values()) and all(model_status.values())
        
        response = {
            "status": "healthy" if overall_healthy else "degraded",
            "version": "1.0.0",
            "models": ["invitro", "invivo"],
            "checks": {
                "data": data_status,
                "models": model_status
            },
            "configuration": {
                "simulation_count": config.SIMULATION_COUNT,
                "time_points": config.TIME_POINTS,
                "debug": config.DEBUG
            }
        }
        
        # Log health check
        logger.info("Health check performed", extra={
            'extra_fields': {
                'status': response["status"],
                'checks': response["checks"],
                'event_type': 'health_check'
            }
        })
        
        return jsonify(response), 200 if overall_healthy else 503
        
    except Exception as e:
        log_error(e, {"endpoint": "/api/health"})
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    API endpoint for model predictions.
    
    Expected JSON input:
    {
        "model_type": "invitro" | "invivo",
        "dose": float,
        "time": float,
        "simulation_count": int (optional, default from config)
    }
    
    Returns JSON with predictions and statistics.
    """
    start_time = time_module.time()
    try:
        data = request.get_json()
        log_request(data or {}, "/api/predict")
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Comprehensive validation
        try:
            validated_data = validate_prediction_request(data)
            model_type = validated_data["model_type"]
            dose = validated_data["dose"]
            time_param = validated_data["time"]  # Renamed to avoid conflict with time module
            simulation_count = validated_data["simulation_count"]
        except ValidationError as e:
            return jsonify({"error": str(e)}), 400
        
        # Run simulation
        log_simulation_start(model_type, dose, time_param, simulation_count)
        simulation_start = time_module.time()
        
        t_vals = np.linspace(0, time_param, config.TIME_POINTS)
        unit = "μM" if model_type == "invitro" else "mg/kg"
        
        dd_vals, necrosis_vals = [], []
        inflammation_vals, kf_vals = [], []
        
        for i in range(simulation_count):
            try:
                params = sample_random_iteration(model_type)
                if model_type == "invitro":
                    sol, _ = run_invitro_model(dose, time_param, params)
                    dd_vals.append(sol[:, -2])      # DNA Damage
                    necrosis_vals.append(sol[:, -1])# Necrosis
                elif model_type == "invivo":
                    sol, _ = run_invivo_model(dose, time_param, params)
                    dd_vals.append(sol[:, 3])       # DNA Damage (DD)
                    necrosis_vals.append(sol[:, 4]) # Cell Death (CD)
                    inflammation_vals.append(sol[:, 5])  # Inflammation (INF)
                    kf_vals.append(sol[:, 6])            # Kidney Failure (KF)
            except Exception as e:
                logger.warning(f"Simulation {i+1}/{simulation_count} failed: {str(e)}")
                if len(dd_vals) == 0:  # If no successful simulations yet, re-raise
                    raise
        
        # Convert to arrays and calculate statistics
        dd_vals = np.array(dd_vals)
        necrosis_vals = np.array(necrosis_vals)
        
        idx = np.argmin(np.abs(t_vals - time_param))
        
        # Prepare response
        response = {
            "model_type": model_type,
            "dose": dose,
            "dose_unit": unit,
            "time": time_param,
            "time_unit": "hours",
            "simulation_count": simulation_count,
            "endpoints": {
                "dna_damage": {
                    "final_value": {
                        "mean": float(np.mean(dd_vals[:, idx])),
                        "std": float(np.std(dd_vals[:, idx])),
                        "median": float(np.median(dd_vals[:, idx])),
                        "min": float(np.min(dd_vals[:, idx])),
                        "max": float(np.max(dd_vals[:, idx])),
                        "percentile_2_5": float(np.percentile(dd_vals[:, idx], 2.5)),
                        "percentile_25": float(np.percentile(dd_vals[:, idx], 25)),
                        "percentile_75": float(np.percentile(dd_vals[:, idx], 75)),
                        "percentile_97_5": float(np.percentile(dd_vals[:, idx], 97.5))
                    },
                    "time_series": {
                        "time_points": t_vals.tolist(),
                        "mean": np.mean(dd_vals, axis=0).tolist(),
                        "std": np.std(dd_vals, axis=0).tolist(),
                        "median": np.median(dd_vals, axis=0).tolist(),
                        "percentile_2_5": np.percentile(dd_vals, 2.5, axis=0).tolist(),
                        "percentile_25": np.percentile(dd_vals, 25, axis=0).tolist(),
                        "percentile_75": np.percentile(dd_vals, 75, axis=0).tolist(),
                        "percentile_97_5": np.percentile(dd_vals, 97.5, axis=0).tolist()
                    }
                },
                "necrosis": {
                    "final_value": {
                        "mean": float(np.mean(necrosis_vals[:, idx])),
                        "std": float(np.std(necrosis_vals[:, idx])),
                        "median": float(np.median(necrosis_vals[:, idx])),
                        "min": float(np.min(necrosis_vals[:, idx])),
                        "max": float(np.max(necrosis_vals[:, idx])),
                        "percentile_2_5": float(np.percentile(necrosis_vals[:, idx], 2.5)),
                        "percentile_25": float(np.percentile(necrosis_vals[:, idx], 25)),
                        "percentile_75": float(np.percentile(necrosis_vals[:, idx], 75)),
                        "percentile_97_5": float(np.percentile(necrosis_vals[:, idx], 97.5))
                    },
                    "time_series": {
                        "time_points": t_vals.tolist(),
                        "mean": np.mean(necrosis_vals, axis=0).tolist(),
                        "std": np.std(necrosis_vals, axis=0).tolist(),
                        "median": np.median(necrosis_vals, axis=0).tolist(),
                        "percentile_2_5": np.percentile(necrosis_vals, 2.5, axis=0).tolist(),
                        "percentile_25": np.percentile(necrosis_vals, 25, axis=0).tolist(),
                        "percentile_75": np.percentile(necrosis_vals, 75, axis=0).tolist(),
                        "percentile_97_5": np.percentile(necrosis_vals, 97.5, axis=0).tolist()
                    }
                }
            }
        }
        
        # Add invivo-specific endpoints
        if model_type == "invivo":
            inflammation_vals = np.array(inflammation_vals)
            kf_vals = np.array(kf_vals)
            
            response["endpoints"]["inflammation"] = {
                "final_value": {
                    "mean": float(np.mean(inflammation_vals[:, idx])),
                    "std": float(np.std(inflammation_vals[:, idx])),
                    "median": float(np.median(inflammation_vals[:, idx])),
                    "min": float(np.min(inflammation_vals[:, idx])),
                    "max": float(np.max(inflammation_vals[:, idx])),
                    "percentile_2_5": float(np.percentile(inflammation_vals[:, idx], 2.5)),
                    "percentile_25": float(np.percentile(inflammation_vals[:, idx], 25)),
                    "percentile_75": float(np.percentile(inflammation_vals[:, idx], 75)),
                    "percentile_97_5": float(np.percentile(inflammation_vals[:, idx], 97.5))
                },
                "time_series": {
                    "time_points": t_vals.tolist(),
                    "mean": np.mean(inflammation_vals, axis=0).tolist(),
                    "std": np.std(inflammation_vals, axis=0).tolist(),
                    "median": np.median(inflammation_vals, axis=0).tolist(),
                    "percentile_2_5": np.percentile(inflammation_vals, 2.5, axis=0).tolist(),
                    "percentile_25": np.percentile(inflammation_vals, 25, axis=0).tolist(),
                    "percentile_75": np.percentile(inflammation_vals, 75, axis=0).tolist(),
                    "percentile_97_5": np.percentile(inflammation_vals, 97.5, axis=0).tolist()
                }
            }

            response["endpoints"]["kidney_failure"] = {
                "final_value": {
                    "mean": float(np.mean(kf_vals[:, idx])),
                    "std": float(np.std(kf_vals[:, idx])),
                    "median": float(np.median(kf_vals[:, idx])),
                    "min": float(np.min(kf_vals[:, idx])),
                    "max": float(np.max(kf_vals[:, idx])),
                    "percentile_2_5": float(np.percentile(kf_vals[:, idx], 2.5)),
                    "percentile_25": float(np.percentile(kf_vals[:, idx], 25)),
                    "percentile_75": float(np.percentile(kf_vals[:, idx], 75)),
                    "percentile_97_5": float(np.percentile(kf_vals[:, idx], 97.5))
                },
                "time_series": {
                    "time_points": t_vals.tolist(),
                    "mean": np.mean(kf_vals, axis=0).tolist(),
                    "std": np.std(kf_vals, axis=0).tolist(),
                    "median": np.median(kf_vals, axis=0).tolist(),
                    "percentile_2_5": np.percentile(kf_vals, 2.5, axis=0).tolist(),
                    "percentile_25": np.percentile(kf_vals, 25, axis=0).tolist(),
                    "percentile_75": np.percentile(kf_vals, 75, axis=0).tolist(),
                    "percentile_97_5": np.percentile(kf_vals, 97.5, axis=0).tolist()
                }
            }
        
        # Log successful completion
        simulation_duration = time_module.time() - simulation_start
        total_duration = time_module.time() - start_time
        log_simulation_complete(model_type, simulation_duration, True)
        log_performance_metric("api_predict_total", total_duration, "seconds", {
            "model_type": model_type,
            "simulation_count": simulation_count,
            "successful_simulations": len(dd_vals)
        })
        
        return jsonify(response)
        
    except Exception as e:
        # Log error with context
        error_duration = time_module.time() - start_time
        log_error(e, {
            "endpoint": "/api/predict",
            "request_data": data if 'data' in locals() else {},
            "duration": error_duration
        })
        
        # Return appropriate error response
        if isinstance(e, ValueError):
            return jsonify({"error": str(e)}), 400
        else:
            return jsonify({
                "error": f"Internal server error: {str(e)}"
            }), 500

@app.route("/api/models", methods=["GET"])
def get_models():
    """Get information about available models."""
    return jsonify({
        "models": {
            "invitro": {
                "description": "6-compartment model for RPTEC/TERT1 cell culture",
                "dose_unit": "μM",
                "typical_dose_range": "1-100 μM",
                "time_unit": "hours",
                "typical_time_range": "1-72 hours",
                "endpoints": ["dna_damage", "necrosis"]
            },
            "invivo": {
                "description": "8-compartment rat kidney model",
                "dose_unit": "mg/kg",
                "typical_dose_range": "1-20 mg/kg",
                "time_unit": "hours", 
                "typical_time_range": "1-700 hours",
                "endpoints": ["dna_damage", "necrosis", "inflammation", "kidney_failure"]
            }
        },
        "configuration": {
            "default_simulation_count": config.SIMULATION_COUNT,
            "default_time_points": config.TIME_POINTS
        }
    })

@app.route("/api/ranges", methods=["GET"])
def get_ranges():
    """Get parameter ranges for validation."""
    try:
        model_type = request.args.get("model_type")
        ranges = get_parameter_ranges(model_type)
        return jsonify(ranges)
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log_error(e, {"endpoint": "/api/ranges"})
        return jsonify({"error": "Internal server error"}), 500

# === Flask route: homepage + form processing ===
@app.route("/", methods=["GET", "POST"])
def index():
    model_type = "invitro"
    dose = None
    time = None
    
    if dose is None:
        dose = 50 if model_type == "invitro" else 5   # invitro→50μM, invivo→5 mg/kg
    if time is None:
        time = 72 if model_type == "invitro" else 700  # invitro→24 h, invivo→48 h

    # Initialize results
    result_dd = result_nec = result_infl = result_kf = None
    plot_dist_dd = plot_dist_nec = plot_dist_infl = plot_dist_kf = None
    plot_time_dd = plot_time_nec = plot_time_infl = plot_time_kf = None

    if request.method == "POST":
        model_type = request.form.get("model", "invitro")  # Capture model type
        
        # Apply validation to web form inputs
        try:
            form_data = {
                "model_type": model_type,
                "dose": request.form.get("dose"),
                "time": request.form.get("time"),
                "simulation_count": config.SIMULATION_COUNT  # Use default for web interface
            }
            validated_data = validate_prediction_request(form_data)
            dose = validated_data["dose"]
            time_param = validated_data["time"]
            model_type = validated_data["model_type"]
        except ValidationError as e:
            result_dd = f"Validation error: {str(e)}"
            return render_template("index.html", result_dd=result_dd)
        except Exception as e:
            result_dd = f"Invalid input: Please enter valid numeric values for dose and time."
            return render_template("index.html", result_dd=result_dd)
        t_vals = np.linspace(0, time_param, config.TIME_POINTS)
        unit = "μM" if model_type == "invitro" else "mg/kg"

        dd_vals, necrosis_vals = [], []
        inflammation_vals, kf_vals = [], []  # Only filled if in vivo

        for _ in range(config.SIMULATION_COUNT):
            params = sample_random_iteration(model_type)
            if model_type == "invitro":
                sol, _ = run_invitro_model(dose, time_param, params)
                dd_vals.append(sol[:, -2])      # DNA Damage
                necrosis_vals.append(sol[:, -1])# Necrosis
            elif model_type == "invivo":
                sol, _ = run_invivo_model(dose, time_param, params)
                dd_vals.append(sol[:, 3])       # DNA Damage (DD)
                necrosis_vals.append(sol[:, 4]) # Cell Death (CD)
                inflammation_vals.append(sol[:, 5])  # Inflammation (INF)
                kf_vals.append(sol[:, 6])            # Kidney Failure (KF)

        # Convert to arrays
        dd_vals = np.array(dd_vals)
        necrosis_vals = np.array(necrosis_vals)
        idx = np.argmin(np.abs(t_vals - time_param))

        # DNA Damage
        mean_dd = dd_vals[:, idx].mean()
        std_dd = dd_vals[:, idx].std()
        result_dd = f"Predicted DNA Damage at {time_param:.1f} h: Mean = {mean_dd:.2f}, Std = {std_dd:.2f}"

        # Prepare endpoint stats for Plotly
        dd_stats = {
            'mean': float(np.mean(dd_vals[:, idx])),
            'median': float(np.median(dd_vals[:, idx])),
            'percentile_2_5': float(np.percentile(dd_vals[:, idx], 2.5)),
            'percentile_25': float(np.percentile(dd_vals[:, idx], 25)),
            'percentile_75': float(np.percentile(dd_vals[:, idx], 75)),
            'percentile_97_5': float(np.percentile(dd_vals[:, idx], 97.5)),
            'time_series': {
                'mean': np.mean(dd_vals, axis=0).tolist(),
                'median': np.median(dd_vals, axis=0).tolist(),
                'percentile_2_5': np.percentile(dd_vals, 2.5, axis=0).tolist(),
                'percentile_25': np.percentile(dd_vals, 25, axis=0).tolist(),
                'percentile_75': np.percentile(dd_vals, 75, axis=0).tolist(),
                'percentile_97_5': np.percentile(dd_vals, 97.5, axis=0).tolist(),
            }
        }

        plot_dist_dd = create_plotly_distribution(dd_vals, "DNA Damage", "red", idx, dd_stats)
        plot_time_dd = create_plotly_timeseries(dd_vals, t_vals, "DNA Damage", "red", dose, unit, dd_stats)

        # Necrosis
        mean_nec = necrosis_vals[:, idx].mean()
        std_nec = necrosis_vals[:, idx].std()
        if model_type == "invivo":
            result_nec = (
                f"Predicted necrosis level at {time_param:.1f} h: "
                f"Mean = {mean_nec:.4f}, Std = {std_nec:.4f} "
                f"(0.0–0.25: minimal; 0.25–0.5: mild; "
                f"0.5–0.75: moderate; 0.75–1.0: marked)"
            )
        else:
            result_nec = f"Predicted fraction of necrotic cells at {time_param:.1f} h: Mean = {mean_nec:.4f}, Std = {std_nec:.4f}"

        # Prepare endpoint stats for Plotly
        nec_stats = {
            'mean': float(np.mean(necrosis_vals[:, idx])),
            'median': float(np.median(necrosis_vals[:, idx])),
            'percentile_2_5': float(np.percentile(necrosis_vals[:, idx], 2.5)),
            'percentile_25': float(np.percentile(necrosis_vals[:, idx], 25)),
            'percentile_75': float(np.percentile(necrosis_vals[:, idx], 75)),
            'percentile_97_5': float(np.percentile(necrosis_vals[:, idx], 97.5)),
            'time_series': {
                'mean': np.mean(necrosis_vals, axis=0).tolist(),
                'median': np.median(necrosis_vals, axis=0).tolist(),
                'percentile_2_5': np.percentile(necrosis_vals, 2.5, axis=0).tolist(),
                'percentile_25': np.percentile(necrosis_vals, 25, axis=0).tolist(),
                'percentile_75': np.percentile(necrosis_vals, 75, axis=0).tolist(),
                'percentile_97_5': np.percentile(necrosis_vals, 97.5, axis=0).tolist(),
            }
        }

        plot_dist_nec = create_plotly_distribution(necrosis_vals, "Necrosis", "blue", idx, nec_stats)
        plot_time_nec = create_plotly_timeseries(necrosis_vals, t_vals, "Necrosis", "blue", dose, unit, nec_stats)

        # If in vivo, also plot Inflammation and Kidney Failure
        if model_type == "invivo":
            inflammation_vals = np.array(inflammation_vals)
            kf_vals = np.array(kf_vals)

            mean_infl = inflammation_vals[:, idx].mean()
            std_infl = inflammation_vals[:, idx].std()
            result_infl = f"Predicted Inflammation at {time_param:.1f} h: Mean = {mean_infl:.2f}, Std = {std_infl:.2f}"

            # Prepare endpoint stats for Plotly
            infl_stats = {
                'mean': float(np.mean(inflammation_vals[:, idx])),
                'median': float(np.median(inflammation_vals[:, idx])),
                'percentile_2_5': float(np.percentile(inflammation_vals[:, idx], 2.5)),
                'percentile_25': float(np.percentile(inflammation_vals[:, idx], 25)),
                'percentile_75': float(np.percentile(inflammation_vals[:, idx], 75)),
                'percentile_97_5': float(np.percentile(inflammation_vals[:, idx], 97.5)),
                'time_series': {
                    'mean': np.mean(inflammation_vals, axis=0).tolist(),
                    'median': np.median(inflammation_vals, axis=0).tolist(),
                    'percentile_2_5': np.percentile(inflammation_vals, 2.5, axis=0).tolist(),
                    'percentile_25': np.percentile(inflammation_vals, 25, axis=0).tolist(),
                    'percentile_75': np.percentile(inflammation_vals, 75, axis=0).tolist(),
                    'percentile_97_5': np.percentile(inflammation_vals, 97.5, axis=0).tolist(),
                }
            }

            plot_dist_infl = create_plotly_distribution(inflammation_vals, "Inflammation", "green", idx, infl_stats)
            plot_time_infl = create_plotly_timeseries(inflammation_vals, t_vals, "Inflammation", "green", dose, unit, infl_stats)

            mean_kf = kf_vals[:, idx].mean()
            std_kf = kf_vals[:, idx].std()
            result_kf = (
                f"Predicted kidney failure level at {time_param:.1f} h: "
                f"Mean = {mean_kf:.2f}, Std = {std_kf:.2f} "
                f"(0.0–0.25: minimal; 0.25–0.5: mild; "
                f"0.5–0.75: moderate; 0.75–1.0: marked)"
            )

            # Prepare endpoint stats for Plotly
            kf_stats = {
                'mean': float(np.mean(kf_vals[:, idx])),
                'median': float(np.median(kf_vals[:, idx])),
                'percentile_2_5': float(np.percentile(kf_vals[:, idx], 2.5)),
                'percentile_25': float(np.percentile(kf_vals[:, idx], 25)),
                'percentile_75': float(np.percentile(kf_vals[:, idx], 75)),
                'percentile_97_5': float(np.percentile(kf_vals[:, idx], 97.5)),
                'time_series': {
                    'mean': np.mean(kf_vals, axis=0).tolist(),
                    'median': np.median(kf_vals, axis=0).tolist(),
                    'percentile_2_5': np.percentile(kf_vals, 2.5, axis=0).tolist(),
                    'percentile_25': np.percentile(kf_vals, 25, axis=0).tolist(),
                    'percentile_75': np.percentile(kf_vals, 75, axis=0).tolist(),
                    'percentile_97_5': np.percentile(kf_vals, 97.5, axis=0).tolist(),
                }
            }

            plot_dist_kf = create_plotly_distribution(kf_vals, "Kidney Failure", "purple", idx, kf_stats)
            plot_time_kf = create_plotly_timeseries(kf_vals, t_vals, "Kidney Failure", "purple", dose, unit, kf_stats)
    
    defaults = {
    "invitro": {"dose": 50, "time": 72},
    "invivo":  {"dose": 5,  "time": 672}
    }

    # Get parameter ranges for the frontend
    param_ranges = get_parameter_ranges()

    # Prepare raw simulation data for client-side time slider (Issue #19)
    simulation_data = None
    if result_dd:  # Only prepare if simulation was run
        simulation_data = {
            'time_points': t_vals.tolist(),
            'model_type': model_type,
            'endpoints': {
                'dd': dd_vals.tolist(),
                'nec': necrosis_vals.tolist()
            }
        }
        if model_type == 'invivo':
            simulation_data['endpoints']['infl'] = inflammation_vals.tolist()
            simulation_data['endpoints']['kf'] = kf_vals.tolist()

    # Store results for report generation (server-side to avoid cookie size limit)
    if result_dd:  # Only store if simulation was run
        session_id = str(uuid.uuid4())
        _report_storage[session_id] = {
            'model_type': model_type,
            'dose': dose,
            'time': time_param if 'time_param' in locals() else time,
            'dose_unit': unit if 'unit' in locals() else ('μM' if model_type == 'invitro' else 'mg/kg'),
            'simulation_count': config.SIMULATION_COUNT,
            'plots': {
                'plot_dist_dd': plot_dist_dd,
                'plot_time_dd': plot_time_dd,
                'plot_dist_nec': plot_dist_nec,
                'plot_time_nec': plot_time_nec,
                'plot_dist_infl': plot_dist_infl,
                'plot_time_infl': plot_time_infl,
                'plot_dist_kf': plot_dist_kf,
                'plot_time_kf': plot_time_kf
            },
            'stats_text': {
                'result_dd': result_dd,
                'result_nec': result_nec,
                'result_infl': result_infl,
                'result_kf': result_kf
            },
            'timestamp': datetime.now().isoformat()
        }
        # Store only the session ID in cookie (small)
        session['report_id'] = session_id

    return render_template(
    "index.html",
    model_type=model_type,
    dose=dose,
    time=time_param if 'time_param' in locals() else time,
    default_invitro_dose=defaults["invitro"]["dose"],
    default_invitro_time=defaults["invitro"]["time"],
    default_invivo_dose=defaults["invivo"]["dose"],
    default_invivo_time=defaults["invivo"]["time"],
    parameter_ranges=param_ranges,
    result_dd=result_dd, result_nec=result_nec,
    plot_dist_dd=plot_dist_dd, plot_time_dd=plot_time_dd,
    plot_dist_nec=plot_dist_nec, plot_time_nec=plot_time_nec,
    result_infl=result_infl, plot_dist_infl=plot_dist_infl, plot_time_infl=plot_time_infl,
    result_kf=result_kf, plot_dist_kf=plot_dist_kf, plot_time_kf=plot_time_kf,
    simulation_data=simulation_data
)


@app.route("/download-report")
@monitor_performance
def download_report():
    """Generate and download HTML report from stored data"""
    if 'report_id' not in session:
        return "No results available. Please run a simulation first.", 400

    report_id = session['report_id']
    if report_id not in _report_storage:
        return "Report data expired. Please run the simulation again.", 400

    from report_generator import generate_html_report

    results_data = _report_storage[report_id]
    html_content = generate_html_report(results_data)

    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model = results_data['model_type']
    filename = f'qAOP_Report_{model}_{timestamp}.html'

    return send_file(
        io.BytesIO(html_content.encode('utf-8')),
        mimetype='text/html',
        as_attachment=True,
        download_name=filename
    )


# === Run the Flask app ===
if __name__ == '__main__':
    logger.info("Starting qAOP Flask application", extra={
        'extra_fields': {
            'host': config.HOST,
            'port': config.PORT,
            'debug': config.DEBUG,
            'simulation_count': config.SIMULATION_COUNT,
            'event_type': 'app_startup'
        }
    })
    
    try:
        app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
    except Exception as e:
        log_error(e, {"event": "app_startup_failed"})
        raise
