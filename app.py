# === Import required packages ===
from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.integrate import odeint
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
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

# === Initialize logging ===
setup_logging()
logger = logging.getLogger("qAOP.app")

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
fixed_params_invitro = [2e6, 2005, 1.24e4, 4.518e1, 2.304e4, 2.232e4, 1e12, 2e12, 3.024e4]
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

def plot_distributions(values, label, color, time_index):
    fig, ax = plt.subplots()
    ax.hist(values[:, time_index], bins=30, density=True, alpha=0.6, color=color, edgecolor='black')
    kde = gaussian_kde(values[:, time_index])
    x_vals = np.linspace(values[:, time_index].min(), values[:, time_index].max(), 300)
    ax.plot(x_vals, kde(x_vals), color='black')
    ax.set_title(f'Distribution of {label}')
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode('utf-8')
    plt.close(fig)
    return plot_data

# === Create time series plot and return as base64 image ===
def plot_time_series(values, time_points, label, color, dose, unit):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Optional: Plot individual traces
    for i in range(values.shape[0]):
        ax.plot(time_points, values[i], alpha=0.2, lw=0.5, color=color)

    # Mean and SD across all simulations
    mean_values = np.mean(values, axis=0)
    std_values = np.std(values, axis=0)

    # Plot mean curve in black
    ax.plot(time_points, mean_values, color="black", linewidth=2, label="Mean")

    ax.set_title(f'{label} Levels Over Time (Dose = {dose:.1f} {unit})')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel(f'{label} Levels' if label != "Necrosis" else 'Necrosis (%)')
    ax.grid(True)
    ax.legend()

    # Save and return base64 image
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_data = base64.b64encode(img.read()).decode('utf-8')
    plt.close(fig)
    return plot_data


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
                        "percentile_25": float(np.percentile(dd_vals[:, idx], 25)),
                        "percentile_75": float(np.percentile(dd_vals[:, idx], 75))
                    },
                    "time_series": {
                        "time_points": t_vals.tolist(),
                        "mean": np.mean(dd_vals, axis=0).tolist(),
                        "std": np.std(dd_vals, axis=0).tolist()
                    }
                },
                "necrosis": {
                    "final_value": {
                        "mean": float(np.mean(necrosis_vals[:, idx])),
                        "std": float(np.std(necrosis_vals[:, idx])),
                        "median": float(np.median(necrosis_vals[:, idx])),
                        "min": float(np.min(necrosis_vals[:, idx])),
                        "max": float(np.max(necrosis_vals[:, idx])),
                        "percentile_25": float(np.percentile(necrosis_vals[:, idx], 25)),
                        "percentile_75": float(np.percentile(necrosis_vals[:, idx], 75))
                    },
                    "time_series": {
                        "time_points": t_vals.tolist(),
                        "mean": np.mean(necrosis_vals, axis=0).tolist(),
                        "std": np.std(necrosis_vals, axis=0).tolist()
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
                    "percentile_25": float(np.percentile(inflammation_vals[:, idx], 25)),
                    "percentile_75": float(np.percentile(inflammation_vals[:, idx], 75))
                },
                "time_series": {
                    "time_points": t_vals.tolist(),
                    "mean": np.mean(inflammation_vals, axis=0).tolist(),
                    "std": np.std(inflammation_vals, axis=0).tolist()
                }
            }
            
            response["endpoints"]["kidney_failure"] = {
                "final_value": {
                    "mean": float(np.mean(kf_vals[:, idx])),
                    "std": float(np.std(kf_vals[:, idx])),
                    "median": float(np.median(kf_vals[:, idx])),
                    "min": float(np.min(kf_vals[:, idx])),
                    "max": float(np.max(kf_vals[:, idx])),
                    "percentile_25": float(np.percentile(kf_vals[:, idx], 25)),
                    "percentile_75": float(np.percentile(kf_vals[:, idx], 75))
                },
                "time_series": {
                    "time_points": t_vals.tolist(),
                    "mean": np.mean(kf_vals, axis=0).tolist(),
                    "std": np.std(kf_vals, axis=0).tolist()
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
    img_dist_dd = img_dist_nec = img_dist_infl = img_dist_kf = None
    img_time_dd = img_time_nec = img_time_infl = img_time_kf = None

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
        img_dist_dd = plot_distributions(dd_vals, "DNA Damage", "red", idx)
        img_time_dd = plot_time_series(dd_vals, t_vals, "DNA Damage", "red", dose, unit)

        # Necrosis
        mean_nec = necrosis_vals[:, idx].mean()
        std_nec = necrosis_vals[:, idx].std()
        result_nec = f"Predicted Necrosis at {time_param:.1f} h: Mean = {mean_nec:.2f}%, Std = {std_nec:.2f}%"
        img_dist_nec = plot_distributions(necrosis_vals, "Necrosis", "blue", idx)
        img_time_nec = plot_time_series(necrosis_vals, t_vals, "Necrosis", "blue", dose, unit)

        # If in vivo, also plot Inflammation and Kidney Failure
        if model_type == "invivo":
            inflammation_vals = np.array(inflammation_vals)
            kf_vals = np.array(kf_vals)

            mean_infl = inflammation_vals[:, idx].mean()
            std_infl = inflammation_vals[:, idx].std()
            result_infl = f"Predicted Inflammation at {time_param:.1f} h: Mean = {mean_infl:.2f}, Std = {std_infl:.2f}"
            img_dist_infl = plot_distributions(inflammation_vals, "Inflammation", "green", idx)
            img_time_infl = plot_time_series(inflammation_vals, t_vals, "Inflammation", "green", dose, unit)

            mean_kf = kf_vals[:, idx].mean()
            std_kf = kf_vals[:, idx].std()
            result_kf = f"Predicted Kidney Failure at {time_param:.1f} h: Mean = {mean_kf:.2f}, Std = {std_kf:.2f}"
            img_dist_kf = plot_distributions(kf_vals, "Kidney Failure", "purple", idx)
            img_time_kf = plot_time_series(kf_vals, t_vals, "Kidney Failure", "purple", dose, unit)
    
    defaults = {
    "invitro": {"dose": 50, "time": 72},
    "invivo":  {"dose": 5,  "time": 672}
    }

    # Get parameter ranges for the frontend
    param_ranges = get_parameter_ranges()
    
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
    image_dist_dd=img_dist_dd, image_time_dd=img_time_dd,
    image_dist_nec=img_dist_nec, image_time_nec=img_time_nec,
    result_infl=result_infl, image_dist_infl=img_dist_infl, image_time_infl=img_time_infl,
    result_kf=result_kf, image_dist_kf=img_dist_kf, image_time_kf=img_time_kf
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
