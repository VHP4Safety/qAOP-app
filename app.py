# === Import required packages ===
from flask import Flask, render_template, request
import numpy as np
from scipy.integrate import odeint
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import base64
import re

# === Initialize the Flask app ===
app = Flask(__name__)

# === Load pre-calibrated parameter samples from file ===
draws_invitro_df = pd.read_csv('./data/draws_cisDDNEC2024-08-05_11-21-18.csv')
draws_invivo_df = pd.read_csv('./data/draws_cisvivo2024-08-07_10-34-00.csv')
pk_pars = (
    pd.read_csv('./data/fit_pk_summary.csv')  # no index_col here
      .set_index('variable')                  # now you can .loc['Plasma0']
)

# === Randomly sample one parameter set from posterior draws ===
def sample_random_iteration(model_type="invitro"):
    if model_type == "invitro":
        sampled_row = draws_invitro_df.sample(1).iloc[0]
        return [
            sampled_row['k_cDD'], sampled_row['d_DD'], sampled_row['k_hillDD'],
            sampled_row['k_inter'], sampled_row['maxdeath'], sampled_row['k_hillnec'],
            sampled_row['p'], sampled_row['h']
        ]
    elif model_type == "invivo":
        sampled_row = draws_invivo_df.sample(1).iloc[0]
        return [
            sampled_row['k_kidDD'], sampled_row['hillDD'], sampled_row['d_DD'], sampled_row['maxdeath'], 
            sampled_row['h1'], sampled_row['k_hillnec'], sampled_row['d_CD'], sampled_row['k_CDINF'], 
            sampled_row['d_INF'], sampled_row['k_INFKF'], sampled_row['hillKF'], sampled_row['INF0']   
        ]
    else:
        raise ValueError("Unknown model type. Choose 'invitro' or 'invivo'.")


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
def run_invitro_model(dose, simulation_time, sampled_params):
    params = fixed_params_invitro + sampled_params
    y0 = initial_conditions_invitro(dose)
    t = np.linspace(0, simulation_time, 100)
    solution = odeint(invitro_model, y0, t, args=(dose, params))
    return solution, t

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

def run_invivo_model(dose, simulation_time, sampled_params):
    params = fixed_params_invivo + sampled_params[:-1]  # last param is INF0
    INF0 = sampled_params[-1]                           # INF0 is last in list
    y0 = initial_conditions_invivo(dose, INF0)
    t = np.linspace(0, simulation_time, 100)
    solution = odeint(invivo_model, y0, t, args=(dose, params))
    return solution, t

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
    return base64.b64encode(img.read()).decode('utf-8')

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
    return base64.b64encode(img.read()).decode('utf-8')


# === Flask route: homepage + form processing ===
@app.route("/", methods=["GET", "POST"])
def index():
    model_type = "invitro"
    dose = None
    time = None

    # Initialize results
    result_dd = result_nec = result_infl = result_kf = None
    img_dist_dd = img_dist_nec = img_dist_infl = img_dist_kf = None
    img_time_dd = img_time_nec = img_time_infl = img_time_kf = None
    dose = time = None

    if request.method == "POST":
        model_type = request.form.get("model", "invitro")  # Capture model type
        try:
            dose_input = float(request.form.get("dose"))
            time = float(request.form.get("time"))
        except (TypeError, ValueError):
            result_dd = "Please enter valid numeric values for dose and time."
            return render_template("index.html", result_dd=result_dd)

        dose = dose_input  # Use as-is; unit depends on model
        t_vals = np.linspace(0, time, 100)
        unit = "μM" if model_type == "invitro" else "mg/kg"

        dd_vals, necrosis_vals = [], []
        inflammation_vals, kf_vals = [], []  # Only filled if in vivo

        for _ in range(250):
            params = sample_random_iteration(model_type)
            if model_type == "invitro":
                sol, _ = run_invitro_model(dose, time, params)
                dd_vals.append(sol[:, -2])      # DNA Damage
                necrosis_vals.append(sol[:, -1])# Necrosis
            elif model_type == "invivo":
                sol, _ = run_invivo_model(dose, time, params)
                dd_vals.append(sol[:, 3])       # DNA Damage (DD)
                necrosis_vals.append(sol[:, 4]) # Cell Death (CD)
                inflammation_vals.append(sol[:, 5])  # Inflammation (INF)
                kf_vals.append(sol[:, 6])            # Kidney Failure (KF)

        # Convert to arrays
        dd_vals = np.array(dd_vals)
        necrosis_vals = np.array(necrosis_vals)
        idx = np.argmin(np.abs(t_vals - time))

        # DNA Damage
        mean_dd = dd_vals[:, idx].mean()
        std_dd = dd_vals[:, idx].std()
        result_dd = f"Predicted DNA Damage at {time:.1f} h: Mean = {mean_dd:.2f}, Std = {std_dd:.2f}"
        img_dist_dd = plot_distributions(dd_vals, "DNA Damage", "red", idx)
        img_time_dd = plot_time_series(dd_vals, t_vals, "DNA Damage", "red", dose, unit)

        # Necrosis
        mean_nec = necrosis_vals[:, idx].mean()
        std_nec = necrosis_vals[:, idx].std()
        result_nec = f"Predicted Necrosis at {time:.1f} h: Mean = {mean_nec:.2f}%, Std = {std_nec:.2f}%"
        img_dist_nec = plot_distributions(necrosis_vals, "Necrosis", "blue", idx)
        img_time_nec = plot_time_series(necrosis_vals, t_vals, "Necrosis", "blue", dose, unit)

        # If in vivo, also plot Inflammation and Kidney Failure
        if model_type == "invivo":
            inflammation_vals = np.array(inflammation_vals)
            kf_vals = np.array(kf_vals)

            mean_infl = inflammation_vals[:, idx].mean()
            std_infl = inflammation_vals[:, idx].std()
            result_infl = f"Predicted Inflammation at {time:.1f} h: Mean = {mean_infl:.2f}, Std = {std_infl:.2f}"
            img_dist_infl = plot_distributions(inflammation_vals, "Inflammation", "green", idx)
            img_time_infl = plot_time_series(inflammation_vals, t_vals, "Inflammation", "green", dose, unit)

            mean_kf = kf_vals[:, idx].mean()
            std_kf = kf_vals[:, idx].std()
            result_kf = f"Predicted Kidney Failure at {time:.1f} h: Mean = {mean_kf:.2f}, Std = {std_kf:.2f}"
            img_dist_kf = plot_distributions(kf_vals, "Kidney Failure", "purple", idx)
            img_time_kf = plot_time_series(kf_vals, t_vals, "Kidney Failure", "purple", dose, unit)

    return render_template(
    "index.html",
    model_type=model_type,
    dose=dose,
    time=time,
    result_dd=result_dd, result_nec=result_nec,
    image_dist_dd=img_dist_dd, image_time_dd=img_time_dd,
    image_dist_nec=img_dist_nec, image_time_nec=img_time_nec,
    result_infl=result_infl, image_dist_infl=img_dist_infl, image_time_infl=img_time_infl,
    result_kf=result_kf, image_dist_kf=img_dist_kf, image_time_kf=img_time_kf
)



# === Run the Flask app (debug mode enabled) ===
if __name__ == "__main__":
    app.run(debug=True)