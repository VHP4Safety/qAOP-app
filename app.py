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
draws_df = pd.read_csv('./data/draws_cisDDNEC2024-08-05_11-21-18.csv')

# === Randomly sample one parameter set from posterior draws ===
def sample_random_iteration():
    sampled_row = draws_df.sample(1).iloc[0]
    return [
        sampled_row['k_cDD'], sampled_row['d_DD'], sampled_row['k_hillDD'],
        sampled_row['k_inter'], sampled_row['maxdeath'], sampled_row['k_hillnec'],
        sampled_row['p'], sampled_row['h']
    ]

# === ODE model defining the qAOP system ===
def qAOP_model(y, t, dose, params):
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

# === Fixed model parameters (from experimental setup or calibration) ===
fixed_params = [2e6, 2005, 1.24e4, 4.518e1, 2.304e4, 2.232e4, 1e12, 2e12, 3.024e4]

# === Define initial conditions based on input dose ===
def initial_conditions(dose):
    Q_api_0 = dose * 301.1 * 1e-15 * 1e12
    Q_bas_0 = dose * 301.1 * 1e-15 * 2e12
    return [Q_api_0, Q_bas_0, 0, 0, 0, 0]

# === Run the ODE solver for a single simulation ===
def run_qAOP_model(dose, simulation_time, sampled_params):
    params = fixed_params + sampled_params
    y0 = initial_conditions(dose)
    t = np.linspace(0, simulation_time, 100)
    solution = odeint(qAOP_model, y0, t, args=(dose, params))
    return solution, t

#def parse_query(query):
#    dose_match = re.search(r'(\d+\.?\d*)\s*(μM|uM|micromolar|mM)', query, re.IGNORECASE)
#    time_match = re.search(r'(\d+\.?\d*)\s*(hours?|h|minutes?|min)', query, re.IGNORECASE)

#    dose = float(dose_match.group(1)) * (1000 if 'mm' in dose_match.group(2).lower() else 1) if dose_match else None
#    time = float(time_match.group(1)) / (60 if 'min' in time_match.group(2).lower() else 1) if time_match else None

#    target = None
#    if 'dna damage' in query.lower():
#        target = 'DNA Damage'
#    elif re.search(r'necrosis|cell death', query, re.IGNORECASE):
#        target = 'Necrosis'
    
#    return dose, time, target

# === Create histogram + KDE plot and return as base64 image ===
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
def plot_time_series(values, time_points, label, color, dose):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Optional: Plot individual traces
    for i in range(values.shape[0]):
        ax.plot(time_points, values[i], alpha=0.2, lw=0.5, color=color)

    # Mean and SD across all simulations
    mean_values = np.mean(values, axis=0)
    std_values = np.std(values, axis=0)

    # Plot mean curve in black
    ax.plot(time_points, mean_values, color="black", linewidth=2, label="Mean")

    ax.set_title(f'{label} Levels Over Time (Dose = {dose:.1f} μM)')
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
    result_dd, result_nec = None, None
    img_dist_dd, img_dist_nec = None, None
    img_time_dd, img_time_nec = None, None
    dose, time = None, None

    if request.method == "POST":
        try:
            dose = float(request.form.get("dose"))
            time = float(request.form.get("time"))
        except (TypeError, ValueError):
            result_dd = "Please enter valid numeric values for dose and time."
            return render_template("index.html", result_dd=result_dd)

        # Simulate both endpoints (250 times)
        necrosis_vals, dd_vals, t_vals = [], [], np.linspace(0, time, 100)
        for _ in range(250):
            params = sample_random_iteration()
            sol, _ = run_qAOP_model(dose, time, params)
            necrosis_vals.append(sol[:, -1])  # NEC
            dd_vals.append(sol[:, -2])        # DD

        necrosis_vals = np.array(necrosis_vals)
        dd_vals = np.array(dd_vals)
        idx = np.argmin(np.abs(t_vals - time))

        # DNA Damage result + plots
        mean_dd = dd_vals[:, idx].mean()
        std_dd = dd_vals[:, idx].std()
        result_dd = f"Predicted DNA Damage at {time:.1f} h for {dose:.1f} μM: Mean = {mean_dd:.2f}, Std = {std_dd:.2f}"
        img_dist_dd = plot_distributions(dd_vals, "DNA Damage", "red", idx)
        img_time_dd = plot_time_series(dd_vals, t_vals, "DNA Damage", "red", dose)

        # Necrosis result + plots
        mean_nec = necrosis_vals[:, idx].mean()
        std_nec = necrosis_vals[:, idx].std()
        result_nec = f"Predicted Necrosis at {time:.1f} h for {dose:.1f} μM: Mean = {mean_nec:.2f}%, Std = {std_nec:.2f}%"
        img_dist_nec = plot_distributions(necrosis_vals, "Necrosis", "blue", idx)
        img_time_nec = plot_time_series(necrosis_vals, t_vals, "Necrosis", "blue", dose)

    return render_template(
        "index.html",
        result_dd=result_dd, result_nec=result_nec,
        image_dist_dd=img_dist_dd, image_time_dd=img_time_dd,
        image_dist_nec=img_dist_nec, image_time_nec=img_time_nec
    )


# === Run the Flask app (debug mode enabled) ===
if __name__ == "__main__":
    app.run(debug=True)