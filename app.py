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

app = Flask(__name__)

# Load parameter estimates
draws_df = pd.read_csv('./data/draws_cisDDNEC2024-08-05_11-21-18.csv')

def sample_random_iteration():
    sampled_row = draws_df.sample(1).iloc[0]
    return [
        sampled_row['k_cDD'], sampled_row['d_DD'], sampled_row['k_hillDD'],
        sampled_row['k_inter'], sampled_row['maxdeath'], sampled_row['k_hillnec'],
        sampled_row['p'], sampled_row['h']
    ]

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

fixed_params = [2e6, 2005, 1.24e4, 4.518e1, 2.304e4, 2.232e4, 1e12, 2e12, 3.024e4]

def initial_conditions(dose):
    Q_api_0 = dose * 301.1 * 1e-15 * 1e12
    Q_bas_0 = dose * 301.1 * 1e-15 * 2e12
    return [Q_api_0, Q_bas_0, 0, 0, 0, 0]

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

@app.route("/", methods=["GET", "POST"])
def index():
    result, img_data = None, None
    dose, time, target = None, None, None

    if request.method == "POST":
        try:
            dose = float(request.form.get("dose"))
            time = float(request.form.get("time"))
            target = request.form.get("target")
        except (TypeError, ValueError):
            result = "Please enter valid numeric values for dose and time."
            return render_template("index.html", result=result)

        if target not in ["DNA Damage", "Necrosis"]:
            result = "Please select a valid target variable."
            return render_template("index.html", result=result)

        necrosis_vals, dd_vals, t_vals = [], [], np.linspace(0, time, 100)
        for _ in range(250):
            params = sample_random_iteration()
            sol, _ = run_qAOP_model(dose, time, params)
            necrosis_vals.append(sol[:, -1])
            dd_vals.append(sol[:, -2])

        necrosis_vals = np.array(necrosis_vals)
        dd_vals = np.array(dd_vals)

        idx = np.argmin(np.abs(t_vals - time))
        if target == "DNA Damage":
            mean_val = dd_vals[:, idx].mean()
            std_val = dd_vals[:, idx].std()
            result = f"Predicted DNA Damage at {time:.1f} h for {dose:.1f} μM: Mean = {mean_val:.2f}, Std = {std_val:.2f}"
            img_data = plot_distributions(dd_vals, "DNA Damage", "red", idx)
        elif target == "Necrosis":
            mean_val = necrosis_vals[:, idx].mean()
            std_val = necrosis_vals[:, idx].std()
            result = f"Predicted Necrosis at {time:.1f} h for {dose:.1f} μM: Mean = {mean_val:.2f}%, Std = {std_val:.2f}%"
            img_data = plot_distributions(necrosis_vals, "Necrosis", "blue", idx)

    return render_template("index.html", result=result, image=img_data)


if __name__ == "__main__":
    app.run(debug=True)