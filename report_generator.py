from datetime import datetime


def generate_html_report(results_data):
    """Generate self-contained HTML report with interactive plots"""

    model_type = results_data['model_type']
    dose = results_data['dose']
    time = results_data['time']
    dose_unit = results_data['dose_unit']
    plots = results_data['plots']
    stats = results_data['stats_text']

    # Model descriptions
    model_desc = {
        'invitro': 'RPTEC/TERT1 cell culture model (6 compartments)',
        'invivo': 'Rat kidney model (8 compartments)'
    }

    # Generate HTML with embedded Plotly plots
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>qAOP Report - {model_type.title()} Model</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        .plot-container {{ margin: 30px 0; }}
        .footer {{ margin-top: 50px; padding-top: 20px; border-top: 2px solid #ddd; color: #7f8c8d; }}
    </style>
</head>
<body>
    <h1>qAOP Prediction Report</h1>

    <h2>Input Parameters</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Model Type</td><td>{model_type.title()}</td></tr>
        <tr><td>Model Description</td><td>{model_desc[model_type]}</td></tr>
        <tr><td>Dose</td><td>{dose} {dose_unit}</td></tr>
        <tr><td>Time Point</td><td>{time} hours</td></tr>
        <tr><td>Monte Carlo Iterations</td><td>{results_data['simulation_count']}</td></tr>
    </table>

    <h2>Results Summary</h2>
    '''

    # Add plots and stats for each endpoint
    endpoints = [
        ('DNA Damage', 'dd', plots.get('plot_dist_dd'), plots.get('plot_time_dd'), stats.get('result_dd')),
        ('Necrosis', 'nec', plots.get('plot_dist_nec'), plots.get('plot_time_nec'), stats.get('result_nec'))
    ]

    if model_type == 'invivo':
        endpoints += [
            ('Inflammation', 'infl', plots.get('plot_dist_infl'), plots.get('plot_time_infl'), stats.get('result_infl')),
            ('Kidney Failure', 'kf', plots.get('plot_dist_kf'), plots.get('plot_time_kf'), stats.get('result_kf'))
        ]

    for name, code, dist_plot, time_plot, stat_text in endpoints:
        if dist_plot and time_plot:
            html += f'''
    <h3>{name}</h3>
    <p><strong>{stat_text}</strong></p>

    <div class="plot-container">
        <h4>Time Course</h4>
        <div id="plot-time-{code}"></div>
    </div>

    <div class="plot-container">
        <h4>Distribution at Selected Time</h4>
        <div id="plot-dist-{code}"></div>
    </div>
    '''

    # Add Plotly rendering script
    html += '''
    <div class="footer">
        <p>Generated: ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
        <p>qAOP Predictor - Cisplatin-Induced Nephrotoxicity Model</p>
        <p><a href="https://github.com/VHP4Safety/qAOP-app">GitHub Repository</a></p>
    </div>

    <script>
    const plotConfig = {
        displayModeBar: true,
        displaylogo: false,
        responsive: true
    };
    '''

    # Render each plot
    for name, code, dist_plot, time_plot, stat_text in endpoints:
        if time_plot:
            html += f'''
    var plotDataTime{code.title()} = {time_plot};
    Plotly.newPlot('plot-time-{code}', plotDataTime{code.title()}.data, plotDataTime{code.title()}.layout, plotConfig);
    '''
        if dist_plot:
            html += f'''
    var plotDataDist{code.title()} = {dist_plot};
    Plotly.newPlot('plot-dist-{code}', plotDataDist{code.title()}.data, plotDataDist{code.title()}.layout, plotConfig);
    '''

    html += '''
    </script>
</body>
</html>'''

    return html
