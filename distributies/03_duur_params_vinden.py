"""
03_duur_params_vinden.py

Dit script lanceert een web-dashboard (Plotly Dash) om de optimale wiskundige
parameters (Bandwidth) te vinden voor het simuleren van de reserveringsduur.

WAAROM DIT DASHBOARD?
Omdat de duur afhankelijk is van het tijdstip op de dag (mensen die om 08:00 boeken,
houden de auto langer dan mensen die om 20:00 boeken), splitsen we de dag in tijdsblokken (buckets).
Met deze tool kunnen we visueel valideren of ons 'Hybride KDE' model de werkelijkheid per tijdsblok goed nabootst.
"""

import json
import warnings
import os
import numpy as np
import pandas as pd
from datetime import datetime
import dash
from dash import dcc, html, Input, Output, State
from sklearn.neighbors import KernelDensity
import scipy.stats as stats
import plotly.graph_objects as go


# ==========================================
# 1. CONFIGURATIE & DATA INLADEN
# ==========================================
FILEPATH = "swsim-2026-02-26.json"
TARGET_VTYPE = 4 # We focussen enkel op personenwagens
SAVE_FILE = "03_saved_kde_buckets.csv"

WEEKDAYS = ["Maandag", "Dinsdag", "Woensdag", "Donderdag", "Vrijdag"]

print(f"Loading {FILEPATH} ...")
try:
    with open(FILEPATH) as f:
        raw = json.load(f)
except FileNotFoundError:
    print(f"[FOUT] Bestand {FILEPATH} niet gevonden.")
    raw = {"jobs": []}

records = []
for job in raw.get("jobs", []):
    if job.get("vehicleTypeId") != TARGET_VTYPE: continue

    fr, to = job.get("period", {}).get("fromDate"), job.get("period", {}).get("toDate")
    if not fr or not to: continue

    try:
        dt_from = pd.to_datetime(fr, utc=True).tz_convert('Europe/Brussels')
        dt_to = pd.to_datetime(to, utc=True).tz_convert('Europe/Brussels')
    except Exception:
        continue

    wd = dt_from.weekday()
    if wd > 4: continue  # Geen weekenden

    dur = (dt_to - dt_from).total_seconds() / 3600.0
    if 0.05 < dur <= 48:
        records.append({"weekday": wd, "hour": dt_from.hour, "duration": dur})

df = pd.DataFrame(records)
print(f"Data geladen! Totaal {len(df)} ritten op werkdagen.")


# ==========================================
# 2. HULPFUNCTIES
# ==========================================
def get_filtered_data(day, start_h, end_h):
    """ Haalt data op voor een specifieke dag en tijdslot (bijv. Maandag 06:00 - 09:00). """
    if df.empty: return np.array([])
    if start_h <= end_h:
        mask = (df['weekday'] == day) & (df['hour'] >= start_h) & (df['hour'] <= end_h)
    else:
        # Handelt nachtelijke 'wraparounds' af (bijv van 22:00 tot 02:00)
        mask = (df['weekday'] == day) & ((df['hour'] >= start_h) | (df['hour'] <= end_h))
    return df[mask]['duration'].values


def bereken_metrics(hist, sim):
    """ Berekent Kolmogorov-Smirnov, Wasserstein en RMSE """
    ks_stat, _ = stats.ks_2samp(hist, sim)
    wass_dist = stats.wasserstein_distance(hist, sim)

    hist_sorted, sim_sorted = np.sort(hist), np.sort(sim)
    ecdf_hist = np.arange(1, len(hist) + 1) / len(hist)
    ecdf_sim = np.searchsorted(sim_sorted, hist_sorted, side='right') / len(sim_sorted)
    rmse = np.sqrt(np.mean((ecdf_hist - ecdf_sim) ** 2))

    return ks_stat, wass_dist, rmse


def optimaliseer_bandwidth_custom(data, p_30):
    """
    NIEUW: In plaats van GridSearchCV (die geen weet heeft van onze hybride snapping),
    testen we zelf bandwidths. We kiezen de bandwidth die de LAAGSTE RMSE oplevert
    TUSSEN de hybride simulatie en de historische data.
    """
    best_bw = 0.5
    best_rmse = float('inf')

    # We proberen 30 waardes tussen 0.05 (heel scherp) en 1.5 (heel glad)
    for bw in np.linspace(0.05, 1.5, 30):
        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data.reshape(-1, 1))
        raw_sim = kde.sample(3000).flatten()  # 3000 is genoeg voor een snelle test
        raw_sim = np.clip(raw_sim, 0.1, max(data) + 5)

        # Pas de snapping toe
        snap_mask = np.random.rand(3000) < p_30
        sim_data = np.where(snap_mask, np.round(raw_sim * 2) / 2, raw_sim)

        # Bereken de uiteindelijke fout op de ECDF
        _, _, rmse = bereken_metrics(data, sim_data)

        if rmse < best_rmse:
            best_rmse = rmse
            best_bw = bw

    return best_bw


def bereken_slimme_bandwidth(data):
    """Fallback voor de tekstuele suggestie"""
    if len(data) < 2: return 0.5
    std = np.std(data)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    a = min(std, iqr / 1.34) if iqr > 0 else std
    h_silverman = 0.9 * a * (len(data) ** -0.2)
    return max(0.1, h_silverman * 1.5)


# ==========================================
# 3. DASH APP SETUP
# ==========================================
app = dash.Dash(__name__)
app.title = "Hybrid KDE Tuner"

app.layout = html.Div(
    style={'backgroundColor': '#0f0f1a', 'color': '#e0e0e0', 'fontFamily': 'Segoe UI, sans-serif', 'minHeight': '100vh',
           'padding': '20px'},
    children=[
        # HEADER
        html.Div(style={'borderBottom': '3px solid #3498db', 'paddingBottom': '10px', 'marginBottom': '20px'},
                 children=[
                     html.H1([html.Span("📉 Hybrid KDE", style={'color': '#3498db'}), " Duration Tuner"]),
                     html.P(
                         "De 'Auto' mode optimaliseert nu op de allerlaagste RMSE, inclusief 30-min snapping effecten!",
                         style={'color': '#888'})
                 ]),

        # BEDIENINGSPANEEL
        html.Div(style={'display': 'flex', 'gap': '30px', 'backgroundColor': '#151530', 'padding': '20px',
                        'borderRadius': '8px', 'border': '1px solid #1e1e3a'}, children=[

            # Sectie 1: Tijd Selectie
            html.Div(style={'borderRight': '1px solid #2a2a4a', 'paddingRight': '30px'}, children=[
                html.H3("1. Tijd Selectie", style={'color': '#fff', 'fontSize': '14px', 'marginBottom': '10px'}),
                html.Div([
                    html.Label("Dag", style={'color': '#999', 'fontSize': '11px'}),
                    dcc.Dropdown(id='day-dropdown', options=[{'label': d, 'value': i} for i, d in enumerate(WEEKDAYS)],
                                 value=0, clearable=False, style={'color': '#000', 'marginBottom': '10px'})
                ]),
                html.Div(style={'display': 'flex', 'gap': '10px'}, children=[
                    html.Div([
                        html.Label("Start Uur", style={'color': '#999', 'fontSize': '11px'}),
                        dcc.Dropdown(id='start-hour', options=[{'label': f"{i:02d}:00", 'value': i} for i in range(24)],
                                     value=6, clearable=False, style={'color': '#000', 'width': '90px'})
                    ]),
                    html.Div([
                        html.Label("Eind Uur", style={'color': '#999', 'fontSize': '11px'}),
                        dcc.Dropdown(id='end-hour', options=[{'label': f"{i:02d}:59", 'value': i} for i in range(24)],
                                     value=9, clearable=False, style={'color': '#000', 'width': '90px'})
                    ])
                ])
            ]),

            # Sectie 2: KDE Parameters
            html.Div(style={'borderRight': '1px solid #2a2a4a', 'paddingRight': '30px', 'flexGrow': '1'}, children=[
                html.H3("2. KDE Bandwidth Settings",
                        style={'color': '#fff', 'fontSize': '14px', 'marginBottom': '10px'}),
                dcc.RadioItems(
                    id='bw-mode',
                    options=[
                        {'label': ' 🤖 Auto (RMSE Optimizer)', 'value': 'auto'},
                        {'label': ' 🛠️ Manual Override', 'value': 'manual'}
                    ],
                    value='auto',
                    inline=True,
                    style={'color': '#e94560', 'fontWeight': 'bold', 'marginBottom': '10px'}
                ),

                html.Div(id='bw-recommendation',
                         style={'color': '#f1c40f', 'fontSize': '12px', 'marginBottom': '15px', 'fontStyle': 'italic'}),

                html.Div(id='manual-bw-container', children=[
                    html.Label("Manual Bandwidth (0.05 - 2.0)", style={'color': '#999', 'fontSize': '11px'}),
                    dcc.Slider(id='bw-slider', min=0.05, max=2.0, step=0.05, value=0.15,
                               marks={i / 10: str(i / 10) for i in range(2, 21, 4)})
                ])
            ]),

            # Sectie 3: Opslaan
            html.Div(children=[
                html.H3("3. Exporteer", style={'color': '#fff', 'fontSize': '14px', 'marginBottom': '10px'}),
                html.Button("💾 Sla Bucket op", id='save-btn', n_clicks=0, style={
                    'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                    'padding': '12px 20px', 'borderRadius': '6px', 'cursor': 'pointer', 'fontWeight': 'bold'
                }),
                html.Div(id='save-status',
                         style={'color': '#2ecc71', 'marginTop': '10px', 'fontSize': '13px', 'maxWidth': '180px'})
            ])
        ]),

        # GRAFIEKEN
        dcc.Loading(
            id="loading-graphs",
            type="circle",
            color="#3498db",
            children=[
                html.Div(style={'display': 'flex', 'gap': '20px', 'marginTop': '20px'}, children=[
                    # Links: Density
                    html.Div(style={'flex': '1', 'backgroundColor': '#151530', 'borderRadius': '8px', 'padding': '10px',
                                    'border': '1px solid #1e1e3a'}, children=[
                        dcc.Graph(id='plot-density', style={'height': '55vh'})
                    ]),
                    # Rechts: ECDF & Metrics
                    html.Div(style={'flex': '1', 'backgroundColor': '#151530', 'borderRadius': '8px', 'padding': '10px',
                                    'border': '1px solid #1e1e3a'}, children=[
                        dcc.Graph(id='plot-ecdf', style={'height': '55vh'})
                    ])
                ])
            ]
        ),

        # Verborgen opslag voor de gebruikte waarden bij het opslaan
        dcc.Store(id='store-current-params')
    ])


# ==========================================
# 4. CALLBACKS (Logica)
# ==========================================

@app.callback(
    Output('manual-bw-container', 'style'),
    Input('bw-mode', 'value')
)
def toggle_mode(mode):
    if mode == 'manual': return {'display': 'block'}
    return {'opacity': '0.4', 'pointerEvents': 'none'}


@app.callback(
    [Output('plot-density', 'figure'),
     Output('plot-ecdf', 'figure'),
     Output('store-current-params', 'data'),
     Output('bw-recommendation', 'children')],
    [Input('day-dropdown', 'value'),
     Input('start-hour', 'value'),
     Input('end-hour', 'value'),
     Input('bw-mode', 'value'),
     Input('bw-slider', 'value')]
)
def update_graphs(day, start_h, end_h, bw_mode, manual_bw):
    data = get_filtered_data(day, start_h, end_h)

    fig_d = go.Figure()
    fig_e = go.Figure()

    layout_base = dict(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       margin=dict(t=50, b=40, l=40, r=20))
    fig_d.update_layout(**layout_base, xaxis_title="Duratie (Uren)", yaxis_title="Dichtheid", bargap=0)
    fig_e.update_layout(**layout_base, xaxis_title="Duratie (Uren)", yaxis_title="Cumulatieve Kans (0 - 1.0)")

    if len(data) < 10:
        msg = "Te weinig data (<10 ritten) voor deze uren."
        fig_d.update_layout(title=msg)
        fig_e.update_layout(title=msg)
        return fig_d, fig_e, None, "💡 Wacht op meer data..."

    # Snapping percentage
    p_30 = np.mean(np.isclose(data % 0.5, 0, atol=0.02))

    # Bepaal de Bandwidth (NU MET CUSTOM OPTIMIZER)
    if bw_mode == 'auto':
        used_bw = optimaliseer_bandwidth_custom(data, p_30)
    else:
        used_bw = manual_bw

    smart_bw_fallback = bereken_slimme_bandwidth(data)
    recommendation_text = f"💡 De RMSE Optimizer koos: ~{used_bw:.2f}" if bw_mode == 'auto' else f"💡 Wiskundige suggestie (Silverman mod): ~{smart_bw_fallback:.2f}"

    # Simuleer voor de uiteindelijke grafiek (Grote sample size voor mooie lijnen)
    kde = KernelDensity(kernel='gaussian', bandwidth=used_bw).fit(data.reshape(-1, 1))
    raw_sim = kde.sample(10_000).flatten()
    raw_sim = np.clip(raw_sim, 0.1, max(data) + 5)

    snap_mask = np.random.rand(10_000) < p_30
    sim_data = np.where(snap_mask, np.round(raw_sim * 2) / 2, raw_sim)

    # Bereken uiteindelijke Foutmarges
    ks, wass, rmse = bereken_metrics(data, sim_data)

    # --- PLOT 1: DENSITY ---
    bin_width = 0.5
    max_val = max(max(data), 24)
    bins = np.arange(0, max_val + bin_width, bin_width)

    # Historisch
    fig_d.add_trace(go.Histogram(x=data, xbins=dict(start=0, end=max_val, size=bin_width),
                                 histnorm='probability density', marker_color='#4575B4',
                                 opacity=0.6, name=f'Historisch (n={len(data)})'))

    # KDE Lijn
    x_smooth = np.linspace(0.01, max_val, 1000).reshape(-1, 1)
    kde_pdf = np.exp(kde.score_samples(x_smooth))
    fig_d.add_trace(go.Scatter(x=x_smooth.flatten(), y=kde_pdf, mode='lines',
                               line=dict(color='#D73027', width=2), name=f'KDE Lijn'))

    # Simulatie (Trappen plot)
    sim_counts, _ = np.histogram(sim_data, bins=bins, density=True)
    x_step, y_step = [], []
    for i in range(len(sim_counts)):
        x_step.extend([bins[i], bins[i + 1]])
        y_step.extend([sim_counts[i], sim_counts[i]])

    fig_d.add_trace(go.Scatter(x=x_step, y=y_step, mode='lines',
                               line=dict(color='#1A9850', width=2), name='Hybride Simulatie'))

    fig_d.update_layout(title=f"Distributie Fit | Bandwidth: {used_bw:.2f} | P(30min): {p_30 * 100:.1f}%",
                        xaxis_range=[0, min(max_val, 30)])

    # --- PLOT 2: ECDF ---
    h_sort, s_sort = np.sort(data), np.sort(sim_data)

    fig_e.add_trace(go.Scatter(x=h_sort, y=np.arange(1, len(data) + 1) / len(data),
                               mode='lines', line_shape='hv', line=dict(color='#4575B4', width=3), name='Historisch'))
    fig_e.add_trace(go.Scatter(x=s_sort, y=np.arange(1, len(sim_data) + 1) / len(sim_data),
                               mode='lines', line_shape='hv', line=dict(color='#1A9850', width=2, dash='dash'),
                               name='Simulatie'))

    fig_e.update_layout(title=f"Foutmarge | RMSE: {rmse * 100:.1f}% | KS: {ks:.2f} | W-Dist: {wass:.2f}u",
                        xaxis_range=[0, min(max_val, 30)])

    param_data = {"bw": float(used_bw), "p30": float(p_30), "n": len(data)}
    return fig_d, fig_e, param_data, recommendation_text


@app.callback(
    Output('save-status', 'children'),
    Input('save-btn', 'n_clicks'),
    [State('day-dropdown', 'value'), State('start-hour', 'value'), State('end-hour', 'value'),
     State('store-current-params', 'data')]
)
def save_bucket(n_clicks, day, start_h, end_h, params):
    if n_clicks == 0 or not params: return ""

    row = {
        "Day": WEEKDAYS[day],
        "Start_Hour": start_h,
        "End_Hour": end_h,
        "N_Jobs": params['n'],
        "Model_Type": "Hybrid_KDE",
        "KDE_Bandwidth": round(params['bw'], 3),
        "P_30_Snap": round(params['p30'], 3),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    df_save = pd.DataFrame([row])
    file_exists = os.path.isfile(SAVE_FILE)
    df_save.to_csv(SAVE_FILE, mode='a', index=False, header=not file_exists)

    return f"Opgeslagen! (BW: {params['bw']:.2f})"


if __name__ == '__main__':
    app.run(debug=True)