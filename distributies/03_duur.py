"""
03_duur.py

Dit analyseert de duur (hoe land een wagen gereserveerd wordt).
Net als starttijd, wordt er een hybride methode gebruikt:
1. KDE: voor de continu vorm te krijgen
2. discrete snapping: Mensen reserveren niet voor "13 minuten", ze ronden meestal af naar meervouden
    van 30 min (bv 1u30).
"""

import matplotlib
matplotlib.use('Agg')
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import scipy.stats as stats
from sklearn.neighbors import KernelDensity
from data_loader import laad_basis_data

# ==============================================================================
# CONFIGURATIE
# ==============================================================================
JSON_DATA_BESTAND = 'swsim-2026-02-26.json'
CSV_MODEL_BESTAND = '03_saved_kde_buckets.csv'
OUTPUT_MAP = '03_duur_metrics'
OUTPUT_JSON_PARAMS = '03_duur_distributies.json'

NAAM_NAAR_DAG = {'Maandag': 0, 'Dinsdag': 1, 'Woensdag': 2, 'Donderdag': 3, 'Vrijdag': 4}

# Kleurenpalet voor de validatieplots
C_HIST = '#4575B4'
C_KDE = '#D73027'
C_SIM = '#1A9850'
C_QQ = '#762A83'


# ==============================================================================
# 1. HULPFUNCTIES
# ==============================================================================
def bereken_metrics(hist, sim):
    """
    Berekent 3 scores om te bewijzen hoe goed onze simulatie is:
    1. Kolmogorov-Smirnov (KS): Test of de algehele vorm van de verdelingen gelijk is.
    2. Wasserstein Distance: De 'Earth Mover's Distance'.
    3. RMSE: De Root Mean Square Error over de cumulatieve (ECDF) lijn.
    """
    ks_stat, _ = stats.ks_2samp(hist, sim)
    wass_dist = stats.wasserstein_distance(hist, sim)

    hist_sorted, sim_sorted = np.sort(hist), np.sort(sim)
    ecdf_hist = np.arange(1, len(hist) + 1) / len(hist)
    ecdf_sim = np.searchsorted(sim_sorted, hist_sorted, side='right') / len(sim_sorted)
    rmse = np.sqrt(np.mean((ecdf_hist - ecdf_sim) ** 2))

    return ks_stat, wass_dist, rmse


# ==============================================================================
# 2. GLOBALE MICRO PLOT
# ==============================================================================
def genereer_globale_micro_plot(df, uit_map):
    """
    Deze functie isoleert puur de minuten van de reserveringsduur.
    Het bewijst visueel waarom 'Snapping' belangrijk is:
        Er zijn pieken rond minuut 0 en 30 en amper data op willekeurige minuten.
    """
    print("\nGenereren van globale micro-schaal (snapping) plot...")
    if df.empty or 'duratie_uren' not in df.columns: return

    df_micro = df.copy()
    # Haal het rest-gedeelte (de minuten) uit de uren (bijv 1.5 uur -> 30 minuten)
    df_micro['minuut_rest'] = np.round(df_micro['duratie_uren'] * 60) % 60
    totaal_ritten = len(df_micro)
    minuten_counts = df_micro['minuut_rest'].value_counts().reindex(range(60), fill_value=0)

    helften = [0, 30]
    kwartieren = [15, 45]
    vijf_minuten = [m for m in range(60) if m % 5 == 0 and m not in helften and m not in kwartieren]

    aantal_30m = sum(minuten_counts[m] for m in helften)
    aantal_15m = sum(minuten_counts[m] for m in kwartieren)
    aantal_5m = sum(minuten_counts[m] for m in vijf_minuten)
    aantal_overig = totaal_ritten - aantal_30m - aantal_15m - aantal_5m

    pct_30m = (aantal_30m / totaal_ritten) * 100
    pct_15m = (aantal_15m / totaal_ritten) * 100
    pct_5m = (aantal_5m / totaal_ritten) * 100
    pct_overig = (aantal_overig / totaal_ritten) * 100

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    kleuren = [
        '#D73027' if m in helften else '#F46D43' if m in kwartieren else '#FDAE61' if m in vijf_minuten else '#4575B4'
        for m in range(60)]
    plt.bar(minuten_counts.index, minuten_counts.values, color=kleuren, width=0.8, alpha=0.9)

    titel = (f"Globale Duratie Distributie (Micro-schaal) - Alle hubs en dagen (n={totaal_ritten})\n"
             f"Halve uren: {pct_30m:.1f}% | Kwartieren: {pct_15m:.1f}% | 5-Minuten: {pct_5m:.1f}% | Overig: {pct_overig:.1f}%")
    plt.title(titel, fontsize=13, fontweight='bold')
    plt.xlabel('Overgebleven minuten van de duratie', fontsize=11)
    plt.ylabel('Aantal Reservaties', fontsize=11)
    plt.xticks(range(0, 60, 5))

    p1 = mpatches.Patch(color='#D73027', label='Meervoud van 30 min (00, 30)')
    p2 = mpatches.Patch(color='#F46D43', label='Meervoud van 15 min (15, 45)')
    p3 = mpatches.Patch(color='#FDAE61', label='Meervoud van 5 min (05, 10, 20...)')
    p4 = mpatches.Patch(color='#4575B4', label='Overig (Willekeurige minuten)')
    plt.legend(handles=[p1, p2, p3, p4], fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(uit_map, "00_globale_duratie_micro.png"), dpi=120)
    plt.close()


# ==============================================================================
# 3. HYBRIDE SIMULATIE (KDE + SNAPPING)
# ==============================================================================
def genereer_bucket_rapport(data, row, uit_map):
    """
    Traint en test het model voor één specifiek tijdsblok (bijv. Maandag 06:00 - 09:00).
    Genereert de validatiegrafiek.
    """
    dag_naam = row['Day']
    st = int(row['Start_Hour'])
    en = int(row['End_Hour'])
    bw = float(row['KDE_Bandwidth'])

    # Haal Snap kans op, gebruik 0.5 als fallback als het niet in de CSV staat
    p30 = float(row.get('P_30_Snap', 0.5))

    bucket_export_data = {
        "model_type": "Hybrid_KDE",
        "kde_bandwidth": bw,
        "p_30_snap": p30,
        "n_jobs": len(data),
        "kde_data": [round(v, 4) for v in data.tolist()]  # VOLLEDIGE dataset voor KDE
    }

    # 1. Train de wiskundige, continue KDE (Kernel Density Estimation) op de historische data
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data.reshape(-1, 1))

    # 2. Altijd eerst willekeurige samples trekken uit de vloeiende KDE lijn
    raw_sim = kde.sample(10_000).flatten()
    raw_sim = np.clip(raw_sim, 0.05, max(data) + 2)

    # 3. Beslis wiskundig of de KDE-waarde 'gesnapt' (afgerond) moet worden naar 30 minuten
    # en ronden dit in p30% van de gevallen af naar de dichtsbijzijnde half uur
    snap_mask = np.random.rand(10_000) < p30
    sim_data = np.where(snap_mask, np.round(raw_sim * 2) / 2, raw_sim)

    # Bereken hoe goed de gesimuleerde data (sim_data) de werkelijkheid (data) nabootst.
    ks, wass, rmse = bereken_metrics(data, sim_data)

    # --- Plotten ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f"Validatie {dag_naam} {st:02d}:00 - {en:02d}:59 | RMSE: {rmse * 100:.2f}% | W-Dist: {wass:.2f}u",
                 fontsize=16, fontweight='bold', y=1.02)

    bin_width = 0.5
    max_val = min(max(max(data), 24), 36)
    bins = np.arange(0, max_val + bin_width, bin_width)

    # Plot 1. Distributie & Pieken (Macro View)
    axs[0].hist(data, bins=bins, density=True, color=C_HIST, alpha=0.6, label=f'Historisch (n={len(data)})')
    x_smooth = np.linspace(0.01, max_val, 500).reshape(-1, 1)
    kde_pdf = np.exp(kde.score_samples(x_smooth))
    axs[0].plot(x_smooth.flatten(), kde_pdf, color=C_KDE, lw=2, label=f'KDE Lijn (bw={bw:.2f})')

    sim_counts, _ = np.histogram(sim_data, bins=bins, density=True)
    x_step, y_step = [], []
    for i in range(len(sim_counts)):
        x_step.extend([bins[i], bins[i + 1]])
        y_step.extend([sim_counts[i], sim_counts[i]])
    axs[0].plot(x_step, y_step, color=C_SIM, lw=2.5, label='Simulatie (KDE + Snapping)')

    axs[0].set_xlim(0, max_val)
    axs[0].set_title('Distributie & Pieken', fontweight='bold')
    axs[0].set_xlabel('Duratie (uur)')
    axs[0].set_ylabel('Dichtheid')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Plot 2. ECDF (Cumulatieve Kans & Foutmarge)
    axs[1].step(np.sort(data), np.arange(1, len(data) + 1) / len(data), color=C_HIST, lw=3, label='Historisch')
    axs[1].step(np.sort(sim_data), np.arange(1, len(sim_data) + 1) / len(sim_data), color=C_SIM, ls='--', lw=2.5,
                label='Simulatie')
    axs[1].set_xlim(0, max_val)
    axs[1].set_title('Cumulatieve Kans (Foutmarge Test)', fontweight='bold')
    axs[1].set_xlabel('Duratie (uur)')
    axs[1].set_ylabel('Kans (0-1.0)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Plot 3. Q-Q Plot (Quantile-Quantile)
    quantiles = np.linspace(0.01, 0.99, 100)
    q_data = np.quantile(data, quantiles)
    q_sim = np.quantile(sim_data, quantiles)

    lim = max(q_data.max(), q_sim.max()) * 1.05
    if lim <= 0: lim = 1.0

    axs[2].scatter(q_sim, q_data, color=C_QQ, alpha=0.6, s=25)
    axs[2].plot([0, lim], [0, lim], color='black', linestyle='--', alpha=0.5, label='Ideale Overlap')
    axs[2].set_xlim(0, lim)
    axs[2].set_ylim(0, lim)
    axs[2].set_title('Quantile-Quantile (Q-Q) Plot', fontweight='bold')
    axs[2].set_xlabel('Gesimuleerde Kwantielen (uur)')
    axs[2].set_ylabel('Historische Kwantielen (uur)')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    pad = os.path.join(uit_map, f"{st:02d}_{en:02d}h.png")
    plt.savefig(pad, dpi=120, bbox_inches='tight')
    plt.close(fig)

    return rmse, wass, ks, bucket_export_data


# ==============================================================================
# HOOFDPROGRAMMA
# ==============================================================================
def main():
    print("=" * 70)
    print("  DURATIE EVALUATIE, PLOTTING & JSON EXPORT")
    print("=" * 70)

    os.makedirs(OUTPUT_MAP, exist_ok=True)
    df = laad_basis_data(JSON_DATA_BESTAND)

    if df.empty or 'duratie_uren' not in df.columns:
        print("[FOUT] Basis data mist of duratie_uren ontbreekt.")
        return

    # De Globale Bar Chart over afrondingsgedrag
    genereer_globale_micro_plot(df, OUTPUT_MAP)

    if not os.path.exists(CSV_MODEL_BESTAND):
        print(f"\n[FOUT] CSV Bestand '{CSV_MODEL_BESTAND}' niet gevonden.")
        return

    df_csv = pd.read_csv(CSV_MODEL_BESTAND)
    rapportage = []
    json_export = {}

    print(f"Genereren van grafieken & opslaan JSON voor {len(df_csv)} bewaarde buckets...\n")

    for _, row in df_csv.iterrows():
        dag_naam = row['Day']
        wd = NAAM_NAAR_DAG.get(dag_naam, -1)
        st = int(row['Start_Hour'])
        en = int(row['End_Hour'])
        bucket_key = f"{wd}_{st}_{en}"

        dag_map = os.path.join(OUTPUT_MAP, f"{wd}_{dag_naam}")
        os.makedirs(dag_map, exist_ok=True)

        if st <= en:
            mask = (df['weekday'] == wd) & (df['start_hour'] >= st) & (df['start_hour'] <= en)
        else:
            mask = (df['weekday'] == wd) & ((df['start_hour'] >= st) | (df['start_hour'] <= en))

        data = df[mask]['duratie_uren'].values
        if len(data) < 5: continue

        rmse, wass, ks, exp_data = genereer_bucket_rapport(data, row, dag_map)
        json_export[bucket_key] = exp_data

        rapportage.append({
            'Dag': dag_naam, 'Tijd': f"{st:02d}:00-{en:02d}:59",
            'N': len(data), 'BW': row['KDE_Bandwidth'],
            'RMSE': rmse, 'W_Dist': wass, 'KS': ks
        })

    # Exporteer de parameters naar JSON voor de hoofdsimulator
    with open(OUTPUT_JSON_PARAMS, 'w', encoding='utf-8') as f:
        json.dump(json_export, f, indent=4)

    df_rap = pd.DataFrame(rapportage)
    if not df_rap.empty:
        print("★ OVERZICHT VAN DE FOUTMARGES ★")
        print("-" * 75)
        print(f"{'Dag':<10} | {'Tijd':<13} | {'Jobs':<5} | {'BW':<5} | {'RMSE':<8} | {'W-Dist':<8} | {'KS':<6}")
        print("-" * 75)
        for _, r in df_rap.iterrows():
            print(
                f"{r['Dag']:<10} | {r['Tijd']:<13} | {r['N']:<5} | {r['BW']:<5.2f} | {r['RMSE'] * 100:>6.2f}% | {r['W_Dist']:>5.2f} u | {r['KS']:.3f}")
        print("-" * 75)
        print(f"Gemiddelde RMSE over alle {len(df_rap)} buckets: {df_rap['RMSE'].mean() * 100:.2f}%")
        print(f"Gemiddelde Wasserstein Afwijking: {df_rap['W_Dist'].mean():.2f} uur")

    print(f"\n[SUCCES] Alle parameters voor de simulatie zijn opgeslagen in: '{OUTPUT_JSON_PARAMS}'")
    print(f"[SUCCES] Alle afbeeldingen zijn netjes opgeslagen in '{OUTPUT_MAP}/'")


if __name__ == '__main__':
    main()