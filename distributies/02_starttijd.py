"""
02_starttijd.py

Annalyseert de historische starttijden van de ritten.
Omdat starttijden in de dag meerdere pieken hebben, (bv ochtend en avond),
is een standaart model (zoals een normaal verdeling) niet geschikt.
Daarom wordt er KDE (kernel density estimation) gebruikt om de complexe vormen na te bootsen.

Er wordt ook een hybride methode toegepast: Het combineert de continu KDE lijn met het menselijk gedrag.
Mensen boeken een rit op het halve uur (bv 8u en 8u30) dan op willekeurige minuten (8u12)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from data_loader import laad_basis_data, VASTE_HUBS, DAG_NAAR_NAAM

# ═══════════════════════════════════════════════════════════════
# CONFIGURATIE
# ═══════════════════════════════════════════════════════════════
JSON_DATA_BESTAND = 'swsim-2026-02-26.json'
JSON_MODEL_BESTAND = '02_starttijd_models_params.json'
OUTPUT_MAP = '02_starttijd_metrics'

# Kleurenpalet voor de grafieken
C_HIST, C_META, C_METB = '#4575B4', '#D73027', '#1A9850'


# ═══════════════════════════════════════════════════════════════
# 1. MODELLEN TRAINEN EN OPSLAAN (OF INLADEN)
# ═══════════════════════════════════════════════════════════════
def train_of_laad_modellen(df):
    """
    Berekent de parameters voor het starttijd model per locatie per dag.
    Het slaat de 'Kans op afronden' (p_half_hour) en de ruwe tijden voor de KDE op in een JSON.
    Als deze JSON al bestaat, wordt deze direct ingeladen om tijd te besparen.
    """
    if os.path.exists(JSON_MODEL_BESTAND):
        print(f"[1/3] Modellen razendsnel ingeladen vanuit {JSON_MODEL_BESTAND}...")
        with open(JSON_MODEL_BESTAND, 'r') as f:
            return json.load(f)

    print(f"[1/3] Modellen trainen en opslaan in {JSON_MODEL_BESTAND}...")
    modellen = {}

    for wd in range(5):
        modellen[str(wd)] = {}
        for hub in VASTE_HUBS:
            df_sub = df[(df['hub'] == hub) & (df['weekday'] == wd)]
            if len(df_sub) < 10: continue # Te weinig data om een betrouwbaar model te maken

            # Bereken Kans op 'Gesnapte' Ritten (exact op minuut 00 of 30)
            # Dit is de parameter die ons vertelt hoe "menselijk/afgerond" het gedrag op deze locatie is.
            p_half = df_sub['minute'].isin([0, 30]).mean()

            # Voor een KDE slaan we de historische tijden (in decimalen, bv 8.5 voor 08:30) op.
            kde_data = df_sub['time_float'].values.tolist()

            modellen[str(wd)][hub] = {
                'p_half_hour': float(p_half),
                'kde_data': kde_data
            }

    # Sla het getrainde gedrag op
    with open(JSON_MODEL_BESTAND, 'w') as f:
        json.dump(modellen, f)

    return modellen


# ═══════════════════════════════════════════════════════════════
# 2. SIMULATIE METHODES & METRICS
# ═══════════════════════════════════════════════════════════════
def simuleer_tijden(kde, num_samples, p_half_hour=None):
    """
    Trek willekeurige starttijden uit het getrainde KDE model.
    - Als p_half_hour None is -> Puur KDE
    - Als p_half_hour gegeven is -> Hybride Model (Trekt tijden uit KDE, maar rond
        een specifiek percentage daarvan af op halve uren, gebaseerd op historisch gedrag).
    """

    # Stap 1: Trek rauwe samples uit de continue KDE.
    # Modulo 24 zorgt ervoor dat tijden net na middernacht (bijv. 24.5) netjes 00:30 worden.
    samples = kde.resample(num_samples)[0] % 24.0

    if p_half_hour is None:
        return samples

    # Stap 2: Hybride Snapping Logica (Methode B)
    # Bepaal voor elke gegenereerde tijd of deze wordt afgerond
    is_gepland = np.random.rand(num_samples) < p_half_hour

    # Rond de tijd af naar het dichtstbijzijnde half uur (bijv. 8.12 -> 8.0)
    gesnapt = np.round(samples * 2) / 2

    # Combineer de afgeronde tijden met de niet-afgeronde (exacte) tijden
    return np.where(is_gepland, gesnapt, samples) % 24.0


def scoor_model(hist_min, sim_min):
    """
    Berekent 3 wiskundige scores om te bewijzen hoe goed onze simulatie de werkelijkheid volgt.
    """

    # 1. Kolmogorov-Smirnov (KS): Test of de vorm van de twee verdelingen statistisch gelijk is.
    ks, _ = stats.ks_2samp(hist_min, sim_min)

    # 2. Wasserstein Distance: Bekend als de 'Earth Mover's Distance'.
    w = stats.wasserstein_distance(hist_min, sim_min)

    # 3. RMSE (Root Mean Square Error): De algemene foutmarge berekend over de cumulatieve lijn.
    hist_sort, sim_sort = np.sort(hist_min), np.sort(sim_min)
    ecdf_hist = np.arange(1, len(hist_sort) + 1) / len(hist_sort)
    idx = np.searchsorted(sim_sort, hist_sort, side='right')
    rmse = np.sqrt(np.mean((ecdf_hist - (idx / len(sim_sort))) ** 2))

    return ks, w, rmse


# ═══════════════════════════════════════════════════════════════
# 3. HOOFDANALYSE & PLOTTING
# ═══════════════════════════════════════════════════════════════
def evalueer_en_plot(df, modellen):
    """
     Deze functie laat het Pure KDE model vergelijken tegen de Hybride Model.
     Het genereert de 3-panel validatieplots die in de presentatie worden gebruikt.
     """
    print("[2/3] Simulatoren laten vechten, metrics berekenen en plotten...")
    os.makedirs(OUTPUT_MAP, exist_ok=True)
    rapportage = []

    for wd in range(5):
        dag_naam = DAG_NAAR_NAAM[wd]
        for hub in VASTE_HUBS:
            if hub not in modellen[str(wd)]: continue

            # --- Setup Data ---
            hub_dir = os.path.join(OUTPUT_MAP, hub)
            os.makedirs(hub_dir, exist_ok=True)

            model_info = modellen[str(wd)][hub]
            hist_times = df[(df['hub'] == hub) & (df['weekday'] == wd)]['time_float'].values

            # Train de KDE met een vaste bandwidth (0.1 zorgt voor een mooie gladde lijn zonder overfitting)
            kde = stats.gaussian_kde(model_info['kde_data'], bw_method=0.1)
            p_half = model_info['p_half_hour']

            # --- Simuleren (10.000 samples voor gladde, betrouwbare lijnen) ---
            sim_A_times = simuleer_tijden(kde, 10000, p_half_hour=None)
            sim_B_times = simuleer_tijden(kde, 10000, p_half_hour=p_half)

            # Hulpfunctie om alleen de minuten uit de tijden te halen (bijv 08:45 -> 45)
            def ext_min(t_array):
                return np.round((t_array % 1) * 60) % 60

            hist_m, sim_A_m, sim_B_m = ext_min(hist_times), ext_min(sim_A_times), ext_min(sim_B_times)

            # --- Scoren ---
            ks_A, wass_A, rmse_A = scoor_model(hist_m, sim_A_m)
            ks_B, wass_B, rmse_B = scoor_model(hist_m, sim_B_m)

            rapportage.append({
                'Hub': hub, 'Dag': dag_naam, 'P_Half': p_half,
                'A_KS': ks_A, 'A_W': wass_A, 'A_R': rmse_A,
                'B_KS': ks_B, 'B_W': wass_B, 'B_R': rmse_B
            })

            # --- Plotten Dashboard ---
            fig, axs = plt.subplots(1, 3, figsize=(22, 6))
            fig.suptitle(f"{hub} op {dag_naam} - Starttijd Evaluatie", fontsize=16, fontweight='bold')

            # Plot 1: Macro Plot (Kijkt naar de verdeling over de 24 uren van de dag)
            axs[0].hist(hist_times, bins=48, density=True, color=C_HIST, alpha=0.5, label='Historisch')
            sns.kdeplot(sim_A_times, ax=axs[0], color=C_META, label='Methode A (KDE)', ls='--')
            sns.kdeplot(sim_B_times, ax=axs[0], color=C_METB, label='Methode B (Hybride)')
            axs[0].set(title="1. Macro: Uren van de dag", xlabel="Uur", ylabel="Dichtheid", xlim=(0, 24))
            axs[0].legend()

            # Plot 2: Micro Plot (Kijkt specifiek naar de 60 minuten in een uur)
            bins = np.arange(-0.5, 60.5, 1)
            axs[1].hist(hist_m, bins=bins, density=True, color=C_HIST, alpha=0.5, label='Historisch')
            axs[1].hist(sim_A_m, bins=bins, density=True, color=C_META, histtype='step', lw=2, label='Methode A')
            axs[1].hist(sim_B_m, bins=bins, density=True, color=C_METB, histtype='step', lw=3, label='Methode B')
            axs[1].set(title=f"2. Micro: Minuut (0-59)\nP_Halve_Uur: {p_half * 100:.1f}%", xlabel="Minuut",
                       xlim=(-1, 60))
            axs[1].legend()

            # Plot 3: Cumulatieve Metrics (Visuele representatie van de foutmarge)
            axs[2].step(np.sort(hist_m), np.linspace(0, 1, len(hist_m)), color=C_HIST, lw=4, label='Historisch')
            axs[2].step(np.sort(sim_A_m), np.linspace(0, 1, len(sim_A_m)), color=C_META, ls='--', lw=2,
                        label=f'Meth. A (W: {wass_A:.1f}, R: {rmse_A * 100:.1f}%)')
            axs[2].step(np.sort(sim_B_m), np.linspace(0, 1, len(sim_B_m)), color=C_METB, ls=':', lw=3,
                        label=f'Meth. B (W: {wass_B:.1f}, R: {rmse_B * 100:.1f}%)')
            axs[2].set(title="3. Micro Cumulatief (Scoring)", xlabel="Minuut", ylabel="Proportie")
            axs[2].legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(hub_dir, f"{dag_naam}_metric_check.png"), dpi=120)
            plt.close()

    return pd.DataFrame(rapportage)


    # ═══════════════════════════════════════════════════════════════
    # HOOFDPROGRAMMA
    # ═══════════════════════════════════════════════════════════════


def main():
    print("=" * 100)
    df = laad_basis_data(JSON_DATA_BESTAND)
    if df is None or df.empty: return

    modellen = train_of_laad_modellen(df)
    df_rap = evalueer_en_plot(df, modellen)

    print("\n[3/3] Cijfers:")
    print("=" * 105)
    print(f"{'Locatie':<8} | {'Dag':<9} | {'P(Halve Uren)':<14} | {'METH. A (KDE)':<23} | {'METH. B (HYBRIDE)':<25}")
    print("-" * 105)

    for _, row in df_rap.iterrows():
        str_A = f"KS:{row['A_KS']:.3f} W:{row['A_W']:>4.1f} R:{row['A_R'] * 100:>4.1f}%"
        str_B = f"KS:{row['B_KS']:.3f} W:{row['B_W']:>4.1f} R:{row['B_R'] * 100:>4.1f}%"
        print(f"{row['Hub']:<8} | {row['Dag']:<9} | {row['P_Half'] * 100:>13.1f}% | {str_A:<23} | {str_B:<25}")

    print("=" * 105)
    print(f"\n[OK] Dashboard plots en scores opgeslagen in: {OUTPUT_MAP}/")


if __name__ == '__main__':
    main()
