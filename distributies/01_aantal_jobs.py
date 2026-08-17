"""
01_aantal_jobs.py

Deze scipt analyseert het dagelijkse gereserveerde ritten per dag.
Omdat het volume sterk schommelt, worden er per dag en per locatie onafhankelijke discrete kansverdeling gefit.

Hierin wordt ook de ruimtelijke correlatie tussen locaties via een soort "anker hub" bepaalt.
Hiermee kunnen feestdagen realistisch over de hele stad plaats vinden.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from data_loader import laad_basis_data

# ═══════════════════════════════════════════════════════════════
# CONFIGURATIE & GLOBALS
# ═══════════════════════════════════════════════════════════════
JSON_DATA_BESTAND = 'swsim-2026-02-26.json'
JSON_MODEL_BESTAND = '01_aantal_jobs_models_params.json'
OUTPUT_MAP = '01_aantal_jobs'
os.makedirs(OUTPUT_MAP, exist_ok=True)

DAG_NAAR_NAAM = {0: 'Maandag', 1: 'Dinsdag', 2: 'Woensdag', 3: 'Donderdag', 4: 'Vrijdag'}
C_HIST = '#4575B4'  # Blauw voor Historisch
C_SIM = '#762A83'   # Paars voor Simulatie
C_THEO = '#D73027'  # Rood voor Theoretisch Model

VASTE_HUBS = ["Punt_1", "Punt_2", "Punt_3", "Punt_4"]
ANKER_HUB = "Punt_1"



# ═══════════════════════════════════════════════════════════════
# 1. DATA INLADEN
# ═══════════════════════════════════════════════════════════════
def laad_data(json_path):
    """
    Laad ruwe data, filter het op de juiste hubs.
    Het maakt een soort kalender. Dagen waarop er geen ritten zijn gereserveerd, worden als '0' is de dataset gemarkeerd.
    """
    print("[1/5] Historische data inladen (via data_loader)...")

    # 1. Haal de data op
    df = laad_basis_data(json_path)

    if df is None or df.empty:
        return None

    # 2. Groepeer per datum en hub om het totaal aantal ritten per dag te tellen
    dag_tellingen = df.groupby(['date', 'weekday', 'hub']).size().reset_index(name='aantal_ritten')
    dag_tellingen['date'] = pd.to_datetime(dag_tellingen['date'])

    # 3. Maak een complete kalender.
    alle_werkdagen = pd.date_range(dag_tellingen['date'].min(), dag_tellingen['date'].max(), freq='B')
    idx = pd.MultiIndex.from_product([alle_werkdagen, VASTE_HUBS], names=['date', 'hub'])
    df_volledig = pd.DataFrame(index=idx).reset_index()
    df_volledig['weekday'] = df_volledig['date'].dt.weekday

    # 4. Voeg de tellingen samen met de kalender en vul de ontbrekende dagen met 0
    df_eind = pd.merge(df_volledig, dag_tellingen, on=['date', 'weekday', 'hub'], how='left').fillna(
        {'aantal_ritten': 0})
    df_eind['aantal_ritten'] = df_eind['aantal_ritten'].astype(int)

    return df_eind


# ═══════════════════════════════════════════════════════════════
# 2. MODELLEN FITTEN (Met RMSE)
# ═══════════════════════════════════════════════════════════════
def bereken_rmse(counts, cdf_func):
    """ Berekent de Root Mean Square Error tussen de werkelijkheid en de CDF van het model. """
    x_sort = np.sort(counts)
    empirisch_cdf = np.arange(1, len(counts) + 1) / len(counts)
    theoretisch_cdf = cdf_func(x_sort)
    return np.sqrt(np.mean((empirisch_cdf - theoretisch_cdf) ** 2))


def fit_modellen_voor_data(counts):
    """
    Fit meerdere discrete kansverdelingen op historische data met behulp van
    Maximum Likelihood Estimation (MLE). Selecteert het model via het
    Akaike Information Criterion (AIC).

    Returns:
        dict: Bevat de modelnaam, gefitte parameters, AIC, RMSE en kans op een 'dip' (p_dip).
    """
    # -------------------------------------------------------------------------
    # 1. DATA PREPARATIE & BASIS STATISTIEKEN
    # -------------------------------------------------------------------------
    # Voorkom deling door nul bij extreem rustige dagen door een bodem in te stellen
    mu_data = max(np.mean(counts), 0.01)
    var_data = max(np.var(counts), 0.01)

    # Check op overdispersie: als de variantie veel groter is dan het gemiddelde,
    is_overdispersed = var_data > mu_data

    # Epsilon voorkomt math domain errors (log(0)) in de likelihood formules
    EPSILON = 1e-10

    modellen_config = []

    # --- MODEL 1: POISSON (De Basislijn) ---
    # Gaat uit van onafhankelijke gebeurtenissen rondom een constant gemiddelde.
    modellen_config.append({
        'naam': 'Poisson',
        'k_params': 1,
        # Voor Poisson is de MLE wiskundig bewezen gelijk aan het gemiddelde van de data.
        # Geen optimalisatie (minimize) nodig. We berekenen direct de NLL.
        'nll_waarde': -np.sum(stats.poisson.logpmf(counts, mu_data)),
        'params': {'mu': mu_data},
        'cdf_func': lambda p, x: stats.poisson.cdf(x, p['mu'])
    })

    # --- MODEL 2: ZERO-INFLATED POISSON (ZIP) ---
    # Bedoeld voor reeksen met onnatuurlijk veel nul-waarden (zoals sluitingsdagen).
    # 'pi' is de kans op een nuldag.
    def nll_zip(p):
        pi, mu = p[0], p[1]
        kans_0 = pi + (1 - pi) * stats.poisson.pmf(0, mu)
        kans_x = (1 - pi) * stats.poisson.pmf(counts, mu)
        pmf_tot = np.where(counts == 0, kans_0, kans_x)
        return -np.sum(np.log(pmf_tot + EPSILON))

    modellen_config.append({
        'naam': 'ZIP',
        'k_params': 2,
        'optimize': True,
        'nll_func': nll_zip,
        'start_guess': [np.mean(counts == 0), mu_data],
        'bounds': [(0, 0.95), (0.01, 500)],
        'extract_params': lambda x: {'pi': x[0], 'mu': x[1]},
        'cdf_func': lambda p, x: np.where(x == 0,
                                          p['pi'] + (1 - p['pi']) * stats.poisson.cdf(0, p['mu']),
                                          p['pi'] + (1 - p['pi']) * stats.poisson.cdf(x, p['mu']))
    })

    # --- MODEL 3: DISCRETE MIXTURE MODEL (DMM) ---
    # Enkel berekend bij overdispersie. Mixt twee Negatief Binomiale Verdelingen (NBD)
    # om twee compleet verschillende 'modellen' te hebben (bijv. vakantie vs normale week).
    if is_overdispersed:
        init_n = max((mu_data ** 2) / (var_data - mu_data), 0.1)
        init_p = min(max(mu_data / var_data, 0.05), 0.99)

        def nll_dmm(p):
            w, n1, p1, n2, p2 = p[0], p[1], p[2], p[3], p[4]
            mix = (w * stats.nbinom.pmf(counts, n1, p1)) + ((1 - w) * stats.nbinom.pmf(counts, n2, p2))
            return -np.sum(np.log(mix + EPSILON))

        def extract_dmm_params(x):
            # Forceer dat Component 1 altijd het model is met het laagste gemiddelde.
            w, n1, p1, n2, p2 = x
            gemiddelde_1 = (n1 * (1 - p1)) / p1
            gemiddelde_2 = (n2 * (1 - p2)) / p2
            if gemiddelde_1 > gemiddelde_2:
                w, n1, p1, n2, p2 = (1 - w), n2, p2, n1, p1
            return {'w': w, 'n1': n1, 'p1': p1, 'n2': n2, 'p2': p2}

        modellen_config.append({
            'naam': 'DMM',
            'k_params': 5,
            'optimize': True,
            'nll_func': nll_dmm,
            'start_guess': [0.3, init_n, init_p, init_n * 2, init_p],
            'bounds': [(0.05, 0.95), (0.1, 200), (0.05, 0.99), (0.1, 200), (0.05, 0.99)],
            'extract_params': extract_dmm_params,
            'cdf_func': lambda p,
                               x: p['w'] * stats.nbinom.cdf(x, p['n1'], p['p1'])
                                  + (1 - p['w']) * stats.nbinom.cdf(x, p['n2'], p['p2'])
        })

    # -------------------------------------------------------------------------
    # 3. EXECUTIE LUS (Train modellen en kies de beste via AIC)
    # -------------------------------------------------------------------------
    best_result = {'aic': float('inf')}

    def bereken_rmse_intern(cdf_theoretisch):
        empirisch_cdf = np.arange(1, len(counts) + 1) / len(counts)
        return np.sqrt(np.mean((empirisch_cdf - cdf_theoretisch(np.sort(counts))) ** 2))

    for config in modellen_config:
        nll_waarde = 0
        parameters = {}

        if config.get('optimize', False):
            res = minimize(config['nll_func'], config['start_guess'], bounds=config['bounds'], method='L-BFGS-B')
            if not res.success: continue
            nll_waarde = res.fun
            parameters = config['extract_params'](res.x)
        else:
            # Voor modellen zonder optimalisatie (zoals standaard Poisson)
            nll_waarde = config['nll_waarde']
            parameters = config['params']

        # AIC straft modellen met te veel parameters af (voorkomt overfitting)
        aic = (2 * config['k_params']) + (2 * nll_waarde)

        # Update de winnaar als deze AIC lager is
        if aic < best_result['aic']:
            best_result = {
                'model': config['naam'],
                'params': parameters,
                'aic': aic,
                'rmse': bereken_rmse_intern(lambda x: config['cdf_func'](parameters, x))
            }

    # Extract wiskundige dip-kans voor latere correlatie
    best_result['p_dip'] = 0.0
    if best_result['model'] in ['ZIP', 'ZINB']:
        best_result['p_dip'] = best_result['params']['pi']
    elif best_result['model'] == 'DMM':
        best_result['p_dip'] = best_result['params']['w']

    return best_result


def laad_of_train_modellen(df):
    """ Laadt bestaande parameters in via het bestand of traint de modellen opnieuw indien nodig. """
    if os.path.exists(JSON_MODEL_BESTAND):
        print("[2/5] Bestaande modellen inladen uit JSON...")
        with open(JSON_MODEL_BESTAND, 'r') as f:
            geladen_data = json.load(f)
            if "modellen" in geladen_data:
                return geladen_data["modellen"], geladen_data.get("condities")
            return geladen_data, None

    print("[2/5] Modellen fitten...")
    modellen = {
        str(wd): {hub: fit_modellen_voor_data(df[(df['weekday'] == wd) & (df['hub'] == hub)]['aantal_ritten'].values)
                  for hub in VASTE_HUBS if len(df[(df['weekday'] == wd) & (df['hub'] == hub)]) > 0} for wd in range(5)}
    return modellen, None


# ═══════════════════════════════════════════════════════════════
# 3. CONDITONELE KANSEN BEREKENEN
# ═══════════════════════════════════════════════════════════════
def bepaal_status(y, m_dict):
    """
    Bekijkt een historische dag (y) en berekent via het model of deze specifieke dag
    als 'Normaal' (0) of als 'Dip/Feestdag' (1) geclassificeerd moet worden.
    """
    mod, p = m_dict['model'], m_dict['params']
    if mod in ['Poisson', 'NBD']: return 0
    if mod in ['ZIP', 'ZINB']: return 1 if y == 0 else 0
    if mod == 'DMM':
        # Check welk component van de mix de hoogste kansmassafunctie (pmf) heeft
        return 1 if (p['w'] * stats.nbinom.pmf(y, p['n1'], p['p1'])) > (
            (1 - p['w']) * stats.nbinom.pmf(y, p['n2'], p['p2'])) else 0
    return None


def bereken_condities(df, modellen):
    """
    Berekent de conditionele kansen tussen de Anker-Hub en de overige hubs.
    (Bv. P(Hub 2 = Dip | Anker = Dip)).
    """
    print("[3/5] Conditionele kansen berekenen op basis van Historische Data...")

    # Label alle historische dagen
    df['state'] = [bepaal_status(row['aantal_ritten'], modellen[str(row['weekday'])][row['hub']]) for _, row in
                   df.iterrows()]
    state_matrix = df.pivot_table(index='date', columns='hub', values='state').dropna()

    condities = {}
    for wd in range(5):
        condities[str(wd)] = {}
        df_wd = state_matrix[state_matrix.index.dayofweek == wd]
        for hub in VASTE_HUBS:
            if hub == ANKER_HUB: continue

            # Tel hoe vaak deze hub een dip was, conditioneel op de status van het anker
            dagen_dip = df_wd[df_wd[ANKER_HUB] == 1]
            dagen_norm = df_wd[df_wd[ANKER_HUB] == 0]

            condities[str(wd)][hub] = {
                'als_anker_is_dip': dagen_dip[hub].mean() if len(dagen_dip) > 0 else 0.0,
                'als_anker_is_normaal': dagen_norm[hub].mean() if len(dagen_norm) > 0 else 0.0
            }
    return condities


def plot_condities_matrix(condities):
    """ Visualiseert de hierboven berekende conditionele kansen in twee heatmaps. """
    hubs = [hub for hub in VASTE_HUBS if hub != ANKER_HUB]
    weekdays = [DAG_NAAR_NAAM[wd] for wd in range(5)]

    dip_matrix = np.array([
        [condities[str(wd)][hub]['als_anker_is_dip'] * 100 for hub in hubs]
        for wd in range(5)
    ])
    normal_matrix = np.array([
        [(1.0 - condities[str(wd)][hub]['als_anker_is_normaal']) * 100 for hub in hubs]
        for wd in range(5)
    ])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    vmin, vmax = 0, 100

    # Plot 1: Anker heeft dip
    im1 = axes[0].imshow(dip_matrix, cmap='YlOrRd', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Als {ANKER_HUB} een dip heeft\nKans dat andere hubs ook dip zijn", fontweight='bold')
    axes[0].set_xticks(range(len(hubs)), labels=hubs)
    axes[0].set_yticks(range(len(weekdays)), labels=weekdays)
    axes[0].set_xlabel('Andere hubs')
    axes[0].set_ylabel('Weekdag')
    for i in range(dip_matrix.shape[0]):
        for j in range(dip_matrix.shape[1]):
            axes[0].text(j, i, f"{dip_matrix[i, j]:.0f}%", ha='center', va='center', color='black' if dip_matrix[i, j] < 60 else 'white')

    # Plot 2: Anker is normaal
    im2 = axes[1].imshow(normal_matrix, cmap='YlGnBu', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Als {ANKER_HUB} een normale dag heeft\nKans dat andere hubs ook normaal zijn", fontweight='bold')
    axes[1].set_xticks(range(len(hubs)), labels=hubs)
    axes[1].set_yticks(range(len(weekdays)), labels=weekdays)
    axes[1].set_xlabel('Andere hubs')
    axes[1].set_ylabel('Weekdag')
    for i in range(normal_matrix.shape[0]):
        for j in range(normal_matrix.shape[1]):
            axes[1].text(j, i, f"{normal_matrix[i, j]:.0f}%", ha='center', va='center', color='black' if normal_matrix[i, j] < 60 else 'white')

    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(f'Conditionele kansmatrix voor anker-hub {ANKER_HUB}', fontsize=14, fontweight='bold')
    out_path = os.path.join(OUTPUT_MAP, 'conditionele_kans_matrix.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Info] Conditionele kansmatrix opgeslagen in '{out_path}'.")


# ═══════════════════════════════════════════════════════════════
# 4. DE SIMULATIE
# ═══════════════════════════════════════════════════════════════
def genereer_ritten(model_dict, is_dip):
    """ Trekt een willekeurige waarde (sample) uit de gefitte kansverdeling. """
    mod, p = model_dict['model'], model_dict['params']
    if mod == 'Poisson': return stats.poisson.rvs(p['mu'])
    if mod == 'NBD': return stats.nbinom.rvs(p['n'], p['p'])
    if mod in ['ZIP', 'ZINB']: return 0 if is_dip else (
        stats.poisson.rvs(p['mu']) if mod == 'ZIP' else stats.nbinom.rvs(p['n'], p['p']))
    if mod == 'DMM': return stats.nbinom.rvs(p['n1'], p['p1']) if is_dip else stats.nbinom.rvs(p['n2'], p['p2'])


def run_simulatie(modellen, condities, num_dagen=2500):
    """
    Genereert een dataset door eerst de Anker-Hub te evalueren en de rest
    van de stad hier op af te stemmen (behoudt de ruimtelijke correlatie).
    """

    print(f"[4/5] Simuleren van {num_dagen} dagen met conditionele Anker-logica...")
    sim_records = []

    for _ in range(num_dagen):
        wd = str(np.random.randint(0, 5))

        # 1. Start bij het Anker
        is_anker_dip = np.random.rand() < modellen[wd][ANKER_HUB]['p_dip']
        row = {'weekday': int(wd), ANKER_HUB: genereer_ritten(modellen[wd][ANKER_HUB], is_anker_dip)}

        # 2. Bepaal de rest van de locaties conditioneel
        for hub in VASTE_HUBS:
            if hub == ANKER_HUB: continue
            kans_op_dip = condities[wd][hub]['als_anker_is_dip'] if is_anker_dip else condities[wd][hub][
                'als_anker_is_normaal']
            row[hub] = genereer_ritten(modellen[wd][hub], np.random.rand() < kans_op_dip)
        sim_records.append(row)

    return pd.DataFrame(sim_records)


# ═══════════════════════════════════════════════════════════════
# 5. VALIDATIE PLOTS MAKEN
# ═══════════════════════════════════════════════════════════════
def bereken_theoretische_pmf(model_type, params, x_as):
    """ Berekent de wiskundige (rode) lijn van het geselecteerde model voor de plot. """
    if model_type == 'Poisson':
        return stats.poisson.pmf(x_as, params['mu'])
    elif model_type == 'ZIP':
        pi, mu = params['pi'], params['mu']
        pmf = (1 - pi) * stats.poisson.pmf(x_as, mu)
        pmf[0] += pi
        return pmf
    elif model_type == 'NBD':
        return stats.nbinom.pmf(x_as, params['n'], params['p'])
    elif model_type == 'ZINB':
        pi, n, p = params['pi'], params['n'], params['p']
        pmf = (1 - pi) * stats.nbinom.pmf(x_as, n, p)
        pmf[0] += pi
        return pmf
    elif model_type == 'DMM':
        w, n1, p1, n2, p2 = params['w'], params['n1'], params['p1'], params['n2'], params['p2']
        return w * stats.nbinom.pmf(x_as, n1, p1) + (1 - w) * stats.nbinom.pmf(x_as, n2, p2)
    return np.zeros_like(x_as)


def plot_en_sla_op(df_hist, df_sim, modellen):
    """ Genereert 5 zeer overzichtelijke validatie-plots per hub per dag. """
    print("[5/5] Submappen aanmaken en 5 individuele grafieken per categorie genereren...")

    for wd in range(5):
        dag_naam = DAG_NAAR_NAAM[wd]
        for hub in VASTE_HUBS:
            hist_counts = df_hist[(df_hist['weekday'] == wd) & (df_hist['hub'] == hub)]['aantal_ritten'].values
            sim_counts = df_sim[df_sim['weekday'] == wd][hub].values
            if len(hist_counts) == 0: continue

            model_info = modellen[str(wd)][hub]
            model_naam = model_info['model']

            # --- MAPPENSTRUCTUUR MAKEN ---
            # Bijv: 01_aantal_jobs/Punt_1/Maandag/
            plot_dir = os.path.join(OUTPUT_MAP, hub, dag_naam)
            os.makedirs(plot_dir, exist_ok=True)

            # Instellingen voor assen
            max_x = int(max(hist_counts.max(), sim_counts.max())) + 2
            x_as = np.arange(0, max_x)

            hist_freq = np.bincount(hist_counts, minlength=max_x)[:max_x] / len(hist_counts)
            sim_freq = np.bincount(sim_counts, minlength=max_x)[:max_x] / len(sim_counts)

            # =================================================================
            # PLOT 1: Theoretisch Model vs Historisch
            # =================================================================
            fig, ax = plt.subplots(figsize=(10, 6))
            theo_pmf = bereken_theoretische_pmf(model_naam, model_info['params'], x_as)

            ax.bar(x_as, hist_freq, color=C_HIST, alpha=0.6, label=f'Historische Data (n={len(hist_counts)})')
            ax.plot(x_as, theo_pmf, color=C_THEO, marker='o', ls='--', lw=2, markersize=5,
                    label=f'Theoretisch Fit ({model_naam})')

            ax.set_title(f"1. Theoretisch Model vs Historisch\n{hub} op {dag_naam}", fontweight='bold')
            ax.set_xlabel("Aantal Ritten per Dag (Eenheden)")
            ax.set_ylabel("Relatieve Frequentie (Kans 0.0 - 1.0)")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "1_Theoretisch_vs_Historisch.png"), dpi=120)
            plt.close(fig)

            # =================================================================
            # PLOT 2: Simulatie vs Historisch PMF (Histogrammen)
            # =================================================================
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x_as - 0.2, hist_freq, width=0.4, color=C_HIST, label='Historisch (Met Ruis)')
            ax.bar(x_as + 0.2, sim_freq, width=0.4, color=C_SIM, label=f'Simulatie (Gecorreleerd, n={len(sim_counts)})')

            ax.set_title(f"2. Simulatie vs Historisch (Kansmassafunctie)\n{hub} op {dag_naam}", fontweight='bold')
            ax.set_xlabel("Aantal Ritten per Dag (Eenheden)")
            ax.set_ylabel("Relatieve Frequentie (Kans 0.0 - 1.0)")
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "2_Simulatie_vs_Historisch_PMF.png"), dpi=120)
            plt.close(fig)

            # =================================================================
            # PLOT 3: Cumulatieve Verdeling (ECDF)
            # =================================================================
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.step(np.sort(hist_counts), np.arange(1, len(hist_counts) + 1) / len(hist_counts), color=C_HIST, lw=3,
                    label='Historisch')
            ax.step(np.sort(sim_counts), np.arange(1, len(sim_counts) + 1) / len(sim_counts), color=C_SIM, ls='--',
                    lw=2, label='Simulatie')

            ax.set_title(f"3. Cumulatieve Verdeling (ECDF)\n{hub} op {dag_naam}", fontweight='bold')
            ax.set_xlabel("Aantal Ritten per Dag (Eenheden)")
            ax.set_ylabel("Cumulatieve Kans (0.0 - 1.0)")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "3_Cumulatieve_Verdeling.png"), dpi=120)
            plt.close(fig)

            # =================================================================
            # PLOT 4: Q-Q Plot (Quantielen)
            # =================================================================
            fig, ax = plt.subplots(figsize=(8, 8))  # Vierkant is beter voor QQ
            q_hist = np.quantile(hist_counts, np.linspace(0.01, 0.99, 100))
            q_sim = np.quantile(sim_counts, np.linspace(0.01, 0.99, 100))
            lim = max(q_hist.max(), q_sim.max()) + 2

            ax.scatter(q_sim + np.random.uniform(-0.3, 0.3, 100), q_hist + np.random.uniform(-0.3, 0.3, 100),
                       color=C_SIM, alpha=0.6)
            ax.plot([0, lim], [0, lim], color='gray', ls='--')

            ax.set_title(f"4. Quantile-Quantile (Q-Q) Plot\n{hub} op {dag_naam}", fontweight='bold')
            ax.set_xlabel("Gesimuleerde Quantielen (Aantal Ritten)")
            ax.set_ylabel("Historische Quantielen (Aantal Ritten)")
            ax.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "4_QQ_Plot.png"), dpi=120)
            plt.close(fig)

            # =================================================================
            # PLOT 5: Gesorteerde Data
            # =================================================================
            fig, ax = plt.subplots(figsize=(10, 6))
            hist_sorted = np.sort(hist_counts)
            sim_sorted = np.sort(sim_counts)

            x_hist = np.linspace(0, 100, len(hist_sorted))
            x_sim = np.linspace(0, 100, len(sim_sorted))

            ax.plot(x_hist, hist_sorted, color=C_HIST, marker='o', markersize=4, linestyle='-', lw=2,
                    label='Historisch (Werkelijkheid)')
            ax.plot(x_sim, sim_sorted, color=C_SIM, linestyle='--', lw=3, label='Simulatie (Geleerde Trend)')

            ax.set_title(f"5. Visuele Data Match (Gesorteerd van rustig naar druk)\n{hub} op {dag_naam}",
                         fontweight='bold')
            ax.set_xlabel("Percentiel van alle geobserveerde/gesimuleerde dagen (%)")
            ax.set_ylabel("Aantal Ritten per Dag (Eenheden)")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "5_Gesorteerde_Data.png"), dpi=120)
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# HOOFDPROGRAMMA
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    df_hist = laad_data(JSON_DATA_BESTAND)
    if df_hist is None: return

    modellen, condities = laad_of_train_modellen(df_hist)

    print("\n--- MODEL OVERZICHT ---")
    for wd in range(5):
        print(f"\n{DAG_NAAR_NAAM[wd].upper()}:")
        for hub in VASTE_HUBS:
            info = modellen[str(wd)][hub]
            print(
                f"{hub:<10} | Model: {info['model']:<5} | Kans op Feestdag/Dip: {info['p_dip'] * 100:>4.1f}% | Kans Normaal: {(1 - info['p_dip']) * 100:>4.1f}% | AIC: {info['aic']:>6.1f} | RMSE: {info['rmse'] * 100:>4.2f}%")

    # Als de matrix nog niet uit caching geladen is, bereken hem dan
    if condities is None:
        condities = bereken_condities(df_hist, modellen)

    plot_condities_matrix(condities)

    # Schrijf parameters en condities weg zodat sampler.py ze snel kan inladen
    export_data = {
        'anker_hub': ANKER_HUB,
        'modellen': modellen,
        'condities': condities
    }

    with open(JSON_MODEL_BESTAND, 'w') as f:
        json.dump(export_data, f, indent=4)
    print(f"[Info] Modellen + condities weggeschreven naar '{JSON_MODEL_BESTAND}'.")

    df_sim = run_simulatie(modellen, condities, num_dagen=2500)
    plot_en_sla_op(df_hist, df_sim, modellen)

    print(f"\n[OK] Klaar! Je overzichtelijke mappenstructuur staat klaar in: '{OUTPUT_MAP}/'")


if __name__ == '__main__':
    main()