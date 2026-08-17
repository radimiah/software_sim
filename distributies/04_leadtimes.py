"""
04_leadtimes.py

Dit script bouwt het model dat bepaalt hoe ver op voorhand medewerkers hun wagen  reserveren (de 'Lead-Time').
Omdat menselijk gedrag sterk verdeeld is in groepen  (planners vs. last-minute beslissers), gebruiken we een Mixture Model.
"""

import os
import json
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# ==============================================================================
# CONFIGURATIE (Jij kan dit aanpassen!)
# ==============================================================================
OUTPUT_MAP = '04_leadtime_metrics'
JSON_OUT_BESTAND = '04_leadtime_distributie.json'

# We gebruiken een Log-Normal Mixture Model.
# - 'weight': De kans dat een boeking in deze categorie valt (wordt genormaliseerd).
# - 'median_hours': Het absolute zwaartepunt van deze groep (in uren vooraf geboekt).
# - 'sigma': De 'spreiding' of onzekerheid. (0.3 is smal, 1.0 is heel breed/willekeurig).
LEADTIME_PARAMS = {
    "last_minute": {
        "weight": 0.20,  # 20% van de werknemers is een last-minute boeker
        "median_hours": 2.0,  # Ze boeken gemiddeld 2 uur voor vertrek
        "sigma": 0.5  # Vrij brede spreiding (sommigen 10 min, anderen 5 uur)
    },
    "vorige_dag": {
        "weight": 0.60,  # Het overgrote deel (60%) boekt een dagje van tevoren
        "median_hours": 24.0,  # De piek ligt op exact 24 uur (1 dag)
        "sigma": 0.4  # Smalle spreiding (het zit vrij geconcentreerd rond die 24u)
    },
    "vorige_week": {
        "weight": 0.20,  # 20% van de werknemers plannen een week vooraf
        "median_hours": 144.0,  # De piek ligt op 6 dagen (144 uur) van tevoren
        "sigma": 0.5  # Zeer brede staart (uitschieters tot wel weken vooraf)
    }
}

# Kleuren voor de plot
KLEUREN = ['#D73027', '#FDAE61', '#4575B4']
C_TOTAAL = '#1A9850'


# ==============================================================================
# 1. WISKUNDIGE BEREKENING (LOG-NORMAL MIXTURE)
# ==============================================================================
def bereken_pdf(x_waarden, params):
    """
    Berekent de kansdichtheid (Probability Density Function) voor een reeks x-waarden (uren).
    Het mengt (mixt) de 3 afzonderlijke log-normale curves tot één grote, complexe curve.
    """

    # Zorg dat de wegingen samen altijd wiskundig exact 1.0 (100%) zijn
    totale_kans = sum(p['weight'] for p in params.values())

    pdf_totaal = np.zeros_like(x_waarden)
    component_pdfs = {}

    for (naam, p) in params.items():
        w = p['weight'] / totale_kans
        mu = np.log(p['median_hours'])
        sigma = p['sigma']

        # Bereken de curve voor specifiek dit onderdeel
        pdf_comp = w * stats.lognorm.pdf(x_waarden, s=sigma, scale=np.exp(mu))
        component_pdfs[naam] = pdf_comp

        # Tel het op bij het geheel
        pdf_totaal += pdf_comp

    return pdf_totaal, component_pdfs


# ==============================================================================
# 2. PLOTTING VAN HET MODEL
# ==============================================================================
def genereer_plot(params, uit_map):
    os.makedirs(uit_map, exist_ok=True)
    print("\n[Plotter] Genereren van Lead-Time distributie grafiek...")

    # We kijken tot 336 uur vooruit (exact 2 weken)
    x = np.linspace(0.1, 336, 2000)
    pdf_totaal, comp_pdfs = bereken_pdf(x, params)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # 1. Plot de losse componenten (Gestippeld, om de 3 groepen te laten zien)
    for i, (naam, pdf) in enumerate(comp_pdfs.items()):
        ax.plot(x, pdf, linestyle='--', color=KLEUREN[i % len(KLEUREN)], lw=2,
                label=f'Fase: {naam} (Mediaan: {params[naam]["median_hours"]}u)')
        ax.fill_between(x, pdf, alpha=0.1, color=KLEUREN[i % len(KLEUREN)])

    # 2. Plot de uiteindelijke samengetelde continue lijn (Dit is wat de simulator echt gebruikt)
    ax.plot(x, pdf_totaal, color=C_TOTAAL, lw=3, label='Totale Lead-Time Distributie (Continu)')

    ax.set_title("Theoretische Lead-Time Distributie (Log-Normal Mixture Model)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tijd tussen Boeking en Vertrek (Uren)", fontsize=11)
    ax.set_ylabel("Dichtheid", fontsize=11)

    # Maak de x-as leesbaar (Dagen aanduiden)
    xticks = [0, 24, 48, 72, 96, 120, 144, 168, 252, 336]
    xlabels = ['0', '1 dag\n(24u)', '2 d.', '3 d.', '4 d.', '5 d.', '6 d.', '1 week\n(168u)', '1.5 w.', '2 weken']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlim(0, 336)

    ax.legend(fontsize=11)

    # Sla op
    pad = os.path.join(uit_map, "00_leadtime_distributie.png")
    plt.tight_layout()
    plt.savefig(pad, dpi=130)
    plt.close()

    print(f" -> Plot opgeslagen in '{pad}'")


# ==============================================================================
# 3. OPSLAAN NAAR JSON VOOR DE SIMULATOR
# ==============================================================================
def sla_op_naar_json(params, bestand):
    """
    Vertaalt onze menselijke parameters (medianen) naar de harde wiskunde (mu)
    en slaat dit op zodat `sampler.py` dit razendsnel kan uitlezen en simuleren.
    """
    totaal_gewicht = sum(p['weight'] for p in params.values())

    json_data = {
        "model_type": "LogNormal_Mixture",
        "description": "Continue lead-time distributie opgesplitst in 3 tijdsblokken.",
        "components": {}
    }

    for naam, p in params.items():
        json_data["components"][naam] = {
            "weight": round(p['weight'] / totaal_gewicht, 4),
            "mu": round(np.log(p['median_hours']), 4),
            "sigma": p['sigma']
        }

    with open(bestand, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)

    print(f" -> Parameters weggeschreven naar '{bestand}'")


# ==============================================================================
# HOOFDPROGRAMMA
# ==============================================================================
def main():
    print("=" * 60)
    print("  LEAD-TIME DISTRIBUTIE GENERATOR")
    print("=" * 60)

    genereer_plot(LEADTIME_PARAMS, OUTPUT_MAP)
    sla_op_naar_json(LEADTIME_PARAMS, JSON_OUT_BESTAND)

    print("\n[OK] Klaar! Je kan de plot bekijken om de parameters te finetunen.")


if __name__ == '__main__':
    main()
