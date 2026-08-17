import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def bereken_run_metrics(df_jobs, actuele_vloot, aantal_dagen):
    metrics = {}
    concurrent_usage = {}
    wait_times_dict = {}
    rejected_hours_dict = {}

    for h, vloot_grootte in actuele_vloot.items():
        if not df_jobs.empty:
            df_h = df_jobs[df_jobs['hub'] == h]
            df_acc = df_h[df_h['status'] == 'Geaccepteerd']
            df_noshow = df_h[df_h['status'] == 'No-Show']  # NIEUW
            df_rej = df_h[df_h['status'] == 'Geweigerd']
        else:
            df_h = pd.DataFrame()
            df_acc = pd.DataFrame()
            df_noshow = pd.DataFrame()
            df_rej = pd.DataFrame()

        tot = len(df_h)
        acc = len(df_acc)
        noshow = len(df_noshow)
        rej = len(df_rej)

        # Service level: Geaccepteerde ritten + No-shows (het systeem heeft zijn best gedaan, de auto was beschikbaar)
        sl = ((acc + noshow) / tot * 100) if tot > 0 else 100.0

        totale_beschikbare_uren = vloot_grootte * aantal_dagen * 24.0
        # No-shows hebben niet gereden, dus die tellen we NIET mee in de bezettingsgraad
        gebruikte_uren = df_acc['duratie_uren'].sum() if not df_acc.empty else 0.0
        bezettingsgraad = (gebruikte_uren / totale_beschikbare_uren * 100) if totale_beschikbare_uren > 0 else 0.0

        if not df_acc.empty and 'wait_time' in df_acc.columns:
            wachttijden = df_acc['wait_time'].dropna().values
            gem_wachttijd = np.mean(wachttijden) if len(wachttijden) > 0 else 0.0
            p95_wachttijd = np.percentile(wachttijden, 95) if len(wachttijden) > 0 else 0.0
            max_wachttijd = np.max(wachttijden) if len(wachttijden) > 0 else 0.0
        else:
            wachttijden = np.array([])
            gem_wachttijd = p95_wachttijd = max_wachttijd = 0.0

        if not df_rej.empty:
            rejected_hours = (df_rej['start_time'] % 24).astype(int).values
        else:
            rejected_hours = np.array([])

        wait_times_dict[h] = wachttijden
        rejected_hours_dict[h] = rejected_hours

        metrics[h] = {
            'totaal': tot,
            'geaccepteerd': acc,
            'geweigerd': rej,
            'no_show': noshow,
            'service_level': sl,
            'bezettingsgraad': bezettingsgraad,
            'gem_wachttijd': gem_wachttijd,
            'p95_wachttijd': p95_wachttijd,
            'max_wachttijd': max_wachttijd
        }

        usage_array = np.zeros(vloot_grootte + 1)
        if not df_acc.empty:
            events = []
            for _, row in df_acc.iterrows():
                events.append((row['start_time'], 1))
                events.append((row['end_time'], -1))

            events.sort(key=lambda x: (x[0], x[1]))
            t_prev = 0.0
            active = 0
            for t_now, change in events:
                if t_now > t_prev:
                    idx = min(active, vloot_grootte)
                    usage_array[idx] += (t_now - t_prev)
                active += change
                t_prev = t_now

            totale_sim_tijd = aantal_dagen * 24.0
            if totale_sim_tijd > t_prev:
                usage_array[0] += (totale_sim_tijd - t_prev)
        else:
            usage_array[0] = aantal_dagen * 24.0

        concurrent_usage[h] = usage_array

    return metrics, concurrent_usage, wait_times_dict, rejected_hours_dict

def schrijf_naar_logboek(pad, tekst, print_console=True):
    with open(pad, 'a', encoding='utf-8') as f:
        f.write(tekst + "\n")
    if print_console:
        print(tekst)


# ═══════════════════════════════════════════════════════════════
# HULPFUNCTIES VOOR PERCENTIEL-ANALYSE (NIEUW)
# ═══════════════════════════════════════════════════════════════
PERCENTIEL_NIVEAUS = [5, 10, 25, 50, 75, 90, 95]


def _bereken_percentiel_data(df_res, hubs):
    """ Berekent per hub de service-level percentielen over alle gesimuleerde jaren. """
    data = {}
    for hub in hubs:
        sl = df_res[f"{hub}_service_level"].values
        data[hub] = {p: float(np.percentile(sl, p)) for p in PERCENTIEL_NIVEAUS}
    return data


def genereer_iteratie_plots(iteratie, df_res, avg_concurrent_usage, actuele_vloot, drempel, output_map, all_wait_times,
                            all_rejected_hours, sl_percentiel=10):
    hubs = sorted(actuele_vloot.keys())

    # 1. Service Level (boxplot) -- nu met percentiel-marker (het niveau dat de vlootbeslissing stuurt)
    plt.figure(figsize=(10, 6))
    sl_cols = [f"{h}_service_level" for h in hubs]
    sns.boxplot(data=df_res[sl_cols], palette="Set2")
    for i, hub in enumerate(hubs):
        sl = df_res[f"{hub}_service_level"].values
        p_waarde = np.percentile(sl, sl_percentiel)
        plt.scatter(i, p_waarde, marker='D', s=90, color='black', zorder=5,
                    label=f'P{sl_percentiel} (beslissing)' if i == 0 else None)
    plt.title(f"Service Level - Iteratie {iteratie}", fontweight='bold')
    plt.ylabel("Service Level (%)")
    plt.xticks(range(len(hubs)), hubs)
    plt.axhline(y=drempel, color='r', linestyle='--', label=f'{drempel}% Drempel')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_map, f"Iter_{iteratie}_Service_Level_Boxplots.png"), dpi=120)
    plt.close()

    # 1b. NIEUW: Service Level Percentielen per hub (gegroepeerde staafgrafiek)
    percentiel_data = _bereken_percentiel_data(df_res, hubs)
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(hubs))
    n_p = len(PERCENTIEL_NIVEAUS)
    width = 0.8 / n_p
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.9, n_p))

    for i, p in enumerate(PERCENTIEL_NIVEAUS):
        waarden = [percentiel_data[h][p] for h in hubs]
        offset = (i - n_p / 2) * width + width / 2
        ax.bar(x + offset, waarden, width, label=f"P{p}", color=colors[i], edgecolor='black', alpha=0.9)

    ax.axhline(y=drempel, color='red', linestyle='--', linewidth=2, label=f'{drempel}% Drempel')
    ax.set_xticks(x)
    ax.set_xticklabels(hubs)
    ax.set_ylabel("Service Level (%)")
    ax.set_title(f"Service Level Percentielen per Hub - Iteratie {iteratie}\n"
                 f"(Vlootbeslissing gebaseerd op P{sl_percentiel})", fontweight='bold')
    ax.legend(loc='lower right', ncol=4, fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_map, f"Iter_{iteratie}_Service_Level_Percentielen.png"), dpi=120)
    plt.close()

    # 1c. NIEUW: Percentage jaren onder de drempel per hub ("risico op een slecht jaar")
    fig, ax = plt.subplots(figsize=(9, 6))
    percentages = []
    for hub in hubs:
        sl = df_res[f"{hub}_service_level"].values
        pct = float(np.mean(sl < drempel) * 100)
        percentages.append(pct)

    kleuren = ['#D73027' if p > 20 else ('#FDAE61' if p > 5 else '#1A9850') for p in percentages]
    bars = ax.bar(hubs, percentages, color=kleuren, edgecolor='black', alpha=0.9)
    ax.set_ylabel(f"% van jaren onder {drempel}% drempel")
    ax.set_title(f"Risico op een Slecht Jaar per Hub - Iteratie {iteratie}", fontweight='bold')
    ax.set_ylim(0, max(100.0, max(percentages) + 10) if percentages else 100.0)
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct:.1f}%",
                ha='center', va='bottom', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_map, f"Iter_{iteratie}_Pct_Jaren_Onder_Drempel.png"), dpi=120)
    plt.close()

    # 1d. NIEUW: Histogram van de service-level verdeling per hub, met drempel + P-marker
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, hub in enumerate(hubs):
        sl = df_res[f"{hub}_service_level"].values
        p_waarde = np.percentile(sl, sl_percentiel)
        axes[i].hist(sl, bins=min(15, max(5, len(sl) // 2)), color='#4575B4', edgecolor='black', alpha=0.8)
        axes[i].axvline(x=drempel, color='red', linestyle='--', linewidth=2, label=f'{drempel}% Drempel')
        axes[i].axvline(x=p_waarde, color='black', linestyle='-', linewidth=2,
                        label=f'P{sl_percentiel} = {p_waarde:.1f}%')
        axes[i].set_title(f"Verdeling Service Level - {hub}", fontweight='bold')
        axes[i].set_xlabel("Service Level (%)")
        axes[i].set_ylabel(f"Aantal jaren (van {len(sl)})")
        axes[i].legend(fontsize=8)
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_map, f"Iter_{iteratie}_Service_Level_Histogrammen.png"), dpi=120)
    plt.close()

    # 2. Gelijktijdig Gebruik
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, hub in enumerate(hubs):
        usage = avg_concurrent_usage[hub]
        x_labels = np.arange(1, len(usage))
        y_values = usage[1:]
        colors = ['#4575B4' if x < len(usage) - 3 else '#D73027' for x in x_labels]
        bars = axes[i].bar(x_labels, y_values, color=colors, edgecolor='black', alpha=0.8)
        axes[i].set_title(f"Gelijktijdig Gebruik - {hub}", fontweight='bold')
        axes[i].set_xlabel("Aantal Auto's Tegelijkertijd op de Baan")
        axes[i].set_ylabel("Uren per jaar (Gemiddeld)")
        axes[i].set_xticks(x_labels)
        max_y = max(y_values) if len(y_values) > 0 else 1
        axes[i].margins(y=0.2)
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:
                axes[i].text(bar.get_x() + bar.get_width() / 2, height + (max_y * 0.02), f"{height:.0f}u", ha='center',
                             va='bottom', fontsize=9, rotation=45, color='black')
        axes[i].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_map, f"Iter_{iteratie}_Gelijktijdig_Gebruik.png"), dpi=120)
    plt.close()

    # 3 & 4. Wachttijden Plots
    has_wait = any(len(w) > 0 and np.max(w) > 0.001 for w in all_wait_times.values())
    if has_wait:
        wait_cols = [f"{h}_gem_wachttijd" for h in hubs]
        p95_cols = [f"{h}_p95_wachttijd" for h in hubs]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.boxplot(data=df_res[wait_cols] * 60, ax=axes[0], palette="Blues")
        axes[0].set_title(f"Gemiddelde Wachttijd - Iteratie {iteratie}", fontweight='bold')
        axes[0].set_ylabel("Wachttijd (Minuten)")
        axes[0].set_xticks(range(len(hubs)))
        axes[0].set_xticklabels(hubs)
        sns.boxplot(data=df_res[p95_cols] * 60, ax=axes[1], palette="Oranges")
        axes[1].set_title(f"95e Percentiel Wachttijd - Iteratie {iteratie}", fontweight='bold')
        axes[1].set_ylabel("Wachttijd (Minuten)")
        axes[1].set_xticks(range(len(hubs)))
        axes[1].set_xticklabels(hubs)
        plt.tight_layout()
        plt.savefig(os.path.join(output_map, f"Iter_{iteratie}_Wachttijden_Boxplots.png"), dpi=120)
        plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        bins = [-1, 0.05, 15, 30, 60, 120, float('inf')]
        labels = ['Direct (0 min)', '1-15 min', '16-30 min', '31-60 min', '1-2 uur', '> 2 uur']
        for i, hub in enumerate(hubs):
            w_min = all_wait_times.get(hub, np.array([])) * 60
            if len(w_min) > 0:
                counts, _ = np.histogram(w_min, bins=bins)
                percentages = (counts / len(w_min)) * 100
            else:
                percentages = np.zeros(len(labels))
            bars = axes[i].bar(labels, percentages, color='#9970AB', edgecolor='black', alpha=0.8)
            axes[i].set_title(f"Hoe lang wacht een klant? - {hub}", fontweight='bold')
            axes[i].set_ylabel("Percentage (%)")
            axes[i].set_ylim(0, 100)
            for bar in bars:
                height = bar.get_height()
                if height > 0.1:
                    axes[i].text(bar.get_x() + bar.get_width() / 2, height + 1.5, f"{height:.1f}%", ha='center',
                                 va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_map, f"Iter_{iteratie}_Wachttijden_Verdeling.png"), dpi=120)
        plt.close()

    # 5. Verdeling van Geweigerde Ritten per Uur
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, hub in enumerate(hubs):
        rej_uren = all_rejected_hours.get(hub, np.array([]))
        uren_counts = pd.Series(rej_uren).value_counts().reindex(range(24), fill_value=0)

        axes[i].bar(uren_counts.index, uren_counts.values, color='#D73027', edgecolor='black', alpha=0.8)
        axes[i].set_title(f"Wanneer is een job Onmogelijk? - {hub}", fontweight='bold')
        axes[i].set_xlabel("Uur van de Dag (00:00 - 23:00)")
        axes[i].set_ylabel(f"Aantal weigeringen (over {len(df_res)} jaren)")
        axes[i].set_xticks(range(0, 24, 2))
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_map, f"Iter_{iteratie}_Onmogelijke_Ritten.png"), dpi=120)
    plt.close()