import os
import pandas as pd
import numpy as np
import concurrent.futures

import config as cfg
from sampler import SWSampler
from job_generator import genereer_jobs_voor_jaar

from algorithms.fcfs import run_fcfs_sim
from algorithms.fcfs_wait import run_fcfs_wait_sim
from algorithms.fcfs_keys import run_fcfs_keys_sim
from metrics_logger import bereken_run_metrics, schrijf_naar_logboek, genereer_iteratie_plots

sampler_instance = None


def init_worker():
    """
    Wordt 1 keer per CPU-core uitgevoerd bij het opstarten van de pool.
    We laden de zware modellen eenmalig in het RAM-geheugen van elke worker.
    """
    global sampler_instance
    np.random.seed()  # Zorg dat elke CPU-core echt willekeurige, unieke getallen genereert
    sampler_instance = SWSampler()


def worker_task(args):
    """ De taak die door elke CPU-core wordt uitgevoerd: het simuleren van exact 1 jaar. """

    run_id, actuele_vloot = args

    # 1. Genereer een heel jaar aan realistische reserveringen
    df_jobs = genereer_jobs_voor_jaar(sampler_instance, cfg.AANTAL_DAGEN, cfg.VRAAG_MULTIPLIER)

    # 2. Pas het gekozen algoritme toe om de agenda's van de auto's te vullen
    if cfg.ALGORITME == 'FCFS_WAIT':
        df_jobs_processed = run_fcfs_wait_sim(df_jobs, actuele_vloot, cfg.BUFFER_TIJD_UREN, cfg.MAX_WACHTTIJD_UREN)
    elif cfg.ALGORITME == 'FCFS_KEYS':
        df_jobs_processed = run_fcfs_keys_sim(df_jobs, actuele_vloot, cfg.BUFFER_TIJD_UREN, cfg.KEYS_NOT_TAKEN)
    else:
        df_jobs_processed = run_fcfs_sim(df_jobs, actuele_vloot, cfg.BUFFER_TIJD_UREN)

    # 3. Bereken direct de prestaties (metrics) van dit gesimuleerde jaar
    metrics, concurrent_usage, wait_times_dict, rejected_hours_dict = bereken_run_metrics(
        df_jobs_processed, actuele_vloot, cfg.AANTAL_DAGEN)

    return run_id, metrics, concurrent_usage, wait_times_dict, rejected_hours_dict, df_jobs_processed


def run_iteratie(iteratie_nr, actuele_vloot, log_pad, executor):
    """
    Voert AANTAL_RUNS gesimuleerde jaren uit voor de huidige vlootgrootte.
    Verzamelt alle resultaten en retourneert de statistieken.
    """
    header = f"\n\n{'=' * 70}\n[ITERATIE {iteratie_nr}] Simuleren met Vloot: {actuele_vloot} (Algoritme: {cfg.ALGORITME})\n{'=' * 70}"
    schrijf_naar_logboek(log_pad, header)

    resultaten_metrics = []
    totale_concurrent_usage = {h: np.zeros(actuele_vloot.get(h, 0) + 1) for h in actuele_vloot.keys()}
    all_wait_times = {h: [] for h in actuele_vloot.keys()}
    all_rejected_hours = {h: [] for h in actuele_vloot.keys()}

    # Verdeel de jaren (runs) over de processorpoule
    taken = [(i, actuele_vloot.copy()) for i in range(cfg.AANTAL_RUNS)]
    futures = [executor.submit(worker_task, t) for t in taken]

    voltooid = 0
    lijst_dfs = []

    # Verzamel de resultaten zodra een CPU-core klaar is met een jaar
    for future in concurrent.futures.as_completed(futures):
        run_id, metrics, concurrent_usage, wait_times_dict, rejected_hours_dict, df_jobs_processed = future.result()

        # Maak een platte dictionary voor makkelijke conversie naar een DataFrame
        flat_metrics = {'run_id': run_id}
        for h, m in metrics.items():
            flat_metrics[f"{h}_service_level"] = m['service_level']
            flat_metrics[f"{h}_geweigerd"] = m['geweigerd']
            flat_metrics[f"{h}_bezettingsgraad"] = m['bezettingsgraad']
            flat_metrics[f"{h}_gem_wachttijd"] = m['gem_wachttijd']
            flat_metrics[f"{h}_p95_wachttijd"] = m['p95_wachttijd']
            flat_metrics[f"{h}_max_wachttijd"] = m['max_wachttijd']
            flat_metrics[f"{h}_no_show"] = m['no_show']

            totale_concurrent_usage[h] += concurrent_usage[h]
            all_wait_times[h].extend(wait_times_dict[h])
            all_rejected_hours[h].extend(rejected_hours_dict[h])

        resultaten_metrics.append(flat_metrics)
        lijst_dfs.append(df_jobs_processed)
        voltooid += 1
        print(f"\r   ⏳ Bezig met simuleren... {voltooid}/{cfg.AANTAL_RUNS} jaren voltooid", end="", flush=True)

    print()

    for h in all_wait_times:
        all_wait_times[h] = np.array(all_wait_times[h])
        all_rejected_hours[h] = np.array(all_rejected_hours[h])

    df_res = pd.DataFrame(resultaten_metrics).set_index('run_id')

    # --- Print totaal aantal uren gewacht in de terminal ---
    totaal_wachturen = 0.0
    for h in all_wait_times:
        if len(all_wait_times[h]) > 0:
            totaal_wachturen += np.nansum(all_wait_times[h])
    print(f"\nTotaal aantal uren gewacht (geaccepteerde jobs over {cfg.AANTAL_RUNS} jaar): {totaal_wachturen:.2f} uren")

    avg_concurrent_usage = {h: (totale_concurrent_usage[h] / cfg.AANTAL_RUNS) for h in actuele_vloot.keys()}
    return df_res, avg_concurrent_usage, all_wait_times, all_rejected_hours


def main():
    os.makedirs(cfg.OUTPUT_MAP, exist_ok=True)
    log_pad = os.path.join(cfg.OUTPUT_MAP, "iteratie_rapport.txt")

    with open(log_pad, 'w', encoding='utf-8') as f:
        f.write(f"=== STRATEGISCHE VLOOT SIMULATIE ===\n")
        f.write(f"Vlootbeslissing gebaseerd op: P{cfg.SL_PERCENTIEL} van het Service Level "
                f"(moet >= {cfg.SL_DREMPEL}% zijn)\n")

    actuele_vloot = cfg.START_VLOOT_CONFIG.copy()
    actieve_hubs = set(actuele_vloot.keys())
    iteratie = 1

    finale_run_actief = False

    # Start de multiprocessing poule EENMALIG op, omzeilt constante memory-reloads
    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:
        while True:
            iteratie_naam = "FINAAL" if finale_run_actief else iteratie

            df_res, avg_usage, all_wait_times, all_rejected_hours = run_iteratie(iteratie_naam, actuele_vloot, log_pad,
                                                                                 executor)

            lower_bounds = {}
            for hub in sorted(actuele_vloot.keys()):
                data_sl = df_res[f"{hub}_service_level"].values

                # --- Gemiddelde-gebaseerde statistieken (Puur informatief voor in het logboek) ---
                gem_sl = np.mean(data_sl)
                fm_sl = 1.96 * (np.std(data_sl, ddof=1) / np.sqrt(cfg.AANTAL_RUNS))
                ci_lower = gem_sl - fm_sl
                ci_upper = gem_sl + fm_sl

                # --- Percentiel-gebaseerde statistieken---
                p05 = np.percentile(data_sl, 5)
                p10 = np.percentile(data_sl, 10)
                p25 = np.percentile(data_sl, 25)
                p50 = np.percentile(data_sl, 50)
                p75 = np.percentile(data_sl, 75)
                p90 = np.percentile(data_sl, 90)
                p95 = np.percentile(data_sl, 95)
                percentiel_beslissing = np.percentile(data_sl, cfg.SL_PERCENTIEL)
                pct_onder_drempel = float(np.mean(data_sl < cfg.SL_DREMPEL) * 100)

                # Sla het gekozen doelniveau op (bijv. P10) om de while-loop aan te sturen
                lower_bounds[hub] = percentiel_beslissing

                # Schrijf een gedetailleerd overzicht naar het logboek
                hub_log = (
                    f"\nLocatie: {hub} (Vloot: {actuele_vloot[hub]} wagens)\n"
                    f"  --- Gemiddelde-gebaseerd (informatief) ---\n"
                    f"  - Gemiddeld Service Level        : {gem_sl:.2f}%\n"
                    f"  - 95% CI van Service Level       : [{ci_lower:.2f}%, {ci_upper:.2f}%]\n"
                    f"  --- Percentiel-gebaseerd (stuurt de vlootbeslissing) ---\n"
                    f"  - P5  Service Level              : {p05:.2f}%\n"
                    f"  - P10 Service Level              : {p10:.2f}%\n"
                    f"  - P25 Service Level              : {p25:.2f}%\n"
                    f"  - P50 Service Level (Mediaan)    : {p50:.2f}%\n"
                    f"  - P75 Service Level              : {p75:.2f}%\n"
                    f"  - P90 Service Level              : {p90:.2f}%\n"
                    f"  - P95 Service Level              : {p95:.2f}%\n"
                    f"  - P{cfg.SL_PERCENTIEL} gebruikt voor beslissing    : {percentiel_beslissing:.2f}%\n"
                    f"  - % jaren onder {cfg.SL_DREMPEL}% drempel        : {pct_onder_drempel:.1f}%\n"
                    f"  --- Overige metrics ---\n"
                    f"  - Gemiddelde Bezettingsgraad     : {df_res[f'{hub}_bezettingsgraad'].mean():.2f}%\n"
                    f"  - Gemiddeld geweigerde ritten    : {df_res[f'{hub}_geweigerd'].mean():.1f} per jaar\n"
                )

                if cfg.ALGORITME == 'FCFS_KEYS' and f'{hub}_no_show' in df_res.columns:
                    hub_log += f"  - Gemiddeld aantal No-Shows      : {df_res[f'{hub}_no_show'].mean():.1f} per jaar\n"

                if cfg.ALGORITME == 'FCFS_WAIT':
                    gem_w = df_res[f'{hub}_gem_wachttijd'].mean()
                    p95_w = df_res[f'{hub}_p95_wachttijd'].mean()
                    max_w = df_res[f'{hub}_max_wachttijd'].max()
                    hub_log += (
                        f"  - Gem. Wachttijd (Geaccepteerd)  : {gem_w * 60:.0f} minuten\n"
                        f"  - 95% Wachttijd (Geaccepteerd)   : {p95_w * 60:.0f} minuten\n"
                        f"  - Max Wachttijd ooit gemeten     : {max_w * 60:.0f} minuten\n"
                    )
                hub_log += f"  - Worst-case jaar (P100)         : {int(df_res[f'{hub}_geweigerd'].max())} ritten geweigerd"
                schrijf_naar_logboek(log_pad, hub_log)

            genereer_iteratie_plots(iteratie_naam, df_res, avg_usage, actuele_vloot, cfg.SL_DREMPEL, cfg.OUTPUT_MAP,
                                    all_wait_times, all_rejected_hours, sl_percentiel=cfg.SL_PERCENTIEL)

            if not cfg.OPTIMALISATIE_AAN:
                schrijf_naar_logboek(log_pad, "\n[Info] Optimalisatie stond uit. Simulatie stopt na 1 run.")
                break

            if finale_run_actief:
                break

            # Evalueer welke hubs kunnen kleinere vloot krijgen
            for hub in list(actieve_hubs):
                if lower_bounds[hub] < cfg.SL_DREMPEL:
                    # Tijdens deze run had deze hub nu te weinig. Voeg er terug 1 toe en stop met optimaliseren van deze hub
                    actuele_vloot[hub] += 1
                    actieve_hubs.remove(hub)
                    schrijf_naar_logboek(log_pad,
                                         f" {hub} zakt onder {cfg.SL_DREMPEL}% (P{cfg.SL_PERCENTIEL}). Vlootgrootte VASTGEZET op {actuele_vloot[hub]} wagens.")
                else:
                    if actuele_vloot[hub] > 1:
                        # Er zijn genoeg wagen, verminder met 1
                        actuele_vloot[hub] -= 1
                        schrijf_naar_logboek(log_pad,
                                             f" {hub} zit veilig op P{cfg.SL_PERCENTIEL}={lower_bounds[hub]:.2f}%. We halen 1 auto weg.")
                    else:
                        actieve_hubs.remove(hub)

            if not actieve_hubs:
                # Alle hubs hebben hun minimum vloot bereikt
                finale_run_actief = True
                schrijf_naar_logboek(log_pad,
                                     f"\n\n{'*' * 70}\n ALLE LOCATIES ZIJN GEOPTIMALISEERD.\n START FINALE VALIDATIE RUN MET EINDVLOOT\n{'*' * 70}")

            iteratie += 1

    schrijf_naar_logboek(log_pad, f"\n\n{'*' * 70}\n EINDOORDEEL BEREIKT\n{'*' * 70}\nEindvloot: {actuele_vloot}\n")
    print(f"\n[OK] Rapport en plots opgeslagen in '{cfg.OUTPUT_MAP}/'")


if __name__ == '__main__':
    main()
