import pandas as pd
import numpy as np

def genereer_jobs_voor_jaar(sampler, dagen, vraag_multiplier):
    """
    Genereert een DataFrame met jobs voor een bepaald aantal dagen.
    Sorteert deze chronologisch op Boekingsmoment (Creation Time).
    """
    records = []
    for dag_nr in range(dagen):
        wd = dag_nr % 5
        week_nr = dag_nr // 5
        absolute_start_van_de_dag = (week_nr * 7 + wd) * 24.0

        vol_dict = sampler.sample_volume(wd)

        for hub, n_jobs in vol_dict.items():
            # Pas de vraag multiplier toe voor what-if scenario's
            n_jobs_adjusted = int(np.floor(n_jobs * vraag_multiplier))
            if np.random.rand() < (n_jobs * vraag_multiplier - n_jobs_adjusted):
                n_jobs_adjusted += 1

            for _ in range(n_jobs_adjusted):
                st_offset = sampler.sample_starttijd(wd, hub)
                du = sampler.sample_duur(wd, st_offset)
                lt = sampler.sample_leadtime()

                start_abs = absolute_start_van_de_dag + st_offset
                eind_abs = start_abs + du
                creation = start_abs - lt

                records.append({
                    'hub': hub,
                    'dag_nr': dag_nr,
                    'week_nr': week_nr,
                    'creation_time': max(0.0, creation),
                    'start_time': start_abs,
                    'end_time': eind_abs,
                    'duratie_uren': du,
                    'lead_time': lt
                })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values('creation_time').reset_index(drop=True)
    return df