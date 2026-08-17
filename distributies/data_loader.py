"""
data_loader.py

Dit is het centrale data lader van de simulatie. Alle andere scripts
(volume, starttijden, duren) halen hun data hier vandaan.
"""

import json
import pandas as pd
import numpy as np

# Globale constanten die in meerdere scripts gebruikt worden
VASTE_HUBS = ["Punt_1", "Punt_2", "Punt_3", "Punt_4"]
DAG_NAAR_NAAM = {0: 'Maandag', 1: 'Dinsdag', 2: 'Woensdag', 3: 'Donderdag', 4: 'Vrijdag'}


def laad_basis_data(json_path):
    """
    Leest de ruwe JSON in en filtert op:
    - Enkel werkdagen (geen weekenden)
    - Enkel auto's (vehicleTypeId == 4)
    - Enkel ritten die starten bij 1 van de 4 vaste hubs
    """
    print("[Data Loader] Historische data inladen...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[FOUT] Bestand '{json_path}' niet gevonden.")
        return pd.DataFrame()

    records = []

    # Koppel de namen aan de exacte GPS coördinaten (latitude, longitude)
    coords_map = {
        "Punt_1": (51.0466, 3.7295), "Punt_2": (51.0653, 3.7303),
        "Punt_3": (51.0794, 3.7460), "Punt_4": (51.0449, 3.7081)
    }

    for job in data.get('jobs', []):
        # 1. Filter: We willen alleen personenwagens simuleren (geen fietsen of bestelbusjes)
        if job.get('vehicleTypeId') == 4:
            s_str = job.get('period', {}).get('fromDate')
            e_str = job.get('period', {}).get('toDate')
            if not s_str or not e_str: continue

            # 2. Tijdzones: Zorg dat alles in Belgische tijd staat (zomertijd/wintertijd correcties)
            try:
                s_dt = pd.to_datetime(s_str, utc=True).tz_convert('Europe/Brussels')
                e_dt = pd.to_datetime(e_str, utc=True).tz_convert('Europe/Brussels')
            except Exception:
                continue

            # 3. Filter: richt op enkel op werknemers tijdens de werkweek
            if s_dt.weekday() >= 5: continue
            if e_dt <= s_dt: continue

            # Haal GPS data op
            lat = job.get('coords', {}).get('latitude', 0)
            lon = job.get('coords', {}).get('longitude', 0)

            # 4. Geografische Mapping (Bounding Box)
            # We zoeken of de GPS-locatie binnen een straal van ~500 meter (0.005 graden) van onze vaste hubs ligt.
            gekozen_hub = next((hub for hub, (h_lat, h_lon) in coords_map.items()
                                if abs(lat - h_lat) < 0.005 and abs(lon - h_lon) < 0.005), None)

            if gekozen_hub:
                duratie_uren = (e_dt - s_dt).total_seconds() / 3600.0
                # 5. Filter: Haal boekingen (korter dan 3 minuten) eruit.
                if duratie_uren > 0.05 and gekozen_hub != 'Onbekend':
                    records.append({
                        'date': s_dt.date(),
                        'weekday': s_dt.weekday(),
                        'hub': gekozen_hub,
                        'start_hour': s_dt.hour,
                        'time_float': s_dt.hour + (s_dt.minute / 60.0),
                        'minute': s_dt.minute,
                        'duratie_uren': duratie_uren
                    })

    return pd.DataFrame(records)


def laad_dag_tellingen(json_path):
    df = laad_basis_data(json_path)
    if df.empty: return None

    dag_tellingen = df.groupby(['date', 'weekday', 'hub']).size().reset_index(name='aantal_ritten')
    dag_tellingen['date'] = pd.to_datetime(dag_tellingen['date'])

    alle_werkdagen = pd.date_range(dag_tellingen['date'].min(), dag_tellingen['date'].max(), freq='B')
    idx = pd.MultiIndex.from_product([alle_werkdagen, VASTE_HUBS], names=['date', 'hub'])

    df_volledig = pd.DataFrame(index=idx).reset_index()
    df_volledig['weekday'] = df_volledig['date'].dt.weekday

    df_eind = pd.merge(df_volledig, dag_tellingen, on=['date', 'weekday', 'hub'], how='left').fillna({'aantal_ritten': 0})
    df_eind['aantal_ritten'] = df_eind['aantal_ritten'].astype(int)

    return df_eind