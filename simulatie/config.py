# --- SIMULATIE PARAMETERS ---
AANTAL_RUNS = 100          # Aantal simulatiejaren per iteratie/run
AANTAL_DAGEN = 260        # 1 jaar aan werkdagen ( 52 * 5 )
BUFFER_TIJD_UREN = 0.25   # 15 minuten poets/buffer tussen ritten
VRAAG_MULTIPLIER = 1.0    # Test scenario's (bv. 1.2 voor 20% meer vraag)

# --- ALGORITME PARAMETERS ---
# Opties: 'FCFS', 'FCFS_WAIT', 'FCFS_KEYS'
ALGORITME = 'FCFS_KEYS'
MAX_WACHTTIJD_UREN = 0.30  # Maximaal aantal uren dat iemand bereid is te wachten (bijv. 0.5u = 30min)
KEYS_NOT_TAKEN = 0.0817  # percentage van hoeveel mensen nooit komen opdagen (op basis van historische data)

OUTPUT_MAP = f'simulatie_resultaten-{ALGORITME}'

# --- STRATEGISCHE BESLISSINGSVARIABELEN ---
OPTIMALISATIE_AAN = True
SL_DREMPEL = 95.0  # minimum waarde voor service level

# Welk percentiel van de gesimuleerde jaren moet minstens de SL_DREMPEL halen?
# Bv. 10 => het 10e percentiel (dus 90% van de jaren) moet >= SL_DREMPEL zijn.
# Lager getal = strenger/conservatiever (grotere vloten), hoger getal = losser.
SL_PERCENTIEL = 10

START_VLOOT_CONFIG = {
    "Punt_1": 20,
    "Punt_2": 14,
    "Punt_3": 10,
    "Punt_4": 7
}