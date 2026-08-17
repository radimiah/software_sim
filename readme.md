## Projectstructuur

Het project bestaat uit twee fases: **1. Datamodellering** en **2. Vlootsimulatie**.

### Fase 1: Datamodellering (Trainen van de verdelingen)
Deze scripts analyseren de historische data (`swsim-2026-02-26.json`) en slaan de wiskundige parameters op in JSON-bestanden in de map `distributies/`.
* `data_loader.py` - Centraal script om de ruwe JSON data in te laden, te filteren (enkel werkdagen, auto's) en aan GPS-hubs te koppelen.
* `01_aantal_jobs.py` - Fit Poisson/ZIP/DMM modellen op dagelijks volume en berekent ruimtelijke correlatie.
* `02_starttijd.py` - Maakt KDE-modellen (Kernel Density) met een hybride afrondingsmechanisme (snapping) voor starttijden.
* `03_duur.py` - Berekent KDE-modellen voor de ritduur, opgedeeld in tijdsblokken (buckets).
* `03_duur_params_vinden.py` - Optioneel dashboard om visueel de beste KDE-bandwidths voor de ritduren te vinden.
* `04_leadtimes.py` - Genereert een Log-Normaal Mixture model dat bepaalt hoe ver op voorhand ritten worden geboekt.

### Fase 2: Simulatie & Optimalisatie
Deze scripts gebruiken de getrainde modellen om duizenden scenario's af te spelen.
* `config.py` - **Centraal configuratiebestand**. Hier stel je het aantal runs, de doelstelling (SL_DREMPEL), het algoritme en de startvloot in.
* `sampler.py` - Gebruikt de parameters uit Fase 1 om willekeurige, realistische virtuele boekingen te genereren.
* `job_generator.py` - Zet de output van `sampler.py` om in een chronologische dataset (kalender) voor één simulatiejaar.
* `core_entities.py` - Bevat de logica van een Hub en een Auto (agendabeheer, botsingen detecteren).
* `metrics_logger.py` - Berekent prestaties (Service Level, wachttijd, no-shows) en genereert de grafieken/dashboards.
* `simulatie.py` - **Het hoofdprogramma**. Genereert jobs over tientallen jaren (via multiprocessing) en haalt iteratief auto's weg totdat het Service Level onder de drempel zakt.

### Algoritmes (`algorithms/`)
* `fcfs.py` - Standaard First-Come-First-Served (auto vrij = boeken, geen auto = weigeren).
* `fcfs_wait.py` - Werknemers kunnen schuiven/wachten op een auto binnen een maximale wachttijd.
* `fcfs_keys.py` - Houdt rekening met "No-Shows" (klanten boeken wel, maar halen sleutel niet op).

---

## Hoe te gebruiken (Stappenplan)

### Stap 1: Benodigdheden installeren
Zorg dat je Python geïnstalleerd hebt. Installeer de benodigde packages via je terminal:
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn dash
```

### Stap 2: Data plaatsen
Zorg dat het historische ruwe databestand (`swsim-2026-02-26.json`) in de distributie-map van het project staat.

### Stap 3: Modellen Trainen (Eenmalig)
Run de onderstaande scripts één voor één. Ze genereren grafieken en maken de nodige `.json` parameterbestanden aan.
(Alle bestanden zijn al in de repository aanwezig, dus dit is optioneel tenzij je de distributies wilt hertrainen.)
```bash
python 01_aantal_jobs.py
python 02_starttijd.py
python 03_duur.py
python 04_leadtimes.py
```

### Stap 4: Configuratie instellen
Open `config.py` om je test-scenario te bepalen.
Belangrijke instellingen:
* `ALGORITME` = Kies tussen `'FCFS'`, `'FCFS_WAIT'`, of `'FCFS_KEYS'`
* `AANTAL_RUNS` = Aantal jaren om te simuleren per iteratie
* `SL_DREMPEL` = Het gewenste Service Level (bijv. `95.0`%)
* `OPTIMALISATIE_AAN` = Zet op `True` om de vloot automatisch af te bouwen tot het minimum.

### Stap 5: Simulatie Starten
Draai het hoofdscript. Het script zal je CPU cores gebruiken om de jaren door te rekenen.
```bash
python simulatie.py
```

---

## Output & Resultaten
Zodra `simulatie.py` klaar is, wordt er een nieuwe map aangemaakt (bijv. `simulatie_resultaten-FCFS_KEYS/`).
Hierin vind je:
1. **`iteratie_rapport.txt`**: Een gedetailleerd logboek met de weggeschreven vlootgrootte en percentiel-statistieken per iteratie.
2. **Grafieken (.png)**:
   * Service Level Boxplots en Percentiel grafieken
   * Gelijktijdig Gebruik (capaciteit) per Hub
   * Risico op een slecht jaar (kans dat Service level onder je drempel zakt)
   * Wachttijden en No-show verdelingen (afhankelijk van je algoritme)
