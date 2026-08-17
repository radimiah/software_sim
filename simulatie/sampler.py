"""
sampler.py

Dit script is de 'rit-Generator' van de simulatie. Het leest alle wiskundige modellen
in die we eerder hebben getraind (volume, starttijd, duur en lead-time) en gebruikt deze
om virtuele reserveringen te genereren.
"""

import os
import json
import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity


class SWSampler:
    def __init__(self):
        """
        Laadt alle JSONs in en 'pre-fit' de KDE modellen zodat dit niet
        tijdens de simulatie loop hoeft te gebeuren.
        """
        print("[Sampler] Initialiseren en Modellen in geheugen laden...")

        # 1. Laad Aantal Jobs (Volume & Correlaties)
        with open('../distributies/01_aantal_jobs_models_params.json', 'r') as f:
            j_jobs = json.load(f)
            self.anker_hub = j_jobs['anker_hub']
            self.jobs_modellen = j_jobs['modellen']
            self.jobs_condities = j_jobs['condities']

        # 2. Laad Starttijden (KDE data per dag per hub)
        with open('../distributies/02_starttijd_models_params.json', 'r') as f:
            self.start_modellen = json.load(f)
            self.start_kdes = {}
            for dag, hubs in self.start_modellen.items():
                self.start_kdes[dag] = {}
                for hub, info in hubs.items():
                    data = np.array(info['kde_data'])
                    if len(data) > 0:
                        # Bouw de Kernel Density Estimator direct op in het geheugen
                        self.start_kdes[dag][hub] = stats.gaussian_kde(data, bw_method=0.1)

        # 3. Laad Duren (KDE data per tijdslot/bucket)
        with open('../distributies/03_duur_distributies.json', 'r') as f:
            self.duur_modellen = json.load(f)
            self.duur_kdes = {}
            for key, info in self.duur_modellen.items():
                if info.get('kde_data') and len(info['kde_data']) > 0:
                    data = np.array(info['kde_data']).reshape(-1, 1)
                    # Train de KDE met de optimaal gevonden bandbreedte (RMSE geoptimaliseerd)
                    self.duur_kdes[key] = KernelDensity(kernel='gaussian', bandwidth=info['kde_bandwidth']).fit(data)

        # 4. Laad Leadtime (Log-Normaal Mixture Model)
        with open('../distributies/04_leadtime_distributie.json', 'r') as f:
            lt_data = json.load(f)
            self.lt_comps = lt_data['components']
            self.lt_names = list(self.lt_comps.keys())

            # Kansen normaliseren (zodat de som exact 1.0 is voor numpy random choice)
            gewichten = np.array([c['weight'] for c in self.lt_comps.values()])
            self.lt_weights = gewichten / gewichten.sum()

    # ═════════════════════════════════════════════════════════
    # SAMPLING FUNCTIES
    # ═════════════════════════════════════════════════════════
    def _sample_discreet(self, mod, p):
        """ Hulpfunctie om zuivere wiskundige waarden te trekken uit de discrete modellen. """
        if mod == 'Poisson': return stats.poisson.rvs(p['mu'])
        if mod == 'NBD': return stats.nbinom.rvs(p['n'], p['p'])
        return 0 # Fallback bij foute model-string

    def sample_volume(self, dag_idx: int):
        """
        Bepaalt het aantal jobs per hub voor een specifieke dag.
        Onderhoudt de ruimtelijke correlatie via het Anker-Hub principe.
        """
        wd = str(dag_idx)
        vol_result = {}

        # ---------------------------------------------------------
        # Stap 1: Bepaal status Anker Hub (Is het een dip/feestdag?)
        # ---------------------------------------------------------
        anker_p_dip = self.jobs_modellen[wd][self.anker_hub]['p_dip']
        # Genereer een getal tussen 0.0 en 1.0 .
        # Is het getal kleiner dan de dip-kans? dan is het een dip dag (true), anders is het een normale dag.
        is_anker_dip = np.random.rand() < anker_p_dip

        # Haal het complete wiskundige model op voor Punt_1 voor deze specifieke dag.
        mod = self.jobs_modellen[wd][self.anker_hub]

        # Check welk soort wiskundig model het best paste bij Punt_1:
        if mod['model'] in ['ZIP', 'ZINB']:
            # Zero-Inflated: Als dip = True, is de hub dicht/extreem rustig (0 ritten).
            if is_anker_dip:
                vol_result[self.anker_hub] = 0
            else:
                vol_result[self.anker_hub] = stats.poisson.rvs(mod['params']['mu'])

        elif mod['model'] == 'DMM':
            # Discrete Mixture Model: Trek uit de 'rustige' of de 'drukke' curve.
            if is_anker_dip:
                vol_result[self.anker_hub] = stats.nbinom.rvs(mod['params']['n1'], mod['params']['p1'])
            else:
                vol_result[self.anker_hub] =  stats.nbinom.rvs(mod['params']['n2'], mod['params']['p2'])

        else:
            # Als het een heel simpel model is zonder dips (gewoon Poisson of NBD), gebruik de hulpfunctie van hierboven.
            vol_result[self.anker_hub] = self._sample_discreet(mod['model'], mod['params'])


        # ---------------------------------------------------------
        # Stap 2: Bepaal volume voor andere hubs afhankelijk van het Anker
        # ---------------------------------------------------------
        for hub in ["Punt_1", "Punt_2", "Punt_3", "Punt_4"]:
            if hub == self.anker_hub: continue

            # Conditionele kans: was het anker een dip? Gebruik dan de bijbehorende kans voor deze hub.
            if is_anker_dip:
                kans_op_dip = self.jobs_condities[wd][hub]['als_anker_is_dip']
            else:
                kans_op_dip = self.jobs_condities[wd][hub]['als_anker_is_normaal']

            # Genereer nu terug een getal of deze specifieke hub echt een dipdag is met de zojuist gevonden kans
            is_dip = np.random.rand() < kans_op_dip

            mod = self.jobs_modellen[wd][hub]   # Haal het wiskundige model op voor deze specifieke hub.

            if mod['model'] in ['ZIP', 'ZINB']:
                if is_dip:
                    vol_result[hub] = 0
                else:
                    vol_result[hub] = stats.poisson.rvs(mod['params']['mu'])

            elif mod['model'] == 'DMM':
                if is_dip:
                    vol_result[hub] = stats.nbinom.rvs(mod['params']['n1'], mod['params']['p1'])
                else:
                    vol_result[hub] = stats.nbinom.rvs(mod['params']['n2'], mod['params']['p2'])
            else:
                vol_result[hub] = self._sample_discreet(mod['model'], mod['params'])

        # Voorbeeld output: {"Punt_1": 15, "Punt_2": 8, "Punt_3": 12, "Punt_4": 4}
        return vol_result

    def sample_starttijd(self, dag_idx: int, hub: str) -> float:
        """ Genereert 1 startuur (Hybride: KDE + Snapping) """

        wd = str(dag_idx)

        # Check: als deze locatie niet is opgeslagenin de modellen, return 12 om een crash te voorkomen
        if hub not in self.start_modellen[wd]: return 12.0

        # Haal de kans op dat iemand op een exact "rond" halfuur boekt (bijv. 08:00 of 08:30).
        p_half = self.start_modellen[wd][hub]['p_half_hour']

        # Haal het getrainde, continue KDE-model op uit het geheugen.
        kde = self.start_kdes[wd].get(hub)

        # Nog een veiligheidscheck: Als het model niet geladen is, retourneer 12:00.
        if kde is None: return 12.0

        # Trek een willekeurige ruwe tijd uit de vloeiende KDE (bijv. 8.347 uur).
        # Modulo 24 zorgt dat waardes > 24 netjes net na middernacht vallen.
        sample = kde.resample(1)[0][0] % 24.0

        # Snapping: Beslis op basis van historische kans of deze specifieke klant afrondt.
        if np.random.rand() < p_half: # voeg snapping toe (afronden)
            # Vermenigvuldig 8.347 met 2 (=16.694), rond af (=17.0), deel door 2 (=8.5). (% 24.0 voor de zekerheid rond middernacht).
            return float(np.round(sample * 2) / 2) % 24.0

        return float(sample) # geen afronding nodig, geen de exacte tijd terug

    def sample_duur(self, dag_idx: int, start_hour: float) -> float:
        """
        Genereert 1 duur. Afhankelijk van de dag en het startuur, zoekt deze
        het juiste tijdsblok (bucket) op, omdat een rit om 08:00 een ander
        duur-patroon heeft dan een rit om 22:00.
        """
        wd = str(dag_idx)
        st_bin = int(np.floor(start_hour))
        gevonden_key = None

        # Zoek in welke tijdsbucket (bijv. "0_0_5" = Maandag 00:00 - 05:00) dit startuur valt
        for key in self.duur_modellen.keys():
            parts = key.split('_')
            # Controleer of de dag klopt en of de startuur (st_bin) precies tussen de start en eindtijd van deze bucket ligt.
            if len(parts) == 3 and parts[0] == wd:
                b_st = int(parts[1])
                b_en = int(parts[2])
                if b_st <= st_bin <= b_en:
                    gevonden_key = key
                    break

        # Haal alle parameters voor deze specifieke tijdsperiode op uit de JSON.
        info = self.duur_modellen.get(gevonden_key)
        if not info: return 2.0  # Fallback

        if gevonden_key in self.duur_kdes:
            raw = self.duur_kdes[gevonden_key].sample(1)[0][0]

            # Snappen (afronden op halve uren) met de p_30_snap waarschijnlijkheid
            # Snap (afronden op half uur) met de p_30_snap waarschijnlijkheid
            if np.random.rand() < info.get('p_30_snap', 0.5):
                return float(np.clip(np.round(raw * 2) / 2, 0.05))

            return float(np.clip(raw, 0.05))

        return 2.0 # Fallback

    def sample_leadtime(self) -> float:
        """
        Genereert 1 leadtime uit het Log-Normal Mixture Model.
        Bepaalt hoever van tevoren de medewerker de boeking in het systeem zet.
        """

        # 1. Bepaal in welke van de 3 groepen de klant valt (Last-minute, Vorige-Dag, Vorige-Week)
        keuze = np.random.choice(self.lt_names, p=self.lt_weights)
        c = self.lt_comps[keuze]

        # 2. Trek een getal uit een Log-Normale verdeling (ideaal voor tijd: geen negatieve getallen, lange staart rechts)
        lt = np.random.lognormal(mean=c['mu'], sigma=c['sigma'])

        # Begrens tussen 6 minuten vooraf en 30 dagen vooraf
        return float(np.clip(lt, 0.1, 720.0))