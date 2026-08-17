import pandas as pd
import numpy as np


class CarKeys:
    def __init__(self, car_id, buffer_tijd):
        self.car_id = car_id
        self.schedule = []
        self.buffer_tijd = buffer_tijd

    def clean_no_shows(self, current_time):
        """
        Verwijdert ritten uit de agenda als de klant niet is komen opdagen
        én de starttijd inmiddels verstreken is op de virtuele klok.
        """
        self.schedule = [
            (s, e, no_show) for s, e, no_show in self.schedule
            if not (no_show and current_time >= s)
        ]

    def can_book(self, start, end, current_time):
        # Maak eerst de agenda schoon (verwijder verlopen no-shows)
        self.clean_no_shows(current_time)

        # Check overlap
        for s, e, _ in self.schedule:
            if start < (e + self.buffer_tijd) and end > (s - self.buffer_tijd):
                return False
        return True

    def book(self, start, end, is_no_show):
        self.schedule.append((start, end, is_no_show))


class HubKeys:
    def __init__(self, name, fleet_size, buffer_tijd):
        self.name = name
        self.buffer_tijd = buffer_tijd
        self.cars = [CarKeys(i, self.buffer_tijd) for i in range(1, fleet_size + 1)]


def run_fcfs_keys_sim(df_jobs, actuele_vloot, buffer_tijd, p_no_show):
    hubs = {h: HubKeys(h, vloot_grootte, buffer_tijd) for h, vloot_grootte in actuele_vloot.items()}

    statuses = []
    wait_times = []
    car_ids = []

    for _, job in df_jobs.iterrows():
        h_name = job['hub']
        req_start = job['start_time']
        req_end = job['end_time']
        creation_time = job['creation_time']  # Het moment dat de klant boekt

        # Bepaal direct of de klant de sleutels wel of niet gaat ophalen
        is_no_show = np.random.rand() < p_no_show

        hub = hubs.get(h_name)
        geboekt = False
        geboekte_auto_id = np.nan

        if hub and len(hub.cars) > 0:
            for car in hub.cars:
                # Check of de auto kan, gegeven de huidige virtuele tijd (creation_time)
                if car.can_book(req_start, req_end, creation_time):
                    car.book(req_start, req_end, is_no_show)
                    geboekt = True
                    geboekte_auto_id = car.car_id
                    break

        if geboekt:
            if is_no_show:
                statuses.append('No-Show')
            else:
                statuses.append('Geaccepteerd')
            wait_times.append(0.0)
            car_ids.append(geboekte_auto_id)
        else:
            statuses.append('Geweigerd')
            wait_times.append(np.nan)
            car_ids.append(np.nan)

    df_res = df_jobs.copy()
    df_res['status'] = statuses
    df_res['wait_time'] = wait_times
    df_res['car_id'] = car_ids

    return df_res