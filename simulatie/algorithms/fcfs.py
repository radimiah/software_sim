import pandas as pd
import numpy as np
from core_entities import Hub


def run_fcfs_sim(df_jobs, actuele_vloot, buffer_tijd):
    hubs = {h: Hub(h, vloot_grootte, buffer_tijd) for h, vloot_grootte in actuele_vloot.items()}

    statuses = []
    wait_times = []
    car_ids = []  # NIEUW: Auto tracking

    for _, job in df_jobs.iterrows():
        h_name = job['hub']
        req_start = job['start_time']
        req_end = job['end_time']

        hub = hubs.get(h_name)
        geboekt = False
        geboekte_auto_id = np.nan

        if hub and len(hub.cars) > 0:
            for car in hub.cars:
                if car.can_book(req_start, req_end):
                    car.book(req_start, req_end)
                    geboekt = True
                    geboekte_auto_id = car.car_id
                    break

        if geboekt:
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