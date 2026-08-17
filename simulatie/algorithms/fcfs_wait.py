import pandas as pd
import numpy as np
from core_entities import Hub

def run_fcfs_wait_sim(df_jobs, actuele_vloot, buffer_tijd, max_wachttijd):
    hubs = {h: Hub(h, vloot_grootte, buffer_tijd) for h, vloot_grootte in actuele_vloot.items()}

    statuses = []
    actual_starts = []
    actual_ends = []
    wait_times = []
    car_ids = []

    for _, job in df_jobs.iterrows():
        h_name = job['hub']
        req_start = job['start_time']
        dur = job['duratie_uren']

        hub = hubs.get(h_name)
        if not hub or len(hub.cars) == 0:
            statuses.append('Geweigerd')
            actual_starts.append(req_start)
            actual_ends.append(req_start + dur)
            wait_times.append(np.nan)
            car_ids.append(np.nan)
            continue

        best_car = None
        best_start = None

        for car in hub.cars:
            t_start = car.find_earliest_slot(req_start, dur, max_wachttijd)
            if t_start is not None:
                if best_start is None or t_start < best_start:
                    best_start = t_start
                    best_car = car
                    if best_start == req_start:
                        break

        if best_car is not None:
            best_car.book(best_start, best_start + dur)
            statuses.append('Geaccepteerd')
            actual_starts.append(best_start)
            actual_ends.append(best_start + dur)
            wait_times.append(best_start - req_start)
            car_ids.append(best_car.car_id)
        else:
            statuses.append('Geweigerd')
            actual_starts.append(req_start)
            actual_ends.append(req_start + dur)
            wait_times.append(np.nan)
            car_ids.append(np.nan)

    df_res = df_jobs.copy()
    df_res['status'] = statuses
    df_res['start_time'] = actual_starts
    df_res['end_time'] = actual_ends
    df_res['wait_time'] = wait_times
    df_res['car_id'] = car_ids

    return df_res