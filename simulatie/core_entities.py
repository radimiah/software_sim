class Car:
    def __init__(self, car_id, buffer_tijd):
        self.car_id = car_id

        # De schedule is een lijst van tuples: [(start1, eind1), (start2, eind2), ...]
        self.schedule = []
        self.buffer_tijd = buffer_tijd

    def can_book(self, start, end):
        """ Controleert overlap, inclusief de verplichte buffer rondom ritten. """
        for s, e in self.schedule:
            if start < (e + self.buffer_tijd) and end > (s - self.buffer_tijd):
                return False

        # Geen enkele botsing gevonden in de hele agenda? Dan is hij vrij.
        return True

    def book(self, start, end):
        """ Voegt de goedgekeurde rit toe aan de kalender van deze specifieke auto. """
        self.schedule.append((start, end))

    def find_earliest_slot(self, req_start, req_dur, max_wait):
        """
         zoekfunctie voor het 'FCFS_WAIT' algoritme.
         Als een auto niet direct vrij is, kijkt deze functie of hij binnen het
         buffer ('max_wait') van de werknemer toch nog ergens vrijkomt.

         Returnt de verschoven starttijd (float) of None (geweigerd).
         """

        # 1. Ideale: Past de rit perfect op het exact gewenste startmoment?
        if self.can_book(req_start, req_start + req_dur):
            return req_start

        # 2. Zo niet, dan hoeven we niet blindelings elke minuut te checken.
        # Er moet enkel gecontroleerd worden op het moment dat een eerdere rit eindigt.
        best_start = None
        for s, e in self.schedule: # s = start en e = end

            # 'possible_start' is het vroegste moment waarop we in theorie de auto kunnen overnemen
            possible_start = e + self.buffer_tijd

            # Check 1: Valt dit nieuwe startmoment wel in de toekomst en binnen de buffer (max_wait) van de werknemer?
            if req_start < possible_start <= req_start + max_wait:

                # Check 2: Als we de rit opschuiven naar dit nieuwe startmoment, kan de klant dan zijn rit
                # nog steeds uitvoeren zonder overlap met andere ritten?
                if self.can_book(possible_start, possible_start + req_dur):
                    if best_start is None or possible_start < best_start:
                        best_start = possible_start # Behoudt enkel de vroegste mogelijke tijd

        return best_start


class Hub:
    def __init__(self, name, fleet_size, buffer_tijd):
        self.name = name
        self.buffer_tijd = buffer_tijd
        self.cars = [Car(i, self.buffer_tijd) for i in range(1, fleet_size + 1)]