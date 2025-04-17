from datetime import datetime
import numpy as np
from .schedule_v2 import Schedule
import sorts
from sorts.radar.radar import Radar
from sorts.radar.tx_rx import Station
from sorts.passes_v2 import Pass

# from sorts.radar.radar_v2 import Radar # TODO: imple and use Radar v2


def calculate_observation(
    tx_station: Station,
    rx_station: Station,
    schedule: Schedule,
    space_object: sorts.SpaceObject,
    tstmp: datetime,
    # stt_tstmp: datetime,
    # end_tstmp: datetime,
    # interpolator=None, # TODO: add interpolator support?
):
    """
    based on `src/sorts/scheduler/observed_parameters.py` -> `calculate_observation()` atm

    wip
    """

    diam = space_object.d
    spin_period = space_object.parameters.get("spin_period", None)
    radar_albedo = space_object.parameters.get("radar_albedo", 1.0)

    states = space_object.get_state(tstmp)

    snr = np.empty((len(schedule.stt_tstmp_us),), dtype=np.float64)
    snr_inch = np.empty((len(schedule.stt_tstmp_us),), dtype=np.float64)
    rcs = np.empty((len(schedule.stt_tstmp_us),), dtype=np.float64)
    keep = np.full((len(schedule.stt_tstmp_us),), True, dtype=bool)

    enus = [
        tx_station.enu(states),
        rx_station.enu(states),
    ]
    ranges = [Pass.calculate_range(enu) for enu in enus]
    range_rates = [Pass.calculate_range_rate(enu) for enu in enus]
