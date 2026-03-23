from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
import spacecoords
from sorts.types import EnuCoordinates
from sorts import types, radar, schedule

logger = logging.getLogger(__name__)


def tracking(
    spobj_ecef_states: types.EcefStates,
    spobj_ecef_states_times: npt.NDArray[types.Datetime64_us],
    tx_station: radar.Station,
    rx_stations: t.Sequence[radar.Station],
    exp_id: types.ExperimentId,
    slice_duration: types.Timedelta64_us,
) -> schedule.ScheduleDataframe:
    """Generate a pointing schedule that tracks the position of a space object."""

    loc_zenith = np.array([0, 0, 1], dtype=np.float64)

    # generate pointings
    tx_pointings: EnuCoordinates = tx_station.enu(spobj_ecef_states[:3])

    tx_pointings_zenith_ang = spacecoords.linalg.vector_angle(
        loc_zenith, tx_pointings, degrees=True
    )
    tx_el_in_range_mask = tx_pointings_zenith_ang <= 90.0 - tx_station.min_elevation
    tx_pointings = tx_pointings[:, tx_el_in_range_mask]

    rxs_pointings: list[EnuCoordinates] = []
    rx_el_in_range_with_tx_masks: list[npt.NDArray[np.bool]] = []
    pure_rx_stations = [stn for stn in rx_stations if stn.uid != tx_station.uid]
    for rx_station in pure_rx_stations:
        rx_pointings: EnuCoordinates = rx_station.enu(spobj_ecef_states[:3])

        rx_pointings_zenith_ang = spacecoords.linalg.vector_angle(
            loc_zenith, rx_pointings, degrees=True
        )
        rx_el_in_range_mask = rx_pointings_zenith_ang <= 90.0 - rx_station.min_elevation

        rx_el_in_range_with_tx_mask = np.logical_and(tx_el_in_range_mask, rx_el_in_range_mask)
        rx_el_in_range_with_tx_masks.append(rx_el_in_range_with_tx_mask)

        rx_pointings = rx_pointings[:, rx_el_in_range_with_tx_mask]
        rxs_pointings.append(rx_pointings)

    tx_sch_time = spobj_ecef_states_times[tx_el_in_range_mask]
    tx_sch_len = len(tx_sch_time)

    tx_sch = schedule.schedule_dataframe.from_ndarrays(
        exp_num=np.full(tx_sch_len, exp_id, dtype=np.int16),
        stn_num=np.full(tx_sch_len, tx_station.uid, dtype=np.int16),
        simult_num=np.full(tx_sch_len, 0, dtype=np.int16),
        start_time=tx_sch_time,
        end_time=tx_sch_time + slice_duration,
        pointing_e=tx_pointings[0, :],
        pointing_n=tx_pointings[1, :],
        pointing_u=tx_pointings[2, :],
    )

    rx_schs: list[schedule.ScheduleDataframe] = []
    for rx_stn, rx_mask, rx_pointings in zip(
        pure_rx_stations, rx_el_in_range_with_tx_masks, rxs_pointings
    ):
        rx_sch_time = spobj_ecef_states_times[rx_mask]
        rx_sch_len = len(rx_sch_time)

        rx_schs.append(
            schedule.schedule_dataframe.from_ndarrays(
                exp_num=np.full(rx_sch_len, exp_id, dtype=np.int16),
                stn_num=np.full(rx_sch_len, rx_stn.uid, dtype=np.int16),
                simult_num=np.full(rx_sch_len, 0, dtype=np.int16),
                start_time=rx_sch_time,
                end_time=rx_sch_time + slice_duration,
                pointing_e=rx_pointings[0, :],
                pointing_n=rx_pointings[1, :],
                pointing_u=rx_pointings[2, :],
            )
        )

    resultant_sch = schedule.ScheduleDataframe((pd.concat([tx_sch, *rx_schs])))
    resultant_sch = resultant_sch.sort_values(by=schedule.ScheduleKey.start_time)
    resultant_sch = resultant_sch.reset_index()

    return resultant_sch
