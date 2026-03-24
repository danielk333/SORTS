from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
import spacecoords
from sorts import types, passage, radar, schedule, interpolation

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
    tx_pointings: types.EnuCoordinates = tx_station.enu(spobj_ecef_states[:3])

    tx_pointings_zenith_ang = spacecoords.linalg.vector_angle(
        loc_zenith, tx_pointings, degrees=True
    )
    tx_el_in_range_mask = tx_pointings_zenith_ang <= 90.0 - tx_station.min_elevation
    tx_pointings = tx_pointings[:, tx_el_in_range_mask]

    rxs_pointings: list[types.EnuCoordinates] = []
    rx_el_in_range_with_tx_masks: list[npt.NDArray[np.bool]] = []
    pure_rx_stations = [stn for stn in rx_stations if stn.uid != tx_station.uid]
    for rx_station in pure_rx_stations:
        rx_pointings: types.EnuCoordinates = rx_station.enu(spobj_ecef_states[:3])

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


# TODO: refine the params, e.g:
#   - passage_time_range instead of passage?
#   - state instead of Interpolator?
def sparse_tracking(
    passages_of_spobj: list[passage.Passage],
    interpolator: interpolation.Interpolator,
    points_per_passage: int,
    tx_station: radar.Station,
    rx_stations: t.Sequence[radar.Station],
    exp_id: types.ExperimentId,
    slice_duration: types.Timedelta64_us,
) -> schedule.ScheduleDataframe:
    """Generate a pointing schedule that sparsely tracks the position of a collection space objects."""

    # early return for empty case
    if len(passages_of_spobj) == 0:
        return schedule.schedule_dataframe.empty()

    observation_times_relative = []
    observation_times = []
    min_time_needed = points_per_passage * slice_duration / np.timedelta64(1, "s")
    for ps in passages_of_spobj:
        pstart_time, pend_time = ps.time_range
        t0 = (pstart_time - ps.epoch) / np.timedelta64(1, "s")
        passage_time = (pend_time - pstart_time) / np.timedelta64(1, "s")
        if passage_time <= min_time_needed:
            continue

        relative_time_sampling = np.linspace(
            0.0, passage_time, num=points_per_passage + 2, endpoint=True
        )
        relative_time_sampling = relative_time_sampling[1:-1]

        observation_times_relative.append(t0 + relative_time_sampling)
        rel_us = (relative_time_sampling.copy() * 1e6).astype("timedelta64[us]")

        observation_times.append(pstart_time + rel_us)

    # TODO: this can be cleaned up quite a lot i feel like
    tx_sch_time_rel = np.concatenate(observation_times_relative)
    tx_sch_time = np.concatenate(observation_times)
    tx_sch_len = len(tx_sch_time)
    tx_pointings: types.EnuCoordinates = tx_station.enu(
        interpolator.get_state(tx_sch_time_rel)[:3, :]
    )
    tx_pointings = tx_pointings / np.linalg.norm(tx_pointings, axis=0)

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
    for rx_stn in rx_stations:
        rx_pointings: types.EnuCoordinates = rx_stn.enu(
            interpolator.get_state(tx_sch_time_rel)[:3, :]
        )
        rx_pointings = rx_pointings / np.linalg.norm(rx_pointings, axis=0)

        rx_schs.append(
            schedule.schedule_dataframe.from_ndarrays(
                exp_num=np.full(tx_sch_len, exp_id, dtype=np.int16),
                stn_num=np.full(tx_sch_len, rx_stn.uid, dtype=np.int16),
                simult_num=np.full(tx_sch_len, 0, dtype=np.int16),
                start_time=tx_sch_time,
                end_time=tx_sch_time + slice_duration,
                pointing_e=rx_pointings[0, :],
                pointing_n=rx_pointings[1, :],
                pointing_u=rx_pointings[2, :],
            )
        )

    resultant_sch = schedule.ScheduleDataframe((pd.concat([tx_sch, *rx_schs])))
    resultant_sch = resultant_sch.sort_values(by=schedule.ScheduleKey.start_time)

    return resultant_sch
