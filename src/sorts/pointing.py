from __future__ import annotations
import logging, typing as t, math
import numpy as np
import numpy.typing as npt
import pandas as pd
import spacecoords
from sorts import types, frames, passage, radar, schedule, interpolation

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

    tx_sch = schedule.from_ndarrays(
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
            schedule.from_ndarrays(
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
    resultant_sch = resultant_sch.sort_values(by=schedule.ScheduleKey.start_time, ignore_index=True)

    return resultant_sch


# TODO: refine the params, e.g:
#   - passage_time_range instead of passage?
#   - state instead of Interpolator?
def sparse_tracking(
    passages_of_spobj: list[passage.Passage],
    interpolator: interpolation.Interpolation,
    points_per_passage: int,
    tx_station: radar.Station,
    rx_stations: t.Sequence[radar.Station],
    exp_id: types.ExperimentId,
    slice_duration: types.Timedelta64_us,
) -> schedule.ScheduleDataframe:
    """Generate a pointing schedule that sparsely tracks the position of a collection space objects."""

    # early return for empty case
    if len(passages_of_spobj) == 0:
        return schedule.empty()

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

    tx_sch = schedule.from_ndarrays(
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
            schedule.from_ndarrays(
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
    resultant_sch = resultant_sch.sort_values(by=schedule.ScheduleKey.start_time, ignore_index=True)

    return resultant_sch


def fence_scanning(
    start_time: types.Datetime64_us,
    end_time: types.Datetime64_us,
    azimuth: types.Float_as_deg,
    min_elevation: types.Float_as_deg,
    scan_range: npt.NDArray[types.Float64_as_m],
    pointings_per_cycle: int,
    tx_station: radar.Station,
    rx_stations: t.Sequence[radar.Station],
    exp_id: types.ExperimentId,
    slice_duration: types.Timedelta64_us,
) -> schedule.ScheduleDataframe:
    """
    Generate a pointing schedule that scan the sky using a circular fence pattern.

    NOTE:
        The param `min_elevation` specify the limit for the desired fence scanning pattern.
        The actual resultant pointings are further limited by the `min_elevation` per stations.
    """

    # TODO: write schedule to disk generally and then chunk load it as needed in the actual
    # simulation, the general simulation pattern will be "1. propagate objects and generate
    # states and make schedule, 2. run simulation, 3. analyze results"

    # The logic of this function:
    # 1. calculate the cycle of tx pointings
    # 2. repeat the cycle of tx pointings to form the tx schedule
    # 3. from the single cycle of tx pointings, we convert it into ECEF location coord and extend them by the `scan_range`
    # 4. using the resultant location coords from previous step,
    #    we convert them to rx station pointings of a cycle in ECEF coord,
    #    and then further back to pointings in ENU coord,
    #    and finally repeat them to form a rx schedule, for each rx station

    tx_schedule_size = math.floor((end_time - start_time) / slice_duration)

    tx_pointings_of_a_cycle = frames.sph_to_cart(
        fence_pattern(
            azimuth=azimuth,
            min_elevation=min_elevation,
            pointings_per_cycle=pointings_per_cycle,
        ),
        degrees=True,
    )

    pointings_per_cycle = tx_pointings_of_a_cycle.shape[1]

    # NOTE: for `np.arange` 'stop param,
    #   - we subtract 'slice_duration' so that only full slice are included
    #   - and add `+1` so that slice with time range `('end_time - 'slice_duration', 'end_time')` is included
    tx_slice_start_time: npt.NDArray[types.Datetime64_us] = np.arange(
        start_time,
        end_time - slice_duration + 1,
        slice_duration,
    )

    # TODO: `tx_schedule_size` is a bit of a mismisnomer, as out-of-range entries might later be removed
    # repeat a cycle of pointings until it is at least the size of `tx_schedule_size`,
    # then trim to exactly `tx_schedule_size` long
    tx_pointing: types.EnuCoordinates = np.tile(
        tx_pointings_of_a_cycle,
        (tx_schedule_size + pointings_per_cycle - 1) // pointings_per_cycle,
    )[:, :tx_schedule_size]

    # mask tx values by min_elevation requirement
    tx_mask = create_mask_by_min_elevation(tx_pointing, tx_station.min_elevation)
    tx_slice_start_time_masked = tx_slice_start_time[tx_mask]
    tx_pointing_masked = tx_pointing[:, tx_mask]

    tx_sch = schedule.from_ndarrays(
        exp_num=np.full(len(tx_slice_start_time_masked), exp_id, dtype=np.int16),
        stn_num=np.full(len(tx_slice_start_time_masked), tx_station.uid, dtype=np.int16),
        simult_num=np.full(len(tx_slice_start_time_masked), 0, dtype=np.int16),
        start_time=tx_slice_start_time_masked,
        end_time=tx_slice_start_time_masked + slice_duration,
        pointing_e=tx_pointing_masked[0, :],
        pointing_n=tx_pointing_masked[1, :],
        pointing_u=tx_pointing_masked[2, :],
    )

    # TODO: `rx_schedule_size` is a bit of a mismisnomer, as out-of-range entries might later be removed
    rx_slice_start_time = tx_slice_start_time.repeat(len(scan_range))
    rx_schedule_size = tx_schedule_size * len(scan_range)
    rx_schs: list[schedule.ScheduleDataframe] = []
    tx_pointings_of_a_cycle_without_translation_ecef: types.EcefCoordinates = frames.enu_to_ecef(
        lat=tx_station.ecef_lat,
        lon=tx_station.ecef_lon,
        alt=tx_station.ecef_alt,
        enu=tx_pointings_of_a_cycle,
        degrees=True,
    )
    rx_pointing_of_a_cycle_ecef: types.EcefCoordinates = (
        tx_pointings_of_a_cycle_without_translation_ecef[:, :, np.newaxis]
        * scan_range[np.newaxis, np.newaxis, :]
        + tx_station.ecef[:, np.newaxis, np.newaxis]
    ).reshape((3, -1))

    for rx_station in rx_stations:
        rx_pointings_of_a_cycle_without_translation_ecef: types.EcefCoordinates = (
            rx_pointing_of_a_cycle_ecef - rx_station.ecef[:, np.newaxis]
        )
        rx_pointings_of_a_cycle_enu: types.EnuCoordinates = frames.ecef_to_enu(
            lat=rx_station.ecef_lat,
            lon=rx_station.ecef_lon,
            alt=rx_station.ecef_alt,
            ecef=rx_pointings_of_a_cycle_without_translation_ecef,
            degrees=True,
        )

        # repeat a cycle of pointings until it is at least the size of `rx_schedule_size`
        # then trim to exactly `rx_schedule_size` long
        rx_pointings_enu: types.EnuCoordinates = np.tile(
            rx_pointings_of_a_cycle_enu,
            (rx_schedule_size + pointings_per_cycle - 1) // pointings_per_cycle,
        )[:, :rx_schedule_size]
        rx_pointings_simult_num = np.arange(rx_schedule_size) % len(scan_range)

        # mask rx values by min_elevation requirement, and has a corresponding tx value
        rx_mask_by_min_elevation = create_mask_by_min_elevation(
            rx_pointings_enu, rx_station.min_elevation
        )
        rx_mask_by_tx_mask = np.isin(rx_slice_start_time, tx_slice_start_time_masked)
        rx_mask = np.logical_and(rx_mask_by_min_elevation, rx_mask_by_tx_mask)
        rx_slice_start_time_masked = rx_slice_start_time[rx_mask]
        rx_pointing_masked = rx_pointings_enu[:, rx_mask]
        rx_pointings_simult_num_masked = rx_pointings_simult_num[rx_mask]

        rx_sch = schedule.from_ndarrays(
            exp_num=np.full(len(rx_slice_start_time_masked), exp_id, dtype=np.int16),
            stn_num=np.full(len(rx_slice_start_time_masked), rx_station.uid, dtype=np.int16),
            simult_num=rx_pointings_simult_num_masked,
            start_time=rx_slice_start_time_masked,
            end_time=rx_slice_start_time_masked + slice_duration,
            pointing_e=rx_pointing_masked[0, :],
            pointing_n=rx_pointing_masked[1, :],
            pointing_u=rx_pointing_masked[2, :],
        )

        rx_schs.append(rx_sch)

    resultant_sch = schedule.ScheduleDataframe((pd.concat([tx_sch, *rx_schs])))
    resultant_sch = resultant_sch.sort_values(by=schedule.ScheduleKey.start_time, ignore_index=True)
    # TODO: re-eval if it is too brutal
    # there will be duplicates if the tx station is also a rx station, we drop the duplicates here
    resultant_sch = resultant_sch.drop_duplicates()
    resultant_sch = resultant_sch.reset_index(drop=True)

    return resultant_sch


def fence_pattern(
    azimuth: types.Float_as_deg,
    min_elevation: types.Float_as_deg,
    pointings_per_cycle: int,
) -> types.AzelrCoordinates_DegM:
    """
    Generate radar pointings that evenly sweep over a symmetrical elevation range, at the given `azimuth`

    Note that the sweeping always strokes in the same direction (not back and forth).
    """

    el = np.linspace(
        min_elevation, 180.0 - min_elevation, num=pointings_per_cycle, dtype=np.float64
    )
    az = np.full(pointings_per_cycle, azimuth, dtype=np.float64)

    # make 0 <= el < 90
    el_over_90deg_mask = el > 90.0
    el[el_over_90deg_mask] = 180.0 - el[el_over_90deg_mask]

    # wrap around az for those with el > 90 deg
    az[el_over_90deg_mask] = np.mod(az[el_over_90deg_mask] + 180.0, 360.0)

    azelr = np.stack(
        [
            az,
            el,
            np.full(pointings_per_cycle, 1.0, dtype=np.float64),
        ],
    )

    return azelr


def create_mask_by_min_elevation(
    pointings: types.EnuCoordinates, min_elevation: float
) -> npt.NDArray[np.bool]:
    loc_zenith = np.array([0, 0, 1], dtype=np.float64)

    _pointings_zenith_ang = spacecoords.linalg.vector_angle(loc_zenith, pointings, degrees=True)
    match _pointings_zenith_ang:
        case np.ndarray():
            pointings_zenith_ang = _pointings_zenith_ang
        case float():
            pointings_zenith_ang = np.array([_pointings_zenith_ang], dtype=np.float64)
        case _:
            raise RuntimeError(
                f"unexpected type of `_pointings_zenith_ang`: {type(_pointings_zenith_ang)}"
            )

    el_in_range_mask = pointings_zenith_ang <= 90.0 - min_elevation

    return el_in_range_mask
