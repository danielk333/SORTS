from __future__ import annotations
import typing as t
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import numpy.typing as npt
import sorts
from sorts.interpolation import Interpolator
from sorts.radar.radars.composite_key import RadarStationCompositeKey
from sorts.radar.tx_rx import Station
from sorts.types import Datetime64_us, Float64_as_sec, Float64_as_m, EcefStates
from sorts import schedule_v2 as schedule
from sorts import scheduler_v2 as scheduler
from sorts.detection_config import ExperimentDetail, Observation


# TODO: rename to `DetectionSystem`?
class DetectionConfigProtocol(t.Protocol):
    def find_passes_time_ranges(
        self,
        dt_s_arr: npt.NDArray[np.float64],
        space_object_states: npt.NDArray[np.float64],
        epoch: datetime,
    ) -> t.Sequence[tuple[Datetime64_us, Datetime64_us]]: ...

    def get_schedule_mask_by_time_range(
        self, time_range: tuple[Datetime64_us, Datetime64_us]
    ) -> npt.NDArray[np.bool]: ...

    def calculate_observation(
        self,
        space_object: sorts.SpaceObject,
        space_object_states_interpolator: Interpolator,
        epoch: datetime,
        schedule_mask: npt.NDArray[np.bool] | None,
    ) -> Observation: ...


@dataclass(kw_only=True)
class SimpleStxSrx(DetectionConfigProtocol):
    tx_station: Station
    tx_schedule: scheduler.Schedule
    rx_station: Station
    rx_schedule: scheduler.Schedule

    exp_num_map: dict[int, ExperimentDetail]

    def find_passes_time_ranges(
        self,
        dt_s_arr: npt.NDArray[Float64_as_sec],
        space_object_states: EcefStates,
        epoch: datetime,
    ):
        time_ranges = find_simultaneous_passes_time_ranges(
            dt_s_arr=dt_s_arr,
            states=space_object_states,
            stations=[self.tx_station, self.rx_station],
            epoch=epoch,
        )

        return time_ranges

    def get_schedule_mask_by_time_range(
        self, time_range: tuple[Datetime64_us, Datetime64_us]
    ) -> npt.NDArray[np.bool]:
        mask = schedule.get_schedule_mask_by_time_range(self.tx_schedule, time_range)
        return mask

    def calculate_observation(
        self,
        space_object: sorts.SpaceObject,
        space_object_states_interpolator: Interpolator,
        epoch: datetime,
        schedule_mask: npt.NDArray[np.bool] | None,
    ):
        return SimpleStxSrx.calculate_observation_from_config(
            self,
            space_object=space_object,
            space_object_states_interpolator=space_object_states_interpolator,
            epoch=epoch,
            schedule_mask=schedule_mask,
        )

    # TODO: maybe making it a standalone func is better?
    @staticmethod
    def calculate_observation_from_config(
        dcfg: SimpleStxSrx,
        space_object: sorts.SpaceObject,
        space_object_states_interpolator: Interpolator,
        epoch: datetime,
        schedule_mask: npt.NDArray[np.bool] | None,
    ):
        """NOTE: We assume the tx and rx time difference is negligible"""

        dt_s_arr: npt.NDArray[Float64_as_sec] = (
            (dcfg.rx_schedule.stt_tstmp_us - np.datetime64(epoch))
            .astype("timedelta64[us]")
            .astype(np.float64)
        ) * 1e-6
        tx_schedule = dcfg.tx_schedule
        rx_schedule = dcfg.rx_schedule

        # apply mask if it exists
        if schedule_mask is not None:
            dt_s_arr = dt_s_arr[schedule_mask]
            tx_schedule = dcfg.tx_schedule.filter_by_mask(schedule_mask)
            rx_schedule = dcfg.rx_schedule.filter_by_mask(schedule_mask)

        obs_size = len(dt_s_arr)

        # TODO: can probably be taken from the `simulation.find_passes`
        spobj_states = space_object_states_interpolator.get_state(dt_s_arr)
        spobj_tx_enu = dcfg.tx_station.enu(spobj_states)  # space object in tx station coordinate
        spobj_rx_enu = dcfg.rx_station.enu(spobj_states)  # space object in rx station coordinate

        range_tx_m: npt.NDArray[Float64_as_m] = np.linalg.norm(spobj_tx_enu[:3, :], axis=0)
        range_rx_m: npt.NDArray[Float64_as_m] = np.linalg.norm(spobj_rx_enu[:3, :], axis=0)

        snr = np.empty((obs_size,), dtype=np.float64)
        # rcs = np.empty((obs_size,), dtype=np.float64) # TODO: chk if needed

        powers = np.empty((obs_size,), dtype=np.float64)

        # pulse_lengths = np.array(
        #     [dcfg.exp_num_map[n].pulse_length for n in tx_schedule.exp_num], dtype=np.float64
        # )  # TODO: chk if needed
        # ipps = np.array(
        #     [dcfg.exp_num_map[n].ipp for n in tx_schedule.exp_num], dtype=np.float64
        # )  # TODO: chk if needed
        powers = np.array(
            [dcfg.exp_num_map[n].power for n in tx_schedule.exp_num], dtype=np.float64
        )
        bandwidths = np.array(
            [dcfg.exp_num_map[n].bandwidth for n in tx_schedule.exp_num], dtype=np.float64
        )
        # duty_cycles = np.array(
        #     [dcfg.exp_num_map[n].duty_cycle for n in tx_schedule.exp_num], dtype=np.float64
        # )  # TODO: chk if needed
        rx_noise_temps = np.array(
            [dcfg.exp_num_map[n].noise_temp for n in rx_schedule.exp_num], dtype=np.float64
        )

        # TODO: vectorize
        tx_gain_arr = np.full((obs_size,), 0.0, dtype=np.float64)
        rx_gain_arr = np.full((obs_size,), 0.0, dtype=np.float64)
        for idx, _ in enumerate(rx_schedule.stt_tstmp_us):
            dcfg.tx_station.beam.sph_point(
                tx_schedule.pointing_az[idx], tx_schedule.pointing_el[idx], degrees=True
            )
            tx_gain_arr[idx] = dcfg.tx_station.beam.gain(spobj_tx_enu[:3, idx])

            dcfg.rx_station.beam.sph_point(
                rx_schedule.pointing_az[idx], tx_schedule.pointing_el[idx], degrees=True
            )
            rx_gain_arr[idx] = dcfg.rx_station.beam.gain(spobj_rx_enu[:3, idx])

        tx_wavelength: float = dcfg.tx_station.beam.wavelength
        # rx_wavelength: float = dcfg.rx_station.beam.wavelength # TODO: chk if needed

        snr = sorts.signals.hard_target_snr(
            tx_gain_arr,
            rx_gain_arr,
            tx_wavelength,
            powers[0],  # TODO: improve: hard-coded from `exp_detail`
            range_tx_m,
            range_rx_m,
            diameter=space_object.d,
            bandwidth=bandwidths[0],  # TODO: improve: hard-coded from `exp_detail`
            rx_noise_temp=rx_noise_temps[0],  # TODO: improve: hard-coded from `exp_detail`
            radar_albedo=space_object.parameters.get("radar_albedo", 1.0),
        )

        # TODO: add `doppler_spread_integrated_snr:` support
        # TODO: add `blind_ranges:` support

        obs = Observation(
            snr=snr,
            range=range_tx_m + range_rx_m,
            range_rx=range_rx_m,
            range_rate=np.full((obs_size,), 1.0, dtype=np.float64),  # TODO: imple
            tx_k=spobj_tx_enu[:3] / range_tx_m,
            rx_k=spobj_rx_enu[:3] / range_rx_m,
            rcs=np.full((obs_size,), 1.0, dtype=np.float64),  # TODO: imple
        )

        return obs


@dataclass
class StxMrx(DetectionConfigProtocol):
    """wip, do not use it"""

    stt_tstmp_us: npt.NDArray[np.datetime64]
    tx_station_key: RadarStationCompositeKey
    rx_station_keys: list[RadarStationCompositeKey]
    tx_schedule: scheduler.Schedule
    rx_schedules: list[scheduler.Schedule]
    exp_num_map: dict[int, ExperimentDetail]

    # TODO: implement or remove
    def find_passes_time_ranges(
        self,
        dt_s_arr: npt.NDArray[np.float64],
        space_object_states: npt.NDArray[np.float64],
        epoch: datetime,
    ) -> t.Sequence[tuple[Datetime64_us, Datetime64_us]]: ...

    # TODO: implement or remove
    def get_schedule_mask_by_time_range(
        self, time_range: tuple[Datetime64_us, Datetime64_us]
    ) -> npt.NDArray[np.bool]: ...

    # TODO: implement or remove
    def calculate_observation(
        self,
        space_object: sorts.SpaceObject,
        space_object_states_interpolator: Interpolator,
        epoch: datetime,
        schedule_mask: npt.NDArray[np.bool] | None,
    ) -> Observation: ...


def find_simultaneous_passes_time_ranges(
    dt_s_arr: npt.NDArray[np.float64],
    states: EcefStates,
    stations: list[Station],
    epoch: datetime,
    fov_kw=None,
) -> t.Sequence[tuple[Datetime64_us, Datetime64_us]]:
    """
    Finds all passes that are simultaneously inside a multiple stations FOV's.
    """

    time_ranges: list[tuple[Datetime64_us, Datetime64_us]] = []
    if fov_kw is None:
        fov_kw = {}

    enu = []
    check = np.full((len(dt_s_arr),), True, dtype=bool)
    for station in stations:
        enu_st = station.enu(states)
        enu.append(enu_st)

        check_st = station.field_of_view(states, **fov_kw)
        check = np.logical_and(check, check_st)

    inds = np.where(check)[0]

    if len(inds) == 0:
        return time_ranges

    dind = np.diff(inds)
    splits = np.where(dind > 1)[0]

    splits = np.insert(splits, 0, -1)
    splits = np.insert(splits, len(splits), len(inds) - 1)
    splits += 1
    for si in range(len(splits) - 1):
        ps_inds = inds[splits[si] : splits[si + 1]]
        if len(ps_inds) == 0:
            continue

        start_time: Datetime64_us = t.cast(
            np.timedelta64, (dt_s_arr[ps_inds[0]] * 1e6).astype("timedelta64[us]")
        ) + np.datetime64(epoch)

        end_time: Datetime64_us = t.cast(
            np.timedelta64, (dt_s_arr[ps_inds[-1]] * 1e6).astype("timedelta64[us]")
        ) + np.datetime64(epoch)

        time_range = (start_time, end_time)
        time_ranges.append(time_range)

    return time_ranges
