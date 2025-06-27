from __future__ import annotations
import typing as t
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import numpy.typing as npt
import sorts
from sorts.interpolation import Interpolator
from sorts.radar.tx_rx import Station
from sorts.types import Datetime64_us, Float64_as_sec, Float64_as_m, EcefStates
from sorts import schedule_v2 as schedule
from sorts import scheduler_v2 as scheduler
from sorts.simulation_v2.observation import Observation
from sorts.simulation_v2.experiment_detail import ExperimentDetail


class StxMrxRadarSystemParamDict(t.TypedDict):
    tx_station: Station
    tx_schedule: scheduler.Schedule
    rx_stations: t.Sequence[Station]
    rx_schedules: t.Sequence[scheduler.Schedule]

    exp_num_map: dict[int, ExperimentDetail]


@dataclass(kw_only=True)
class StxMrxRadarSystemParam:
    tx_station: Station
    tx_schedule: scheduler.Schedule
    rx_stations: t.Sequence[Station]
    rx_schedules: t.Sequence[scheduler.Schedule]

    exp_num_map: dict[int, ExperimentDetail]

    @classmethod
    def from_dict(cls, d: StxMrxRadarSystemParamDict) -> StxMrxRadarSystemParam:
        return StxMrxRadarSystemParam(**d)


class StxMrxRadarSystem:
    @t.overload
    def __init__(self, param: StxMrxRadarSystemParam): ...

    @t.overload
    def __init__(self, param: StxMrxRadarSystemParamDict): ...

    def __init__(self, param: StxMrxRadarSystemParam | StxMrxRadarSystemParamDict):
        if isinstance(param, StxMrxRadarSystemParam):
            self.param = param
        else:
            self.param = StxMrxRadarSystemParam.from_dict(param)

    def find_passes_time_ranges(
        self,
        dt_s_arr: npt.NDArray[Float64_as_sec],
        space_object_states: EcefStates,
        epoch: datetime,
    ):
        time_ranges = find_simultaneous_passes_time_ranges(
            dt_s_arr=dt_s_arr,
            states=space_object_states,
            stations=[self.param.tx_station, *self.param.rx_stations],
            epoch=epoch,
        )

        return time_ranges

    def get_schedule_mask_by_time_range(
        self, time_range: tuple[Datetime64_us, Datetime64_us]
    ) -> npt.NDArray[np.bool]:
        return self.param.tx_schedule.create_mask_by_time_range(time_range)

    # TODO: rename to `calculate_observation_per_station_pass`?
    # TODO: there is a note about assuming the tx and rx time difference is negligible.
    #   tx-rx time difference is used to calc range so this cannot be true.
    #   likely it is a related assumption regarding similar terms (e.g. in schedule), and should be cleaned up.
    # TODO: we need mask per (tx, rx) schedule?
    def calculate_observation_per_rx_station(
        self,
        rx_station_index: int,
        space_object: sorts.SpaceObject,
        space_object_states_interpolator: Interpolator,
        epoch: datetime,
        # TODO: remove, and calc the mask using `start_time`, `end_time` ?
        schedule_mask: npt.NDArray[np.bool] | None,
        time_range: tuple[Datetime64_us, Datetime64_us],
    ) -> Observation:
        tx_station = self.param.tx_station
        tx_schedule = self.param.tx_schedule
        rx_station = self.param.rx_stations[rx_station_index]
        rx_schedule = self.param.rx_schedules[rx_station_index]

        dt_s_arr: npt.NDArray[Float64_as_sec] = (
            (rx_schedule.stt_tstmp_us - np.datetime64(epoch))
            .astype("timedelta64[us]")
            .astype(np.float64)
        ) * 1e-6
        tx_schedule = tx_schedule
        rx_schedule = rx_schedule

        # apply mask if it exists
        if schedule_mask is not None:
            dt_s_arr = dt_s_arr[schedule_mask]
            tx_schedule = tx_schedule.filter_by_mask(schedule_mask)
            rx_schedule = rx_schedule.filter_by_mask(schedule_mask)

        obs_size = len(dt_s_arr)

        # TODO: can probably be taken from the `simulation.find_passes`
        spobj_states = space_object_states_interpolator.get_state(dt_s_arr)
        spobj_tx_enu = tx_station.enu(spobj_states)  # space object in tx station coordinate
        spobj_rx_enu = rx_station.enu(spobj_states)  # space object in rx station coordinate

        range_tx_m: npt.NDArray[Float64_as_m] = np.linalg.norm(spobj_tx_enu[:3, :], axis=0)
        range_rx_m: npt.NDArray[Float64_as_m] = np.linalg.norm(spobj_rx_enu[:3, :], axis=0)

        snr = np.empty((obs_size,), dtype=np.float64)
        powers = np.empty((obs_size,), dtype=np.float64)

        # pulse_lengths = np.array(
        #     [self.param.exp_num_map[n].pulse_length for n in tx_schedule.exp_num], dtype=np.float64
        # )  # TODO: chk if needed
        # ipps = np.array(
        #     [self.param.exp_num_map[n].ipp for n in tx_schedule.exp_num], dtype=np.float64
        # )  # TODO: chk if needed
        powers = np.array(
            [self.param.exp_num_map[n].power for n in tx_schedule.exp_num], dtype=np.float64
        )
        bandwidths = np.array(
            [self.param.exp_num_map[n].bandwidth for n in tx_schedule.exp_num], dtype=np.float64
        )
        # duty_cycles = np.array(
        #     [self.param.exp_num_map[n].duty_cycle for n in tx_schedule.exp_num], dtype=np.float64
        # )  # TODO: chk if needed
        rx_noise_temps = np.array(
            [self.param.exp_num_map[n].noise_temp for n in rx_schedule.exp_num], dtype=np.float64
        )

        # TODO: vectorize
        tx_gain_arr = np.full((obs_size,), 0.0, dtype=np.float64)
        rx_gain_arr = np.full((obs_size,), 0.0, dtype=np.float64)
        for idx, _ in enumerate(rx_schedule.stt_tstmp_us):
            tx_station.beam.sph_point(
                tx_schedule.pointing_az[idx], tx_schedule.pointing_el[idx], degrees=True
            )
            tx_gain_arr[idx] = tx_station.beam.gain(spobj_tx_enu[:3, idx])

            rx_station.beam.sph_point(
                rx_schedule.pointing_az[idx], tx_schedule.pointing_el[idx], degrees=True
            )
            rx_gain_arr[idx] = rx_station.beam.gain(spobj_rx_enu[:3, idx])

        tx_wavelength: float = tx_station.beam.wavelength
        # rx_wavelength: float = rx_station.beam.wavelength # TODO: chk if needed

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
            time_range=time_range,
            snr=snr,
            range=range_tx_m + range_rx_m,
            range_rx=range_rx_m,
            range_rate=np.full((obs_size,), 1.0, dtype=np.float64),  # TODO: imple
            tx_k=spobj_tx_enu[:3] / range_tx_m,
            rx_k=spobj_rx_enu[:3] / range_rx_m,
        )

        return obs

    def calculate_observation(
        self,
        space_object: sorts.SpaceObject,
        space_object_states_interpolator: Interpolator,
        epoch: datetime,
        schedule_mask: npt.NDArray[np.bool] | None,
        time_range: tuple[Datetime64_us, Datetime64_us],
    ):
        obss = [
            self.calculate_observation_per_rx_station(
                idx,
                space_object=space_object,
                space_object_states_interpolator=space_object_states_interpolator,
                epoch=epoch,
                schedule_mask=schedule_mask,
                time_range=time_range,
            )
            for idx in range(len(self.param.rx_stations))
        ]

        return [obs for obs in obss]


# TODO: move it into a helper/util func file?
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
