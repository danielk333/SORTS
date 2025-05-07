from __future__ import annotations
import typing as t
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import numpy.typing as npt
import sorts
from sorts.radar.radar import Radar
from sorts.radar.radars.composite_key import RadarStationCompositeKey
from sorts.radar.tx_rx import Station
from sorts.calculations import Observation, ExperimentDetail
from sorts import passes_v2 as passes
from sorts import scheduler_v2 as scheduler
from sorts import controller_v2 as controller


# TODO: rename to `DetectionSystem`?
class DetectionConfigProtocol(t.Protocol):
    def find_passes(
        self,
        dt_s_arr: npt.NDArray[np.float64],
        space_object_states: npt.NDArray[np.float64],
        epoch: datetime,
    ) -> list[passes.Pass]: ...

    def calculate_observations(
        self,
        space_objects: list[sorts.SpaceObject],
        epoch: datetime,
        schedule_mask: npt.NDArray[np.bool] | None,
    ) -> Observation: ...


@dataclass
class SimpleStxSrx(DetectionConfigProtocol):
    stt_tstmp_us: npt.NDArray[np.datetime64]
    "the timestamp which observations can be made"

    tx_station: Station
    rx_station: Station

    tx_schedule: scheduler.Schedule
    "the tx radar station schedule, can contain more entries than `stt_tstmp_us`"
    rx_schedule: scheduler.Schedule
    "the rx radar station schedule, can contain more entries than `stt_tstmp_us`"

    exp_num_map: dict[int, ExperimentDetail]

    # TODO: implement or remove
    def find_passes(
        self,
        dt_s_arr: npt.NDArray[np.float64],
        space_object_states: npt.NDArray[np.float64],
        epoch: datetime,
    ) -> list[passes.Pass]: ...

    # TODO: implement or remove
    def calculate_observations(
        self,
        space_objects: list[sorts.SpaceObject],
        epoch: datetime,
        schedule_mask: npt.NDArray[np.bool] | None,
    ) -> Observation: ...


# TODO: dissolve existing `SimpleStxSrx` and rename this to `SimpleStxSrx`
@dataclass(kw_only=True)
class StxSrx(DetectionConfigProtocol):
    # option 1
    # tx_station: Station
    # rx_station: Station
    # schedule_dict: dict[RadarStationCompositeKey, scheduler.Schedule]

    # option 2
    tx_station: Station
    tx_schedule: scheduler.Schedule
    rx_station: Station
    rx_schedule: scheduler.Schedule

    exp_num_map: dict[int, ExperimentDetail]

    # TODO: remove or complete the implementation of this helper method
    # @classmethod
    # def from_radar(cls, radar: Radar) -> StxSrx:
    #     return cls(
    #         tx_station=radar
    #         rx_station=
    #         schedule=
    #     )

    def find_passes(
        self,
        dt_s_arr: npt.NDArray[np.float64],
        space_object_states: npt.NDArray[np.float64],
        epoch: datetime,
    ):
        ps_objs = passes.find_simultaneous_passes(
            dt_arr=dt_s_arr,
            states=space_object_states,
            stations=[self.tx_station, self.rx_station],
            radar_station_composite_keys=[
                ("radar", "tx", "0"),
                ("radar", "rx", "0"),
            ],  # TODO: tmp hard-coded here, should use each radars' own logic/prop when implemented
            epoch=epoch,
        )

        return ps_objs

    def calculate_observations(
        self,
        space_objects: list[sorts.SpaceObject],
        epoch: datetime,
        schedule_mask: npt.NDArray[np.bool] | None,
    ):
        return StxSrx.calculate_observations_from_config(
            self,
            space_objects=space_objects,
            epoch=epoch,
            schedule_mask=schedule_mask,
        )

    # TODO: maybe making it a standalone func is better?
    @staticmethod
    def calculate_observations_from_config(
        dcfg: StxSrx,
        space_objects: list[sorts.SpaceObject],
        epoch: datetime,
        schedule_mask: npt.NDArray[np.bool] | None,
    ):
        """NOTE: We assume the tx and rx time difference is negligible"""

        dt_s_arr: npt.NDArray[np.float64] = (
            (dcfg.rx_schedule.stt_tstmp_us - np.datetime64(epoch))
            .astype("timedelta64[us]")
            .astype(np.float64)
        ) * 1e-6
        tx_schedule = dcfg.tx_schedule
        rx_schedule = dcfg.rx_schedule

        # apply mask if it exist
        if schedule_mask is not None:
            dt_s_arr = dt_s_arr[schedule_mask]
            tx_schedule = dcfg.tx_schedule.filter_by_mask(schedule_mask)
            rx_schedule = dcfg.rx_schedule.filter_by_mask(schedule_mask)

        obs_size = len(dt_s_arr)

        spobjs_states = [spobj.get_state(dt_s_arr) for spobj in space_objects]

        # TODO: loop over the spobjs_states instead of just handling one
        spobj = space_objects[0]
        states = spobjs_states[0]

        spobj_tx_enu = dcfg.tx_station.enu(states)  # space object in tx station coordinate
        spobj_rx_enu = dcfg.rx_station.enu(states)  # space object in rx station coordinate

        range_tx_m = np.linalg.norm(spobj_tx_enu[:3, :], axis=0)
        range_rx_m = np.linalg.norm(spobj_rx_enu[:3, :], axis=0)

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
            powers,  # TODO: improve: hard-coded from `exp_detail`
            range_tx_m,
            range_rx_m,
            diameter=spobj.d,
            bandwidth=bandwidths[0],  # TODO: improve: hard-coded from `exp_detail`
            rx_noise_temp=rx_noise_temps[0],  # TODO: improve: hard-coded from `exp_detail`
            radar_albedo=spobj.parameters.get("radar_albedo", 1.0),
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
    def find_passes(
        self,
        dt_s_arr: npt.NDArray[np.float64],
        space_object_states: npt.NDArray[np.float64],
        epoch: datetime,
    ) -> list[passes.Pass]: ...

    # TODO: implement or remove
    def calculate_observations(
        self,
        space_objects: list[sorts.SpaceObject],
        epoch: datetime,
        schedule_mask: npt.NDArray[np.bool] | None,
    ) -> Observation: ...


DetectionConfig: t.TypeAlias = t.Union[SimpleStxSrx, StxSrx, StxMrx]


# TODO: should move to other file? e.g. calculations?
# TODO: should vectorize and work against an array of SpaceObject?
def calculate_simple_stx_srx_observations(
    dcfg: SimpleStxSrx, space_object: sorts.SpaceObject, epoch: datetime
) -> Observation:
    dt_arr: npt.NDArray[np.float64] = (
        (dcfg.stt_tstmp_us - np.datetime64(epoch)).astype("timedelta64[us]").astype(np.float64)
    ) * 1e-6
    states = space_object.get_state(dt_arr)

    space_object_tx_enu = dcfg.tx_station.enu(states)  # space object in tx station coordinate
    space_object_rx_enu = dcfg.rx_station.enu(states)  # space object in rx station coordinate

    range_tx_m = np.linalg.norm(space_object_tx_enu[:3, :], axis=0)
    range_rx_m = np.linalg.norm(space_object_rx_enu[:3, :], axis=0)

    snr = np.empty((len(dcfg.stt_tstmp_us),), dtype=np.float64)
    rcs = np.empty((len(dcfg.stt_tstmp_us),), dtype=np.float64)

    powers = np.empty((len(dcfg.stt_tstmp_us),), dtype=np.float64)
    t_slices = np.empty((len(dcfg.stt_tstmp_us),), dtype=np.float64)
    txrx_on = np.full((len(dcfg.stt_tstmp_us),), False, dtype=bool)

    pulse_lengths = np.array(
        [dcfg.exp_num_map[n].pulse_length for n in dcfg.tx_schedule.exp_num], dtype=np.float64
    )
    ipps = np.array([dcfg.exp_num_map[n].ipp for n in dcfg.tx_schedule.exp_num], dtype=np.float64)
    powers = np.array(
        [dcfg.exp_num_map[n].power for n in dcfg.tx_schedule.exp_num], dtype=np.float64
    )
    bandwidths = np.array(
        [dcfg.exp_num_map[n].bandwidth for n in dcfg.tx_schedule.exp_num], dtype=np.float64
    )
    duty_cycles = np.array(
        [dcfg.exp_num_map[n].duty_cycle for n in dcfg.tx_schedule.exp_num], dtype=np.float64
    )
    rx_noise_temps = np.array(
        [dcfg.exp_num_map[n].noise_temp for n in dcfg.tx_schedule.exp_num], dtype=np.float64
    )

    # TODO: vectorize
    tx_gain_arr = np.full(len(dcfg.stt_tstmp_us), 0.0, dtype=np.float64)
    rx_gain_arr = np.full(len(dcfg.stt_tstmp_us), 0.0, dtype=np.float64)
    for idx, _ in enumerate(dcfg.stt_tstmp_us):
        dcfg.tx_station.beam.sph_point(
            dcfg.tx_schedule.pointing_az[idx], dcfg.tx_schedule.pointing_el[idx], degrees=True
        )
        tx_gain_arr[idx] = dcfg.tx_station.beam.gain(space_object_tx_enu[:3, idx])

        dcfg.rx_station.beam.sph_point(
            dcfg.rx_schedule.pointing_az[idx], dcfg.tx_schedule.pointing_el[idx], degrees=True
        )
        rx_gain_arr[idx] = dcfg.rx_station.beam.gain(space_object_rx_enu[:3, idx])

    tx_wavelength: float = dcfg.tx_station.beam.wavelength
    rx_wavelength: float = dcfg.rx_station.beam.wavelength

    snr = sorts.signals.hard_target_snr(
        tx_gain_arr,
        rx_gain_arr,
        tx_wavelength,
        powers,  # TODO: improve: hard-coded from `exp_detail`
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
        range_rate=np.full(len(dcfg.stt_tstmp_us), 1.0, dtype=np.float64),  # TODO: imple
        tx_k=space_object_tx_enu[:3] / range_tx_m,
        rx_k=space_object_rx_enu[:3] / range_rx_m,
        rcs=np.full(len(dcfg.stt_tstmp_us), 1.0, dtype=np.float64),  # TODO: imple
    )

    return obs


__all__ = [
    "DetectionConfigProtocol",
    "SimpleStxSrx",
    "StxSrx",
    "StxMrx",
    "DetectionConfig",
    "calculate_simple_stx_srx_observations",
]
