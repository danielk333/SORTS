import logging, typing as t
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pyorb
import sorts
from sorts.interpolation import Interpolator
from sorts.radar.tx_rx import Station
from sorts.types import Float64_as_sec, Float64_as_m, EcefStates, Datetime64_us
from sorts.simulation_v2.observation import Observation
from sorts.schedule_v2 import Schedule, ExperimentDetail
from sorts.simulation_v2.simulation_protocol import SimulationProtocol
from sorts.simulation_v2.helpers import find_simultaneous_passes_time_ranges

logger = logging.getLogger(__name__)


class SpaceObjectDtSampler(t.Protocol):
    def __call__(
        self, orbit: pyorb.Orbit, start_time: datetime, end_time: datetime
    ) -> npt.NDArray[Float64_as_sec]: ...


@dataclass(kw_only=True)
class StxMrxSimulationParam:
    tx_station: Station
    tx_schedule: Schedule
    rx_stations: t.Sequence[Station]
    rx_schedules: t.Sequence[Schedule]

    exp_num_map: dict[int, ExperimentDetail]

    epoch: datetime
    start_time: datetime
    end_time: datetime

    space_objects: list[sorts.SpaceObject]

    # TODO: support different sampler for different obj?
    # TODO: probably taking a function + a args/kwargs obj is more pythonic
    space_objects_dt_sampler_s: SpaceObjectDtSampler

    # TODO: rename to `space_objects_dt_s_interpolator`
    space_objects_dt_interpolator_s: type[Interpolator]


# TODO: further generalize it into MtxMrx?
class StxMrxSimulation(SimulationProtocol):
    def __init__(self, param: StxMrxSimulationParam):
        self.param = param

        # TODO: these are short cuts to access internal states of `Simulation` (e.g. for plotting)
        #   need to be removed or exposed more properly
        self._spobjs_states_interps: list[Interpolator] = []

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

    def propagate_and_sample_space_objects_states(self):
        """
        Use the sampler the get the delta time of space object within the simulation `start_time` and `end_time`
        """

        spobjs_smpl_dt_s_arr: list[npt.NDArray[Float64_as_sec]] = [
            self.param.space_objects_dt_sampler_s(
                spobj.state,
                self.param.start_time,
                self.param.end_time,
            )
            for spobj in self.param.space_objects
        ]

        spobjs_smpl_states: list[EcefStates] = [
            spobj.get_state(spobj_smpl_dt_s_arr)
            for spobj, spobj_smpl_dt_s_arr in zip(self.param.space_objects, spobjs_smpl_dt_s_arr)
        ]

        return spobjs_smpl_dt_s_arr, spobjs_smpl_states

    def calculate_observations(self) -> list[list[Observation]]:
        obss: list[list[Observation]] = []

        spobjs_smpl_dt_s_arr, spobjs_smpl_states = self.propagate_and_sample_space_objects_states()
        spobjs_states_interps = [
            create_space_object_states_interpolator(
                self.param.space_objects_dt_interpolator_s, spobj_smpl_states, spobj_smpl_dt_s_arr
            )
            for spobj_smpl_dt_s_arr, spobj_smpl_states in zip(
                spobjs_smpl_dt_s_arr, spobjs_smpl_states
            )
        ]
        self._spobjs_states_interps = spobjs_states_interps

        for spobj, spobj_smpl_dt_s_arr, spobj_smpl_states, spobj_states_interp in zip(
            self.param.space_objects,
            spobjs_smpl_dt_s_arr,
            spobjs_smpl_states,
            spobjs_states_interps,
        ):
            time_ranges = find_simultaneous_passes_time_ranges(
                dt_s_arr=spobj_smpl_dt_s_arr,
                states=spobj_smpl_states,
                stations=[self.param.tx_station, *self.param.rx_stations],
                epoch=self.param.epoch,
            )
            masks_for_spobj: list[npt.NDArray[np.bool]] = [
                self.param.tx_schedule.create_mask_by_time_range(time_range)
                for time_range in time_ranges
            ]

            # TODO: improvements needed; this is only works for StxSrx case, where calculate_observation gives out 1 element list
            # TODO: use for-loop + mutation instead of nested for-comprehension for better readability
            obs_for_spobj: list[Observation] = [
                obs
                # TODO: remove enumerate; it was used as tmp replacement for `for mask in masks_for_spobj`
                for time_range_idx, time_range in enumerate(time_ranges)
                for obs in [
                    self.calculate_observation_per_rx_station(
                        idx,
                        space_object=spobj,
                        space_object_states_interpolator=spobj_states_interp,
                        epoch=self.param.epoch,
                        schedule_mask=masks_for_spobj[time_range_idx],
                        time_range=time_range,
                    )
                    for idx in range(len(self.param.rx_stations))
                ]
            ]

            obss.append(obs_for_spobj)

        return obss

    # NOTE: kept for ref until the class is stablized
    # def run(self) -> dict[RadarStationCompositeKey, dict]: ...


# TODO: implement or remove
@dataclass(kw_only=True)
class SimulationResult:
    pass


def create_space_object_states_interpolator(
    interpolator: type[Interpolator],
    states: EcefStates,
    sample_dt_s_arr: npt.NDArray[Float64_as_sec],
):
    states_interp = interpolator(states, sample_dt_s_arr)
    return states_interp
