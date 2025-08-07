from __future__ import annotations
import logging, typing as t
from datetime import datetime
import numpy as np
import numpy.typing as npt
import pyorb
import sorts
from sorts.interpolation import Interpolator
from sorts.radar.tx_rx import Station
from sorts.utils import to_pydatetime, to_datetime64_us
from sorts.types import Datetime_like, Float64_as_sec, Float64_as_m, EcefStates, Datetime64_us
from sorts.simulation_v2.passage import (
    ExperimentPassage,
    find_passages,
    split_passage_by_schedule,
)
from sorts.simulation_v2.observation import Observation
from sorts.schedule_v2 import Schedule, ExperimentDetail

logger = logging.getLogger(__name__)


class SpaceObjectDsecSampler(t.Protocol):
    def __call__(
        self, orbit: pyorb.Orbit, start_time: Datetime_like, end_time: Datetime_like
    ) -> npt.NDArray[Float64_as_sec]: ...


class Spec(t.TypedDict):
    """A TypedDict of params"""

    tx_station: Station
    tx_schedule: Schedule
    rx_stations: t.Sequence[Station]
    rx_schedules: t.Sequence[Schedule]
    exp_num_map: dict[int, ExperimentDetail]
    epoch: Datetime_like
    start_time: Datetime_like
    end_time: Datetime_like
    space_objects: t.Sequence[sorts.SpaceObject]
    dsec_sampler: SpaceObjectDsecSampler  # TODO: support different sampler for different obj?
    interpolator_class: type[Interpolator]


# TODO: reduce duplication with `Spec`
class State(t.TypedDict):
    """A TypedDict of params"""

    # TODO: these are short cuts to access internal states of `Simulation` (e.g. for plotting)
    #   need to be removed or exposed more properly
    _spobjs_states_interps: list[Interpolator]


def sample_and_propagate_pace_objects_states(
    sampler: SpaceObjectDsecSampler,
    spobjs: t.Sequence[sorts.SpaceObject],
    start_time: Datetime64_us,
    end_time: Datetime64_us,
) -> tuple[list[npt.NDArray[Float64_as_sec]], list[EcefStates]]:
    """
    Use the sampler the get the delta time of space object within the simulation `start_time` and `end_time`
    """

    spobjs_smpl_dsec: list[npt.NDArray[Float64_as_sec]] = [
        sampler(spobj.state, start_time, end_time) for spobj in spobjs
    ]

    spobjs_smpl_states: list[EcefStates] = [
        spobj.get_state(spobj_smpl_dsec) for spobj, spobj_smpl_dsec in zip(spobjs, spobjs_smpl_dsec)
    ]

    return spobjs_smpl_dsec, spobjs_smpl_states


# TODO: there was a note about assuming the tx and rx time difference is negligible.
#   tx-rx time difference is used to calc range so this cannot be true.
#   likely it is a related assumption regarding similar terms (e.g. in schedule), and should be cleaned up.
# TODO: we need mask per (tx, rx) schedule?
def calculate_observation_per_experiment_passage(
    spec: Spec,
    experiment_passage: ExperimentPassage,
    # TODO: move `space_object_states_interpolator` outside? use NamedTuple? maybe even pregenerate the state?
    space_object_states_interpolator: Interpolator,
) -> Observation:
    # TODO: can probably be simplified?
    rx_station_index = next(
        (
            i
            for i, s in enumerate(spec["rx_stations"])
            if s.uid == experiment_passage["rx_station"].uid
        )
    )

    tx_station = experiment_passage["tx_station"]
    tx_schedule = spec["tx_schedule"]
    rx_station = experiment_passage["rx_station"]
    rx_schedule = spec["rx_schedules"][rx_station_index]

    schedule_mask = spec["rx_schedules"][rx_station_index].create_mask_by_time_range(
        experiment_passage["time_range"]
    )

    dsec: npt.NDArray[Float64_as_sec] = (
        rx_schedule.start_time - experiment_passage["epoch"]
    ).astype(np.float64) * 1e-6

    dsec = dsec[schedule_mask]
    tx_schedule = tx_schedule.filter_by_mask(schedule_mask)
    rx_schedule = rx_schedule.filter_by_mask(schedule_mask)

    obs_size = len(dsec)

    spobj_states = space_object_states_interpolator.get_state(dsec)
    spobj_tx_enu = tx_station.enu(spobj_states)  # space object in tx station coordinate
    spobj_rx_enu = rx_station.enu(spobj_states)  # space object in rx station coordinate

    range_tx_m: npt.NDArray[Float64_as_m] = np.linalg.norm(spobj_tx_enu[:3, :], axis=0)
    range_rx_m: npt.NDArray[Float64_as_m] = np.linalg.norm(spobj_rx_enu[:3, :], axis=0)

    snr = np.empty((obs_size,), dtype=np.float64)
    powers = np.empty((obs_size,), dtype=np.float64)

    # pulse_lengths = np.array(
    #     [spec["exp_num_map"][n]["pulse_length"] for n in tx_schedule.exp_num], dtype=np.float64
    # )  # TODO: chk if needed
    # ipps = np.array(
    #     [spec["exp_num_map"][n]["ipp"] for n in tx_schedule.exp_num], dtype=np.float64
    # )  # TODO: chk if needed
    powers = np.array(
        [spec["exp_num_map"][n]["power"] for n in tx_schedule.exp_num], dtype=np.float64
    )
    bandwidths = np.array(
        [spec["exp_num_map"][n]["bandwidth"] for n in tx_schedule.exp_num],
        dtype=np.float64,
    )
    # duty_cycles = np.array(
    #     [spec["exp_num_map"][n]["duty_cycle"] for n in tx_schedule.exp_num], dtype=np.float64
    # )  # TODO: chk if needed
    rx_noise_temps = np.array(
        [spec["exp_num_map"][n]["noise_temp"] for n in rx_schedule.exp_num],
        dtype=np.float64,
    )

    # TODO: check with daniel on how to vectorize
    #   passing in a ndarray of pointing will trigger exception when calculating gain
    #   refs:
    #   - `pyant/beam.py` `L235` `assert vector_cnt <= max_vectors, "Too many vector valued parameters"`
    #   - `pyant/models/array.py` `L185` `params, shape = self.get_parameters(ind, named=True, max_vectors=0)`
    tx_gain_arr = np.full((obs_size,), 0.0, dtype=np.float64)
    rx_gain_arr = np.full((obs_size,), 0.0, dtype=np.float64)
    for idx, _ in enumerate(rx_schedule.start_time):
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
        powers,
        range_tx_m,
        range_rx_m,
        diameter=experiment_passage["space_object"].d,
        bandwidth=bandwidths,
        rx_noise_temp=rx_noise_temps,
        radar_albedo=experiment_passage["space_object"].parameters.get("radar_albedo", 1.0),
    )

    # TODO: add `doppler_spread_integrated_snr:` support
    # TODO: add `blind_ranges:` support

    obs = Observation(
        id=f'{rx_station_index}-({str(experiment_passage["time_range"][0])}, {str(experiment_passage["time_range"][1])})',  # TODO: revisit
        experiment_passage=experiment_passage,
        snr=snr,
        range=range_tx_m + range_rx_m,
        range_rx=range_rx_m,
        range_rate=np.full((obs_size,), 1.0, dtype=np.float64),  # TODO: imple
        tx_k=spobj_tx_enu[:3] / range_tx_m,
        rx_k=spobj_rx_enu[:3] / range_rx_m,
    )

    return obs


def calculate_observations(spec: Spec, state: State) -> list[Observation]:
    obss: list[Observation] = []

    spobjs_smpl_dsec, spobjs_smpl_states = sample_and_propagate_pace_objects_states(
        sampler=spec["dsec_sampler"],
        spobjs=spec["space_objects"],
        start_time=to_datetime64_us(spec["start_time"]),
        end_time=to_datetime64_us(spec["end_time"]),
    )
    spobjs_states_interps = [
        spec["interpolator_class"](spobj_smpl_states, spobj_smpl_dsec)
        for spobj_smpl_dsec, spobj_smpl_states in zip(spobjs_smpl_dsec, spobjs_smpl_states)
    ]
    state["_spobjs_states_interps"] = spobjs_states_interps

    for spobj, spobj_smpl_dsec, spobj_smpl_states, spobj_states_interp in zip(
        spec["space_objects"],
        spobjs_smpl_dsec,
        spobjs_smpl_states,
        spobjs_states_interps,
    ):
        exp_passages: list[ExperimentPassage] = []
        for rx_station, rx_schedule in zip(spec["rx_stations"], spec["rx_schedules"]):
            passages = find_passages(
                dt=spobj_smpl_dsec,
                space_object=spobj,
                states=spobj_smpl_states,
                tx_station=spec["tx_station"],
                rx_station=rx_station,
                epoch=spec["epoch"],
            )

            for passage in passages:
                exp_passages_ = split_passage_by_schedule(
                    passage=passage,
                    schedule=rx_schedule,
                    exp_num_map=spec["exp_num_map"],
                )

                exp_passages.extend(exp_passages_)

        # TODO: improvements needed; this only works for StxSrx case, where calculate_observation gives out 1 element list
        for exp_passage in exp_passages:
            obs = calculate_observation_per_experiment_passage(
                spec=spec,
                experiment_passage=exp_passage,
                space_object_states_interpolator=spobj_states_interp,
            )
            obss.append(obs)

    return obss


class StxMrxSimulation:
    def __init__(self, spec: Spec, state: State):
        self.spec: Spec = spec
        self.state: State = state

    @classmethod
    def from_spec(cls, spec: Spec) -> StxMrxSimulation:
        sim = StxMrxSimulation(
            spec=spec,
            state={
                "_spobjs_states_interps": [],
            },
        )

        return sim

    def calculate_observations(self):
        global calculate_observations

        if self.state is None:
            raise RuntimeError("Cannot calculate observations when `state` prop is `None`")
        else:
            return calculate_observations(self.spec, self.state)
