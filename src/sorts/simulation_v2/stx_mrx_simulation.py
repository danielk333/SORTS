from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import pyorb
import sorts
from tqdm import tqdm
from sorts.interpolation import Interpolator
from sorts.radar.tx_rx import Station, StationId
from sorts.utils import to_datetime64_us
from sorts.types import Datetime_Like, Float64_as_sec, EcefStates, Datetime64_us
from sorts.schedule_v2.schedule import (
    Schedule,
    TimeRangeIndexer,
    ScheduleNdarrayDict2,
    ExperimentDetail,
)
from sorts.simulation_v2.passage import Passage, find_passages
from sorts.simulation_v2.simulation_unit import SimulationUnit
from sorts.simulation_v2.observation import Observation, ObservationIndexer

logger = logging.getLogger(__name__)


class SpaceObjectDsecSampler(t.Protocol):
    def __call__(
        self, orbit: pyorb.Orbit, start_time: Datetime_Like, end_time: Datetime_Like
    ) -> npt.NDArray[Float64_as_sec]: ...


class Spec(t.TypedDict):
    """A TypedDict of params"""

    tx_station: Station
    tx_schedule: ScheduleNdarrayDict2
    rx_stations: t.Sequence[Station]
    rx_schedules: t.Sequence[ScheduleNdarrayDict2]
    exp_detail_map: dict[int, ExperimentDetail]
    epoch: Datetime_Like
    start_time: Datetime_Like
    end_time: Datetime_Like
    space_objects: t.Sequence[sorts.SpaceObject]
    dsec_sampler: SpaceObjectDsecSampler  # TODO: support different sampler for different obj?
    interpolator_class: type[Interpolator]


class State(t.TypedDict):
    """A TypedDict of params"""

    space_object_sample_dsec: list[npt.NDArray[Float64_as_sec]]
    space_object_sample_states: list[EcefStates]
    space_object_interpolators: list[Interpolator]
    observations: list[Observation]


def sample_and_propagate_pace_objects_states(
    sampler: SpaceObjectDsecSampler,
    spobjs: t.Sequence[sorts.SpaceObject],
    start_time: Datetime64_us,
    end_time: Datetime64_us,
) -> tuple[list[npt.NDArray[Float64_as_sec]], list[EcefStates]]:
    """
    Use the sampler to get the delta time of space object within the simulation `start_time` and `end_time`,
    then get the space object states at those delta time using the propagator in the space object.

    Returns a list of sampled delta seconds and a list of corresponding states.
    """

    spobjs_smpl_dsec: list[npt.NDArray[Float64_as_sec]] = []
    for spobj in tqdm(spobjs, total=len(spobjs)):
        spobjs_smpl_dsec.append(sampler(spobj.state, start_time, end_time))

    spobjs_smpl_states: list[EcefStates] = []
    for spobj, spobj_smpl_dsec in tqdm(zip(spobjs, spobjs_smpl_dsec), total=len(spobjs)):
        spobjs_smpl_states.append(spobj.get_state(spobj_smpl_dsec))

    return spobjs_smpl_dsec, spobjs_smpl_states


def derive_schedule_indexers_per_tx_rx_station_pair(
    passages: list[Passage],
) -> dict[tuple[StationId, StationId], list[TimeRangeIndexer]]:
    """Derive a list of schedule indexer for each tx-rx station pair found in the give passages"""

    sch_indexers: dict[tuple[StationId, StationId], list[TimeRangeIndexer]] = {}

    for passage in passages:
        tx_station_id = passage["tx_station"].uid
        rx_station_id = passage["rx_station"].uid

        if (tx_station_id, rx_station_id) in sch_indexers:
            sch_indexers[(tx_station_id, rx_station_id)].append(passage["time_range"])
        else:
            sch_indexers[(tx_station_id, rx_station_id)] = [passage["time_range"]]

    return sch_indexers


def derive_observation_indexers(
    passages: list[Passage],
    tx_sch: Schedule,
    rx_schs: list[Schedule],
) -> list[ObservationIndexer]:
    """Derive an indexer for each observation"""

    obs_indexers: list[ObservationIndexer] = []

    k = Schedule.keys

    rx_schedule_map: dict[str, Schedule] = {
        sch._data.attrs[sch.attr_keys["stn_id"]]: sch for sch in rx_schs
    }

    for passage in passages:
        tx_sch_obss = tx_sch.filter_by_time_range(passage["time_range"]).split_by_measurements()

        rx_sch = rx_schedule_map[passage["rx_station"].uid]
        rx_sch_obss = rx_sch.filter_by_time_range(passage["time_range"]).split_by_measurements()

        for tx_sch_obs in tx_sch_obss:
            for rx_sch_obs in rx_sch_obss:
                obs_indexers.append(
                    {
                        "tx_indexer": tx_sch_obs._data[k["start_time"]],
                        "rx_indexer": rx_sch_obs._data[k["start_time"]],
                    }
                )

    return obs_indexers


# TODO: can be dissolved?
def calculate_observations(
    spec: Spec,
    spobjs_smpl_dsec: list[npt.NDArray[Float64_as_sec]],
    spobjs_smpl_states: list[EcefStates],
    spobjs_interpolators: list[Interpolator],
) -> list[SimulationUnit]:
    """
    Calculate the observations.
    """

    passages: list[Passage] = []
    sim_units: list[SimulationUnit] = []

    pbar = tqdm(desc="simulating observation", total=len(spec["space_objects"]))
    for spobj, spobj_smpl_dsec, spobj_smpl_states, spobj_states_interp in zip(
        spec["space_objects"],
        spobjs_smpl_dsec,
        spobjs_smpl_states,
        spobjs_interpolators,
    ):
        logger.debug(f"{spobj} calculating")
        for rx_station in spec["rx_stations"]:
            found_passages = find_passages(
                dt=spobj_smpl_dsec,
                space_object=spobj,
                states=spobj_smpl_states,
                tx_station=spec["tx_station"],
                rx_station=rx_station,
                epoch=spec["epoch"],
            )
            passages.extend(found_passages)

        indexers_dict = derive_schedule_indexers_per_tx_rx_station_pair(passages)

        # TODO: minor cleanup needed, old code for schedule is making is a bit more messy than needed
        for stn_id_pair, indexers in indexers_dict.items():
            rx_stn_idx = [stn.uid for stn in spec["rx_stations"]].index(stn_id_pair[1])

            sim_unit = SimulationUnit.from_passages_over_tx_rx_station_pair(
                indexers=indexers,
                spobj=spobj,
                spobj_interp=spobj_states_interp,
                tx_stn=spec["tx_station"],
                rx_stn=spec["rx_stations"][rx_stn_idx],
                tx_sch=Schedule.from_ndarrays_2(spec["tx_schedule"]),
                rx_sch=Schedule.from_ndarrays_2(spec["rx_schedules"][rx_stn_idx]),
            )
            sim_unit.simulate()

            sim_units.append(sim_unit)

        pbar.update(1)
    pbar.close()

    # TODO: update `Observation` class, (and return list of `Observation` here?)
    # TODO: call `derive_observation_indexers`
    return sim_units


# TODO: we need to enforce each station to has a unique id (`.uid` prop)
#   either in the simulation class or in related station getter like `get_radar`
class StxMrxSimulation:
    def __init__(self, spec: Spec, state: State):
        self.spec: Spec = spec
        self.state: State = state

    @classmethod
    def from_spec(cls, spec: Spec) -> StxMrxSimulation:
        sim = StxMrxSimulation(
            spec=spec,
            state={
                "space_object_sample_dsec": [],
                "space_object_sample_states": [],
                "space_object_interpolators": [],
                "observations": [],
            },
        )

        return sim

    def run(self):
        logger.debug("starting stx mrx sim")
        spobjs_smpl_dsec, spobjs_smpl_states = sample_and_propagate_pace_objects_states(
            sampler=self.spec["dsec_sampler"],
            spobjs=self.spec["space_objects"],
            start_time=to_datetime64_us(self.spec["start_time"]),
            end_time=to_datetime64_us(self.spec["end_time"]),
        )
        logger.debug("sample and propagate done")
        spobjs_interpolators = [
            self.spec["interpolator_class"](spobj_smpl_states, spobj_smpl_dsec)
            for spobj_smpl_dsec, spobj_smpl_states in zip(spobjs_smpl_dsec, spobjs_smpl_states)
        ]
        logger.debug("interpolators done")

        self.state["space_object_sample_dsec"] = spobjs_smpl_dsec
        self.state["space_object_sample_states"] = spobjs_smpl_states
        self.state["space_object_interpolators"] = spobjs_interpolators

        sim_units = calculate_observations(
            self.spec, spobjs_smpl_dsec, spobjs_smpl_states, spobjs_interpolators
        )
        logger.debug("calc obs done")
        # self.state["observations"] = obss

        return sim_units
