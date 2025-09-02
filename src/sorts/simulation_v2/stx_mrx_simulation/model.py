from __future__ import annotations
import logging, typing as t
import numpy.typing as npt
import pyorb
import sorts
from tqdm import tqdm
from sorts.interpolation import Interpolator
from sorts.radar import Station
from sorts.utils import to_datetime64_us
from sorts.types import Datetime_Like, Float64_as_sec, EcefStates
from sorts.schedule_v2 import Schedule, ExperimentDetail
from sorts.simulation_v2 import passage
from sorts.simulation_v2.observation import Observation, ObservationIndexer
from sorts.simulation_v2.stx_mrx_simulation import funcs

logger = logging.getLogger(__name__)


class SpaceObjectDsecSampler(t.Protocol):
    def __call__(
        self, orbit: pyorb.Orbit, start_time: Datetime_Like, end_time: Datetime_Like
    ) -> npt.NDArray[Float64_as_sec]: ...


class Spec(t.TypedDict):
    """A TypedDict of params"""

    tx_station: Station
    tx_schedule: Schedule
    rx_stations: t.Sequence[Station]
    rx_schedules: t.Sequence[Schedule]
    exp_detail_map: dict[int, ExperimentDetail]
    epoch: Datetime_Like
    start_time: Datetime_Like
    end_time: Datetime_Like
    space_objects: t.Sequence[sorts.SpaceObject]
    dsec_sampler: SpaceObjectDsecSampler  # TODO: support different sampler for different obj?
    interpolator_class: type[Interpolator]


# TODO: can be dissolved, when we have SimulationUnit now?
class State(t.TypedDict):
    """A TypedDict of params"""

    space_object_sample_dsec: list[npt.NDArray[Float64_as_sec]]
    space_object_sample_states: list[EcefStates]
    space_object_interpolators: list[Interpolator]
    observations: list[Observation]


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

        spobjs_smpl_dsec, spobjs_smpl_states = funcs.sample_and_propagate_pace_objects_states(
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

        passages_lists = funcs.find_passages(
            spec=self.spec,
            spobjs_smpl_dsec=spobjs_smpl_dsec,
            spobjs_smpl_states=spobjs_smpl_states,
        )
        logger.debug("find_passages done")

        sim_units = funcs.derive_simulation_units(
            spec=self.spec,
            passages_lists=passages_lists,
            spobjs_interpolators=spobjs_interpolators,
        )
        logger.debug("derive_simulation_units done")

        pbar = tqdm(desc="simulating", total=len(sim_units))

        for sim_unit in sim_units:
            sim_unit.simulate()
            pbar.update(1)
        logger.debug("simulation done")

        pbar.close()

        # TODO: simplification neeeded, update `Observation` class, (and return list of `Observation` here?)
        passage_obs_idxers_pairs: list[tuple[passage.Passage, list[ObservationIndexer]]] = []
        for passages in passages_lists:
            pairs = [
                (
                    ps,
                    funcs.derive_observation_indexers(
                        passage=ps,
                        tx_sch=self.spec["tx_schedule"],
                        rx_schs=self.spec["rx_schedules"],
                    ),
                )
                for ps in passages
            ]
            passage_obs_idxers_pairs.extend(pairs)

        return sim_units, passage_obs_idxers_pairs
