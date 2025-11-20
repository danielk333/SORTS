from __future__ import annotations
import logging, typing as t, pickle
from pathlib import Path
import numpy.typing as npt
import pyorb
import sorts
from tqdm import tqdm
# TODO: fix `schedule` namespace overlap issues in this file
from sorts import schedule, controller, simulation
from sorts.types import Datetime_Like, Float64_as_sec, Datetime64_us, Float64_as_sec, EcefStates
from sorts.utils import to_datetime64_us
from sorts.radar import Station, StationId
from sorts.simulation import Passage
from sorts.interpolation import Interpolator
from sorts.schedule import Schedule, ExperimentDetailMap
from sorts.simulation.stx_mrx_simulation.simulation_unit import (
    SimulationUnit,
    FromPassagesOverTxRxStationPairParam,
    Observation,
)

logger = logging.getLogger(__name__)


sim_unit_fname_tpl = "sim_unit.{id}.pickle"


class SpaceObjectDsecSampler(t.Protocol):
    def __call__(
        self,
        orbit: pyorb.Orbit,
        epoch: Datetime_Like,
        start_time: Datetime_Like,
        end_time: Datetime_Like,
    ) -> npt.NDArray[Float64_as_sec]: ...


def sample_and_propagate_space_objects_states(
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
    for spobj in tqdm(spobjs, desc="sampling spobjs dt", total=len(spobjs)):
        spobjs_smpl_dsec.append(
            sampler(spobj.state, to_datetime64_us(spobj.epoch), start_time, end_time)
        )

    spobjs_smpl_states: list[EcefStates] = []
    for spobj, spobj_smpl_dsec in tqdm(
        zip(spobjs, spobjs_smpl_dsec),
        desc="propagating spobjs states at sampled dt",
        total=len(spobjs),
    ):
        spobjs_smpl_states.append(spobj.get_state(spobj_smpl_dsec))

    return spobjs_smpl_dsec, spobjs_smpl_states


def group_passages_by_tx_rx_station_pair(
    passages: t.Sequence[Passage],
) -> dict[tuple[StationId, StationId], list[Passage]]:
    groupped_passages: dict[tuple[StationId, StationId], list[Passage]] = {}

    for passage in passages:
        tx_station_id = passage.tx_station.uid
        rx_station_id = passage.rx_station.uid

        if (tx_station_id, rx_station_id) in groupped_passages:
            groupped_passages[(tx_station_id, rx_station_id)].append(passage)
        else:
            groupped_passages[(tx_station_id, rx_station_id)] = [passage]

    return groupped_passages


# TODO: its name is confusing with `prepare_simulation_unit_params`; and maybe its func can be merged as well?
def derive_simulation_unit_params(
    sch: Schedule,  # TODO rename this param to `schedule` after refactoring `schedule` module
    station_map: dict[StationId, Station],
    exp_detail_map: ExperimentDetailMap,
    space_objects: t.Sequence[sorts.SpaceObject],
    passages_lists: t.Sequence[t.Sequence[Passage]],
    spobjs_interpolators: t.Sequence[Interpolator],
) -> list[FromPassagesOverTxRxStationPairParam]:
    """
    Derive a list of param for the `from_passages_over_tx_rx_station_pair` constructor of `SimulationUnit`

    NOTE: Integers (casted to `str`) are used as `SimulationUnit`s' id
    """

    params: list[FromPassagesOverTxRxStationPairParam] = []

    for spobj, passages_of_a_spobj, spobj_states_interp in zip(
        space_objects, passages_lists, spobjs_interpolators
    ):
        _SK = schedule._K

        groupped_passages = group_passages_by_tx_rx_station_pair(passages_of_a_spobj)

        for stn_id_pair, passages in groupped_passages.items():
            tx_stn = station_map[stn_id_pair[0]]
            rx_stn = station_map[stn_id_pair[1]]

            filtered_sch = schedule.filter_by_time_ranges(sch, [ps.time_range for ps in passages])

            # filter by station id
            filtered_sch = filtered_sch.loc[
                {
                    _SK.multi_index: (
                        slice(None),
                        (stn_id_pair[0], stn_id_pair[1]),
                        slice(None),
                        slice(None),
                    )
                }
            ]

            params.append(
                FromPassagesOverTxRxStationPairParam(
                    id=str(len(params)),
                    passages=passages,
                    spobj=spobj,
                    spobj_interp=spobj_states_interp,
                    tx_station=tx_stn,
                    rx_station=rx_stn,
                    schedule=filtered_sch,
                    exp_detail_map=exp_detail_map,
                )
            )

    return params


def find_passages(
    station_map: dict[StationId, Station],
    station_id_pairs: t.Sequence[tuple[StationId, StationId]],
    space_objects: t.Sequence[sorts.SpaceObject],
    epoch: Datetime_Like,
    spobjs_smpl_dsec: list[npt.NDArray[Float64_as_sec]],
    spobjs_smpl_states: list[EcefStates],
) -> list[list[Passage]]:
    """
    Find passages for each space objects over the simulation period.

    Returns a `list[Passage]` per space object.
    """

    passages_list: list[list[Passage]] = []

    for spobj, spobj_smpl_dsec, spobj_smpl_states in zip(
        space_objects,
        spobjs_smpl_dsec,
        spobjs_smpl_states,
    ):
        passages_of_spobj: list[Passage] = []

        for stn_id_pair in station_id_pairs:
            tx_stn = station_map[stn_id_pair[0]]
            rx_stn = station_map[stn_id_pair[1]]

            passages_of_spobj.extend(
                simulation.funcs.find_passages(
                    dt=spobj_smpl_dsec,
                    space_object=spobj,
                    states=spobj_smpl_states,
                    tx_station=tx_stn,
                    rx_station=rx_stn,
                    epoch=epoch,
                )
            )

        passages_list.append(passages_of_spobj)

    return passages_list


# TODO: should be tailored per experiment?
def iter_mpi_simulation_results(save_dir: Path):
    for fpath in save_dir.glob(sim_unit_fname_tpl.format(id="*")):
        with open(fpath, "rb") as f:
            sim_unit: SimulationUnit = pickle.load(f)
            yield sim_unit


# TODO: we need to enforce each station to has a unique id (`.uid` prop)
#   either in the simulation class or in related station getter like `get_radar`
class StxMrxSimulation:
    """
    NOTE: This is intended as an internal constructor, please use the constructor methods to create instances.
    """

    def __init__(
        self,
        station_map: dict[StationId, Station],
        station_id_pairs: t.Sequence[tuple[StationId, StationId]],
        schedule: Schedule,
        exp_detail_map: ExperimentDetailMap,
        epoch: Datetime_Like,
        start_time: Datetime_Like,
        end_time: Datetime_Like,
        space_objects: t.Sequence[sorts.SpaceObject],
        dsec_sampler: SpaceObjectDsecSampler,
        interpolator_class: type[Interpolator],
    ):
        self.station_map = station_map
        self.station_id_pairs = station_id_pairs
        self.schedule = schedule
        self.exp_detail_map = exp_detail_map
        self.epoch = epoch
        self.start_time = start_time
        self.end_time = end_time
        self.space_objects = space_objects
        self.dsec_sampler = dsec_sampler  # TODO: support different sampler for different obj?
        # TODO: we need to implement falback mechanism,
        #   e.g. a `Legendre8` `Interpolator` requires >=8 points, but sometime it might get less than that
        self.interpolator_class = interpolator_class

        self.sim_units: list[SimulationUnit] = []
        self.obss: list[Observation] = []

    @classmethod
    def from_controllers(
        cls,
        controllers: t.Sequence[controller.ControllerBase],
        schedule: Schedule,
        epoch: Datetime_Like,
        start_time: Datetime_Like,
        end_time: Datetime_Like,
        space_objects: t.Sequence[sorts.SpaceObject],
        dsec_sampler: SpaceObjectDsecSampler,
        interpolator_class: type[Interpolator],
    ):
        """A constructor method"""
        # TODO: - the exp details are already computed outside? Should the `controllers` field be
        # removed? or this classmethod? or what?

        stn_map: dict[StationId, Station] = {}
        stn_id_pairs_set: set[tuple[StationId, StationId]] = set()
        exp_detail_map: ExperimentDetailMap = {}

        for ctrl in controllers:
            stn_map.update(ctrl.get_station_map())

            for pairs in ctrl.get_experiment_id_station_id_pairs_map().values():
                stn_id_pairs_set.update(pairs)

            exp_detail = ctrl.get_experiment_detail()
            exp_detail_map[exp_detail.id] = exp_detail

        return cls(
            station_map=stn_map,
            station_id_pairs=list(stn_id_pairs_set),
            schedule=schedule,
            exp_detail_map=exp_detail_map,
            epoch=epoch,
            start_time=start_time,
            end_time=end_time,
            space_objects=space_objects,
            dsec_sampler=dsec_sampler,
            interpolator_class=interpolator_class,
        )

    def prepare_simulation_unit_params(self) -> list[FromPassagesOverTxRxStationPairParam]:
        spobjs_smpl_dsec, spobjs_smpl_states = sample_and_propagate_space_objects_states(
            sampler=self.dsec_sampler,
            spobjs=self.space_objects,
            start_time=to_datetime64_us(self.start_time),
            end_time=to_datetime64_us(self.end_time),
        )
        logger.debug("sample and propagate done")

        spobjs_interpolators = [
            self.interpolator_class(spobj_smpl_states, spobj_smpl_dsec)
            for spobj_smpl_dsec, spobj_smpl_states in zip(spobjs_smpl_dsec, spobjs_smpl_states)
        ]
        logger.debug("interpolators done")

        passages_lists = find_passages(
            station_map=self.station_map,
            station_id_pairs=self.station_id_pairs,
            space_objects=self.space_objects,
            epoch=self.epoch,
            spobjs_smpl_dsec=spobjs_smpl_dsec,
            spobjs_smpl_states=spobjs_smpl_states,
        )
        logger.debug("find_passages done")

        sim_units_param = derive_simulation_unit_params(
            sch=self.schedule,
            station_map=self.station_map,
            exp_detail_map=self.exp_detail_map,
            space_objects=self.space_objects,
            passages_lists=passages_lists,
            spobjs_interpolators=spobjs_interpolators,
        )
        # filter away param with empty schedule
        sim_units_param = [
            p for p in sim_units_param if len(p.schedule[schedule._K.multi_index]) > 0
        ]
        logger.info(f"prepare_simulation_unit_params done")

        return sim_units_param

    def run(self) -> tuple[list[Observation], list[SimulationUnit]]:
        logger.debug("starting stx mrx sim")

        self.sim_units = []
        self.obss = []

        sim_units_param = self.prepare_simulation_unit_params()

        pbar = tqdm(desc="simulating", total=len(sim_units_param))

        for param in sim_units_param:
            sim_unit = SimulationUnit.from_passages_over_tx_rx_station_pair(param)
            self.sim_units.append(sim_unit)

            sim_unit.simulate()
            self.obss.extend(sim_unit.observations)
            pbar.update(1)
        logger.debug("simulation done")

        pbar.close()

        return self.obss, self.sim_units
