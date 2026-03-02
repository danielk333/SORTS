from __future__ import annotations
import logging, typing as t, pickle
from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyorb
import sorts
from tqdm import tqdm
from sorts import types, schedule, controller, simulation
from sorts.types import Datetime_Like, Float64_as_sec, Datetime64_us, EcefStates
from sorts.utils import to_datetime64_us
from sorts.radar import Station, StationId
from sorts.interpolated_propagation import InterpolatedPropagation
from .types import Passage
from .simulation_unit import (
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
        # TODO: make sure this is not broken
        tx_station_id = passage.tx_station.uid
        rx_station_id = passage.rx_stations[0].uid

        if (tx_station_id, rx_station_id) in groupped_passages:
            groupped_passages[(tx_station_id, rx_station_id)].append(passage)
        else:
            groupped_passages[(tx_station_id, rx_station_id)] = [passage]

    return groupped_passages


def find_passages(
    station_map: dict[StationId, Station],
    station_id_pairs: t.Sequence[tuple[StationId, StationId]],
    space_objects: t.Sequence[sorts.SpaceObject],
    epoch: Datetime_Like,
    spobjs_smpl_dsec: list[npt.NDArray[Float64_as_sec]],
    spobjs_smpl_states: list[EcefStates],
) -> dict[int, list[Passage]]:
    """
    Find passages for each space objects over the simulation period.

    Returns a `list[Passage]` per space object.
    """

    passages_map: dict[int, list[Passage]] = {}

    for spobj_idx, (spobj, spobj_smpl_dsec, spobj_smpl_states) in enumerate(
        zip(
            space_objects,
            spobjs_smpl_dsec,
            spobjs_smpl_states,
        )
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

        passages_map[spobj_idx] = passages_of_spobj

    return passages_map


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

    save_table_name: t.Final = "_StxMrxSimulation"

    def __init__(
        self,
        station_map: dict[StationId, Station],
        station_id_pairs: t.Sequence[tuple[StationId, StationId]],
        schedule_db: schedule.ScheduleDb,
        exp_detail_map: types.ExperimentDetailMap,
        epoch: Datetime_Like,
        start_time: Datetime_Like,
        end_time: Datetime_Like,
        space_objects: t.Sequence[sorts.SpaceObject],
        interpolated_propagations: t.Sequence[InterpolatedPropagation],
        passages: dict[int, list[Passage]] | None = None,
        progress: bool = False,
    ):
        self.station_map = station_map
        self.station_id_pairs = station_id_pairs
        self.schedule_db = schedule_db
        self.exp_detail_map = exp_detail_map
        self.epoch = epoch
        self.start_time = start_time
        self.end_time = end_time
        self.space_objects = space_objects
        self.interpolated_propagations = interpolated_propagations
        self.progress = progress
        self.passages = passages

        # indexed by space object index in spobj list
        self.sim_units: dict[int, list[SimulationUnit]] = {}
        self.obss: dict[int, list[Observation]] = {}

    @classmethod
    def from_controllers(
        cls,
        controllers: t.Sequence[controller.ControllerBase],
        schedule: schedule.ScheduleDb,
        epoch: Datetime_Like,
        start_time: Datetime_Like,
        end_time: Datetime_Like,
        space_objects: t.Sequence[sorts.SpaceObject],
        interpolated_propagations: t.Sequence[InterpolatedPropagation],
        passages: dict[int, list[Passage]] | None = None,
    ):
        """A constructor method"""
        # TODO: - the exp details are already computed outside? Should the `controllers` field be
        # removed? or this classmethod? or what?
        #
        # Notes from Hin, 2025-11-21:
        #   - both `exp_id_stn_id_pairs_map` and `ExperimentDetail` are currently owned by the controller;
        #   - the func `priority_scheduling` evolved to requires `exp_id_stn_id_pairs_map` at some point,
        #     and therefore it is sometimes found as an explicitly variable in simulation experiment file as well
        #   - we can re-work info flow later but this is needed atm

        schedule_db = schedule

        stn_map: dict[StationId, Station] = {}
        stn_id_pairs_set: set[tuple[StationId, StationId]] = set()
        exp_detail_map: types.ExperimentDetailMap = {}

        for ctrl in controllers:
            stn_map.update(ctrl.get_station_map())

            for pairs in ctrl.get_experiment_id_station_id_pairs_map().values():
                stn_id_pairs_set.update(pairs)

            exp_detail = ctrl.get_experiment_detail()
            exp_detail_map[exp_detail.id] = exp_detail

        return cls(
            station_map=stn_map,
            station_id_pairs=list(stn_id_pairs_set),
            schedule_db=schedule_db,
            exp_detail_map=exp_detail_map,
            epoch=epoch,
            start_time=start_time,
            end_time=end_time,
            space_objects=space_objects,
            interpolated_propagations=interpolated_propagations,
            passages=passages,
        )

    def prepare_simulation_unit_params(
        self,
    ) -> dict[int, list[FromPassagesOverTxRxStationPairParam]]:
        epoch = to_datetime64_us(self.epoch)

        # TODO: this is ugly and can be fixed
        dsecs = [
            (interp.times - epoch) / np.timedelta64(1, "s")
            for interp in self.interpolated_propagations
        ]
        states = [interp.states for interp in self.interpolated_propagations]
        if self.passages is None:
            passages_map = find_passages(
                station_map=self.station_map,
                station_id_pairs=self.station_id_pairs,
                space_objects=self.space_objects,
                epoch=self.epoch,
                spobjs_smpl_dsec=dsecs,
                spobjs_smpl_states=states,
            )
            # for obj, objps in enumerate(passages_lists):
            #     print(f"- {obj}")
            #     for ps in objps:
            #         print(f"--  {ps.time_range[0]}")
            logger.debug("find_passages done")
        else:
            passages_map = self.passages

        sim_units_param: dict[int, list[FromPassagesOverTxRxStationPairParam]] = {}
        for spobj_idx, spobj in enumerate(self.space_objects):
            sim_unit_params: list[FromPassagesOverTxRxStationPairParam] = []

            spobj_states_interp = self.interpolated_propagations[spobj_idx].interpolator
            passages_of_a_spobj = passages_map[spobj_idx]

            groupped_passages = group_passages_by_tx_rx_station_pair(passages_of_a_spobj)

            for stn_id_pair, passages in groupped_passages.items():
                tx_stn = self.station_map[stn_id_pair[0]]
                rx_stn = self.station_map[stn_id_pair[1]]

                tx_rx_pointing_pairs = pd.concat(
                    [
                        self.schedule_db.get_tx_rx_pointing_pairs(
                            start_time=ps.time_range[0],
                            end_time=ps.time_range[1],
                            tx_stn_num=stn_id_pair[0],
                            rx_stn_num=stn_id_pair[1],
                        )
                        for ps in passages
                    ]
                )
                tx_rx_pointing_pairs = tx_rx_pointing_pairs.sort_values(
                    by=types.SimulationUnitKey.time
                )

                # NOTE: Integers (casted to `str`) are used as `SimulationUnit`s' id
                sim_unit_params.append(
                    FromPassagesOverTxRxStationPairParam(
                        id=str(len(sim_units_param)),
                        passages=passages,
                        spobj=spobj,
                        spobj_interp=spobj_states_interp,
                        tx_station=tx_stn,
                        rx_station=rx_stn,
                        tx_rx_pointing_pairs=tx_rx_pointing_pairs,
                        exp_detail_map=self.exp_detail_map,
                    )
                )
            sim_unit_params = [p for p in sim_unit_params if len(p.tx_rx_pointing_pairs) > 0]
            sim_units_param[spobj_idx] = sim_unit_params

        # filter away param with empty schedule
        logger.debug("prepare_simulation_unit_params done")

        return sim_units_param

    def run(self) -> tuple[dict[int, list[Observation]], dict[int, list[SimulationUnit]]]:
        logger.debug("starting stx mrx sim")

        self.sim_units = {}
        self.obss = {}

        sim_units_param = self.prepare_simulation_unit_params()

        pbar = None
        if self.progress:
            pbar = tqdm(desc="simulating", total=len(sim_units_param))

        for spobj_idx, params in sim_units_param.items():
            self.sim_units[spobj_idx] = []
            self.obss[spobj_idx] = []
            for param in params:
                sim_unit = SimulationUnit.from_passages_over_tx_rx_station_pair(param)
                self.sim_units[spobj_idx].append(sim_unit)

                sim_unit.simulate()
                self.obss[spobj_idx].extend(sim_unit.observations)
            if self.progress and pbar is not None:
                pbar.update(1)
        if self.progress and pbar is not None:
            pbar.close()
        logger.debug("simulation done")

        return self.obss, self.sim_units
