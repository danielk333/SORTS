from __future__ import annotations
import logging, typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd
import sorts
from sorts import (
    types,
    utils,
    signals,
    space_object,
    interpolation,
    radar,
    schedule,
    controller,
    passage,
    simulation,
)
from sorts.types import Datetime_Like, Float64_as_sec, EcefStates
from sorts.radar import Station, StationId
from sorts.interpolated_propagation import InterpolatedPropagation
from sorts.simulation import tx_rx_pair_state

logger = logging.getLogger(__name__)


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
        schedule_db: schedule.ScheduleDb,
        exp_detail_map: types.ExperimentDetailMap,
        epoch: Datetime_Like,
        start_time: Datetime_Like,
        end_time: Datetime_Like,
        space_objects: t.Sequence[sorts.SpaceObject],
        interpolated_propagations: t.Sequence[InterpolatedPropagation],
        passages: list[passage.Passage],
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
        passages: list[passage.Passage],
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
        self, passages_map: dict[int, list[passage.Passage]]
    ) -> dict[int, list[FromPassagesOverTxRxStationPairParam]]:
        sim_units_param: dict[int, list[FromPassagesOverTxRxStationPairParam]] = {}

        for spobj_idx, spobj in enumerate(self.space_objects):
            sim_unit_params: list[FromPassagesOverTxRxStationPairParam] = []

            spobj_states_interp = self.interpolated_propagations[spobj_idx].interpolator
            passages_of_a_spobj = passages_map[spobj_idx]

            groupped_passages = group_passages_by_tx_rx_station_pair(passages_of_a_spobj)

            for stn_id_pair, passages in groupped_passages.items():
                tx_stn = self.station_map[stn_id_pair[0]]
                rx_stn = self.station_map[stn_id_pair[1]]

                tx_rx_pointing_pairs = get_pointing_pairs_by_stn_id_pair_passages(
                    stn_id_pair=stn_id_pair,
                    passages=passages,
                    schedule_db=self.schedule_db,
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
                        tx_rx_pointing_pairs=schedule.TxRxPointingPairs(tx_rx_pointing_pairs),
                        exp_detail_map=self.exp_detail_map,
                    )
                )
            sim_unit_params = [p for p in sim_unit_params if len(p.tx_rx_pointing_pairs) > 0]
            sim_units_param[spobj_idx] = sim_unit_params

        # filter away param with empty schedule
        logger.debug("prepare_simulation_unit_params done")

        return sim_units_param


# TODO: the docstring is copied from `SimulationUnit`, and needs update
@dataclass
class FromPassagesOverTxRxStationPairParam:
    """
    Contains all the params and results for a unit of simulation calculation.

    Notes about the state data:
    - It is stored in a private attribute `_state`
    - It can contain data for more than 1 passage
    - The dataset does not always contains all the columns,
        which ones are available depends on what calculation have been done.
    """

    id: str
    passages: list[passage.Passage]
    spobj: space_object.SpaceObject
    spobj_interp: interpolation.Interpolator
    tx_station: radar.Station
    rx_station: radar.Station
    tx_rx_pointing_pairs: schedule.TxRxPointingPairs # TODO: this is a tmp solution, should refactor this type and dataflow; # fmt: skip
    exp_detail_map: types.ExperimentDetailMap


# TODO: tmp; should be simplified
SimulationResult = t.NewType("SimulationResult", dict[int, list[tx_rx_pair_state.TxRxPairState]])
"""
`NewType` of `dict[int, list[tx_rx_pair_state.TxRxPairState]]`.
Indexed by space object index in spobj list (not `oid` of `SpaceObject`).
"""


# TODO: move to `passage` module?
def group_passages_by_tx_rx_station_pair(
    passages: t.Sequence[passage.Passage],
) -> dict[tuple[StationId, StationId], list[passage.Passage]]:
    """
    Group passages by tx-rx station pair.

    For system with multi-rx station, the same passage will be referenced multiple times after the grouping,
    once per unqiue tx-rx pair.
    """

    groupped_passages: dict[tuple[StationId, StationId], list[passage.Passage]] = {}

    for passage in passages:
        for rx_station in passage.rx_stations:
            # TODO: make sure this is not broken
            tx_station_id = passage.tx_station.uid
            rx_station_id = rx_station.uid

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
) -> dict[int, list[passage.Passage]]:
    """
    Find passages for each space objects over the simulation period.
    """

    passages_map: dict[int, list[passage.Passage]] = {}

    for spobj_idx, (spobj, spobj_smpl_dsec, spobj_smpl_states) in enumerate(
        zip(
            space_objects,
            spobjs_smpl_dsec,
            spobjs_smpl_states,
        )
    ):
        passages_of_spobj: list[passage.Passage] = []

        for stn_id_pair in station_id_pairs:
            tx_stn = station_map[stn_id_pair[0]]
            rx_stn = station_map[stn_id_pair[1]]

            passages_of_spobj.extend(
                passage.find_passages(
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


def get_pointing_pairs_by_stn_id_pair_passages(
    stn_id_pair: tuple[radar.StationId, radar.StationId],
    passages: list[passage.Passage],
    schedule_db: schedule.ScheduleDb,
) -> schedule.TxRxPointingPairs:
    """Create `TxRxPointingPairs` from a list of `Passage`, sorted by time in ascending order."""

    tx_rx_pointing_pairs = pd.concat(
        [
            schedule_db.get_tx_rx_pointing_pairs(
                start_time=ps.time_range[0],
                end_time=ps.time_range[1],
                tx_stn_num=stn_id_pair[0],
                rx_stn_num=stn_id_pair[1],
            )
            for ps in passages
        ]
    )
    tx_rx_pointing_pairs = tx_rx_pointing_pairs.sort_values(
        by=simulation.TxRxPairStateKey.time, ascending=True
    )

    return schedule.TxRxPointingPairs(tx_rx_pointing_pairs)


def gather_tx_rx_pointing_pairs(
    passages: list[passage.Passage],
    schedule_db: schedule.ScheduleDb,
) -> dict[tuple[radar.StationId, radar.StationId], schedule.TxRxPointingPairs]:
    """
    Find the unique tx-rx station pairs among the `passages`,
    then for each pair, gather a `TxRxPointingPairs` from the schedule when the passages pass over the them.
    """

    passages_by_tx_rx_stn_pair = group_passages_by_tx_rx_station_pair(passages)

    pointing_pairs_dict = {
        stn_id_pair: get_pointing_pairs_by_stn_id_pair_passages(
            stn_id_pair=stn_id_pair,
            passages=passages,
            schedule_db=schedule_db,
        )
        for stn_id_pair, passages in passages_by_tx_rx_stn_pair.items()
    }

    return pointing_pairs_dict


def gather_tx_rx_pair_state(
    passages: list[passage.Passage],
    schedule_db: schedule.ScheduleDb,
) -> dict[tuple[radar.StationId, radar.StationId], tx_rx_pair_state.TxRxPairState]:
    """
    Find the unique tx-rx station pairs among the `passages`,
    then for each pair, gather a `TxRxPairState` from the schedule when the passages pass over the them.
    """

    _K = tx_rx_pair_state.TxRxPairStateKey

    pointing_pairs_dict = gather_tx_rx_pointing_pairs(passages=passages, schedule_db=schedule_db)
    pair_state_dict = {
        key: tx_rx_pair_state.TxRxPairState(
            pointing_pairs.set_index([_K.exp_num, _K.rx_simult_num, _K.time])
        )
        for key, pointing_pairs in pointing_pairs_dict.items()
    }

    return pair_state_dict


def simulate(
    state: tx_rx_pair_state.TxRxPairState,
    spobj: space_object.SpaceObject,
    spobj_interp: interpolation.Interpolator,
    tx_station: radar.Station,
    rx_station: radar.Station,
    exp_detail_map: types.ExperimentDetailMap,
) -> tx_rx_pair_state.TxRxPairState:
    """
    Run TX RX simulation calculations.

    Returns:
        The updated state/data.
    """

    _K = tx_rx_pair_state.TxRxPairStateKey

    if tx_station.wavelength is None:
        # TODO: remove this hack; see issues #25 for details
        raise RuntimeError(
            "A hack of injecting `frequency` into `tx_stn.frequency` is currently required for calling `hard_target_snr`"
        )

    epoch = utils.to_datetime64_us(spobj.epoch)
    dsec = (
        (state.index.get_level_values(_K.time).to_numpy() - epoch)
        / np.timedelta64(1, "s")
    ) # fmt: skip
    spobj_states = spobj_interp.get_state(dsec)
    spobj_tx_enu = tx_station.enu(spobj_states)
    spobj_rx_enu = rx_station.enu(spobj_states)

    range_tx: npt.NDArray[types.Float64_as_m] = np.linalg.norm(spobj_tx_enu[:3, :], axis=0)
    range_rx: npt.NDArray[types.Float64_as_m] = np.linalg.norm(spobj_rx_enu[:3, :], axis=0)

    # TODO: can likely use assignment by slice/indexing instead of looping
    powers = np.array(
        [exp_detail_map[n].power for n in state.index.get_level_values(_K.exp_num).to_numpy()],
        dtype=np.float64,
    )
    bandwidths = np.array(
        [exp_detail_map[n].bandwidth for n in state.index.get_level_values(_K.exp_num).to_numpy()],
        dtype=np.float64,
    )
    rx_noise_temps = np.array(
        [exp_detail_map[n].noise_temp for n in state.index.get_level_values(_K.exp_num).to_numpy()],
        dtype=np.float64,
    )

    state = tx_rx_pair_state.calc_gain(
        state=state,
        tx_stn=tx_station,
        rx_stn=rx_station,
        spobj_tx_enu=spobj_tx_enu,
        spobj_rx_enu=spobj_rx_enu,
    )

    snr = signals.hard_target_snr(
        gain_tx=state[_K.gain_tx].to_numpy(),
        gain_rx=state[_K.gain_rx].to_numpy(),
        wavelength=tx_station.wavelength,
        power_tx=powers,
        range_tx_m=range_tx,
        range_rx_m=range_rx,
        diameter=spobj.d,
        bandwidth=bandwidths,
        rx_noise_temp=rx_noise_temps,
        radar_albedo=spobj.properties.get("radar_albedo", 1.0),
    )
    state[_K.snr] = snr

    state[_K.tx_range] = np.linalg.norm(spobj_tx_enu[:3, :], axis=0)

    state[_K.rx_range] = np.linalg.norm(spobj_rx_enu[:3, :], axis=0)

    state[_K.two_way_range] = range_tx + range_rx
    v_tx = np.sum(spobj_tx_enu[:3, :] * spobj_tx_enu[3:, :], axis=0) / range_tx
    v_rx = np.sum(spobj_rx_enu[:3, :] * spobj_rx_enu[3:, :], axis=0) / range_rx
    state[_K.two_way_range_rate] = v_tx + v_rx

    return state
