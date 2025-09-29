"""
Functions for core functionalities of this subpackage

- Intended to be imported as a whole module when consuming
"""

from __future__ import annotations
import logging, typing as t
import numpy as np
import numpy.typing as npt
import sorts
from tqdm import tqdm
from sorts.interpolation import Interpolator
from sorts.radar import Station, StationId
from sorts.types import Float64_as_sec, EcefStates, Datetime64_us, EnuCoordinates
from sorts.schedule import Schedule
from sorts.simulation.types import Passage
from sorts.simulation import funcs
from .simulation_unit import SimulationUnit
from .observation import ObservationIndexer, Observation


if t.TYPE_CHECKING:
    from .simulation_unit import StateData
    from .simulation_unit import FromPassagesOverTxRxStationPairParam
    from .stx_mrx_simulation import Spec, SpaceObjectDsecSampler


logger = logging.getLogger(__name__)


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
        spobjs_smpl_dsec.append(sampler(spobj.state, start_time, end_time))

    spobjs_smpl_states: list[EcefStates] = []
    for spobj, spobj_smpl_dsec in tqdm(
        zip(spobjs, spobjs_smpl_dsec),
        desc="propagating spobjs states at sampled dt",
        total=len(spobjs),
    ):
        spobjs_smpl_states.append(spobj.get_state(spobj_smpl_dsec))

    return spobjs_smpl_dsec, spobjs_smpl_states


# TODO: use `TxRxTuple` type for return value?
# TODO: can be combined with 'stx_mrx_simulation.funcs.find_passages'?
def group_passages_by_tx_rx_station_pair(
    passages: list[Passage],
) -> dict[tuple[StationId, StationId], list[Passage]]:
    groupped_passages: dict[tuple[StationId, StationId], list[Passage]] = {}

    for passage in passages:
        tx_station_id = passage["tx_station"].uid
        rx_station_id = passage["rx_station"].uid

        if (tx_station_id, rx_station_id) in groupped_passages:
            groupped_passages[(tx_station_id, rx_station_id)].append(passage)
        else:
            groupped_passages[(tx_station_id, rx_station_id)] = [passage]

    return groupped_passages


# TODO: go through its logic again; similar to `get_indexer_per_measurement`,
#   now we have `stn_num`, `simult_num` in index, things can likely be done differently
def derive_observations(
    passages: list[Passage], schedule: Schedule, sim_unit: SimulationUnit
) -> list[Observation]:
    """Derive observations"""

    obss: list[Observation] = []

    # NOTE: xarray simplify/collapse MultiIndex when filtering a level to an exact value,
    #   we filter on the top level "multi_index' with a tuple here to prevent it
    tx_schdata = schedule._data.loc[
        {Schedule._K.multi_index: (slice(None), slice(None), sim_unit.tx_station.uid, slice(None))}
    ]
    rx_schdata = schedule._data.loc[
        {Schedule._K.multi_index: (slice(None), slice(None), sim_unit.rx_station.uid, slice(None))}
    ]
    tx_schedule = Schedule(tx_schdata)
    rx_schedule = Schedule(rx_schdata)

    for passage in passages:
        tx_obs_idxers = tx_schedule.filter_by_time_range(
            passage["time_range"]
        ).get_indexer_per_measurement(is_split_simult=False, is_copy=True)

        rx_obs_idxers = rx_schedule.filter_by_time_range(
            passage["time_range"]
        ).get_indexer_per_measurement(is_split_simult=True, is_copy=True)

        for tx_obs_idxer in tx_obs_idxers:
            for rx_obs_idxer in rx_obs_idxers:
                indexer = ObservationIndexer(tx=tx_obs_idxer, rx=rx_obs_idxer)
                obs = Observation(
                    passage=passage,
                    indexer=indexer,
                    simulation_unit=sim_unit,
                    tx_schedule=tx_schedule,
                    rx_schedule=rx_schedule,
                )
                obss.append(obs)

    return obss


def find_passages(
    spec: Spec,
    spobjs_smpl_dsec: list[npt.NDArray[Float64_as_sec]],
    spobjs_smpl_states: list[EcefStates],
) -> list[list[Passage]]:
    """
    Find passages for each space objects over the simulation period.

    Returns a `list[Passage]` per space object.
    """

    passages_list: list[list[Passage]] = []

    for spobj, spobj_smpl_dsec, spobj_smpl_states in zip(
        spec["space_objects"],
        spobjs_smpl_dsec,
        spobjs_smpl_states,
    ):
        passages_of_spobj: list[Passage] = []

        # TODO: this assume spec["rx_stations"] ordering is the same as spec["rx_schedules"],
        #   which should be correct but should not be relied on.
        for rx_station in spec["rx_stations"]:
            passages_of_spobj.extend(
                funcs.find_passages(
                    dt=spobj_smpl_dsec,
                    space_object=spobj,
                    states=spobj_smpl_states,
                    tx_station=spec["tx_station"],
                    rx_station=rx_station,
                    epoch=spec["epoch"],
                )
            )

        passages_list.append(passages_of_spobj)

    return passages_list


# TODO: minor cleanup needed
#   - `enumerate` to get space_objects by `idx` can likely be simplified, with small adj in params/props
#   - the loops might be simplified a bit as well
def derive_simulation_unit_params(
    spec: Spec,
    passages_lists: list[list[Passage]],
    spobjs_interpolators: list[Interpolator],
) -> list[FromPassagesOverTxRxStationPairParam]:
    """
    Derive a list of param for the `from_passages_over_tx_rx_station_pair` constructor of `SimulationUnit`

    NOTE: Integers (casted to `str`) are used as `SimulationUnit`s' id
    """

    params: list[FromPassagesOverTxRxStationPairParam] = []

    for spobj, passages_of_a_spobj, spobj_states_interp in zip(
        spec["space_objects"], passages_lists, spobjs_interpolators
    ):
        groupped_passages = group_passages_by_tx_rx_station_pair(passages_of_a_spobj)

        for stn_id_pair, passages in groupped_passages.items():
            rx_stn = next((stn for stn in spec["rx_stations"] if stn.uid == stn_id_pair[1]))

            params.append(
                {
                    "id": str(len(params)),
                    "passages": passages,
                    "spobj": spobj,
                    "spobj_interp": spobj_states_interp,
                    "tx_station": spec["tx_station"],
                    "rx_station": rx_stn,
                    "schedule": spec["schedule"],
                }
            )

    return params


def calc_gain(
    state_data: StateData,
    tx_stn: Station,
    rx_stn: Station,
    spobj_tx_enu: EnuCoordinates,
    spobj_rx_enu: EnuCoordinates,
) -> StateData:
    # NOTE: used lazy import here to avoid circular import
    from .simulation_unit import _K

    size = len(state_data[_K.time])

    # NOTE: looping is needed becase passing in a ndarray of pointing will trigger exception when calculating gain
    #   refs:
    #   - `pyant/beam.py` `L235` `assert vector_cnt <= max_vectors, "Too many vector valued parameters"`
    #   - `pyant/models/array.py` `L185` `params, shape = self.get_parameters(ind, named=True, max_vectors=0)`
    tx_gain_arr = np.full(size, 0.0, dtype=np.float64)
    rx_gain_arr = np.full(size, 0.0, dtype=np.float64)
    for idx in range(len(state_data[_K.time])):
        tx_stn.beam.point(state_data[_K.tx_pointing][:, 0].to_numpy())
        tx_gain_arr[idx] = tx_stn.beam.gain(spobj_tx_enu[:3, idx])

        rx_stn.beam.point(state_data[_K.rx_pointing][:, 0].to_numpy())
        rx_gain_arr[idx] = rx_stn.beam.gain(spobj_rx_enu[:3, idx])

    state_data[_K.gain_tx] = (_K.time, tx_gain_arr)
    state_data[_K.gain_rx] = (_K.time, rx_gain_arr)

    return state_data
