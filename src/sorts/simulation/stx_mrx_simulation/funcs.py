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
from .observation import Observation


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


# TODO: better move to `simulation_unit` module?
# TODO: go through its logic again; similar to `get_indexer_per_measurement`,
#   now we have `stn_num`, `simult_num` in index, things can likely be done differently
def derive_observations(
    passages: list[Passage], schedule: Schedule, sim_unit: SimulationUnit
) -> list[Observation]:
    """Derive observations"""

    obss: list[Observation] = []

    # early return for empty cases
    # NOTE: this is particularly needed because `.loc` will throw KeyError for non-existence keys
    # TODO: add test case for empty case?
    if (
        len(passages) == 0
        or not (schedule._data[Schedule._K.stn_num] == sim_unit.tx_station.uid).any()
        or not (schedule._data[Schedule._K.stn_num] == sim_unit.rx_station.uid).any()
    ):
        return obss

    # NOTE: xarray simplify/collapse MultiIndex when filtering a level to an exact value,
    #   we filter on the top level "multi_index' with a tuple here to prevent it
    tx_schdata = schedule._data.loc[
        {Schedule._K.multi_index: (slice(None), sim_unit.tx_station.uid, slice(None), slice(None))}
    ]
    rx_schdata = schedule._data.loc[
        {Schedule._K.multi_index: (slice(None), sim_unit.rx_station.uid, slice(None), slice(None))}
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

        # for tx_obs_idxer in tx_obs_idxers:
        #     for rx_obs_idxer in rx_obs_idxers:
        #         indexer = ObservationIndexer(tx=tx_obs_idxer, rx=rx_obs_idxer)
        #         obs = Observation(
        #             passage=passage,
        #             indexer=indexer,
        #             simulation_unit=sim_unit,
        #             tx_schedule=tx_schedule,
        #             rx_schedule=rx_schedule,
        #         )
        #         obss.append(obs)

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

        for stn_id_pair in spec["station_id_pairs"]:
            tx_stn = spec["station_map"][stn_id_pair[0]]
            rx_stn = spec["station_map"][stn_id_pair[1]]

            passages_of_spobj.extend(
                funcs.find_passages(
                    dt=spobj_smpl_dsec,
                    space_object=spobj,
                    states=spobj_smpl_states,
                    tx_station=tx_stn,
                    rx_station=rx_stn,
                    epoch=spec["epoch"],
                )
            )

        passages_list.append(passages_of_spobj)

    return passages_list


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
            tx_stn = spec["station_map"][stn_id_pair[0]]
            rx_stn = spec["station_map"][stn_id_pair[1]]

            filtered_sch = spec["schedule"].filter_by_time_ranges(
                [ps["time_range"] for ps in passages]
            )

            params.append(
                {
                    "id": str(len(params)),
                    "passages": passages,
                    "spobj": spobj,
                    "spobj_interp": spobj_states_interp,
                    "tx_station": tx_stn,
                    "rx_station": rx_stn,
                    "schedule": filtered_sch,
                    "exp_detail_map": spec["exp_detail_map"],
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

    vector_len = len(state_data[_K.multi_index])

    # will be populated to [tx_gain_arr, rx_gain_arr]
    gain_arr_list: list[npt.NDArray[np.float64]] = []

    for beam, spobj_stn_enu in zip([tx_stn.beam, rx_stn.beam], [spobj_tx_enu, spobj_rx_enu]):
        # we broadcast_to/reshape the beam params according to the input state length
        # TODO: the `gain` method being dependent on beam's states are not helpful here;
        #   we need a gain func that take all param as args
        # NOTE: we need to mutate `beam.parameters` here,
        #   but such mutation would create unexpect conditions if we directly mutate it.
        #   therefore we mutate on a copy of `beam.parameters` and restore the original one afterwards
        #   (since those `beam.parameters` are state that will be bounded with the life time of the object)
        #   (e.g. if the same `tx_stn` is used in another `calc_gain`, the mutation from prev will persist)

        # early return for empty cases
        # NOTE: this is particularly needed because some `.gain` does not work with empty parameters (e.g. beam.parameters["pointing"])
        # TODO: add test case for empty case?
        if len(state_data[_K.multi_index]) == 0:
            gain_arr_list.append(np.empty(0, dtype=np.float64))

        else:
            orig_beam_params = beam.parameters
            mut_beam_params = beam.parameters.copy()
            beam.parameters = mut_beam_params

            for key, val in mut_beam_params.items():
                if key == "pointing":
                    beam.parameters["pointing"] = state_data[_K.tx_pointing].to_numpy()
                if key in beam.parameters_shape:
                    shape: tuple[int, ...] = beam.parameters_shape[key]
                    beam.parameters[key] = np.broadcast_to(
                        val.reshape((*shape, 1)), (*shape, vector_len)
                    )
                else:
                    beam.parameters[key] = np.full(vector_len, val, dtype=np.float64)

            gain_arr_list.append(beam.gain(spobj_stn_enu[:3]))
            beam.parameters = orig_beam_params

    state_data[_K.gain_tx] = (_K.multi_index, gain_arr_list[0])
    state_data[_K.gain_rx] = (_K.multi_index, gain_arr_list[1])

    return state_data
