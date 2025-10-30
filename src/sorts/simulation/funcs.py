"""
Functions for core functionalities of this subpackage

- Intended to be imported as a whole module when consuming
"""

# TODO: this module can be moved to top level as helper funcs of sorts pkg?

import typing as t, random
import numpy as np
import numpy.typing as npt
from sorts.types import Datetime64_us, EcefStates, Float64_as_sec, Datetime_Like
from sorts.utils import to_datetime64_us
from sorts.radar import Station
from sorts.space_object import SpaceObject
from .types import SimultaneousPassage, Passage, SpaceObjectJacobianTuple


def find_simultaneous_passages(
    dt: npt.NDArray[Float64_as_sec],
    space_object: SpaceObject,
    states: EcefStates,
    tx_station: Station,
    rx_stations: t.Sequence[Station],
    epoch: Datetime_Like,
    fov_kw=None,
) -> list[SimultaneousPassage]:
    """
    Finds all passages that are simultaneously inside all tx, rx stations' FOV.
    """
    # NOTE: based on the `find_passes` func in `src/sorts/passes.py`

    epoch = to_datetime64_us(epoch)

    passages: list[SimultaneousPassage] = []
    if fov_kw is None:
        fov_kw = {}

    enu = []
    check = np.full((len(dt),), True, dtype=bool)
    for station in [tx_station, *rx_stations]:
        enu_st = station.enu(states)
        enu.append(enu_st)

        check_st = station.field_of_view(states, **fov_kw)
        check = np.logical_and(check, check_st)

    inds = np.where(check)[0]

    if len(inds) == 0:
        return passages

    dind = np.diff(inds)
    splits = np.where(dind > 1)[0]

    splits = np.insert(splits, 0, -1)
    splits = np.insert(splits, len(splits), len(inds) - 1)
    splits += 1
    for si in range(len(splits) - 1):
        ps_inds = inds[splits[si] : splits[si + 1]]
        if len(ps_inds) == 0:
            continue

        start_time: Datetime64_us = t.cast(
            np.timedelta64, (dt[ps_inds[0]] * 1e6).astype("timedelta64[us]")
        ) + np.datetime64(epoch)

        end_time: Datetime64_us = t.cast(
            np.timedelta64, (dt[ps_inds[-1]] * 1e6).astype("timedelta64[us]")
        ) + np.datetime64(epoch)

        time_range = (start_time, end_time)
        passages.append(
            SimultaneousPassage(
                space_object=space_object,
                tx_station=tx_station,
                rx_stations=list(rx_stations),
                epoch=epoch,
                time_range=time_range,
            )
        )

    return passages


def find_passages(
    dt: npt.NDArray[Float64_as_sec],
    space_object: SpaceObject,
    states: EcefStates,
    tx_station: Station,
    rx_station: Station,
    epoch: Datetime_Like,
    fov_kw=None,
) -> list[Passage]:
    """
    Finds all passages that are simultaneously inside a tx-rx station pair's FOV.
    """

    passages = find_simultaneous_passages(
        dt=dt,
        space_object=space_object,
        states=states,
        tx_station=tx_station,
        rx_stations=[rx_station],
        epoch=epoch,
        fov_kw=fov_kw,
    )

    passages = [
        Passage(
            space_object=ps.space_object,
            tx_station=ps.tx_station,
            rx_station=rx_station,
            epoch=ps.epoch,
            time_range=ps.time_range,
        )
        for ps in passages
    ]

    return passages


def duplicate_and_perturbate_space_objects(
    spobjs: list[SpaceObject], pert_ratio: float
) -> list[SpaceObjectJacobianTuple]:
    # duplicate list items
    spobjs_jacobian_tuples = [(spobj, spobj, spobj, spobj, spobj, spobj) for spobj in spobjs]

    # TODO: confirm with daniel the perturbation
    # perturbate
    for spobjs_jacobian_tuple in spobjs_jacobian_tuples:
        for idx, spobj in enumerate(spobjs_jacobian_tuple):
            # the original spobj are left intact
            if idx != 0:
                perturbated_param = {
                    pkey: pval * (1 + random.random() * pert_ratio)
                    for pkey, pval in spobj.parameters.items()
                    if pkey not in ["m"]
                }
                spobj.update(**perturbated_param)

    return spobjs_jacobian_tuples
