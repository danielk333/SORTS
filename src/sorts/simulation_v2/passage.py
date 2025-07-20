import typing as t
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from sorts.types import Datetime64_us, EcefStates, Float64_as_sec
from sorts.radar.tx_rx import Station
from sorts.space_object import SpaceObject


@dataclass(kw_only=True)
class Passage:
    # id: int # TODO: revisit if this is needed
    # TODO: add ENU and/or ECEF states?

    space_object: SpaceObject
    tx_station: Station
    rx_station: Station
    time_range: tuple[Datetime64_us, Datetime64_us]
    """The start time and end time of the passage, inclusive on both ends"""


def find_passages(
    dts: npt.NDArray[Float64_as_sec],
    space_object: SpaceObject,
    states: EcefStates,
    tx_station: Station,
    rx_station: Station,
    epoch: datetime,
    fov_kw=None,
) -> list[Passage]:
    """
    Finds all find_passages that are simultaneously inside a tx-rx station pair's FOV.
    """
    # NOTE: based on the `find_passes` func in `src/sorts/passes.py`

    passages: list[Passage] = []
    if fov_kw is None:
        fov_kw = {}

    enu = []
    check = np.full((len(dts),), True, dtype=bool)
    for station in [tx_station, rx_station]:
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
            np.timedelta64, (dts[ps_inds[0]] * 1e6).astype("timedelta64[us]")
        ) + np.datetime64(epoch)

        end_time: Datetime64_us = t.cast(
            np.timedelta64, (dts[ps_inds[-1]] * 1e6).astype("timedelta64[us]")
        ) + np.datetime64(epoch)

        time_range = (start_time, end_time)
        passages.append(
            Passage(
                space_object=space_object,
                tx_station=tx_station,
                rx_station=rx_station,
                time_range=time_range,
            )
        )

    return passages
