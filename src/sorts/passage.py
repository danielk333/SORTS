import typing as t
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from sorts import types, radar
from sorts.types import Datetime64_us, EcefStates, Float64_as_sec, Datetime_Like, StateType
from sorts.utils import to_datetime64_us
from sorts.space_object import SpaceObject


@dataclass(kw_only=True)
class Passage:
    """Represent a passage of a space object over the field of view of a single TX, multiple simultaneous RX radar stations."""

    space_object: SpaceObject
    tx_station: radar.Station
    rx_stations: list[radar.Station]
    epoch: types.Datetime64_us
    time_range: types.TimeRange_us
    """The start time and end time of the passage, a right-open interval"""


def find_simultaneous_passages(
    dt: npt.NDArray[Float64_as_sec],
    space_object: SpaceObject,
    states: EcefStates,
    tx_station: radar.Station,
    rx_stations: t.Sequence[radar.Station],
    epoch: Datetime_Like,
    fov_kw=None,
) -> list[Passage]:
    """
    Finds all passages that are simultaneously inside all tx, rx stations' FOV.
    """
    # NOTE: based on the `find_passes` func in `src/sorts/passes.py`

    epoch = to_datetime64_us(epoch)

    passages: list[Passage] = []
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

        start_time: Datetime64_us = (
            t.cast(np.timedelta64, (dt[ps_inds[0]] * 1e6).astype("timedelta64[us]")) + epoch
        )

        end_time: Datetime64_us = (
            t.cast(np.timedelta64, (dt[ps_inds[-1]] * 1e6).astype("timedelta64[us]")) + epoch
        )

        time_range = (start_time, end_time)
        passages.append(
            Passage(
                space_object=space_object,
                tx_station=tx_station,
                rx_stations=list(rx_stations),
                epoch=epoch,
                time_range=time_range,
            )
        )

    return passages


def group_passages_by_tx_rx_station_pair(
    passages: t.Sequence[Passage],
) -> dict[tuple[radar.StationId, radar.StationId], list[Passage]]:
    """
    Group passages by tx-rx station pair.

    For system with multi-rx station, the same passage will be referenced multiple times after the grouping,
    once per unqiue tx-rx pair.
    """

    groupped_passages: dict[tuple[radar.StationId, radar.StationId], list[Passage]] = {}

    for passage in passages:
        for rx_station in passage.rx_stations:
            tx_station_id = passage.tx_station.uid
            rx_station_id = rx_station.uid

            if (tx_station_id, rx_station_id) in groupped_passages:
                groupped_passages[(tx_station_id, rx_station_id)].append(passage)
            else:
                groupped_passages[(tx_station_id, rx_station_id)] = [passage]

    return groupped_passages
