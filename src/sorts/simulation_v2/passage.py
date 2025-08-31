import typing as t
import numpy as np
import numpy.typing as npt
from sorts.types import Datetime64_us, EcefStates, Float64_as_sec, Datetime_Like
from sorts.utils import to_datetime64_us
from sorts.radar.tx_rx import Station
from sorts.space_object import SpaceObject
from sorts import schedule_v2 as schedule
from sorts.schedule_v2.schedule import Schedule, ScheduleNdarrayDict2, ExperimentDetail


class Passage(t.TypedDict):
    """
    A TypedDict of params. Represent a passage of a space object over the field of view of a TX-RX radar station pair.
    """

    # id: int # TODO: revisit if this is needed
    # TODO: add ENU and/or ECEF states?

    space_object: SpaceObject
    tx_station: Station
    rx_station: Station
    epoch: Datetime64_us
    time_range: tuple[Datetime64_us, Datetime64_us]
    """The start time and end time of the passage, a right-open interval"""


# TODO: can be removed?
# TODO: rename to `MeasurementPassage`?
class ExperimentPassage(Passage):
    """
    A TypedDict of params.
    Represent a passage of a space object over the field of view of a TX-RX radar station pair,
    with additional data that facilitate the measurement calculations.

    Contain these fields in addition to those in TypedDict `Passage`:
    - `experiment_detail`
    - `tx_schedule` (The TX schedule during the passage.)
    - `rx_schedule` (The RX schedule during the passage.)
    """

    experiment_detail: ExperimentDetail
    tx_schedule: ScheduleNdarrayDict2
    rx_schedule: ScheduleNdarrayDict2


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
    Finds all find_passages that are simultaneously inside a tx-rx station pair's FOV.
    """
    # NOTE: based on the `find_passes` func in `src/sorts/passes.py`

    epoch = to_datetime64_us(epoch)

    passages: list[Passage] = []
    if fov_kw is None:
        fov_kw = {}

    enu = []
    check = np.full((len(dt),), True, dtype=bool)
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
            np.timedelta64, (dt[ps_inds[0]] * 1e6).astype("timedelta64[us]")
        ) + np.datetime64(epoch)

        end_time: Datetime64_us = t.cast(
            np.timedelta64, (dt[ps_inds[-1]] * 1e6).astype("timedelta64[us]")
        ) + np.datetime64(epoch)

        time_range = (start_time, end_time)
        passages.append(
            {
                "space_object": space_object,
                "tx_station": tx_station,
                "rx_station": rx_station,
                "epoch": epoch,
                "time_range": time_range,
            }
        )

    return passages
