import typing as t
from datetime import datetime
import numpy as np
import numpy.typing as npt
from sorts.radar.tx_rx import Station
from sorts.types import Datetime64_us, EcefStates


def find_simultaneous_passes_time_ranges(
    dt_s_arr: npt.NDArray[np.float64],
    states: EcefStates,
    stations: list[Station],
    epoch: datetime,
    fov_kw=None,
) -> t.Sequence[tuple[Datetime64_us, Datetime64_us]]:
    """
    Finds all passes that are simultaneously inside a multiple stations FOV's.
    """

    time_ranges: list[tuple[Datetime64_us, Datetime64_us]] = []
    if fov_kw is None:
        fov_kw = {}

    enu = []
    check = np.full((len(dt_s_arr),), True, dtype=bool)
    for station in stations:
        enu_st = station.enu(states)
        enu.append(enu_st)

        check_st = station.field_of_view(states, **fov_kw)
        check = np.logical_and(check, check_st)

    inds = np.where(check)[0]

    if len(inds) == 0:
        return time_ranges

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
            np.timedelta64, (dt_s_arr[ps_inds[0]] * 1e6).astype("timedelta64[us]")
        ) + np.datetime64(epoch)

        end_time: Datetime64_us = t.cast(
            np.timedelta64, (dt_s_arr[ps_inds[-1]] * 1e6).astype("timedelta64[us]")
        ) + np.datetime64(epoch)

        time_range = (start_time, end_time)
        time_ranges.append(time_range)

    return time_ranges
