"""Encapsulates a fundamental component of tracking space objects:
a pass over a geographic location.
Also provides convenience functions for finding passes given states
and stations and sorting structures of passes in particular ways.

"""

import typing as t
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from .radar.radars.composite_key import RadarStationCompositeKey
from .radar.tx_rx import Station


@dataclass(kw_only=True)
class Pass:
    """Saves the local coordinate data for a single pass.
    Optionally also indicates the location of that pass in a bigger dataset.

    TODO: rename to RadarPass or similar to avoid potential identifier clash with python `pass` keyword?
    TODO: better member field names, was kept for compatiblity during v1 -> v2 dev
    """

    t: npt.NDArray[np.datetime64]  # TODO: better naming
    enu: list[npt.NDArray] | npt.NDArray  # TODO: shape should be (3,n), not (6,n)
    radar_station_composite_keys: list[RadarStationCompositeKey]

    def get_deltatime_ndarray(self, epoch: datetime | None = None) -> npt.NDArray[np.float64]:
        _epoch: np.datetime64 = np.datetime64(epoch) or self.t[0]

        deltatime_arr = self.t - _epoch  # deltatime in numpy timedelta64
        deltatime_arr = deltatime_arr / np.timedelta64(1, "s")  # deltatime in numpy float64

        return deltatime_arr


def find_simultaneous_passes(
    dt_arr: npt.NDArray[np.float64],
    states: npt.NDArray[np.float64],
    stations: list[Station],
    radar_station_composite_keys: list[RadarStationCompositeKey],
    epoch: datetime,
    fov_kw=None,
) -> list[Pass]:
    """
    Finds all passes that are simultaneously inside a multiple stations FOV's.

    Parameters
    ----------
    dt_arr
        Vector of times in seconds to use as a base to find passes.
    states
        ECEF states of the object to find passes for.
    stations
        Radar stations that defines the FOV's.
    radar_station_composite_keys
        RadarStationCompositeKeys of the stations
    epoch
        the datetime where the deltatime `dt_arr` is based on

    WIP

    TODO: probably don't need both `stations` and `radar_station_composite_keys`

    """
    passes = []
    if fov_kw is None:
        fov_kw = {}

    enu = []
    check = np.full((len(dt_arr),), True, dtype=bool)
    for station in stations:
        enu_st = station.enu(states)
        enu.append(enu_st)

        check_st = station.field_of_view(states, **fov_kw)
        check = np.logical_and(check, check_st)

    inds = np.where(check)[0]

    if len(inds) == 0:
        return passes

    dind = np.diff(inds)
    splits = np.where(dind > 1)[0]

    splits = np.insert(splits, 0, -1)
    splits = np.insert(splits, len(splits), len(inds) - 1)
    splits += 1
    for si in range(len(splits) - 1):
        ps_inds = inds[splits[si] : splits[si + 1]]
        if len(ps_inds) == 0:
            continue

        ps = Pass(
            t=(dt_arr[ps_inds] * 1e6).astype("timedelta64[us]") + np.datetime64(epoch),
            enu=[xv[:, ps_inds] for xv in enu],
            radar_station_composite_keys=radar_station_composite_keys,
        )

        passes.append(ps)

    return passes
