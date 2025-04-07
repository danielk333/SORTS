#!/usr/bin/env python

"""Encapsulates a fundamental component of tracking space objects:
a pass over a geographic location.
Also provides convenience functions for finding passes given states
and stations and sorting structures of passes in particular ways.

"""
import datetime

import numpy as np
import numpy.typing as npt


class Pass:
    """Saves the local coordinate data for a single pass.
    Optionally also indicates the location of that pass in a bigger dataset.
    """

    # TODO: ndarray of enu is of shape (6,n), but should be (3,n)
    def __init__(self, t, enu, epoch=None, station_id: int | list[int] = 0):
        self.t = t  # time
        self.enu: list[npt.NDArray] | npt.NDArray = enu  # of shapes: (3, n) | ((3, n), ..., k)

        self.station_id = station_id

    def __str__(self):
        str_ = "Pass "
        if self.station_id is not None:
            str_ += f"Station {self.station_id} | "
        str_ += (
            f"Rise {str(datetime.timedelta(seconds=self.start()))}"
            f"({(self.end() - self.start())/60.0:.1f} min)"
            f"{str(datetime.timedelta(seconds=self.end()))} Fall"
        )

        return str_

    def __repr__(self):
        return str(self)


def find_passes(t, states, station, cache_data=True):
    """Find passes inside the FOV of a radar station given a series of times for a space object.

    :param numpy.ndarray t: Vector of times in seconds to use as a base to find passes.
    :param numpy.ndarray states: ECEF states of the object to find passes for.
    :param sorts.Station station: Radar station that defines the FOV, or a list of radar stations.
    :return: list of passes
    :rtype: sorts.Pass

    """
    passes = []

    if isinstance(station, list):
        check = np.logical_or.reduce([st.field_of_view(states) for st in station])
    else:
        check = station.field_of_view(states)
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

        if cache_data:
            enu = station.enu(states[:3, :])
            ps = Pass(
                t=t[ps_inds],
                enu=enu[:, ps_inds],
                inds=ps_inds,
                cache=True,
            )
        else:
            ps = Pass(
                t=None,
                enu=None,
                inds=ps_inds,
                cache=True,
            )
            ps._start = t[ps_inds].min()
            ps._end = t[ps_inds].max()

        passes.append(ps)

    return passes


def find_simultaneous_passes(t, states, stations, cache_data=True, fov_kw=None):
    """Finds all passes that are simultaneously inside a multiple stations FOV's.

    :param numpy.ndarray t: Vector of times in seconds to use as a base to find passes.
    :param numpy.ndarray states: ECEF states of the object to find passes for.
    :param list of sorts.Station stations: Radar stations that defines the FOV's.
    :return: list of passes
    :rtype: list of sorts.Pass

    """
    passes = []
    if fov_kw is None:
        fov_kw = {}

    enu = []
    check = np.full((len(t),), True, dtype=bool)
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
        if cache_data:
            ps = Pass(
                t=t[ps_inds],
                enu=[xv[:, ps_inds] for xv in enu],
                inds=ps_inds,
                cache=True,
                station_id=[None, None],
            )
        else:
            ps = Pass(
                t=None,
                enu=None,
                inds=ps_inds,
                cache=True,
                station_id=[None, None],
            )
            ps._start = t[ps_inds].min()
            ps._end = t[ps_inds].max()

        passes.append(ps)

    return passes


def group_passes(passes):
    """Takes a list of passes structured as
    [tx][rx][pass] and find all simultaneous passes
    and groups them according to [tx], resulting in a [tx][pass][rx] structure.
    """

    def overlap(ps1, ps2):
        return ps1.start() <= ps2.end() and ps2.start() <= ps1.end()

    grouped_passes = []
    for tx_passes in passes:
        grouped_passes.append([])

        # first flatten
        flat_passes = [x for rx_passes in tx_passes for x in rx_passes]

        if len(flat_passes) > 0:
            grouped_passes[-1].append([flat_passes[0]])
        else:
            continue

        for x in range(1, len(flat_passes)):
            for y in range(len(grouped_passes[-1])):
                member = False
                for gps in grouped_passes[-1][y]:
                    if overlap(gps, flat_passes[x]):
                        member = True
                        break

                if member:
                    member_id = y
                    break

            if member:
                grouped_passes[-1][member_id].append(flat_passes[x])
            else:
                grouped_passes[-1].append([flat_passes[x]])

    return grouped_passes
