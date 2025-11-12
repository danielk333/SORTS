#!/usr/bin/env python

import numpy as np
import spacecoords
from sorts import frames


def set_station_ecef(station):
    station["ecef"] = frames.geodetic_to_ITRS(
        station["lat"], station["lon"], station["alt"], degrees=True
    )
    ecef_lla = spacecoords.spherical.cart_to_sph(station["ecef"], degrees=True)
    station["ecef_lat"] = ecef_lla[1]
    station["ecef_lon"] = 90 - ecef_lla[0]
    station["ecef_alt"] = ecef_lla[2]


def field_of_view(station, states):
    """Determines the field of view of the station.
    Should be vectorized over second dimension of states.
    Needs to return a numpy boolean array with `True` when the state is inside the FOV.

    Used to determine when a "pass" is occurring based on the input ECEF states and times.
    The default implementation is a minimum elevation check.
    """
    zenith = np.array([0, 0, 1], dtype=np.float64)

    enu_states = enu(station, states[:3, :])

    zenith_ang = spacecoords.linalg.vector_angle(zenith, enu_states, degrees=True)
    check = zenith_ang < 90.0 - station.get("min_elevation", 0)

    return check


def enu(station, ecefs):
    """Converts a set of ECEF states to local ENU coordinates using geocentric zenith."""
    rel_ = ecefs.copy()
    rel_[:3, :] = rel_[:3, :] - station["ecef"][:, None]
    rel_[:3, :] = frames.ecef_to_enu(
        station["ecef_lat"],
        station["ecef_lon"],
        station["ecef_alt"],
        rel_[:3, :],
        degrees=True,
    )
    if ecefs.shape[0] > 3:
        rel_[3:, :] = frames.ecef_to_enu(
            station["ecef_lat"],
            station["ecef_lon"],
            station["ecef_alt"],
            rel_[3:, :],
            degrees=True,
        )
    return rel_


def point_ecef(station, point):
    """Point Station beam in location of ECEF coordinate. Returns local pointing direction."""
    k = frames.ecef_to_enu(
        station["ecef_lat"],
        station["ecef_lon"],
        station["ecef_alt"],
        point,
        degrees=True,
    )
    k_norm = np.linalg.norm(k, axis=0)
    station["beam"].point(k / k_norm)
    return k
