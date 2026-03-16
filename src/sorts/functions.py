#!/usr/bin/env python

"""Miscellaneous functions"""

import numpy as np
import scipy.constants
import spacecoords
import pyorb


def equidistant_sampling(orbit, start_t, end_t, max_dpos=1e3, eccentricity_tol=0.3):
    """Find the temporal sampling of an orbit which is sufficient to achieve a
    maximum spatial separation. Assume elliptic orbit and uses Keplerian propagation
    to find sampling, does not take perturbation patterns into account. If
    eccentricity is small, uses periapsis speed and uniform sampling in time.

    :param pyorb.Orbit orbit: Orbit to find temporal sampling of.
    :param float start_t: Start time in seconds
    :param float end_t: End time in seconds
    :param float max_dpos: Maximum separation between evaluation points in meters.
    :param float eccentricity_tol: Minimum eccentricity below which the orbit is
        approximated as a circle and temporal samples are uniform in time.
    :return: Vector of sample times in seconds.
    :rtype: numpy.ndarray
    """
    if len(orbit) > 1:
        raise ValueError(f"Cannot use vectorized orbits: len(orbit) = {len(orbit)}")

    if orbit.e <= eccentricity_tol:
        r = pyorb.elliptic_radius(0.0, orbit.a, orbit.e, degrees=False)
        v = pyorb.orbital_speed(r, orbit.a, orbit.G * (orbit.M0 + orbit.m))[0]
        return np.arange(start_t, end_t, max_dpos / v)

    tmp_orb = orbit.copy()
    tmp_orb.auto_update = False

    tmp_orb.propagate(start_t)
    period = tmp_orb.period

    t_curr = start_t
    t = [t_curr]
    t_repeat = None
    while t_curr < end_t:
        if t_curr - start_t > period:
            if t_repeat is None:
                t_repeat = len(t)
            dt = t[-t_repeat + 1] - t[-t_repeat]
            t_curr += dt
        else:
            v = tmp_orb.speed[0]
            dt = max_dpos / v
            t_curr += dt
            tmp_orb.propagate(dt)

        t.append(t_curr)
    return np.array(t, dtype=np.float64)


def calculate_range(enu):
    """Norm of the ENU coordinates."""
    return np.linalg.norm(enu[:3, :], axis=0)


def calculate_range_rate(enu):
    """Projected ENU velocity along the ENU range."""
    return np.sum(enu[3:, :] * (enu[:3, :] / np.linalg.norm(enu[:3, :], axis=0)), axis=0)


def calculate_zenith_angle(enu, radians=False):
    """Zenith angle of the ENU coordinates."""
    return spacecoords.linalg.vector_angle(
        np.array([0, 0, 1], dtype=np.float64), enu[:3, :], degrees=not radians
    )


def signal_delay(st1, st2, ecef):
    """Signal delay due to speed of light between station-1 to ecef position to station-2"""
    r1 = np.linalg.norm(ecef - st1.ecef[:, None], axis=0)
    r2 = np.linalg.norm(ecef - st1.ecef[:, None], axis=0)
    dt = (r1 + r2) / scipy.constants.c
    return dt


def instantaneous_to_coherrent(gain, groups, N_IPP, IPP_scale=1.0, units="dB"):
    """Using pulse encoding schema, subgroup setup and coherent integration setup; convert from instantaneous gain to coherently integrated gain.

    :param float gain: Instantaneous gain, linear units or in dB.
    :param int groups: Number of subgroups from witch signals are coherently combined, assumes subgroups are identical.
    :param int N_IPP: Number of pulses to coherently integrate.
    :param float IPP_scale: Scale the IPP effective length in case e.g. the IPP is the same but the actual TX length is lowered.
    :param str units: If string equals 'dB', assume input and output units should be dB, else use linear scale.

    :return float: Gain after coherent integration, linear units or in dB.
    """
    if units == "dB":
        return gain + 10.0 * np.log10(groups * N_IPP * IPP_scale)
    else:
        return gain * (groups * N_IPP * IPP_scale)


def coherrent_to_instantaneous(gain, groups, N_IPP, IPP_scale=1.0, units="dB"):
    """Using pulse encoding schema, subgroup setup and coherent integration setup; convert from coherently integrated gain to instantaneous gain.

    :param float gain: Coherently integrated gain, linear units or in dB.
    :param int groups: Number of subgroups from witch signals are coherently combined, assumes subgroups are identical.
    :param int N_IPP: Number of pulses to coherently integrate.
    :param float IPP_scale: Scale the IPP effective length in case e.g. the IPP is the same but the actual TX length is lowered.
    :param str units: If string equals 'dB', assume input and output units should be dB, else use linear scale.

    :return float: Instantaneous gain, linear units or in dB.
    """
    if units == "dB":
        return gain - 10.0 * np.log10(groups * N_IPP * IPP_scale)
    else:
        return gain / (groups * N_IPP * IPP_scale)
