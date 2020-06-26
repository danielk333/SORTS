#!/usr/bin/env python

'''

'''

#Python standard import
import time
import os

#Third party import
import numpy as np

#Local import


sec = np.timedelta64(1000000000, 'ns')
'''numpy.datetime64: Interval of 1 second
'''

def jd_to_mjd(jd):
    '''Convert Julian Date (relative 12h Jan 1, 4713 BC) to Modified Julian Date (relative 0h Nov 17, 1858)
    '''
    return jd - 2400000.5


def mjd_to_jd(mjd):
    '''Convert Modified Julian Date (relative 0h Nov 17, 1858) to Julian Date (relative 12h Jan 1, 4713 BC)
    '''
    return mjd + 2400000.5


def npdt2date(dt):
    '''Converts a numpy datetime64 value to a date tuple

    :param numpy.datetime64 dt: Date and time (UTC) in numpy datetime64 format

    :return: tuple (year, month, day, hours, minutes, seconds, microsecond)
             all except usec are integer
    '''

    t0 = np.datetime64('1970-01-01', 's')
    ts = (dt - t0)/sec

    it = int(np.floor(ts))
    usec = 1e6 * (dt - (t0 + it*sec))/sec

    tm = time.localtime(it)
    return tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, usec


def npdt2mjd(dt):
    '''Converts a numpy datetime64 value (UTC) to a modified Julian date
    '''
    return (dt - np.datetime64('1858-11-17'))/np.timedelta64(1, 'D')


def mjd2npdt(mjd):
    '''Converts a modified Julian date to a numpy datetime64 value (UTC)
    '''
    day = np.timedelta64(24*3600*1000*1000, 'us')
    return np.datetime64('1858-11-17') + day * mjd


def unix2npdt(unix):
    '''Converts unix seconds to a numpy datetime64 value (UTC)
    '''
    return np.datetime64('1970-01-01') + np.timedelta64(1000*1000,'us')*unix


def npdt2unix(dt):
    '''Converts a numpy datetime64 value (UTC) to unix seconds
    '''
    return (dt - np.datetime64('1970-01-01'))/np.timedelta64(1,'s')


def gstime(jdut1):
    '''This function finds the greenwich sidereal time (iau-82).

    *References:* Vallado 2007, 193, Eq 3-43

    Author: David Vallado 719-573-2600 7 jun 2002
    Adapted to Python, Daniel Kastinen 2018

    :param float jdut1: Julian date of ut1 in days from 4713 bc

    :return: Greenwich sidereal time in radians, 0 to :math:`2\pi`
    :rtype: float
    '''
    twopi      = 2.0*np.pi;
    deg2rad    = np.pi/180.0;

    # ------------------------  implementation   ------------------
    tut1= ( jdut1 - 2451545.0 ) / 36525.0

    temp = - 6.2e-6 * np.multiply(np.multiply(tut1,tut1),tut1) + 0.093104 * np.multiply(tut1,tut1) \
           + (876600.0 * 3600.0 + 8640184.812866) * tut1 + 67310.54841

    # 360/86400 = 1/240, to deg, to rad
    temp = np.fmod( temp*deg2rad/240.0,twopi )

    # ------------------------ check quadrants --------------------
    temp = np.where(temp < 0.0, temp+twopi, temp)


    gst = temp
    return gst


def GMST1982(jd_ut1):
    """Return the angle of Greenwich Mean Standard Time 1982 given the JD.
    This angle defines the difference between the idiosyncratic True
    Equator Mean Equinox (TEME) frame of reference used by SGP4 and the
    more standard Pseudo Earth Fixed (PEF) frame of reference.


    *Reference:* AIAA 2006-6753 Appendix C.

    Original work Copyright (c) 2013-2018 Brandon Rhodes under the MIT license
    Modified work Copyright (c) 2019 Daniel Kastinen

    :param float jd_ut1: UT1 Julian date.
    :returns: tuple of (Earth rotation [rad], Earth angular velocity [rad/day])

    """
    _second = 1.0 / (24.0 * 60.0 * 60.0)

    T0 = 2451545.0

    t = (jd_ut1 - T0) / 36525.0
    g = 67310.54841 + (8640184.812866 + (0.093104 + (-6.2e-6) * t) * t) * t
    dg = 8640184.812866 + (0.093104 * 2.0 + (-6.2e-6 * 3.0) * t) * t
    theta = (jd_ut1 % 1.0 + g * _second % 1.0) * 2.0 * np.pi
    theta_dot = (1.0 + dg * _second / 36525.0) * 2.0 * np.pi
    return theta, theta_dot