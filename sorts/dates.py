#!/usr/bin/env python

'''

'''

#Python standard import
import time
import datetime
import os

#Third party import
import numpy as np

#Local import

os.environ['TZ'] = 'GMT'
time.tzset()

def date_to_unix(year, month, day, hour, minute, second):
    '''Convert date to unix time in seconds


    :param int year: Year as integer. Years preceding 1 A.D. should be 0 or negative. The year before 1 A.D. is 0, 10 B.C. is year -9.
    :param int month: Month as integer, Jan = 1, Feb. = 2, etc.
    :param int day:  Day
    :param int hour:  Hour in 24h format
    :param int minute:  Minute
    :param float second:  Second, may contain fractional part.
    :return: Unix time in seconds
    :rtype: float
    '''
    t = datetime.datetime(year, month, day, hour, minute, second)
    return time.mktime(t.timetuple())


def unix_to_date(unix):
    '''Convert unix time in seconds to UTC date datetime object

    :param float unix: Unix time in seconds.
    :return: Datetime object in UTC
    :rtype: datetime.datetime
    '''
    return datetime.datetime.utcfromtimestamp(unix)


def unix_to_datestr(unix):
    '''Convert unix time in seconds to Gregorian calendar UTC date-time formatted string

    :param float unix: Unix time in seconds.
    :return: Gregorian calendar UTC date-time formatted string
    :rtype: str
    '''
    return unix_to_date(unix).strftime('%Y-%m-%dT%H:%M:%S')


def unix_to_datestrf(x):
    '''Convert unix time in seconds to Gregorian calendar UTC date-time formatted string
    
    Different implementation?

    :param float unix: Unix time in seconds.
    :return: Gregorian calendar UTC date-time formatted string
    :rtype: str
    '''
    return "%s"%(unix_to_date(x).strftime('%Y-%m-%dT%H:%M:%S.%f'))


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


def npdt_to_date(dt):
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


def npdt_to_mjd(dt):
    '''Converts a numpy datetime64 value (UTC) to a modified Julian date
    '''
    return (dt - np.datetime64('1858-11-17'))/np.timedelta64(1, 'D')


def mjd_to_npdt(mjd):
    '''Converts a modified Julian date to a numpy datetime64 value (UTC)
    '''
    day = np.timedelta64(24*3600*1000*1000, 'us')
    return np.datetime64('1858-11-17') + day * mjd


def unix_to_npdt(unix):
    '''Converts unix seconds to a numpy datetime64 value (UTC)
    '''
    return np.datetime64('1970-01-01') + np.timedelta64(1000*1000,'us')*unix


def npdt_to_unix(dt):
    '''Converts a numpy datetime64 value (UTC) to unix seconds
    '''
    return (dt - np.datetime64('1970-01-01'))/np.timedelta64(1,'s')

