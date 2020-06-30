#!/usr/bin/env python

'''Defines what a radar observation schema is in the form of a class.

'''

#Python standard import
from abc import ABC, abstractmethod
import inspect


#Third party import
import numpy as np

#Local import
from .. import frames

class Scan(ABC):
    '''Encapsulates the observation schema of a radar system, i.e. its "scan".

    :param str coordinates: The coordinate system used, can be :code:`'azelr'`, :code:`'ned'` or :code:`'enu'`

    **Pointing function:**
    
    The pointing function must follow the following standard:

     * Take in time in seconds past reference epoch in seconds as first argument.
     * Take any number of keyword arguments.
     * It must return the pointing coordinates as an `(3,)` or an `(3,n)` numpy ndarray.
     * Units are in meters.
     * Should be vectorized according to time as the second axis.
     
    Example pointing function:
    
    .. code-block:: python

        import numpy as np
        
        

    **Coordinate systems:**

     :azelr: Azimuth, Elevation, Range in degrees east of north and above horizon.
     :ned: Cartesian coordinates in North, East, Down.
     :enu: Cartesian coordinates in East, North, Up.


    '''
    def __init__(self, coordinates='enu'):
        self.coordinates = coordinates.lower()


    @abstractmethod
    def dwell(self, t):
        '''The current dwell time of the scan. 
        '''
        pass


    @abstractmethod
    def min_dwell(self):
        '''If there are dynamic dwell times, this is the minimum dwell time. Otherwise, returns same as :code:`dwell`.
        '''
        pass


    def copy(self):
        '''Return a copy of the current instance of :class:`radar_scans.RadarScan`.
        '''
        pass


    def check_dwell_tx(self, tx):
        '''Checks if the transmitting antenna pulse pattern and coherent integration schema is compatible with the observation schema. Raises an Exception if not.
        
            :param sorts.radar.TX tx: The antenna that should perform this scan.
        '''
        time_slice = tx.n_ipp*tx.ipp
        dwell_time = self.min_dwell()
        if dwell_time is not None:
            if time_slice > dwell_time:
                raise Exception(f'TX time_slice of {time_slice:.2f} s incompatible with minimum dwell time of scan {dwell_time:.2f} s')


    @abstractmethod
    def pointing(self, t):
        pass


    def enu_pointing(self, t):
        '''Returns the instantaneous pointing in East, North, Up (ENU) local coordinates.
        
            :param float/numpy.ndarray t: Seconds past a reference epoch to retrieve the pointing at.
        '''
        point = self.pointing(t)

        if self.coordinates == 'ned':
            point[2,...] = -point[2,...]
        elif self.coordinates == 'enu':
            pass
        elif self.coordinates == 'azel':
            point = frames.sph_to_cart(point)

        return point
    

    def station_pointing(self, t, ant):
        '''Returns the instantaneous WGS84 ECEF pointing direction and the radar geographical location in WGS84 ECEF coordinates.
        
            :param float t: Seconds past a reference epoch to retrieve the pointing at.
        '''
        point = self.local_pointing(t)

        if self.coordinates == 'ned':
            k0 = coord.ned_to_ecef(ant.lat, ant.lon, ant.alt, point, radians=False)
        elif self.coordinates == 'enu':
            k0 = coord.enu_to_ecef(ant.lat, ant.lon, ant.alt, point, radians=False)
        elif self.coordinates == 'azel':
            k0 = coord.azel_ecef(ant.lat, ant.lon, ant.alt, point, radians=False)
        
        return k0
