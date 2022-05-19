#!/usr/bin/env python

'''Defines what a radar observation schema is in the form of a class.

'''

#Python standard import
from abc import ABC, abstractmethod


#Third party import
import numpy as np

#Local import
from ...transformations import frames


class Scan(ABC):
    '''Encapsulates the observation schema of a radar system, i.e. its "scan".

    :param str coordinates: The coordinate system used, can be :code:`'azelr'`, :code:`'ned'` or :code:`'enu'`. If `azelr` is used, degrees are assumed.

    **Pointing function:**
    
    The pointing function must follow the following standard:

     * Take in time in seconds past reference epoch in seconds as first argument.
     * Take any number of keyword arguments.
     * It must return the pointing coordinates as an `(3,)`, `(3,n)` or `(3,n,m)` numpy ndarray where `n` is the length of the input time vector and `m` is the number of simultaneous pointing directions.
     * Units are in meters.
     * Should be vectorized according to time as the second axis.
     
    Example pointing function:
    
    .. code-block:: python

        import numpy as np
        #TODO

    **Coordinate systems:**

     :azelr: Azimuth, Elevation, Range in degrees east of north and above horizon.
     :ned: Cartesian coordinates in North, East, Down.
     :enu: Cartesian coordinates in East, North, Up.


    '''
    def __init__(self, coordinates='enu'):
        self.coordinates = coordinates.lower()


    def dwell(self, t):
        '''The current dwell time of the scan. 
        '''
        return None


    def min_dwell(self):
        '''If there are dynamic dwell times, this is the minimum dwell time. Otherwise, returns same as :code:`dwell`.
        '''
        return None


    def cycle(self):
        '''The cycle time of the scan if applicable.
        '''
        return None


    def copy(self):
        '''Return a copy of the current instance.
        '''
        raise NotImplementedError()


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
        elif self.coordinates == 'azelr':
            if len(point.shape) == 3:
                p_ = np.zeros(point.shape, dtype=point.dtype)
                for ind in range(point.shape[2]):
                    p_[:,:,ind] = frames.sph_to_cart(point[:,:,ind], radians=False)
                point = p_
                del p_
            else:
                point = frames.sph_to_cart(point, radians=False)

        return point
    

    def _transform_ecef(self, point, ant):
        if self.coordinates == 'ned':
            k0 = frames.ned_to_ecef(ant.lat, ant.lon, ant.alt, point, radians=False)
        elif self.coordinates == 'enu':
            k0 = frames.enu_to_ecef(ant.lat, ant.lon, ant.alt, point, radians=False)
        elif self.coordinates == 'azelr':
            k0 = frames.azel_to_ecef(ant.lat, ant.lon, ant.alt, point[0,...], point[1,...], radians=False)
        return k0


    def ecef_pointing(self, t, ant):
        '''Returns the instantaneous WGS84 ECEF pointing direction and the radar geographical location in WGS84 ECEF coordinates.
        
            :param float t: Seconds past a reference epoch to retrieve the pointing at.
        '''
        point = self.pointing(t)

        if len(point.shape) == 3:
            k0 = np.zeros(point.shape, dtype=point.dtype)
            for ind in range(point.shape[2]):
                k0[:,:,ind] = self._transform_ecef(point[:,:,ind], ant)
        else:
            k0 = self._transform_ecef(point, ant)
        return k0
