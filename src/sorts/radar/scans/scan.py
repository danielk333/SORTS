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
    '''
        Encapsulates the observation scheme of a radar system, i.e. its "scan".

        .. rubric:: Pointing function
        
        The pointing function must follow the following standard:
    
         * Take in time in seconds past reference epoch in seconds as first argument.
         * Take any number of keyword arguments.
         * It must return the pointing coordinates as an `(3,)`, `(3,n)` or `(3,n,m)` numpy ndarray where `n` is the length of the input time vector and `m` is the number of simultaneous pointing directions.
         * Units are in meters.
         * Should be vectorized according to time as the second axis.
         
        .. rubric:: Example pointing function :

        This example ``pointing`` function picks random scanning points within a set of predifined
        pointing directions with elevation between 30 and 90 degrees

        .. code-block:: python
    
            def pointing(self, t):
                t = np.asarray(t)
                az = np.linspace(0, 180, 10)
                el = np.linspace(30, 90, 10)

                ind_az = np.random.choice(len(ez), len(t))
                ind_el = np.random.choice(len(al), len(t))

                azelr = np.empty((3, np.size(t)), dtype=np.float64)
                azelr[0,...] = az[ind_az]
                azelr[1,...] = el[ind_el]
                azelr[2,...] = 1.0
                return azelr
    
        .. rubric:: Coordinate systems :
        
        There are 3 possible coordinates systems which can be used in the output of the :func:`Scan.pointing` function :
         * **azelr :** Azimuth, Elevation, Range in degrees east of north and above horizon.
         * **ned :** Cartesian coordinates in North, East, Down.
         * **enu :** Cartesian coordinates in East, North, Up.

        .. rubric:: Constructor  :

        Parameters
        ----------
        coordinates : str 
            The coordinate system used, can be :code:`'azelr'`, :code:`'ned'` or :code:`'enu'`. 
            If `azelr` is used, degrees are assumed.
    '''
    
    def __init__(self, coordinates='enu'):
        ''' Default class constructor. '''
        self.coordinates = coordinates.lower()
        ''' 
        Coordinate system used to define the scanning scheme. 

        There are 3 possible coordinates systems which can be used in the output of the :func:`Scan.pointing` function :
         * **azelr :** Azimuth, Elevation, Range in degrees east of north and above horizon.
         * **ned :** Cartesian coordinates in North, East, Down.
         * **enu :** Cartesian coordinates in East, North, Up.
        '''


    def dwell(self, t):
        ''' The current dwell time of the scan. 
        
        **Dwell** time corresponds to the time interval between two consecutive
        pointing directions in the sequence.

        Parameters
        ----------
        t : float / numpy.ndarray
            Time points at which we want to get the scanning **dwell time** (in seconds).

        Returns
        -------
        dwell : float / numpy.ndarray
            Current **radar dwell** time (in seconds).
        '''
        return None


    def min_dwell(self):
        ''' If there are dynamic dwell times, returns the minimum dwell time. 
        Otherwise, returns same as :ref:`Scan.dwell`. '''
        return None


    def cycle(self):
        ''' Cycle time of the scan if applicable. '''
        return None


    def copy(self):
        ''' Returns a copy of the current instance. '''
        raise NotImplementedError()


    def check_dwell_tx(self, tx):
        ''' Checks if the transmitting antenna pulse pattern and coherent integration 
        scheme is compatible with the observation schema. Raises an Exception if not.
    
        Parameters
        ----------
        tx : :class:`TX<sorts.radar.system.station.TX>` 
            Station which compatibility with the scanning scheme is tested.

        Raises
        ------
        Exception : 
            If the time slice of the :class:`TX<sorts.radar.system.station.TX>` station is not 
            compatible with the dwell time of the scan.
        '''
        time_slice = tx.n_ipp*tx.ipp
        dwell_time = self.min_dwell()
        if dwell_time is not None:
            if time_slice > dwell_time:
                raise Exception(f'TX time_slice of {time_slice:.2f} s incompatible with minimum dwell time of scan {dwell_time:.2f} s')


    @abstractmethod
    def pointing(self, t):
        ''' Returns the sequence of radar pointing directions at each given time points.

         The pointing function must follow the following standard:
    
         * Take in time in seconds past reference epoch in seconds as first argument.
         * Take any number of keyword arguments.
         * It must return the pointing coordinates as an `(3,)`, `(3,n)` or `(3,n,m)` numpy ndarray where `n` is the length of the input time vector and `m` is the number of simultaneous pointing directions.
         * Units are in meters.
         * Should be vectorized according to time as the second axis.
         
        .. rubric:: Example pointing function :

        This example ``pointing`` function picks random scanning points within a set of predifined
        pointing directions with elevation between 30 and 90 degrees

        .. code-block:: python
    
            def pointing(self, t):
                t = np.asarray(t)
                az = np.linspace(0, 180, 10)
                el = np.linspace(30, 90, 10)

                ind_az = np.random.choice(len(ez), len(t))
                ind_el = np.random.choice(len(al), len(t))

                azelr = np.empty((3, np.size(t)), dtype=np.float64)
                azelr[0,...] = az[ind_az]
                azelr[1,...] = el[ind_el]
                azelr[2,...] = 1.0
                return azelr

        Parameters
        ----------
        t : float / numpy.ndarray (N,)
            Time point(s) (relative to the reference epoch) where the pointing directions are computed.
            The epoch of reference is relative to the definition of the pointing function.

        Returns
        -------
        pointing : numpy.ndarray (3, N)
            pointing directions programmed by the scanning scheme at each time point. 

            .. note::
                The pointing directions returned by the :attr:`Scan.pointing` function are given relative to
                the coordinate frame of the :class:`Scan` object (see :attr:`Scan.coordinates`)
        '''
        pass


    def enu_pointing(self, t):
        ''' Returns the instantaneous pointing in East, North, Up (ENU) local coordinates.
        
        Parameters
        ----------
        t : float / numpy.ndarray 
            Time point(s) (relative to the reference epoch) where the pointing directions are computed.
            The epoch of reference is relative to the definition of the pointing function.

        Returns
        -------
        pointing : numpy.ndarray (3, N)
            pointing directions programmed by the scanning scheme at each time point in the **ENU coordinate frame**. 
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
    

    def _transform_ecef(self, point, station):
        ''' Transforms a set of points given in the coordinate system of the :class:`Scan` object in the 
        ECEF reference frame.

        Parameters
        ----------
        point : np.ndarray (3, N, M)
            Set of N points (given in the local reference frame M stations) to be converted to the 
            ECEF frame.

            .. note::
                Those points must be given in the same coordinate system as the :class:`Scan` object.
                See :attr:`Scan` for more information about the possible reference frames available.

        station : list of :class:`sorts.Station<sorts.radar.system.station.Station>` (M,)
            Stations with respect to which the ``points`` are given.

        Returns
        -------
        k0 : np.ndarray (3, N, M)
            points in the ECEF frame of reference.
        '''
        station = np.atleast_1d(station)

        pos = np.ndarray((3, len(station)))
        
        for station_ind, st in enumerate(station):
            pos[:, station_ind] = [st.lat, st.lon, st.alt]
        
        if self.coordinates == 'ned':
            k0 = frames.ned_to_ecef(pos[0], pos[1], pos[2], point, radians=False)
        elif self.coordinates == 'enu':
            k0 = frames.enu_to_ecef(pos[0], pos[1], pos[2], point, radians=False)
        elif self.coordinates == 'azelr':
            k0 = frames.azel_to_ecef(pos[0], pos[1], pos[2], point[0,...], point[1,...], radians=False)
        
        return k0


    def ecef_pointing(self, t, station):
        ''' Returns the instantaneous WGS84 ECEF pointing direction and the radar 
        geographical location in WGS84 ECEF coordinates.
    
        Parameters
        ----------
        t : float
            Time point(s) (relative to the reference epoch) where the pointing directions are computed.
            The epoch of reference is relative to the definition of the pointing function.
        station : :class:`sorts.Station<sorts.radar.system.station.Station>`
            Station with respect to which the pointing direction is computed.

        Returns
        -------
        k0 : numpy.ndarray (3, N)
            Poiting directions at each time point given by ``t`` in the ECEF coordinate frame.
        '''
        t = np.atleast_1d(t)
        point = self.pointing(t)

        if len(point.shape) == 3:
            k0 = np.zeros(point.shape, dtype=point.dtype)
            for ind in range(point.shape[2]):
                k0[:,:,ind] = self._transform_ecef(point[:,:,ind], station).reshape(3, -1)
        else:            
            k0 = self._transform_ecef(point, station).reshape(3, -1)
        return k0
