#!/usr/bin/env python

'''

'''

import numpy as np

from .scan import Scan

class Fence(Scan):
    '''General fence scan.

    The objective of the :class:`Fence` scanning scheme is to increase the radar measurement surface by successively
    pointing the beam of the radar at fixed azimuths and variable elevations (between :math:`\\pm \\theta_{min}`).

    .. rubric:: Scan parametrization :

    The pointing direction :math:`p_i` can be written in the azelr coordinate frame as the 
    vector :math:`[\\phi, \\theta_i, r]^T`. The *Azimuth* :math:`\\phi` is set to be fixed 
    while the *elevation* :math:`\\theta_i` can be written as 

    .. math:: \\theta_i = \\Delta \\theta t_i - \\theta_{min} \\text{  , } t_i \\in [0, t_{cycle}]

    Where :math:`\\Delta \\theta =  2 \\theta_{min}/N`, with :math:`N` the number of scanning directions per
    cycle.

    .. rubric:: Constructor :

    Parameters
    ----------
    azimuth : float, default=0.0
        Azimuth of the beam (deg).
    min_elevation : float, default=30.0
        Minimum elevation of the beam. ``min_elevation`` also usually define the FOV of a station (in degrees).
    dwell : float, default=0.2
        Dwell time of the scan (s).
    num : int, default=20
        Number of successive pointing directions per cycle of the scan.

    Examples
    --------
    As a simple example, consider a :class:`Fence` scan at azimuth 180.0 deg, min elevation 30 deg and
    with 10 individual directions per cycle (time slice duration of 1.0s).

    >>> import sorts
    >>> fence = sorts.scans.Fence(180.0, 30, 1, 10)

    To evaluate the result of the ``pointing`` function, we create a time array of 10 elements :
    
    >>> t = np.arange(0.0, 9.0)
    >>> fence.pointing(t)
    array([[180.        , 180.        , 180.        , 180.        ,
            180.        ,   0.        ,   0.        ,   0.        ,
              0.        ],
           [ 30.        ,  43.33333333,  56.66666667,  70.        ,
             83.33333333,  83.33333333,  70.        ,  56.66666667,
             43.33333333],
           [  1.        ,   1.        ,   1.        ,   1.        ,
              1.        ,   1.        ,   1.        ,   1.        ,
              1.        ]])
    
    To plot the results of the ``pointing`` function, we can use the :ref:`plotting` module implemented
    in sorts:

    >>> radar = sorts.radars.eiscat3d
    >>> sorts.plotting.plot_scanning_sequence(fence, station=radar.tx[0], earth=True, ax=None, plot_local_normal=True, max_range=1000e3)

    .. figure:: ../../../../figures/scans_fence.png

        Example :class:`Fence` scanning scheme (azimuth of 180.0 deg, min elevation of 30.0 deg and 
        dwell time of 1s with 10 individual points). In *blue*, the local vertical vector/in *red*, 
        the pointing direction vector of the tx station of the **EISCAT_3D radar**.
    '''
    def __init__(self, azimuth=0.0, min_elevation=30.0, dwell=0.2, num=20):
        super().__init__(coordinates='azelr')
        self._dwell = dwell
        ''' Dwell time of the scan (in seconds). '''
        self.num = num
        ''' Number of pointing directions to generate per scan cycle. '''
        self.min_elevation = min_elevation
        ''' Minimum elevation of the beam. ``min_elevation`` also usually defines the Field of view of the station (in degrees).'''
        self.azimuth = azimuth
        ''' Azimuth of the beam (in degrees). '''

        # generate the azimuth and elevation arrays
        self._az = np.empty((num,), dtype=np.float64)
        self._el = np.linspace(min_elevation, 180-min_elevation, num=num, dtype=np.float64)

        # remove all points outside of the FOV
        inds_ = self._el > 90.0
        self._az[inds_] = np.mod(self.azimuth + 180.0, 360.0)
        self._az[np.logical_not(inds_)] = self.azimuth

        self._el[inds_] = 180.0 - self._el[inds_]


    def dwell(self, t=None):
        if t is None:
            return self._dwell
        else:
            if isinstance(t, float) or isinstance(t, int):
                return self._dwell
            else:
                return np.ones(t.shape, dtype=t.dtype)*self._dwell


    def min_dwell(self):
        return self._dwell


    def cycle(self):
        return self.num*self._dwell


    def pointing(self, t):
        ''' Returns the sequence of radar pointing directions at each given time points.

        The ``pointing`` function of the :class:`Fence` class returns a sequence pointing
        directions in the *azelr* coordinate frame (Azimuth, Elevation, Radius). 

        Parameters
        ----------
        t : float / numpy.ndarray (N,)
            Time point(s) (relative to the reference epoch) where the pointing directions are computed.
            The epoch of reference is relative to the definition of the pointing function.

        Returns
        -------
        pointing : numpy.ndarray (3, N)
            pointing directions programmed by the scanning scheme at each time point in the `azelr` coordinate
            frame.

        Examples
        --------
        See :class:`Fence` for an in-depth example on how to generate a :class:`Fence` scanning scheme.
        '''
        t = np.mod(t, self.cycle()-1e-8)

        # we add 1e-10 to t/self._dwell to prevent rounding errors (for example, 0.99999999999999 will be rounded to 1, but 0.9999 will be rounded to 0
        ind = np.floor(t/self._dwell + 1e-8).astype(int)

        azelr = np.empty((3, np.size(ind)), dtype=np.float64)
        azelr[0,...] = self._az[ind]
        azelr[1,...] = self._el[ind]
        azelr[2,...] = 1.0
        
        return azelr