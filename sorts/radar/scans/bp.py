#!/usr/bin/env python

'''
Defines the :class:`Beampark` scanning scheme.
'''

import numpy as np

from .scan import Scan

class Beampark(Scan):
    ''' Defines the static :class:`Beampark` scanning scheme.

    .. rubric:: description :

    The Beampark scanning scheme is one of main methods used for the 
    discovery of new :class:`space objects<sorts.targets.space_object.SpaceObject>`.

    The beam is parameterized by its *azimuth* and *elevation* and **remains fixed in the 
    station coordinate frame**. Given the statistical repartion of Space objects on orbit, 
    we can estimate the average expected number of counts as a function of altitude. 

    Parameters
    ----------
    azimuth : float / numpy.ndarray, default=0.0
        Azimuth of the beam.
    elevation : float / numpy.ndarray, default=90.0
        Elevation of the beam.
    dwell : float, default=0.1
        Dwell time between two consecutive pointing directions.

    Examples
    --------
    As a simple example, consider a :class:`Beampark` scan at azimuth 180.0 deg and elevation 75 deg (
    time slice duration of 0.1s).

    >>> import sorts
    >>> bp = sorts.scans.Beampark(180.0, 75.0, 0.1)

    To evaluate the result of the ``pointing`` function, we create a time array of 10 elements :
    
    >>> t = np.arange(0.0, 9.0)
    >>> bp.pointing(t)
    array([[180., 180., 180., 180., 180., 180., 180., 180., 180.],
           [ 75.,  75.,  75.,  75.,  75.,  75.,  75.,  75.,  75.],
           [  1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.]])
    
    To plot the results of the ``pointing`` function, we can use the :ref:`plotting` module implemented
    in sorts:

    >>> radar = sorts.radars.eiscat3d
    >>> sorts.plotting.plot_scanning_sequence(bp, station=radar.tx[0], earth=True, ax=None, plot_local_normal=True, max_range=1000e3)

    .. figure:: ../../../../figures/scans_beampark.png

        Example :class:`Beampark` scanning scheme (azimuth of 180.0 deg, elevation of 75.0 deg and 
        dwell time of 0.1s). In *blue*, the local vertical vector/in *red*, the pointing direction vector of the 
        tx station of the **EISCAT_3D radar**.
    '''
    def __init__(self, azimuth=0.0, elevation=90.0, dwell=0.1):
        super().__init__(coordinates='azelr')
        self.elevation = elevation
        ''' Elevation of the beam. '''
        self.azimuth = azimuth
        ''' Azimuth of the beam. '''
        self._dwell = dwell
        ''' Scanning dwell time. '''


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
        return None


    def pointing(self, t):
        ''' Returns the sequence of radar pointing directions at each given time points.

        The ``pointing`` function of the :class:`Beampark` class returns a set of static pointing
        directions in the *azelr* coordinate frame (Azimuth, Elevation, Radius). If ``azimuth``
        and ``elevation`` were given as arrays (N,), then the output pointing directions will be 
        picked according to the sequence parameterized by the ``azimuth`` and ``elevation`` arrays.

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
        See :class:`Beampark` for an in-depth example on how to generate a :class:`Beampark` scanning scheme.
        '''
        if isinstance(t, float) or isinstance(t, int):
            shape = (3, )
        else:
            shape = (3, np.size(t))

        if hasattr(self.elevation, '__len__') and np.size(self.elevation) > 1:
            shape += (len(self.elevation), )
        elif hasattr(self.azimuth, '__len__') and np.size(self.azimuth) > 1:
            shape += (len(self.azimuth), )

        azelr = np.empty(shape, dtype=np.float64)

        if hasattr(self.azimuth, '__len__') and np.size(self.azimuth) > 1:
            for ind in range(len(self.azimuth)):
                if len(shape) == 2:
                    azelr[0,ind] = self.azimuth[ind]
                else:
                    azelr[0,:,ind] = self.azimuth[ind]
        else:
            azelr[0,...] = self.azimuth

        if hasattr(self.elevation, '__len__') and np.size(self.elevation) > 1:
            for ind in range(len(self.elevation)):
                if len(shape) == 2:
                    azelr[1,ind] = self.elevation[ind]
                else:
                    azelr[1,:,ind] = self.elevation[ind]
        else:
            azelr[1,...] = self.elevation

        azelr[2,...] = 1.0
        return azelr