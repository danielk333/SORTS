#!/usr/bin/env python

'''

'''

import numpy as np

from .scan import Scan

class RandomUniform(Scan):
    '''Uniform randomly distributed points in the FOV.

    The :class:`RandomUniform` scan class uniformly generates a set of random pointing
    directions over the integrality of the FOV. 

    .. rubric:: Pointing direction parametrization :

    The pointing direction :math:`p_i` at time :math:`t_i` can be written as a vector 
    in the ``spherical`` coordinate frame:

    .. math::       p_i = [r, \\theta_i, \\phi_i]^T

    In the :class:`RandomUniform` scanning algorithm, :math:`\\phi` is generated randomly
    between :math:`[0, 2 \\pi]`. :math:`\\theta` is calculated by performing a linear interpolation
    between the min altitude (given by ``min_elevation`` :math:`\\pi/2 - \\theta_{min}`) and 1.0 over the half sphere of radius 1:

    .. math::       \\theta_i = \\arccos{[\\nu_i (1 - \\sin(\\theta_{min}) + \\sin(\\theta_{min}))]}

    Where :math:`\\nu_i` is a uniform random number between 0 and 1. The final pointing direction is then computed by performing a 
    transformation to the ENU coordinate frame by setting :math:`r=1`.

    .. rubric:: Constructor

    Parameters
    ----------
    min_elevation : float, default=30.0
        Minimum elevation of the beam. ``min_elevation`` also usually define the FOV of a station (in degrees).
    dwell : float, default=0.1
        Dwell time of the scan (in seconds).
    cycle_num : int, default=10000
        Number of pointing directions per cycle.

    Examples
    --------
    As a simple example, consider a :class:`RandomUniform` scan performing the uniform random scanning of 
    a FOV defined by a minimum elevation of 30 degrees. The number of pointing directions per cycle is set 
    to be 10 and the dwell time as 1.0 second. 

    >>> import sorts
    >>> rand_uniform = sorts.scans.RandomUniform(min_elevation=30.0, dwell=1.0, cycle_num=10)

    To evaluate the result of the ``pointing`` function, we create a time array of 25 elements :
    
    >>> t = np.linspace(0.0, 9.0, 10)
    >>> rand_uniform.pointing(t)
    array([[0.44747891 0.44747891 0.44747891 0.44747891 0.44747891 0.44747891
          0.44747891 0.44747891 0.44747891 0.44747891]
         [0.2441659  0.2441659  0.2441659  0.2441659  0.2441659  0.2441659
          0.2441659  0.2441659  0.2441659  0.2441659 ]
         [0.86031717 0.86031717 0.86031717 0.86031717 0.86031717 0.86031717
          0.86031717 0.86031717 0.86031717 0.86031717]])
    
    To plot the results of the ``pointing`` function, we can use the :ref:`plotting` module implemented
    in sorts:

    >>> radar = sorts.radars.eiscat3d
    >>> sorts.plotting.plot_scanning_sequence(rand_uniform, station=radar.tx[0], earth=True, ax=None, plot_local_normal=True, max_range=1000e3)
    
    .. figure:: ../../../../figures/scans_rand_uniform.png

        Example :class:`RandomUniform` scanning scheme. In *blue*, the local vertical vector/in *red*, 
        the pointing direction vector of the tx station of the **EISCAT_3D radar**.
    '''
    def __init__(self, min_elevation=30.0, dwell=0.1, cycle_num=10000):
        super().__init__(coordinates='enu')
        self._dwell = dwell
        ''' Dwell time of the scan (in seconds). '''
        self.num = cycle_num
        ''' Number of pointing directions per scanning cycle. '''
        self.min_elevation = min_elevation
        ''' Minimum elevation of the beam. ``min_elevation`` also usually defines the Field of view of the station (in degrees).'''

        min_z = np.sin(np.radians(min_elevation))

        # generate random points
        phi = 2*np.pi*np.random.rand(self.num)
        theta = np.arccos(np.random.rand(self.num)*(1 - min_z) + min_z)

        # convert spherical points to enu
        k = np.empty((3, self.num), dtype=np.float64)
        k[0,:] = np.cos(phi)*np.sin(theta)
        k[1,:] = np.sin(phi)*np.sin(theta)
        k[2,:] = np.cos(theta)

        self.pointings = k
        ''' Pointing direction in the ENU coordinate frame. '''


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

        The ``pointing`` function of the :class:`RandomUniform` class returns a set of randomly generated
        pointing directions in the ENU coordinate frame within the FOV of the scan.

        Parameters
        ----------
        t : float / numpy.ndarray (N,)
            Time point(s) (relative to the reference epoch) where the pointing directions are computed.
            The epoch of reference is relative to the definition of the pointing function.

        Returns
        -------
        pointing : numpy.ndarray (3, N)
            pointing directions programmed by the scanning scheme at each time point in the `ENU` coordinate
            frame.

        Examples
        --------
        See :class:`RandomUniform` for an in-depth example on how to generate a :class:`RandomUniform` 
        scanning scheme.
        '''
        ind = (np.mod(t/self.cycle(), 1)*self.num).astype(np.int)
        return self.pointings[:,ind]