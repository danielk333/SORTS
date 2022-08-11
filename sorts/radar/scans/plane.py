#!/usr/bin/env python

'''

'''

import numpy as np

from .scan import Scan

class Plane(Scan):
    ''' A uniform sampling of a horizontal plane. 

    The :class:`Plane` scanning scheme performs a *uniform sampling* of a plane with normal the local vertical
    vector of the station. 

    .. rubric:: Scan parametrization :

    The pointing directions of the scan are outputted in the **ENU** coordinate frame of the station, meaning that the 
    ith pointing direction can be represented as :math:`p_i = [x_i, y_i, z]^T`

    .. note:: 
        Since the plane is locally tangent to the surface of the earth, :math:`z` will be constant for all pointing directions
        of the scan.

    The :math:`x` and :math:`y` coordinates of the pointing direction vector in the ENU frame can be expressed as:

    .. math::       x_i = = \\Delta x \\cdot t_i - \\frac{L_x}{2} + x_{offset}\\text{  , } t_i \\in [0, t_{cycle}]

    .. math::       y_i = = \\Delta y \\cdot t_i - \\frac{L_y}{2} + y_{offset} \\text{  , } t_i \\in [0, t_{cycle}]

    and :math:`\\Delta x = L_x/N_x`, :math:`\\Delta y = L_y/N_y`, with :math:`L_x` and :math:`L_y` respectively the size
    of the scanning surface along the x and y directions, :math:`N_x` and :math:`N_y` the number of points on the surface along
    the x and y directions, :math:`x_{offset}` and :math:`y_{offset}` the offsets along x and y with respect to the normal
    vertical vector of the station.

    .. note::
        Since the FOV of the station is limited, all the points which elevation is lower than the minimum elevation of the 
        station will be discarded.

    .. rubric:: Constructor :

    Parameters
    ----------
    min_elevation : float, default=30.0
        Minimum elevation of the beam. ``min_elevation`` also usually defines the Field of view of
        the station (in degrees). 
    altitude : float, default=200e3
        Altitude of the plane defining the scanning surface (in meters).
    x_size : float, default=50e3
        Size along the local x direction (*East*) of the plane defining the scanning surface (in meters).
    y_size : float, default=50e3
        Size along the local y direction (*North*) of the plane defining the scanning surface (in meters).
    x_num : int, default=20
        Number of scanning points along the local x direction (*East*) of the plane defining the scanning surface.
    y_num : int, default=20
        Number of scanning points along the local y direction (*North*) of the plane defining the scanning surface.
    dwell : float, default=0.1
        Dwell time of the scan (in seconds).
    x_offset : float, default=0.0
        Offset from the local vertical vector of the station along the local x direction (*East*) of the plane defining 
        the scanning surface (in meters).
    y_offset : float, default=0.0
        ffset from the local vertical vector of the station along the local y direction (*North*) of the plane defining 
        the scanning surface (in meters).

    Examples
    --------
    As a simple example, consider a :class:`Plane` scan performing the uniform scanning of 
    a 1000x1000km plane at altitude 1000km. The scan is defined such that there are 5 possible coordinates 
    for each pointing direction along x and  along y. The x and y offsets are set to 0 to ensure that the plane center
    is directly over the station. Finally the dwell time is set to 1 second, meaning that the whole scanning scheme 
    lasts for 25 seconds.

    >>> import sorts
    >>> plane = sorts.scans.Plane(min_elevation=30.0, altitude=1000e3, x_size=1000e3, y_size=1000e3, x_num=5, y_num=5, dwell=1.0, x_offset=0.0, y_offset=0.0)

    To evaluate the result of the ``pointing`` function, we create a time array of 25 elements :
    
    >>> t = np.arange(0.0, 24.0)
    >>> plane.pointing(t)
    array([[-0.40824829 -0.43643578 -0.4472136  -0.43643578 -0.40824829 -0.21821789
          -0.23570226 -0.24253563 -0.23570226 -0.21821789  0.          0.
           0.          0.          0.          0.21821789  0.23570226  0.24253563
           0.23570226  0.21821789  0.40824829  0.43643578  0.4472136   0.43643578
           0.40824829]
         [-0.40824829 -0.21821789  0.          0.21821789  0.40824829 -0.43643578
          -0.23570226  0.          0.23570226  0.43643578 -0.4472136  -0.24253563
           0.          0.24253563  0.4472136  -0.43643578 -0.23570226  0.
           0.23570226  0.43643578 -0.40824829 -0.21821789  0.          0.21821789
           0.40824829]
         [ 0.81649658  0.87287156  0.89442719  0.87287156  0.81649658  0.87287156
           0.94280904  0.9701425   0.94280904  0.87287156  0.89442719  0.9701425
           1.          0.9701425   0.89442719  0.87287156  0.94280904  0.9701425
           0.94280904  0.87287156  0.81649658  0.87287156  0.89442719  0.87287156
           0.81649658]])

    
    To plot the results of the ``pointing`` function, we can use the :ref:`plotting` module implemented
    in sorts:

    >>> radar = sorts.radars.eiscat3d
    >>> sorts.plotting.plot_scanning_sequence(plane, station=radar.tx[0], earth=True, ax=None, plot_local_normal=True, max_range=1000e3)

    .. figure:: ../../../../figures/scans_plane.png

        Example :class:`Plane` scanning scheme. In *blue*, the local vertical vector/in *red*, 
        the pointing direction vector of the tx station of the **EISCAT_3D radar**.
    '''
    def __init__(self, min_elevation=30.0, altitude=200e3, x_size=50e3, y_size=50e3, x_num=20, y_num=20, dwell=0.1, x_offset=0.0, y_offset=0.0):
        super().__init__(coordinates='enu')
        self._dwell = dwell
        ''' Dwell time of the scan (in seconds). '''
        self.min_elevation = min_elevation
        ''' Minimum elevation of the beam. ``min_elevation`` also usually defines the Field of view of the station (in degrees).'''
        self.altitude = altitude
        ''' Altitude of the plane defining the scanning surface (in meters). '''
        
        self.x_size = x_size
        ''' Size along the local x direction (*East*) of the plane defining the scanning surface (in meters). '''
        self.y_size = y_size
        ''' Size along the local y direction (*North*) of the plane defining the scanning surface (in meters). '''
        self.x_num = x_num
        ''' Number of scanning points along the local x direction (*East*) of the plane defining the scanning surface. '''
        self.y_num = y_num
        ''' Number of scanning points along the local y direction (*North*) of the plane defining the scanning surface. '''

        k = np.empty((3, x_num*y_num), dtype=np.float64)

        # create the mesh grid for the pointing directions
        xv, yv = np.meshgrid(
            np.linspace(-x_size*0.5, x_size*0.5, num=self.x_num, endpoint=True) + x_offset,
            np.linspace(-y_size*0.5, y_size*0.5, num=self.y_num, endpoint=True) + y_offset,
            sparse=False, 
            indexing='ij',
         )

        k[0,:] = xv.flatten()
        k[1,:] = yv.flatten()
        k[2,:] = altitude

        # normalize each pointing direction
        k = k/np.linalg.norm(k, axis=0)

        # remove all points outside of the FOV
        min_z = np.sin(np.radians(min_elevation))
        k = k[:, k[2,:] >= min_z]

        self.num = k.shape[1]
        ''' Number of pointing directions. '''
        self.pointings = k
        ''' Pointing directions in the ENU frame of reference of the station. '''


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

        The ``pointing`` function of the :class:`Plane` class returns a set of static pointing
        directions in the ENU coordinate frame *uniformly sampled* from a horizontal plane at altitude
        z. 

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
        See :class:`Plane` for an in-depth example on how to generate a :class:`Plane` scanning scheme.
        '''
        ind = (np.mod(t/self.cycle(), 1)*self.num).astype(np.int)
        return self.pointings[:,ind]