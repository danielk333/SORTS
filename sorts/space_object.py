#!/usr/bin/env python

'''Defines a space object. Encapsulates orbital elements, propagation and related methods.


**Example:**

Using space object for propagation.

.. code-block:: python

    import numpy as n
    import matplotlib.pyplot as plt

    


Using space object with a different propagator.

.. code-block:: python

    import numpy as n
    import matplotlib.pyplot as plt




'''

#Python standard import
import copy

#Third party import
import numpy as np
import pyorb
from astropy.time import Time


class SpaceObject(object):
    '''Encapsulates a object in space who's dynamics is governed in time by a propagator.

    The relation between the Cartesian and Kepler states are a direct transformation according to the below orientation rules.
    If the Kepler elements are given in a Inertial system, to reference the Cartesian to a Earth-fixed system a earth rotation transformation
    must be applied externally of the method.

    #TODO: THIS DOC MUST BE UPDATED TOO


    **Orientation of the ellipse in the coordinate system:**
       * For zero inclination :math:`i`: the ellipse is located in the x-y plane.
       * The direction of motion as True anoamly :math:`\\nu`: increases for a zero inclination :math:`i`: orbit is anti-coockwise, i.e. from +x towards +y.
       * If the eccentricity :math:`e`: is increased, the periapsis will lie in +x direction.
       * If the inclination :math:`i`: is increased, the ellipse will rotate around the x-axis, so that +y is rotated toward +z.
       * An increase in Longitude of ascending node :math:`\Omega`: corresponds to a rotation around the z-axis so that +x is rotated toward +y.
       * Changing argument of perihelion :math:`\omega`: will not change the plane of the orbit, it will rotate the orbit in the plane.
       * The periapsis is shifted in the direction of motion.
       * True anomaly measures from the +x axis, i.e :math:`\\nu = 0` is located at periapsis and :math:`\\nu = \pi` at apoapsis.
       * All anomalies and orientation angles reach between 0 and :math:`2\pi`

       *Reference:* "Orbital Motion" by A.E. Roy.
    

    **Variables:**
       * :math:`a`: Semi-major axis
       * :math:`e`: Eccentricity
       * :math:`i`: Inclination
       * :math:`\omega`: Argument of perihelion
       * :math:`\Omega`: Longitude of ascending node
       * :math:`\\nu`: True anoamly


    :ivar pyorb.Orbit orbit: Orbit instance
    :ivar int oid: Identifying object ID
    :ivar float C_D: Drag coefficient
    :ivar float C_R: Radiation pressure coefficient
    :ivar float A: Area [:math:`m^2`]
    :ivar float m: Mass [kg]
    :ivar float d: Diameter [m]
    :ivar float mjd0: Epoch for state [BC-relative JD]
    :ivar float prop: Propagator instance, child of :class:`~base_propagator.PropagatorBase`
    :ivar dict propagator_options: Propagator initialization keyword arguments
    :ivar dict propagator_args: Propagator call keyword arguments

    The constructor creates a space object using Kepler elements.

    :param float A: Area in square meters
    :param float m: Mass in kg
    :param float C_D: Drag coefficient
    :param float C_R: Radiation pressure coefficient
    :param float mjd0: Epoch for state
    :param int oid: Identifying object ID
    :param float d: Diameter in meters
    :param PropagatorBase propagator: Propagator class pointer
    :param dict propagator_options: Propagator initialization keyword arguments
    :param dict propagator_args: Keyword arguments to be passed to the propagator call
    :param dict kwargs: All additional keywords are used to initialize the orbit
    
    :Keyword arguments:
        * *a* (``float``) -- Semi-major axis in meters
        * *e* (``float``) -- Eccentricity
        * *i* (``float``) -- Inclination in degrees
        * *aop* (``float``) -- Argument of perigee in degrees
        * *raan* (``float``) -- Right ascension of the ascending node in degrees
        * *mu0* (``float``) -- Mean anomaly in degrees
        * *x* (``float``) -- X-position
        * *y* (``float``) -- Y-position
        * *z* (``float``) -- Z-position
        * *vx* (``float``) -- X-velocity
        * *vy* (``float``) -- Y-velocity
        * *vz* (``float``) -- Z-velocity

    '''

    default_parameters = dict(
        d = 0.01,
        C_D = 2.3,
        A = 1.0,
        m = 1.0,
        C_R = 1.0,
    )

    def __init__(self,
            propagator,
            propagator_options = {},
            propagator_args = {},
            parameters = {},
            epoch=Time(57125.7729, format='mjd'),
            oid=1,
            **kwargs
        ):

        self.oid = oid
        self.parameters = copy.copy(SpaceObject.default_parameters)
        self.parameters.update(parameters)
        
        #assume MJD if not "Time" object
        if not isinstance(epoch, Time):
            epoch = Time(epoch, format='mjd', scale='utc')

        self.epoch = epoch

        if 'state' in kwargs:
            self.state = kwargs['state']
        else:
            if 'aop' in kwargs:
                kwargs['omega'] = kwargs.pop('aop')
            if 'raan' in kwargs:
                kwargs['Omega'] = kwargs.pop('raan')
            if 'mu0' in kwargs:
                kwargs['anom'] = kwargs.pop('mu0')

            self.state = pyorb.Orbit(
                M0 = kwargs.get('M_cent', pyorb.M_earth), 
                degrees = True,
                type='mean',
                auto_update = True, 
                direct_update = True,
                num = 1,
                m = self.parameters.get('m', 0.0),
                **kwargs
            )

        self.__propagator = propagator
        self.propagator_options = propagator_options
        self.propagator_args = propagator_args

        self.propagator = propagator(**propagator_options)


    def copy(self):
        '''Returns a copy of the SpaceObject instance.
        '''
        return SpaceObject(
            propagator = self.__propagator,
            propagator_options = copy.deepcopy(self.propagator_options),
            propagator_args = copy.deepcopy(self.propagator_args),
            epoch=self.oid,
            oid=self.oid,
            state=copy.copy(self.state),
            parameters = copy.deepcopy(self.parameters),
        )


    @property
    def m(self):
        '''Object mass, if changed the Kepler elements stays constant
        '''
        return self.parameters['m']
    
    @m.setter
    def m(self, val):
        self.parameters['m'] = val
        if isinstance(self.state, pyorb.Orbit):
            self.state.m[0] = val            
            self.state.calculate_cartesian()
            self.state.calculate_kepler()


    @property
    def d(self):
        if 'd' in self.parameters:
            diam = self.parameters['d']
        elif 'diam' in self.parameters:
            diam = self.parameters['diam']
        elif 'r' in self.parameters:
            diam = self.parameters['r']*2
        elif 'A' in self.parameters:
            diam = np.sqrt(self.parameters['A']/np.pi)*2
        else:
            raise ValueError('Space object does not have a diameter parameter: cannot calculate SNR/RCS')
        return diam


    @property
    def orbit(self):
        if isinstance(self.state, pyorb.Orbit):
            return self.state
        else:
            raise AttributeError('SpaceObject state is not "pyorb.Orbit"')



    def propagate(self, dt):
        '''Propagate and change the epoch of this space object if the state is a `pyorb.Orbit`.
        '''

        if 'in_frame' in self.propagator.settings and 'out_frame' in self.propagator.settings:
            out_frame = self.propagator.settings['out_frame']
            self.propagator.set(out_frame=self.propagator.settings['in_frame'])

            state = self.get_state(np.array([dt], dtype=np.float64))

            self.propagator.set(out_frame=out_frame)
        else:
            state = self.get_state(np.array([dt], dtype=np.float64))


        self.mjd0 = self.mjd0 + dt/(3600.0*24.0)

        x, y, z, vx, vy, vz = state_frame.flatten()

        self.update(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
        )


    def update(self, **kwargs):
        '''If a `pyorb.Orbit` is present, updates the orbital elements and Cartesian state vector of the space object.

        Can update any of the related state parameters, all others will automatically update.

        Cannot update Keplerian and Cartesian elements simultaneously.

        :param float a: Semi-major axis in km
        :param float e: Eccentricity
        :param float i: Inclination in degrees
        :param float aop/omega: Argument of perigee in degrees
        :param float raan/Omega: Right ascension of the ascending node in degrees
        :param float mu0/anom: Mean anomaly in degrees
        :param float x: X position in km
        :param float y: Y position in km
        :param float z: Z position in km
        :param float vx: X-direction velocity in km/s
        :param float vy: Y-direction velocity in km/s
        :param float vz: Z-direction velocity in km/s
        '''
        if 'aop' in kwargs:
            kwargs['omega'] = kwargs.pop('aop')
        if 'raan' in kwargs:
            kwargs['Omega'] = kwargs.pop('raan')
        if 'mu0' in kwargs:
            kwargs['anom'] = kwargs.pop('mu0')

        self.orbit.update(**kwargs)



    def __str__(self):
        p = '\nSpace object {}: {}:\n'.format(self.oid,repr(self.epoch))
        p+= str(self.state) + '\n'
        p+= f'Parameters: ' + ', '.join([f'{key}={val:.3f}' for key,val in self.parameters.items()])
        return p


    def get_position(self,t):
        '''Gets position at specified times using propagator instance.

        :param float/list/numpy.ndarray t: Time relative epoch in seconds.

        :return: Array of positions as a function of time.
        :rtype: numpy.ndarray of size (3,len(t))
        '''
        ecefs = self.get_state(t)
        return ecefs[:3,:]


    def get_velocity(self,t):
        '''Gets velocity at specified times using propagator instance.

        :param float/list/numpy.ndarray t: Time relative epoch in seconds.

        :return: Array of positions as a function of time.
        :rtype: numpy.ndarray of size (3,len(t))
        '''
        ecefs = self.get_state(t)
        return ecefs[3:,:]


    def get_state(self, t):
        '''Gets ECEF state at specified times using propagator instance.

        :param float/list/numpy.ndarray t: Time relative epoch in seconds.

        :return: Array of state (position and velocity) as a function of time.
        :rtype: numpy.ndarray of size (6,len(t))
        '''
        if isinstance(t, Time):
            t = t - self.epoch
        elif not isinstance(t,np.ndarray):
            if not isinstance(t,list):
                t = [t]
            t = np.array(t,dtype=np.float64)
        
        kw = {}
        kw.update(self.propagator_args)
        kw.update(self.parameters)

        ecefs = self.propagator.propagate(
            t = t,
            state0 = self.state,
            epoch = self.epoch,
            **kw
        )
        return ecefs

