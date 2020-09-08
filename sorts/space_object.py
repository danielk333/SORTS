#!/usr/bin/env python

'''Defines a space object. Encapsulates orbital elements, propagation and related methods.


**Example:**

Using space object for propagation.

.. code-block:: python

    import numpy as n
    import matplotlib.pyplot as plt

    ecefs=o.get_state(t)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)
    .draw_earth_grid(ax)

    ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],'-',alpha=0.5,color="black")
    plt.title("Orbital propagation test")
    plt.show()


Using space object with a different propagator.

.. code-block:: python

    import numpy as n
    import matplotlib.pyplot as plt


    o = so.SpaceObject(
        a=7000, e=0.0, i=69,
        raan=0, aop=0, mu0=0,
        C_D=2.3, A=1.0, m=1.0,
        C_R=1.0, oid=42,
        mjd0=57125.7729,
        propagator = PropagatorOrekit,
        propagator_options = {
            'in_frame': 'TEME',
            'out_frame': 'ITRF',
        },
    )

    t=np.linspace(0,24*3600,num=1000, dtype=np.float)
    ecefs=o.get_state(t)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)
    plothelp.draw_earth_grid(ax)

    ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],'-',alpha=0.5,color="black")
    plt.title("Orbital propagation test")
    plt.show()


'''

#Python standard import
import copy

#Third party import
import numpy as np
import pyorb
from astropy.time import Time


#Local import
from .constants import R_earth


class SpaceObject(object):
    '''Encapsulates a object in space who's dynamics is governed in time by a propagator.

    The relation between the Cartesian and Kepler states are a direct transformation according to the below orientation rules.
    If the Kepler elements are given in a Inertial system, to reference the Cartesian to a Earth-fixed system a earth rotation transformation
    must be applied externally of the method.


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
    def __init__(self,
            propagator,
            d=0.01,
            C_D=2.3,
            A=1.0,
            m=1.0,
            epoch=Time(57125.7729, format='mjd'),
            oid=1,
            M_cent = pyorb.M_earth,
            C_R = 1.0,
            propagator_options = {},
            propagator_args = {},
            **kwargs
        ):
        if 'aop' in kwargs:
            kwargs['omega'] = kwargs.pop('aop')
        if 'raan' in kwargs:
            kwargs['Omega'] = kwargs.pop('raan')
        if 'mu0' in kwargs:
            kwargs['anom'] = kwargs.pop('mu0')

        self.orbit = pyorb.Orbit(
            M0 = M_cent, 
            degrees = True,
            type='mean',
            auto_update = True, 
            direct_update = True,
            num = 1,
            m = m,
            **kwargs
        )

        self.oid = oid
        self.C_D = C_D
        self.C_R = C_R
        self.A = A
        self.d = d
        self.epoch = epoch
        self._propagator_cls = propagator
        self.propagator_options = propagator_options
        self.propagator = propagator(**propagator_options)
        self.propagator_args = propagator_args


    def copy(self):
        '''Returns a copy of the SpaceObject instance.
        '''
        new_so = SpaceObject(
            d=self.d,
            C_D=self.C_D,
            A=self.A,
            m=self.m,
            epoch=self.epoch,
            oid=self.oid,
            M_cent = self.M_cent,
            C_R = self.C_R,
            propagator = self._propagator_cls,
            propagator_options = copy.copy(self.propagator_options),
            propagator_args = copy.copy(self.propagator_args),
        )
        new_so.orbit.kepler = self.orbit._kep.copy()


    @property
    def m(self):
        '''Object mass, if changed the Kepler elements stays constant
        '''
        return self.orbit.m[0]
    
    @m.setter
    def m(self, val):
        self.orbit.m[0] = val
        self.orbit.calculate_cartesian()


    @property
    def diam(self):
        return self.d
    
    @diam.setter
    def diam(self, val):
        self.d = val


    def propagate(self, dt):
        '''Propagate and change the epoch of this space object
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
        '''Updates the orbital elements and Cartesian state vector of the space object.

        Can update any of the related state parameters, all others will automatically update.

        Cannot update Keplerian and Cartesian elements simultaneously.

        :param float a: Semi-major axis in km
        :param float e: Eccentricity
        :param float i: Inclination in degrees
        :param float aop: Argument of perigee in degrees
        :param float raan: Right ascension of the ascending node in degrees
        :param float mu0: Mean anomaly in degrees
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
        p = '\nSpace object {} at epoch {} MJD:\n'.format(self.oid,self.epoch)
        p+= str(self.orbit) + '\n'
        p+= 'PARAMETERS: diameter = {:.3f} m, drag coefficient = {:.3f}, albedo = {:.3f}, area = {:.3f}, mass = {:.3f} kg\n'.format(self.d,self.C_D,self.C_R,self.A,self.m)
        if len(self.propagator_args) > 0:
            p+= 'ADDITIONAL KEYWORDS: ' + ', '.join([ '{} = {}'.format(key,val) for key,val in self.propagator_args.items() ])
        return p


    def __enter__(self):
        return self

    def get_position(self,t):
        '''Gets position at specified times using propagator instance.

        :param float/list/numpy.ndarray t: Time relative epoch in seconds.

        :return: Array of positions as a function of time.
        :rtype: numpy.ndarray of size (3,len(t))
        '''
        ecefs = self.get_state(t)
        return(ecefs[:3,:])


    def get_velocity(self,t):
        '''Gets velocity at specified times using propagator instance.

        :param float/list/numpy.ndarray t: Time relative epoch in seconds.

        :return: Array of positions as a function of time.
        :rtype: numpy.ndarray of size (3,len(t))
        '''
        ecefs = self.get_state(t)
        return(ecefs[3:,:])

    
    def __exit__(self, exc_type, exc_value, traceback):
        pass


    def get_state(self,t):
        '''Gets ECEF state at specified times using propagator instance.

        :param float/list/numpy.ndarray t: Time relative epoch in seconds.

        :return: Array of state (position and velocity) as a function of time.
        :rtype: numpy.ndarray of size (6,len(t))
        '''
        if not isinstance(t,np.ndarray):
            if not isinstance(t,list):
                t = [t]
            t = np.array(t,dtype=np.float)
        
        ecefs = self.propagator.propagate(
            t = t, 
            state0 = np.squeeze(self.orbit.cartesian), 
            epoch=self.epoch,
            C_D=self.C_D,
            C_R=self.C_R,
            A=self.A,
            m=self.m,
            **self.propagator_args
        )
        return ecefs

