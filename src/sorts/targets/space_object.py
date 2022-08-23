#!/usr/bin/env python

''' Defines a space object. Encapsulates orbital elements, propagation and related methods. '''

#Python standard import
import copy

#Third party import
import numpy as np
import pyorb
from astropy.time import Time, TimeDelta


class SpaceObject(object):
    ''' Encapsulates a object in space which dynamics are governed by :ref:`propagators`.

    The state of the object is stored in a :class:`pyorb.Orbit` instance which allows for the direct transformation between
    Cartesian and Kepler state vectors. 

    The orbit of a space object obeys the following convention :
       * For zero inclination :math:`i`: the ellipse is located in the x-y plane.
       * The direction of motion as True anoamly :math:`\\nu`: increases for a zero inclination :math:`i`: orbit is anti-coockwise, i.e. from +x towards +y.
       * If the eccentricity :math:`e`: is increased, the periapsis will lie in +x direction.
       * If the inclination :math:`i`: is increased, the ellipse will rotate around the x-axis, so that +y is rotated toward +z.
       * An increase in Longitude of ascending node :math:`\\Omega`: corresponds to a rotation around the z-axis so that +x is rotated toward +y.
       * Changing argument of perihelion :math:`\\omega`: will not change the plane of the orbit, it will rotate the orbit in the plane.
       * The periapsis is shifted in the direction of motion.
       * True anomaly measures from the +x axis, i.e :math:`\\nu = 0` is located at periapsis and :math:`\\nu = \\pi` at apoapsis.
       * All anomalies and orientation angles reach between 0 and :math:`2\\pi`

       *Reference:* "Orbital Motion" by A.E. Roy.
    
    
    Parameters
    ----------
    propagator : :class:`sorts.Propagator<sorts.targets.propagator.base>` 
        :class:`sorts.Propagator<sorts.targets.propagator.base>`  class used to propagate the states of the space object.
    propagator_options : dict
        :class:`sorts.Propagator<sorts.targets.propagator.base>`  initialization keyword arguments.
    propagator_args : dict
        Keyword arguments to be passed to the propagator call.
    parameters : dict
        Space object parameters. Some parameters are by default recognized by the :class:`SpaceObject`.
    epoch : float / :class:`astropy.time.Time`
        Epoch for the state, if not an astropy :class:`astropy.time.Time` is given, it is assumed the input float is `format='mjd', scale='utc'`.
    oid : int
        Object ID
    kwargs : dict
        All additional keywords are used to initialize the orbit. If the keyword `state` is given, that input variable 
        is simply saved as the state. This is useful for e.g. :class:`sorts.propagator.SGP4<sorts.targets.propagator.pysgp4.SGP4>` 
        that does not use regular cartesian states for propagation.
    
        * *a* (``float``)    -- Semi-major axis in meters
        * *e* (``float``)    -- Eccentricity
        * *i* (``float``)    -- Inclination in degrees
        * *aop* (``float``)  -- Argument of perigee in degrees
        * *raan* (``float``) -- Right ascension of the ascending node in degrees
        * *mu0* (``float``)  -- Mean anomaly in degrees
        * *x* (``float``)    -- X-position
        * *y* (``float``)    -- Y-position
        * *z* (``float``)    -- Z-position
        * *vx* (``float``)   -- X-velocity
        * *vy* (``float``)   -- Y-velocity
        * *vz* (``float``)   -- Z-velocity


    Examples
    --------
    Using space object for propagation.

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from astropy.time import Time

        from sorts.propagator import Kepler
        from sorts import SpaceObject

        options = dict(
            settings=dict(
                in_frame='GCRS',
                out_frame='GCRS',
            ),
        )

        t = np.linspace(0,3600*24.0*2,num=5000)

        obj = SpaceObject(
            Kepler,
            propagator_options = options,
            a = 7000e3, 
            e = 0.0, 
            i = 69, 
            raan = 0, 
            aop = 0, 
            mu0 = 0, 
            epoch = Time(57125.7729, format='mjd'),
            parameters = dict(
                d = 0.2,
            )
        )

        print(obj)

        states = obj.get_state(t)

        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(states[0,:], states[1,:], states[2,:],"-b")

        max_range = np.linalg.norm(states[0:3,0])*2

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        plt.show()
    '''

    default_parameters = dict(
        C_D = 2.3,
        m = 1.0,
        C_R = 1.0,
    )
    ''' Default space object parameters.
    
    Values 
     - C_D = 2.3
     - m = 1.0
     - C_R = 1.0

    .. seealso::
        - :attr:`SpaceObject.C_D` : object drag coefficient.
        - :attr:`SpaceObject.m` : object mass.
        - :attr:`SpaceObject.C_R` : object radiation pressure coefficient.
    '''

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
        ''' float : Object index. ''' 
        self.parameters = copy.copy(SpaceObject.default_parameters)
        ''' dict : Parameters for the space object. ''' 
        self.parameters.update(**parameters)
                
        #assume MJD if not "Time" object
        if not isinstance(epoch, Time):
            epoch = Time(epoch, format='mjd', scale='utc')

        self._epoch = epoch
        ''' dict : Parameters for the space object. ''' 

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

        # add orbit properties to class
        for key in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
            property_code = f"""@property   
            \ndef prop(self,):
            \n    val = self.state.{key}
            \n    return val

            \n@prop.setter
            \ndef prop(self, value):
            \n    self.state.{key} = value

            \nsetattr(self.__class__, '{key}', prop)
            """
            exec(property_code)

        # add property to class
        for key in parameters:
            if key not in pyorb.Orbit.UPDATE_KW and not hasattr(self, key):
                @property
                def prop(self):
                    return self.parameters[key]

                @prop.setter
                def prop(self, value):
                    self.parameters[key] = value

                # add attribute to class to enable calls like space_object.X
                setattr(self.__class__, key, prop)

        self.__propagator = propagator
        ''' Propagator class used for the propagation of the space object states. '''
        self.propagator_options = propagator_options
        ''' Space object propagator options. '''
        self.propagator_args = propagator_args
        ''' Arguments used by the propagator to propagates the space object states. '''
        self.propagator = propagator(**propagator_options)
        ''' Propagator class used for the propagation of the space object states. '''


    def copy(self):
        ''' Performs a deepcopy of the :class:`SpaceObject` instance. 
        '''
        return SpaceObject(
            propagator=self.__propagator,
            propagator_options=copy.deepcopy(self.propagator_options),
            propagator_args=copy.deepcopy(self.propagator_args),
            epoch=copy.deepcopy(self._epoch),
            oid=self.oid,
            state=copy.deepcopy(self.state),
            parameters=copy.deepcopy(self.parameters),
        )


    @property
    def m(self):
        ''' Space object mass (in kg), if changed the Kepler elements stays constant. '''
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
        ''' Space object diameter (in meters). '''
        if 'd' in self.parameters:
            diam = self.parameters['d']
        elif 'diam' in self.parameters:
            diam = self.parameters['diam']
        elif 'r' in self.parameters:
            diam = self.parameters['r']*2
        elif 'A' in self.parameters:
            diam = np.sqrt(self.parameters['A']/np.pi)*2
        else:
            raise AttributeError('Space object does not have a diameter parameter or any way to calculate one')
            
        return diam


    @property
    def C_R(self):
        ''' Object radiation pressure coefficient. '''
        return self.parameters['C_R']


    @property
    def C_D(self):
        ''' Object drag coefficient. '''
        return self.parameters['C_D']


    @property
    def orbit(self):
        ''' Space object orbit instance encapsulating the `state` if the latter is an orbit. '''
        if isinstance(self.state, pyorb.Orbit):
            return self.state
        else:
            raise AttributeError('SpaceObject state is not "pyorb.Orbit"')



    @property
    def out_frame(self):
        ''' Output reference frame of the computed space object states used by the propagator. 

        .. seealso::
            - :ref:`propagators` : for more information about the possible input/output frames.
        '''
        if 'out_frame' not in self.propagator.settings:
            return None 

        return self.propagator.settings['out_frame']
            

    @out_frame.setter
    def out_frame(self, val):
        ''' Output reference frame of the computed space object states used by the propagator. 

        .. seealso::
            - :ref:`propagators` : for more information about the possible input/output frames.
        '''
        if 'settings' not in self.propagator_options:
            self.propagator_options['settings'] = {}
            
        self.propagator_options['settings']['out_frame'] = val
        self.propagator.settings['out_frame'] = val


    @property
    def in_frame(self):
        ''' Input reference frame of the computed space object states used by the propagator. 

        .. seealso::
            - :ref:`propagators` : for more information about the possible input/output frames.
        '''
        if 'in_frame' not in self.propagator.settings:
            return None 

        return self.propagator.settings['in_frame']


    @in_frame.setter
    def in_frame(self, val):
        ''' Input reference frame of the computed space object states used by the propagator. 

        .. seealso::
            - :ref:`propagators` : for more information about the possible input/output frames.
        '''
        if 'settings' not in self.propagator_options:
            self.propagator_options['settings'] = {}
            
        self.propagator_options['settings']['in_frame'] = val
        self.propagator.settings['in_frame'] = val


    @property
    def epoch(self):
        ''' Space obejct reference epoch. '''
        return self._epoch 


    def propagate(self, dt):
        ''' Propagate and change the epoch of this space object if the state is a `pyorb.Orbit`. 

        Updates the state and epoch of the object.

        Parameters
        ----------
        dt : float
            Time interval through which the object is propagated in time relative to the current epoch of the object (in seconds). 
            The new state of the object will correspond to the state at time :math:`t_0 + dt`.

        Returns 
        -------
        None
        '''

        if 'in_frame' in self.propagator.settings and 'out_frame' in self.propagator.settings:
            out_frame = self.propagator.settings['out_frame']
            self.propagator.set(out_frame=self.propagator.settings['in_frame'])

            state = self.get_state(np.array([dt], dtype=np.float64))

            self.propagator.set(out_frame=out_frame)
            
        else:
            state = self.get_state(np.array([dt], dtype=np.float64))


        self._epoch = self._epoch + TimeDelta(dt, format='sec')

        x, y, z, vx, vy, vz = state.flatten()

        self.update(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
        )


    def update(self, **kwargs):
        ''' Updates the orbital elements and Cartesian state vector of the space object if a :class:`pyorb.Orbit` is present.

        Can update any of the related state parameters, all others will automatically update. This function cannot be used
        to update Keplerian and Cartesian elements simultaneously.
        
        Parameters
        ----------
        a : float
            Semi-major axis in km.
        e : float
            Eccentricity.
        i : float
            Inclination in degrees.
        aop / omega : float
            Argument of perigee in degrees.
        raan / Omega : float
            Right ascension of the ascending node in degrees.
        mu0 / anom : float
            Mean anomaly in degrees.
        x : float
            X position in km.
        y : float
            Y position in km.
        z : float
            Z position in km.
        vx : float
            X-direction velocity in km/s.
        vy : float
            Y-direction velocity in km/s.
        vz : float
            Z-direction velocity in km/s.

        Returns
        -------
        None
        '''
        if not isinstance(self.state, pyorb.Orbit):
            raise ValueError(f'Cannot update non-Orbit state ({type(self.state)})')

        # remove duplicated elements in the parameters to update
        if 'aop' in kwargs:
            kwargs['omega'] = kwargs.pop('aop')
        if 'raan' in kwargs:
            kwargs['Omega'] = kwargs.pop('raan')
        if 'mu0' in kwargs:
            kwargs['anom'] = kwargs.pop('mu0')

        # adds a new property if not already present.
        for key in kwargs:
            if key not in pyorb.Orbit.UPDATE_KW:
                self.parameters[key] = kwargs[key]

                # add property to class
                @property
                def prop(self):
                    return self.parameters[key]

                @prop.setter
                def prop(self, value):
                    self.parameters[key] = value

                # add attribute to class to enable calls like space_object.X
                setattr(self.__class__, key, prop)

        # update the pyorb orbit
        self.orbit.update(**kwargs)



    def __str__(self):
        ''' Overloaded __str__ method. '''
        p = '\nSpace object {}: {}:\n'.format(self.oid, repr(self._epoch))
        
        p+= str(self.state) + '\n'
        p+= f'Parameters: ' + ', '.join([
            f'{key}={val}'
            for key,val in self.parameters.items()
        ])
        return p


    def get_position(self,t):
        ''' Gets position at specified times using propagator instance.
        
        This function propagates the states of the space object to fet all the
        states at each specified time point relative to the space object epoch.

        Parameters
        ----------
        t : float / list / numpy.ndarray (N,)
            Time relative epoch in seconds.
        
        Returns
        -------
        ecefs : numpy.ndarray (3, N)
            Array of positions as a function of time.
        '''
        ecefs = self.get_state(t)
        return ecefs[:3,:]


    def get_velocity(self,t):
        ''' Gets velocity at specified times using propagator instance.
        
        This function propagates the states of the space object to fet all the
        states at each specified time point relative to the space object epoch.

        Parameters
        ----------
        t : float / list / numpy.ndarray (N,)
            Time relative epoch in seconds.
        
        Returns
        -------
        ecefs : numpy.ndarray (3, N)
            Array of velocities as a function of time.
        '''
        ecefs = self.get_state(t)
        return ecefs[3:,:]


    def get_state(self, t):
        ''' Gets ECEF state at specified times using propagator instance.

        This function propagates the states of the space object to fet all the
        states at each specified time point relative to the space object epoch.
        
        Parameters
        ----------
        t : int/float/list/numpy.ndarray/astropy.time.Time/astropy.time.TimeDelta
            Time relative epoch in seconds.
        
        Returns
        -------
        states : numpy.ndarray of size (6,len(t))
            Array of state (position and velocity) as a function of time.
        '''
        kw = {}
        kw.update(self.propagator_args)
        kw.update(self.parameters)

        ret = self.propagator.propagate(
            t = t,
            state0 = self.state,
            epoch = self._epoch,
            **kw
        )

        #if propagator returns something non-standard, just return that
        #Otherwise, ensures a 2d-object is returned
        if isinstance(ret, np.ndarray):
            if len(ret.shape) == 1:
                ret = ret.reshape((6,1))

        return ret