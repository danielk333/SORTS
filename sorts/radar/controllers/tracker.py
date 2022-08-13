#!/usr/bin/env python

'''#TODO

'''

import numpy as np

from . import radar_controller
from ..system import Radar
from ..scheduler import RadarSchedulerBase
from .. import radar_controls

from sorts.common import interpolation

class Tracker(radar_controller.RadarController):
    ''' Generates a control sequence tracking a set of ECEF points.

    The :class:`Tracker` controller generates a set of tracking controls allowing for 
    the tracking of a set of ECEF points in time. If the points correspond to the
    consecutive states (in the ECEF frame) of a :class:`Space object<sorts.targets.space_object.SpaceObject>`,
    the tracking controls will effectively allow the radar to track the space object in time
    (see the :ref:`example below<tracker_controller_example>`).

    .. note::
        The tracking of a space object assumes that our knowledge of the object's orbit at time
        :math:`t_0` is sufficient to propagate the orbit in time to determine the consecutive 
        ECEF points to be targetted by the radar system.

    When coherent/incoherent integration is used, the implementation of the controller allows for
    multiple pointing directions per time slice (but this number is set to be constant).

    .. seealso::

        * :class:`sorts.Radar<sorts.radar.system.radar.Radar>` : class encapsulating the radar system.
        * :class:`sorts.RadarController<sorts.radar.controllers.radar_congirtroller.RadarController>` : radar controller base class.
        * :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>` : class encapsulating radar control sequences.
        * :class:`sorts.SpaceObject<sorts.targets.space_object.SpaceObject>` : class encapsulating a space object.
    
    Parameters
    ----------
    profiler : :class:`sorts.Profiler<sorts.common.profing.Profiler>`, default=None
        Profiler instance used to monitor the computation performances of the class methods. 
    logger : :class:`logging.Logger`, default=None
        Logger instance used to log the computation status of the class methods.
    
    Examples
    --------
    .. _tracker_controller_example:

    This example showcases the generation of multiple controls sequences allowing to track a 
    :class:`Space object<sorts.targets.space_object.SpaceObject>` passing over the EISCAT_3D radar system:

    .. code-block:: Python

        import numpy as np
        import matplotlib.pyplot as plt

        import sorts

        # RADAR system definition
        eiscat3d = sorts.radars.eiscat3d

        # controller and simulation parameters
        max_points = 100
        end_t = 3600*12
        t_slice = 10
        tracking_period = 20
        states_per_slice = 10

        # Propagator
        Prop_cls = sorts.propagator.Kepler
        Prop_opts = dict(
            settings = dict(
                out_frame='ITRS',
                in_frame='TEME',
            ),
        )
        # Object definition
        space_object = sorts.SpaceObject(
                Prop_cls,
                propagator_options = Prop_opts,
                a = 7200.0e3, 
                e = 0.1,
                i = 80.0,
                raan = 86.0,
                aop = 0.0,
                mu0 = 50.0,
                epoch = 53005.0,
                parameters = dict(
                    d = 0.1,
                ),
            )
        # create state time array
        t_states = sorts.equidistant_sampling(
            orbit = space_object.state, 
            start_t = 0, 
            end_t = end_t, 
            max_dpos=50e3,
        )

        # create tracking controller
        tracker_controller = sorts.controllers.Tracker()

        # get object states/passes in ECEF frame
        object_states = space_object.get_state(t_states)
        eiscat_passes = sorts.find_simultaneous_passes(t_states, object_states, eiscat3d.tx + eiscat3d.rx)

        # plot results
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plotting station ECEF positions and earth grid
        for tx in eiscat3d.tx:
            ax.plot([tx.ecef[0]], [tx.ecef[1]], [tx.ecef[2]],'or')

        for rx in eiscat3d.rx:
            ax.plot([rx.ecef[0]], [rx.ecef[1]], [rx.ecef[2]],'og')

        sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

        # plotting object states
        ax.plot(object_states[0], object_states[1], object_states[2], "--b", alpha=0.2)

        # compute and plot controls for each pass
        for pass_id in range(np.shape(eiscat_passes)[0]):
            # get states within pass to generate tracking controls
            tracking_states = object_states[:, eiscat_passes[pass_id].inds]
            t_states_i = t_states[eiscat_passes[pass_id].inds]
            
            # generate controls
            t_controller = np.arange(t_states_i[0], t_states_i[-1] + tracking_period, tracking_period)
            controls = tracker_controller.generate_controls(t_controller, eiscat3d, t_states_i, tracking_states, t_slice=t_slice, max_points=max_points, states_per_slice=states_per_slice)
            
            # plot states being tracked
            ax.plot(tracking_states[0], tracking_states[1], tracking_states[2], "-", color="blue")
            
            # plot beam directions over each control period
            for period_id in range(controls.n_periods):
                ctrl = controls.get_pdirs(period_id)
                sorts.plotting.plot_beam_directions(ctrl, eiscat3d, ax=ax, tx_beam=True, rx_beam=True, zoom_level=0.9, azimuth=10, elevation=10)

        plt.show()

    .. figure:: ../../../../figures/example_tracker_controller.png

    '''

    META_FIELDS = radar_controller.RadarController.META_FIELDS

    def __init__(self, profiler=None, logger=None, **kwargs):
        ''' Default class constructor. '''
        super().__init__(profiler=profiler, logger=logger, **kwargs)
        
        self.meta['controller_type'] = self.__class__
        
        if self.logger is not None:
            self.logger.info(f'Tracker:init')
    

    def retreive_target_states(
            self, 
            controls, 
            t_states,
            state_interpolator,
            states_per_slice,
            ):
        ''' Retreives target's states given the time array and the state interpolator.

        This function removes the control time slices and periods which are outside of the
        field of view of the radar system. Then, it performs an interpolation to retrieve all
        the intermediate states when ``t_states`` has less elements that there are pointing
        directions.

        Parameters
        ----------
        controls : :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            RadarControls instance containing the current control sequence being generated.
        t_states : numpy.ndarray (N,)
            State time points (in seconds).
        state_interpolator : :class:`sorts.Interpolator<sorts.common.interpolation.Interpolator>`
            Interpolation algorithm used to estimate the states of the object (i.e. Legendre8, ...)
        states_per_slice : int
            Number of target states per slice. 
            This number can be used when multiple measurements are performed during a given 
            time slice (i.e. for coherent integration)
    
        Returns
        -------
        target_states : numpy.ndarray (n_periods, 6, ...)
            Interpolated states targeted by the radar system.
        t_states_final : numpy.ndarray (n_periods, 6, ...)
            Final time array corresponding to the interpolated states.
        '''
        n_control_periods = controls.n_periods
        target_states   = np.empty((n_control_periods, ), dtype=object)
        t_states_final  = np.empty((n_control_periods, ), dtype=object)
        
        # find states within the control interval
        states_msk = np.logical_and(t_states >= controls.t[0][0], t_states <= controls.t[-1][-1] + controls.t_slice[-1][-1])
        keep = np.full((n_control_periods,), False, dtype=bool)
            
        if np.size(np.where(states_msk)[0]) > 0:
            t_shape = np.shape(controls.t) # [scheduling slice/control subarray][time points]
            
            # get start and end points of the object pass
            t_start = t_states[states_msk][0]
            t_end = t_states[states_msk][-1]

            flag_found_pass = False
            
            # get the states for each time sub-array
            for period_id in range(controls.n_periods):   
                if controls.t[period_id] is None:
                    continue
                    
                pass_msk = np.logical_and(controls.t[period_id] >= t_start, controls.t[period_id] <= t_end) # get all time slices in the pass

                if np.size(np.where(pass_msk)) < len(pass_msk): 
                    if self.logger is not None:
                        self.logger.warning(f"tracker:retreive_target_states: some incomplete tracking control slices have been discarded between t={controls.t[period_id][0]} and t={controls.t[period_id][-1]} seconds (control sub array {period_id})")
                
                # if there are some time slices inside the pass
                if np.size(np.where(pass_msk)[0]) > 0:
                    flag_found_pass = True      
                    keep[period_id] = True

                    dt_states = controls.t_slice[period_id][pass_msk]/float(states_per_slice) # time interval between states
                    t_states = np.repeat(controls.t[period_id][pass_msk], states_per_slice).astype(np.float64) # initializes space object state sampling time array
                    
                    # add intermediate time points to sample space object states
                    for ix in range(states_per_slice):
                        t_states[ix::states_per_slice] = t_states[ix::states_per_slice] + ix*dt_states

                    t_states_final[period_id]     = t_states
                    target_states[period_id]      = state_interpolator.get_state(t_states_final[period_id])[0:3, :].astype(np.float64) #[scheduling slice/control subarray][time points][xyz]
                else:                    
                    if flag_found_pass is True:
                        break

        # remove control periods where no space object states are present
        controls.remove_periods(keep)

        return target_states[keep], t_states_final[keep]


    def compute_pointing_directions(
            self, 
            controls,
            period_id, 
            args, 
            ):
        ''' Computes the radar pointing directions for each time slice over a given control
        period.

        This function computes the pointing directions allowing the radar to track a set of ECEF points.

        The station pointing direction corresponds to the normalized Poynting vector of the transmitted 
        signal (i.e. its direction of propagation / direction of the beam). It is possible to have multiple 
        pointing direction vectors for a sinle station per time slice. This feature can therefore be used 
        to model digital beam steering of phased array receiver antennas (see radar_eiscat3d).

        .. seealso::
            :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>` : class encapsulating a radar control sequence.
            :attr:`sorts.RadarControls.get_pdirs<sorts.radar.radar_controls.RadarControls.get_pdirs>` : function used to compute the pointing directions associated with a given control sequence. 

        This function will be called by the :attr:`RadarControls.get_pdirs<sorts.radar.radar_controls.RadarControls.get_pdirs>` 
        in order to generate the pointing direction controls over each control period. Therefore, 
        this function is usually called after the generation of station property controls is completed. 
    
        .. note::
            The choice of allowing the user to compute separatly the pointing directions from the property controls
            can be justified by the fact that pointing direction arrays can easily reach millions of values and computing
            them all at once can quickly **fill up the RAM** of desktop computers.

        Parameters
        ----------
        controls : :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            RadarControls instance containing the current control sequence being generated.
        period_id : int
            Control period index.
        args : list of variables
            The list of additional variables required by the function:

            - interpolated_states : numpy.ndarray (n_periods, 6, n_dirs_per_period)
                Set of interpolated states to be targetted by the radar.
            - t_dirs : numpy.ndarray (n_periods, n_dirs_per_period)
                Pointing direction time array.

            >>> interpolated_states, t_dirs = args
        
        Returns 
        -------
        pdirs : dict
            Pointing direction computation results. The data is organized as a dictionnary with 3 keys :

            - "tx":
                Contains the pointing directions of all radar :class:`sorts.TX<sorts.radar.system.station.TX>` stations.
                The data within ``pdirs["tx"]`` is organized as follows :

                >>> pdirs['tx'][txi, 0, i, j]

                With :

                - txi :
                    Index of the :class:`sorts.TX<sorts.radar.system.station.TX>` station within the :attr:`Radar.tx<sorts.radar.system.radar.Radar.tx>` list.
                - i :
                    :math:`i^{th}` component of the pointing direction. :math:`i \\in [\\![ 0, 3 [\\![`
                - j : 
                    :math:`j^{th}` time point.
                    Beware that since there can be multiple pointing directions per time slice, the number of pointing directions
                    for a single station is greater or equal to the number of time slices of the control sequence.

            - "rx":
                Contains the pointing directions of all radar :class:`sorts.RX<sorts.radar.system.station.RX>` stations.
                The data within ``pdirs["rx"]`` is organized as follows :

                >>> pdirs['rx'][rxi, txi, i, j]

                With :

                - txi :
                    Index of the :class:`sorts.RX<sorts.radar.system.station.RX>` station within the :attr:`Radar.rx<sorts.radar.system.radar.Radar.rx>` list.
                - i :
                    :math:`i^{th}` component of the pointing direction. :math:`i \\in [\\![ 0, 3 [\\![`
                - j : 
                    :math:`j^{th}` time point.
                    Beware that since there can be multiple pointing directions per time slice, the number of pointing directions
                    for a single station is greater or equal to the number of time slices of the control sequence.
            
            - "t": 
                Contains the pointing direction time array. When there are more than one pointing direction per time slice, the
                number of time points within the pointing direction time array will be greater than the number of time slices within the 
                control sequence.

        Examples
        --------
        Consider a control sequence ``controls`` generated by a radar controller performing a scan over 9 time slices. 
        We will assume that the tracking controller performs 2 tracking updates per time slice (one at :math:`t=t^k` and
        one at :math:`t=t^k+\\Delta t^k/2`). The tracking period will be assumed to be 1 second and the time slice 0.5 seconds.
        The controlled radar system is the EISCAT_3D radar, **comprised of 3 receiving stations and 1 transmitting station**.

        The control period is set by a radar scheduler such that:

            - :math:`t_0 = 0` seconds
            - :math:`\\Delta t_{sched} = 5` seconds

        This yields the following controller time slice arrays:
        
        >>> controls.t 
        array([[0., 1., 2., 3., 4.], [5., 6., 7., 8.]])
        >>> controls.t_slice
        array([[0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])

        Since digital beam steering allows for the simultaneous stanning of multiple points, 
        the pointing direction time arrays will be:

        >>> controls.pdirs[0]["t"]
        array([0., 0.25, 1., 1.25, 2., 2.25, 3., 3.25, 4., 4.25])
        >>> controls.pdirs[1]["t"]
        array([5., 5.25, 6., 6.25, 7., 7.25, 8., 8.25])

        The shape of the pointing direction arrays will therefore be:

        >>> controls.pdirs[0]["tx"].shape
        (1, 1, 3, 10)
        >>> controls.pdirs[1]["tx"].shape
        (1, 1, 3, 8)
        >>> controls.pdirs[0]["rx"].shape
        (3, 1, 3, 10)
        >>> controls.pdirs[1]["rx"].shape
        (3, 1, 3, 8)

        To get the pointing direction of the first scan of the 2nd RX station during the time slice t = 7 seconds, we need
        to run:

        >>> controls.get_pdirs(1)["rx"][1, 0, :, 4]
        
        And to get the second one:

        >>> controls.get_pdirs(1)["rx"][1, 0, :, 5]

        To get the pointing direction of the first scan of the only TX station during the time slice t = 3 seconds, we need
        to run:

        >>> controls.get_pdirs(0)["tx"][0, 0, :, 6]

        and to get the second:

        >>> controls.get_pdirs(0)["tx"][0, 0, :, 7]
        '''
        interpolated_states, t_dirs = args

        # compute pointing directions for each control sub time array      
        if interpolated_states[period_id] is not None:
            if self.profiler is not None:
                self.profiler.start('Tracker:generate_controls:compute_beam_orientation')
            
            # initializing results
            pointing_direction = dict()
        
            # get the position of the Tx/Rx stations
            tx_ecef = np.array([tx.ecef for tx in controls.radar.tx], dtype=np.float64) # get the position of each Tx station (ECEF frame)
            rx_ecef = np.array([rx.ecef for rx in controls.radar.rx], dtype=np.float64) # get the position of each Rx station (ECEF frame)
            
            if self.profiler is not None:
                self.profiler.start('Tracker:generate_controls:compute_beam_orientation:tx')

            # Compute Tx pointing directions
            tx_dirs = interpolated_states[period_id][None, None, :, :] - tx_ecef[:, None, :, None]
            del tx_ecef
            
            pointing_direction['tx'] = tx_dirs/np.linalg.norm(tx_dirs, axis=2)[:, :, None, :] # the beam directions are given as unit vectors in the ecef frame of reference
            del tx_dirs
            
            if self.profiler is not None:
                self.profiler.stop('Tracker:generate_controls:compute_beam_orientation:tx') 
                self.profiler.start('Tracker:generate_controls:compute_beam_orientation:rx') 
            
            # compute Rx pointing direction
            rx_dirs = interpolated_states[period_id][None, None, :, :]  - rx_ecef[:, None, :, None]     
            del rx_ecef

            rx_dirs = np.repeat(rx_dirs, len(controls.radar.tx), axis=1)
            
            # save computation results
            pointing_direction['rx'] = rx_dirs/np.linalg.norm(rx_dirs, axis=2)[:, :, None, :] # the beam directions are given as unit vectors in the ecef frame of reference
            del rx_dirs

            pointing_direction['t'] = t_dirs[period_id]

            if self.profiler is not None:
                self.profiler.stop('Tracker:generate_controls:compute_beam_orientation:rx')
                self.profiler.stop('Tracker:generate_controls:compute_beam_orientation')
        else:
            pointing_direction = dict()

            pointing_direction['rx'] = []
            pointing_direction['tx'] = []
            pointing_direction['t'] = []

        return pointing_direction


    def generate_controls(
            self, 
            t, 
            radar, 
            t_states,
            target_states, 
            t_slice=0.1, 
            states_per_slice=1,
            interpolator=interpolation.Linear,
            scheduler=None,
            priority=None, 
            max_points=100,
            beam_enabled=True,
            cache_pdirs=False,
            ):
        ''' Generates RADAR tracking controls for a given radar and sampling time and target. 
        This method can be called multiple times to generate different controls for different radar systems.
        
        Once instanciated, the the class can be used multiple times to generate different 
        static controls for different radar systems, parameters and control intervals 
        (see the :ref:`example below<tracker_controller_generate_controls_example>`).
        
        .. seealso::
            :class:`sorts.Radar<sorts.radar.system.radar.Radar>` : class encapsulating a radar system.

        Parameters
        ----------
        t : numpy.ndarray 
            Time slice start time (in seconds).
        radar : :class:`sorts.Radar<sorts.radar.system.radar.Radar>`
            Radar instance to be controlled.
        t_states : numpy.ndarray (6, M)
            Time points :math:`t_i` where the target states have been propagated (in seconds).
        target_states : numpy.ndarray (M,)
            ECEF points to be tracked by the radar system (given in the ECEF frame):

            .. math::   \\mathbf{x}^i = [x^i, y^i, z^i, v_x^i, v_y^i, v_z^i]^T

        t_slice : float / numpy.ndarray, default=0.1 
            Time slice durations. The duration of the time slice for a given control must be less than or equal 
            to the time step between two consecutive time slices.
        state_per_slice : int, default=1
            Number of target points per time slice. Numbers greated than unity can for example be used to model 
            incoherently/coherently integrated measurements.
        interpolator : interpolation.Interpolator, default=:class:`interpolation.Linear<sorts.common.interpolation.Linear>`
            Interpolation algorithm instance used to reconstruct the states of the target at each needed time point.
        scheduler : :class:`sorts.RadarSchedulerBase<sorts.radar.scheduler.base.RadarSchedulerBase>`, default=None
            RadarSchedulerBase instance used for time synchronization between control periods.
            This parameter is useful when multiple controls are sent to a given scheduler.
            If the scheduler is not provided, the control periods will be generated using the ``max_points``
            parameter.
        priority : int, default=None
            Priority of the generated controls, only used by the static priority scheduler to choose between 
            overlapping controls. 
            Low numbers indicate a high control prioriy.
        max_points : int, default=100
            Max number of points for a given control array computed simultaneously. 
            This number is used to limit the impact of computations over RAM.

            .. note::
                Lowering this number might increase computation time, while increasing this number might cause
                memory problems depending on the available RAM on your machine.

        cache_pdirs : bool, default=False
            If ``True``, the pointing directions will be computed and stored. Calling the function 
            :attr:`RadarControls.get_pdirs<sorts.radar.radar_controls.RadarControls.get_pdirs>` will return
            the pre-computed pointing directions. 
            If ``False``, the pointing directions will be computed at each :attr:`RadarControls.get_pdirs
            <sorts.radar.radar_controls.RadarControls.get_pdirs>` call.
            
            .. note::
                Enabling the option increases RAM usage, but decreases computation time. 

        Returns
        -------
        controls : :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            Radar control structure containing the list of instructions needed to perform the operation
            defined by the type of radar controller.

        Examples
        --------
        .. _tracker_controller_generate_controls_example:

        This example showcases the generation of multiple controls sequences allowing to track a 
        :class:`Space object<sorts.targets.space_object.SpaceObject>` passing over the EISCAT_3D radar system:

        .. code-block:: Python

            import numpy as np
            import matplotlib.pyplot as plt

            import sorts

            # RADAR system definition
            eiscat3d = sorts.radars.eiscat3d

            # controller and simulation parameters
            max_points = 100
            end_t = 3600*12
            t_slice = 10
            tracking_period = 20
            states_per_slice = 10

            # Propagator
            Prop_cls = sorts.propagator.Kepler
            Prop_opts = dict(
                settings = dict(
                    out_frame='ITRS',
                    in_frame='TEME',
                ),
            )
            # Object definition
            space_object = sorts.SpaceObject(
                    Prop_cls,
                    propagator_options = Prop_opts,
                    a = 7200.0e3, 
                    e = 0.1,
                    i = 80.0,
                    raan = 86.0,
                    aop = 0.0,
                    mu0 = 50.0,
                    epoch = 53005.0,
                    parameters = dict(
                        d = 0.1,
                    ),
                )
            # create state time array
            t_states = sorts.equidistant_sampling(
                orbit = space_object.state, 
                start_t = 0, 
                end_t = end_t, 
                max_dpos=50e3,
            )

            # create tracking controller
            tracker_controller = sorts.controllers.Tracker()

            # get object states/passes in ECEF frame
            object_states = space_object.get_state(t_states)
            eiscat_passes = sorts.find_simultaneous_passes(t_states, object_states, eiscat3d.tx + eiscat3d.rx)

            # plot results
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plotting station ECEF positions and earth grid
            for tx in eiscat3d.tx:
                ax.plot([tx.ecef[0]], [tx.ecef[1]], [tx.ecef[2]],'or')

            for rx in eiscat3d.rx:
                ax.plot([rx.ecef[0]], [rx.ecef[1]], [rx.ecef[2]],'og')

            sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

            # plotting object states
            ax.plot(object_states[0], object_states[1], object_states[2], "--b", alpha=0.2)

            # compute and plot controls for each pass
            for pass_id in range(np.shape(eiscat_passes)[0]):
                # get states within pass to generate tracking controls
                tracking_states = object_states[:, eiscat_passes[pass_id].inds]
                t_states_i = t_states[eiscat_passes[pass_id].inds]
                
                # generate controls
                t_controller = np.arange(t_states_i[0], t_states_i[-1] + tracking_period, tracking_period)
                controls = tracker_controller.generate_controls(t_controller, eiscat3d, t_states_i, tracking_states, t_slice=t_slice, max_points=max_points, states_per_slice=states_per_slice)
                
                # plot states being tracked
                ax.plot(tracking_states[0], tracking_states[1], tracking_states[2], "-", color="blue")
                
                # plot beam directions over each control period
                for period_id in range(controls.n_periods):
                    ctrl = controls.get_pdirs(period_id)
                    sorts.plotting.plot_beam_directions(ctrl, eiscat3d, ax=ax, tx_beam=True, rx_beam=True, zoom_level=0.9, azimuth=10, elevation=10)

            plt.show()

        .. figure:: ../../../../figures/example_tracker_controller.png
        '''
        # add new profiler entry
        if self.profiler is not None:
            self.profiler.start('Tracker:generate_controls')

        if not issubclass(interpolator, interpolation.Interpolator):
            raise TypeError(f"interpolator must be an instance of {interpolation.Interpolator}.")
        else:
            state_interpolator = interpolator(target_states, t_states)
            
            if self.logger is not None:
                self.logger.info(f"Tracker:generate_controls -> creating state interpolator {state_interpolator}")

        # output data initialization
        controls = radar_controls.RadarControls(radar, self, scheduler=scheduler, priority=priority, logger=self.logger, profiler=self.profiler)  # the controls structure is defined as a dictionnary of subcontrols
        controls.meta["interpolator"] = interpolator

        controls.set_time_slices(t, t_slice, max_points=max_points)

        # split time array into scheduler periods and target states if a scheduler is attached to the controls
        target_states_interp, t_states_interp = self.retreive_target_states(controls, t_states, state_interpolator, states_per_slice)
        
        # Compute controls
        pdir_args = (target_states_interp, t_states_interp)
        controls.set_pdirs(pdir_args, cache_pdirs=cache_pdirs)

        radar_controller.RadarController.coh_integration(controls, radar, t_slice)

        # TODO : include this in radar -> RadarController.coh_integration(self.radar, self.meta['dwell'])

        if self.profiler is not None:
            self.profiler.stop('Tracker:generate_controls')
        
        return controls