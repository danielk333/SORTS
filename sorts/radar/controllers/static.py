#!/usr/bin/env python

'''#TODO

'''

import numpy as np

from . import radar_controller
from ..scans import Beampark
from ..system import Radar
from .. import radar_controls


class Static(radar_controller.RadarController):
    ''' Generates static radar controls. 
    
    The :class:`Static` class is used to create RADAR static controls (also called Beampark experiement). 
    Static radar controls point the beam of the transmitter stations in a fixed direction 
    with respect to the local reference frame of the station. 

    Beampark experiments are widely used in the field of space awareness and monitoring because they
    allow for the rather simple detection and orbit determination of many space objects. 

    If beam steering is allowed, receivers can then perform multiple scans of
    the TX beam at multiple ranges from the TX stations per time slice
    (see the :ref:`example below<static_controller_example>`). 

    Once instanciated, the class can be used multiple times to generate different 
    static controls for different radar systems, parameters and control intervals.

    .. seealso::

        * :class:`sorts.Radar<sorts.radar.system.radar.Radar>` : class encapsulating the radar system.
        * :class:`sorts.RadarController<sorts.radar.controllers.radar_controller.RadarController>` : radar controller base class.
        * :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>` : class encapsulating radar control sequences.
        * :class:`sorts.Beampark<sorts.radar.scans.bp.Beampark>` : Beampark scanning scheme.

    Parameters
    ----------
    profiler : :class:`sorts.Profiler<sorts.common.profing.Profiler>`, default=None
        Profiler instance used to monitor the computation performances of the class methods. 
    logger : :class:`logging.Logger`, default=None
        Logger instance used to log the computation status of the class methods.
    
    Examples
    --------
    .. _static_controller_example:

    This short example showcases the generation of radar static controls using the :class:`Static` radar controller
    class:

    .. code-block:: Python

        import numpy as np
        import matplotlib.pyplot as plt

        import sorts

        # RADAR definition
        eiscat3d = sorts.radars.eiscat3d

        # Controller parameters
        end_t = 100.0
        t_slice = 0.1
        max_points = 100

        # create scheduler and controller
        static_controller = sorts.Static()

        t = np.arange(0.0, end_t, t_slice)
        controls = static_controller.generate_controls(t, eiscat3d, t_slice=t_slice, max_points=max_points)

        # plot the first control period
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plotting station ECEF positions and earth grid
        sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)
        for tx in eiscat3d.tx:
            ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')

        for rx in eiscat3d.rx:
            ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

        # plot pointing directions
        period_id = 0 # get results over 1st control period
        ctrl = controls.get_pdirs(period_id)
        ax = sorts.plotting.plot_beam_directions(ctrl, eiscat3d, ax=ax, zoom_level=0.95, azimuth=10, elevation=10)
            
        plt.show()

    .. figure:: ../../../../figures/example_static_controller.png

    '''

    META_FIELDS = radar_controller.RadarController.META_FIELDS + []

    def __init__(self, profiler=None, logger=None, **kwargs):
        super().__init__(profiler=profiler, logger=logger, **kwargs)
                
        if self.logger is not None:
            self.logger.info('Static:init')
   
    def compute_pointing_directions(
            self, 
            controls,
            period_id, 
            args, 
        ):
        ''' Computes the radar pointing directions for each time slice over a given control
        period.

        This function computes the pointing directions associated with the static :class:`Beampark<sorts.radar.scans.bp.Beampark>` 
        scanning scheme.

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
        args : variable or list of variables
            Distance (or array of distances if beam-steering is used) from the transmitter stations which will be 
            scanned by the receiver stations. 

            >>> r = args 
        
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
        Consider a control sequence ``controls`` generated by a radar controller performing a scan over 9 time slices. The
        controller uses digital beam steering over the stations such that during each time slice, the receiver stations are 
        able to target two points of the radar beam. The controlled radar system is the EISCAT_3D radar, **comprised of 3 
        receiving stations and 1 transmitting station**.

        The control period is set by a radar scheduler such that:

            - :math:`t_0 = 0` seconds
            - :math:`\\Delta t_{sched} = 5` seconds

        This yields the following controller time slice arrays:
        
        >>> controls.t 
        array([[0., 1., 2., 3., 4.], [5., 6., 7., 8.]])
        >>> controls.t_slice
        array([[0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])

        Since digital beam steering allows for the simultaneous stanning of multiple points, the pointing direction time arrays
        will be:

        >>> controls.pdirs[0]["t"]
        array([0., 0., 1., 1., 2., 2., 3., 3., 4., 4.])
        >>> controls.pdirs[1]["t"]
        array([5., 5., 6., 6., 7., 7., 8., 8.])

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
        r = args

        if self.profiler is not None:
            self.profiler.start('Scanner:pointing_direction:compute_controls_subarray:tx') 
    
        # initializing results
        pointing_direction = dict()

        # get the position of the Tx/Rx stations
        tx_ecef = np.array([tx.ecef for tx in controls.radar.tx], dtype=np.float64) # get the position of each Tx station (ECEF frame)
        rx_ecef = np.array([rx.ecef for rx in controls.radar.rx], dtype=np.float64) # get the position of each Rx station (ECEF frame)
        
        # get Tx pointing directions
        # [ind_tx][x, y, z][t]
        points = controls.meta["scan"].ecef_pointing(controls.t[period_id], controls.radar.tx).reshape((len(tx_ecef), 3, -1))
 
        # Compute Tx pointing directions
        pointing_direction['tx'] = np.repeat(points[:, None, :, :], len(r), axis=3) # the beam directions are given as unit vectors in the ecef frame of reference
        
        if self.profiler is not None:
            self.profiler.stop('Scanner:pointing_direction:compute_controls_subarray:tx') 
            self.profiler.start('Scanner:pointing_direction:compute_controls_subarray:rx') 
        
        # get Rx target points on the Tx beam
        point_rx_to_tx = np.repeat(points[:, None, :, :], len(r), axis=3)*np.tile(r, len(points[0, 0]))[None, None, None, :] + tx_ecef[:, None, :, None] # compute the target points for the Rx stations
        del tx_ecef, points
        point_rx = np.repeat(point_rx_to_tx, len(controls.radar.rx), axis=0) 
        del point_rx_to_tx
        
        # compute actual pointing direction
        rx_dirs = point_rx - rx_ecef[:, None, :, None]
        del point_rx, rx_ecef
    
        # save computation results

        pointing_direction['rx'] = rx_dirs/np.linalg.norm(rx_dirs, axis=2)[:, :, None, :] # the beam directions are given as unit vectors in the ecef frame of reference
        pointing_direction['t'] = np.repeat(controls.t[period_id], len(r))

        if self.profiler is not None:
            self.profiler.stop('Scanner:pointing_direction:compute_controls_subarray:rx')
            
        return pointing_direction


    def generate_controls(
            self, 
            t, 
            radar, 
            azimuth=0.0, 
            elevation=90.0, 
            t_slice=0.1, 
            r=np.linspace(300e3,1000e3,num=10), 
            scheduler=None,
            priority=None, 
            max_points=100,
            cache_pdirs=False,
            ):
        ''' Generates RADAR static controls in a given direction. 

        This method is used to create RADAR static controls. Static radar controls 
        point the beam of the transmitter stations in a fixed direction with respect 
        to the local reference frame of the station. 

        If beam steering is allowed, receivers can then perform multiple scans of
        the TX beam at multiple ranges from the TX stations (see the :ref:`example 
        below<static_controller_generate_controls_example>`). 

        Once instanciated, the the class can be used multiple times to generate different 
        static controls for different radar systems, parameters and control intervals.
        
        .. seealso::
            :class:`sorts.Radar<sorts.radar.system.radar.Radar>` : class encapsulating a radar system.

        Parameters
        ----------
        t : numpy.ndarray (N,)
            Time points at which the controls are to be generated (in seconds).
        radar : :class:`sorts.Radar<sorts.radar.system.radar.Radar>`
            Radar instance to be controlled.
        azimuth : float, default=0.0
            Azimuth of the target beam (in degrees).
        elevation : float, default=90.0
            Elevation of the target beam (in degrees).
        t_slice : float / numpy.ndarray (N,), default=0.1
            Array of time slice durations (in seconds). 
            The duration of the time slice must be less than or equal to the time step between two 
            consecutive time slice starting points.
        r : float / numpy.ndarray (M,), default=np.linspace(300e3,1000e3,num=10)
            Array of ranges from the transmitter beam at which the receiver will perform scans during a 
            given time slice (in meters).
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
        This short example showcases the generation of radar static controls using the :class:`Static` radar controller
        class:

        .. _static_controller_generate_controls_example:

        .. code-block:: Python

            import numpy as np
            import matplotlib.pyplot as plt

            import sorts

            # RADAR definition
            eiscat3d = sorts.radars.eiscat3d

            # Controller parameters
            end_t = 100.0
            t_slice = 0.1
            max_points = 100

            # create scheduler and controller
            static_controller = sorts.Static()

            t = np.arange(0.0, end_t, t_slice)
            controls = static_controller.generate_controls(t, eiscat3d, t_slice=t_slice, max_points=max_points)

            # plot the first control period
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plotting station ECEF positions and earth grid
            sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)
            for tx in eiscat3d.tx:
                ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')

            for rx in eiscat3d.rx:
                ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

            # plot pointing directions
            period_id = 0 # get results over 1st control period
            ctrl = controls.get_pdirs(period_id)
            ax = sorts.plotting.plot_beam_directions(ctrl, eiscat3d, ax=ax, zoom_level=0.95, azimuth=10, elevation=10)
                
            plt.show()

        .. figure:: ../../../../figures/example_static_controller.png

        '''
        # add new profiler entry
        if self.profiler is not None:
            self.profiler.start('Static:generate_controls')

        # controls computation initialization
        scan = Beampark(azimuth=azimuth, elevation=elevation, dwell=t_slice)
        
        # output data initialization
        controls = radar_controls.RadarControls(radar, self, scheduler=scheduler, priority=priority) 
        controls.meta["scan"] = scan

        controls.set_time_slices(t, t_slice, max_points=max_points)

        # compute controls
        pdir_args = r # set ``compute_pointing_directions`` args
        controls.set_pdirs(pdir_args, cache_pdirs=cache_pdirs)

        radar_controller.RadarController.coh_integration(controls, radar, scan.dwell())

        if self.profiler is not None:
            self.profiler.stop('Static:generate_controls')
        
        return controls