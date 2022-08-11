#!/usr/bin/env python

'''This module is used to define the radar controller

'''
#Python standard import
from abc import ABC, abstractmethod
import copy
import ctypes

#Third party import
import numpy as np

#Local import
from ..scheduler import RadarSchedulerBase
from sorts import clibsorts

class RadarController(ABC):
    ''' Defines the fundamental structure of a radar controller. 

    The objective of the Radar controller is to generate a sequence of instructions to be followed by the 
    :class:`Radar system<sorts.radar.system.radar.Radar>` stations (Tx/Rx) in order to accomplish a given
    objective (i.e. track an object, scan a given area, ...). The **controller type** is defined by the
    type of action performed by the radar system when the radar controls are used (scanning, tracking, ...). 
    
    The controller generates a :class:`radar controls structure <sorts.radar.radar_controls.RadarControls>` 
    which is interpretable by the :class:`Radar<sorts.radar.system.radar.Radar>` system.

    The implementation of the :class:`RadarController` class allows one controller instance to generate multiple 
    controls over different control intervals, radar systems, controller parameters, ... As such, the role of 
    the radar controller is to define how the controls will be generated according to a set of input arguments.
    
    .. seealso::
        :class:`sorts.Radar<sorts.radar.system.radar.Radar>` : class encapsulating a radar system.

    Parameters
    ----------
    profiler : :class:`sorts.Profiler<sorts.common.profing.Profiler>`, default=None
        Profiler instance used to monitor the computation performances of the class methods. 
    logger : :class:`logging.Logger`, default=None
        Logger instance used to log the computation status of the class methods.
    '''
    
    META_FIELDS = [
        'controller_type',
    ]
    ''' Radar controller metadata entries. '''

    def __init__(self, profiler=None, logger=None):
        ''' Default class constructor. '''
        self.logger = logger
        ''' Logger instance used to log the computation status of the class methods. '''
        self.profiler = profiler
        ''' Profiler instance used to monitor the computation performances of the class methods. '''

        # set controller metadata        
        self.meta = dict()
        ''' Radar controller metadata. 
        
        .. seealso::
            :attr:`RadarController.META_FIELDS` : Radar controller metadata entries.
        '''
        self.meta['controller_type'] = self.__class__

    @abstractmethod
    def generate_controls(self, t, radar, **kwargs):
        ''' Generates the radar control sequence. 

        This method is used to generate the radar control sequence needed to achieve the objective set 
        by the type of radar controller (tracking, scanning, ...). 
        
        Parameters
        ----------
        t : numpy.ndarray (N,)
            Time points at which the controls are to be generated (in seconds).
        radar : :class:`sorts.Radar<sorts.radar.system.radar.Radar>` 
            Radar instance to be controlled.
       
        Returns
        -------
        controls : :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            Radar control structure containing the list of instructions needed to perform the operation
            defined by the type of radar controller.

        .. seealso::
            :class:`sorts.Radar<sorts.radar.system.radar.Radar>` : class encapsulating a radar system.
        '''
        pass


    def compute_pointing_directions(
        self, 
        controls,
        period_id, 
        args):
        ''' Computes the radar pointing directions for each time slice over a given control
        period.

        The station pointing direction corresponds to the normalized Poynting vector of the transmitted 
        signal (i.e. its direction of propagation / direction of the beam). It is possible to have multiple 
        pointing direction vectors for a sinle station per time slice. This feature can therefore be used 
        to model digital beam steering of phased array receiver antennas (see radar_eiscat3d).

        This function will be called by the 
        :attr:`RadarControls.get_pdirs<sorts.radar.radar_controls.RadarControls.get_pdirs>` 
        in order to generate the pointing direction controls over each control period. Therefore, 
        this function is usually called after the generation of station property controls is completed. 
    
        .. note::
            The choice of allowing the user to compute separatly the pointing directions from the property controls
            can be justified by the fact that pointing direction arrays can easily reach millions of values and computing
            them all at once can cause RAM problems.

        Parameters
        ----------
        controls : :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            Radar controls instance generated by the :attr:`RadarController.generate_controls` method.
        period_id : int
            Control period index.
        args : variable or list of variables
            Variable or list of aditional variables needed by the custom implementation of the 
            :attr:`RadarController.compute_pointing_directions` method.
        
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
                    Index of the :class:`sorts.TX<sorts.radar.system.station.TX>` station within the :attr:`Radar.tx<sorts.
                    radar.system.radar.Radar.tx>` list.
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
                    Index of the :class:`sorts.RX<sorts.radar.system.station.RX>` station within the :attr:`Radar.rx<sorts.
                    radar.system.radar.Radar.rx>` list.
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
        
        .. seealso::
            :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>` : class encapsulating a radar control sequence.
            :attr:`sorts.RadarControls.get_pdirs<sorts.radar.radar_controls.RadarControls.get_pdirs>` : function used to compute the pointing directions associated with a given control sequence. 

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
        raise Exception("No pointing directions generation function defined.")


    @staticmethod
    def coh_integration(controls, radar, dwell):
        ''' Sets the coherent integration settings based on the time slice duration (dwell time). '''
        dwell = np.atleast_1d(dwell)

        for tx in radar.tx:
            controls.add_property_control("n_ipp", tx, (dwell/tx.ipp).astype(int))
            controls.add_property_control("coh_int_bandwidth", tx, 1.0/(tx.pulse_length*tx.n_ipp))