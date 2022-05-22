#!/usr/bin/env python

'''#TODO

'''

import numpy as np

from .radar_controller import RadarController
from ..scans import Beampark


class Static(RadarController):
    '''
    Purpose 
    -------
    
    This class can is used to create RADAR static controls. Once instanciated, the the class can be used multiple times to generate different static controls for different radar systems.
    
    Examples
    ----------
    
    :numpy.ndarray t: Time points at which the controls are to be generated [s]
    :radar.system.Radar radar: Radar instance to be controlled
    :float azimuth: Azimuth of the target beam
    :float azimuth: Elevation of the target beam
    :scans.Scan scan: Scan instance used to generate the scanning controls
    :float/numpy.ndarray dwell: Dwell time of the scan. The dwell time shall be smaller than the controller's time step.
    :numpy.ndarray r (optional): Array of ranges from the transmitter where the receivers need to target simultaneously at a given time t [m]

    
    Return value
    ----------
    
    Dictionnary containing the controls to be applied to the radar to perform the required scanning scheme. In the case of the Scanner controller, the controls are the following.

    - "t"
        1D array of time points at which the controls need to be executed by the radar
    
    - "dwell"
        1D array of the scanning dwell times
    
    - "on_state"
        1D array of on/off states of the radar. If the value for a given index k is 0, then the radar is "off" at time t[k], if instead the value is 1, then the radar is "on" at time t[k],
    
    - "beam_direction_tx"
        Array of unit vectors representing the target direction of the beam for a given Tx station at time t. 
        To ensure that this controls can be understood by the radar and the scheduler, the orientation data is given in the following standard:
        - Dimension 0 : 
            Tx station index. In the case where there is only one Tx station considered, the value of this index will be 0.
       
        - Dimension 1 : 
            position index (0, 1, 2) = (x, y, z)
        
        - Dimension 2 : 
            time index
    
    - "beam_direction_rx"
        Array of unit vectors representing the target direction of the beam for a given Rx station at time t. 
        To ensure that this controls can be understood by the radar and the scheduler, the orientation data is given in the following standard:
        - Dimension 0 : 
            Rx station index. in the case where there is only one Rx station considered, the value of this index will be 0.
        
        - Dimension 1 : 
            Associated Tx station index. Since the Rx stations must target the same points as the Tx stations, each Rx station gets at least as many beam orientation controls as there are Tx stations. In the case where there is only one Tx station considered, the value of the index will be 0.
       
        - Dimension 2 : 
            Range index : one Rx station can simultaneously target the same Tx beam at multiple ranges (from the Tx station), those ranges are given when calling the generate_controls method by setting the argument r.
       
        - Dimension 3 : 
            position index (0, 1, 2) = (x, y, z)
        
        - Dimension 4 : 
            time index 
         
    Examples 
    --------
    
    Suppose that there is a sinle Tx station performing a scan at 100 different time points. The shape of the output "beam_direction_tx" array will be (1, 3, 100). 
    
    - To get the z component of the beam direction of the Tx station at the second time step, one must call : 
        
        >>> ctrl = controls["beam_direction_tx"][0, 2, 2]
        
    - To get the x component of the beam direction of the Tx station at the 35th time step, one must call :
    
        >>> ctrl = controls["beam_direction_tx"][0, 0, 34]
    

    Suppose that there is a sinle Tx station performing a scan at 100 different time points, that there are 2 Rx stations performing simultaneous scans at 10 different ranges. The shape of the output "beam_direction_rx" array will be (2, 1, 10, 3, 100). 
    
    - To get the z component of the beam direction of the first Rx station with respect to the second simultaneous scan at range r of the only Tx beam at the second time step, one must call : 
         
         >>> ctrl = controls["beam_direction_rx"][0, 0, 1, 2, 2]
     
    - To get the y component of the beam direction of the second Rx station with respect to the 5th simultaneous scan at range r of the only Tx beam at the 80th time step, one must call 
         
        >>> ctrl = controls["beam_direction_rx"][1, 0, 4, 1, 79]
        
      '''

    META_FIELDS = RadarController.META_FIELDS + [
        'scan_type',
    ]

    def __init__(self, profiler=None, logger=None, **kwargs):
        super().__init__(profiler=profiler, logger=logger, **kwargs)
        
        self.meta['scan_type'] = self.__class__
        
        if self.logger is not None:
            self.logger.info(f'Static:init')

    def generate_controls(self, t, radar, azimuth=0.0, elevation=90.0, dwell=0.1, r=np.linspace(300e3,1000e3,num=10), priority=-1):
        '''
        Purpose 
        -------
        
        Generate RADAR static controls for a given radar and sampling time. This method can be called multiple times to generate different controls for different radar systems.
        
        Parameters
        ----------
        
        :numpy.ndarray t: Time points at which the controls are to be generated [s]
        :radar.system.Radar radar: Radar instance to be controlled
        :float azimuth: Azimuth of the target beam
        :float azimuth: Elevation of the target beam
        :scans.Scan scan: Scan instance used to generate the scanning controls
        :float/numpy.ndarray dwell: Dwell time of the scan. The dwell time shall be smaller than the controller's time step.
        :numpy.ndarray r (optional): Array of ranges from the transmitter where the receivers need to target simultaneously at a given time t [m]
        :int priority (optional): Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority scheduler algorithms.
        
        Return value
        ----------
        
        Dictionnary containing the controls to be applied to the radar to perform the required scanning scheme. In the case of the Scanner controller, the controls are the following.
    
        - "t"
            1D array of time points at which the controls need to be executed by the radar
        
        - "dwell"
            1D array of the scanning dwell times
        
        - "on_state"
            1D array of on/off states of the radar. If the value for a given index k is 0, then the radar is "off" at time t[k], if instead the value is 1, then the radar is "on" at time t[k],
        
        - "beam_direction_tx"
            Array of unit vectors representing the target direction of the beam for a given Tx station at time t. 
            To ensure that this controls can be understood by the radar and the scheduler, the orientation data is given in the following standard:
            - Dimension 0 : 
                Tx station index. In the case where there is only one Tx station considered, the value of this index will be 0.
           
            - Dimension 1 : 
                position index (0, 1, 2) = (x, y, z)
            
            - Dimension 2 : 
                time index
        
        - "beam_direction_rx"
            Array of unit vectors representing the target direction of the beam for a given Rx station at time t. 
            To ensure that this controls can be understood by the radar and the scheduler, the orientation data is given in the following standard:
            - Dimension 0 : 
                Rx station index. in the case where there is only one Rx station considered, the value of this index will be 0.
            
            - Dimension 1 : 
                Associated Tx station index. Since the Rx stations must target the same points as the Tx stations, each Rx station gets at least as many beam orientation controls as there are Tx stations. In the case where there is only one Tx station considered, the value of the index will be 0.
           
            - Dimension 2 : 
                Range index : one Rx station can simultaneously target the same Tx beam at multiple ranges (from the Tx station), those ranges are given when calling the generate_controls method by setting the argument r.
           
            - Dimension 3 : 
                position index (0, 1, 2) = (x, y, z)
            
            - Dimension 4 : 
                time index 
                
        - "priority"
            Priority of the generated controls, only used by the scheduler to choose between overlapping controls. Low numbers indicate a high control prioriy. -1 is used for dynamic priority scheduler algorithms.
                    
             
        Examples 
        --------
        
        Suppose that there is a sinle Tx station performing a scan at 100 different time points. The shape of the output "beam_direction_tx" array will be (1, 3, 100). 
        
        - To get the z component of the beam direction of the Tx station at the second time step, one must call : 
            
            >>> ctrl = controls["beam_direction_tx"][0, 2, 2]
            
        - To get the x component of the beam direction of the Tx station at the 35th time step, one must call :
        
            >>> ctrl = controls["beam_direction_tx"][0, 0, 34]
        

        Suppose that there is a sinle Tx station performing a scan at 100 different time points, that there are 2 Rx stations performing simultaneous scans at 10 different ranges. The shape of the output "beam_direction_rx" array will be (2, 1, 10, 3, 100). 
        
        - To get the z component of the beam direction of the first Rx station with respect to the second simultaneous scan at range r of the only Tx beam at the second time step, one must call : 
             
             >>> ctrl = controls["beam_direction_rx"][0, 0, 1, 2, 2]
         
        - To get the y component of the beam direction of the second Rx station with respect to the 5th simultaneous scan at range r of the only Tx beam at the 80th time step, one must call 
             
            >>> ctrl = controls["beam_direction_rx"][1, 0, 4, 1, 79]
            
          '''
        # checks input values to make sure they are compatible with the implementation of the function
        if not isinstance(priority, int): raise TypeError("the priority must be an integer.")
        else:
            if priority < -1: raise ValueError("the priority must be positive [0; +inf] or equal to -1.")
            
        if not isinstance(radar, Radar): raise TypeError(f"the radar must be an instance of {Radar}.")
        
        # add support for both arrays and floats
        t = np.asarray(t)
        if len(np.shape(t)) > 1: raise TypeError("t must be a 1-dimensional array or a float")
        
        # add new profiler entry
        if self.profiler is not None:
            self.profiler.start('Static:generate_controls')
        
        # generate the static beam with the required characteristics
        scan = Beampark(azimuth = azimuth, elevation=elevation, dwell=dwell)

        # generate controls structure
        controls = dict()  # the controls structure is defined as a dictionnary of subcontrols
        
        controls["t"] = t.copy() # save the time points of the controls
        controls["dwell"] = np.ones(np.size(t))*dwell # save the dwell time of each time point
        controls["on_state"] = np.ones(np.size(t)) # save on/off state of the radar for each time point (we assume that the radar is fully on at each control step)
        controls["priority"] = priority # set the controls priority
        
        # get the coordinates of the Tx target points 
        t = t*0.0 # set the time array to 0
        points = scan.ecef_pointing(t, radar.tx)
        
        # get station positions
        tx_ecef = np.array([tx.ecef for tx in radar.tx], dtype=float).reshape((len(radar.tx), 3, 1)) # get the position of each Tx station (ECEF frame)
        rx_ecef = np.array([rx.ecef for rx in radar.rx], dtype=float) # get the position of each Rx station (ECEF frame)
        
        if self.profiler is not None:
            self.profiler.start('Static:generate_controls:compute_tx_beam_directions')

        # compute Tx pointing direction
        point_tx = points + tx_ecef
        point_rx_to_tx = points[:, None, :, :]*r[None, :, None, None] + tx_ecef
        del points

        tx_dirs = point_tx - tx_ecef
        controls['beam_direction_tx'] = tx_dirs/np.linalg.norm(tx_dirs, axis=1) # the beam directions are given as unit vectors in the ecef frame of reference
        del tx_dirs
        
        if self.profiler is not None:
            self.profiler.stop('Static:generate_controls:compute_tx_beam_directions') 
            self.profiler.start('Static:generate_controls:compute_rx_beam_directions')
        
        # get Rx pointing directions        
        point_rx = np.repeat(point_rx_to_tx[None, :], len(radar.rx), axis=0)
        
        # compute directions for stations where tx and rx < 200 meters apart => same location for pointing
        rx_close_to_tx = np.linalg.norm(tx_ecef - rx_ecef.transpose(), axis=1) < 200.0
        inds_rx_close_to_tx = np.array(np.where(rx_close_to_tx)) # [txinds, rxinds]
        del rx_close_to_tx
        
        point_rx[inds_rx_close_to_tx[1], :, :, :] = point_tx[None, None, inds_rx_close_to_tx[0], :, :]
        del inds_rx_close_to_tx
        
        rx_dirs = point_rx - rx_ecef[:, None, None, :, None]
        controls['beam_direction_rx'] = rx_dirs/np.linalg.norm(rx_dirs, axis=3)[:, :, :, None] # the beam directions are given as unit vectors in the ecef frame of reference

        # TODO : include this in radar -> RadarController.coh_integration(self.radar, self.meta['dwell'])

        if self.profiler is not None:
            self.profiler.stop('Static:generate_controls')
            
        return controls