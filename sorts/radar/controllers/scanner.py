#!/usr/bin/env python

'''#TODO

'''

import numpy as np

from .radar_controller import RadarController


class Scanner(RadarController):
    '''
        This class is used to create scanning controls for a given radar system.
        To create a control, one instanciates the scanner class by giving a profiler and logger depending on the application. 
        Then, calling the function generate_controls() with the radar to be controlled and the scan algorithm to be executed will return the corresponding array of controls.
        
        Only one Scanner controller is needed to create multiple controls for multiple scan methods and radars.
    '''

    META_FIELDS = RadarController.META_FIELDS + [
        'scan_type',
        'dwell',
    ]

    def __init__(
            self, 
            profiler=None, 
            logger=None, 
            **kwargs
            ):
        '''
            Creates a Scanner controller to generate radar scanning controls. Only one Scanner controller is needed to create multiple controls for differnet scan methods and radars.
            
            :param Profiler profiler: Profiler instance for checking function performance.
            :param logging.Logger logger: Logger instance for logging the execution of the function.
            
            :rtype: controllers.Scanner
            :return: Scanner : Scanner used to generate scanning controls
        '''
        super().__init__(profiler=profiler, logger=logger, **kwargs)

        if self.logger is not None:
            self.logger.info('Scanner:init')
        
    def generate_controls(
            self, 
            t, 
            radar, 
            scan,
            r=np.linspace(300e3, 1000e3, num=10), 
            as_altitude=False, 
            ):
        '''
            Generate RADAR controls for a given radar, scanning algorithm and sampling time. This method can be called multiple times to generate different controls for different radar systems.
                            
            :param numpy.ndarray t: Time points at which the controls are to be generated [s]
            :param radar.system.Radar radar: Radar instance to be controlled
            :param scans.Scan scan: Scan instance used to generate the scanning controls
            :param numpy.ndarray r (optional): Array of ranges from the transmitter where the receivers need to target simultaneously at a given time t [m]
            :param boolean as_altitude (optional): True if r is given as the altitude 
            
            :rtype: controllers.Scanner
            :return: :dict() controls: Dictionnary containing the controls to be applied to the radar to perform the required scanning scheme. In the case of the Scanner controller, the controls are the following :
                - "t" : 1D array of time points at which the controls need to be executed by the radar
                - "dwell" : 1D array of the scanning dwell times
                - "on_state" : 1D array of on/off states of the radar. If the value for a given index k is 0, then the radar is "off" at time t[k], if instead the value is 1, then the radar is "on" at time t[k],
                
                - "beam_direction_tx" : Array of unit vectors representing the target direction of the beam for a given Tx station at time t. To ensure that this controls can be understood by the radar and the scheduler, the orientation data is given in the following standard:
                    - Dimension 0 : Tx station index. In the case where there is only one Tx station considered, the value of this index will be 0.
                    - Dimension 1 : position index (0, 1, 2) = (x, y, z)
                    - Dimension 2 : time index
                    
                  examples : Suppose that there is a sinle Tx station performing a scan at 100 different time points. The shape of the output "beam_direction_tx" array will be (1, 3, 100). 
                      - To get the z component of the beam direction of the Tx station at the second time step, one must call controls["beam_direction_tx"][0, 2, 2]
                      - To get the x component of the beam direction of the Tx station at the 35th time step, one must call controls["beam_direction_tx"][0, 0, 34]
                
                - "beam_direction_rx" : Array of unit vectors representing the target direction of the beam for a given Rx station at time t. To ensure that this controls can be understood by the radar and the scheduler, the orientation data is given in the following standard:
                    - Dimension 0 : Rx station index. in the case where there is only one Rx station considered, the value of this index will be 0.
                    - Dimension 1 : Associated Tx station index. Since the Rx stations must target the same points as the Tx stations, each Rx station gets at least as many beam orientation controls as there are Tx stations. In the case where there is only one Tx station considered, the value of the index will be 0.
                    - Dimension 2 : Range index : one Rx station can simultaneously target the same Tx beam at multiple ranges (from the Tx station), those ranges are given when calling the generate_controls method by setting the argument r.
                    - Dimension 3 : position index (0, 1, 2) = (x, y, z)
                    - Dimension 4 : time index
                    
                  examples : Suppose that there is a sinle Tx station performing a scan at 100 different time points, that there are 2 Rx stations performing simultaneous scans at 10 different ranges. The shape of the output "beam_direction_rx" array will be (2, 1, 10, 3, 100). 
                      - To get the z component of the beam direction of the first Rx station with respect to the second simultaneous scan at range r of the only Tx beam at the second time step, one must call controls["beam_direction_rx"][0, 0, 1, 2, 2]
                      - To get the y component of the beam direction of the second Rx station with respect to the 5th simultaneous scan at range r of the only Tx beam at the 80th time step, one must call controls["beam_direction_rx"][1, 0, 4, 1, 79]
        '''
        if self.profiler is not None:
            self.profiler.start('Scanner:generate_controls')
            
        # add support for both arrays and floats
        t = np.asarray(t)
        N = np.size(t)
        
        # controls array initialization
        controls = dict()
        
        controls['t'] = t
        controls['dwell'] = scan.dwell(t)
        controls['on_state'] = np.ones([N])
        
        if self.profiler is not None:
            self.profiler.start('Scanner:generate_controls:compute_tx_beam_directions')
        
        # get Tx pointing directions
        points = scan.ecef_pointing(t, radar.tx) # [ind_tx][x, y, z][t]
        tx_ecef = np.array([tx.ecef for tx in radar.tx]).reshape((len(radar.tx), 3, 1))
        
        # get Tx pointing directions and target points for Rx
        if as_altitude:
            r = r/points[:, 2, :]
            
        point_tx = points + tx_ecef
        point_rx_to_tx = points[:, None, :, :]*r[None, :, None, None] + tx_ecef
        del points
            
        # Compute Tx pointing directions
        tx_dirs = point_tx - tx_ecef
        controls['beam_direction_tx'] = tx_dirs/np.linalg.norm(tx_dirs, axis=1)
        del tx_dirs
        
        if self.profiler is not None:
            self.profiler.stop('Scanner:generate_controls:compute_tx_beam_directions') 
            self.profiler.start('Scanner:generate_controls:compute_rx_beam_directions')
        
        # get Rx pointing directions        
        rx_ecef = np.array([rx.ecef for rx in radar.rx], dtype=float)
        point_rx = np.repeat(point_rx_to_tx[None, :], len(radar.rx), axis=0)
        
        # compute directions for stations where tx and rx < 200 meters apart => same location for pointing
        rx_close_to_tx = np.linalg.norm(tx_ecef - rx_ecef.transpose(), axis=1) < 200.0
        inds_rx_close_to_tx = np.array(np.where(rx_close_to_tx)) # [txinds, rxinds]
        del rx_close_to_tx
        
        point_rx[inds_rx_close_to_tx[1], :, :, :] = point_tx[None, None, inds_rx_close_to_tx[0], :, :]
        del inds_rx_close_to_tx
        
        rx_dirs = point_rx - rx_ecef[:, None, None, :, None]
        controls['beam_direction_rx'] = rx_dirs/np.linalg.norm(rx_dirs, axis=3)[:, :, :, None]
        
        if self.profiler is not None:
            self.profiler.stop('Scanner:generate_controls:compute_rx_beam_directions')
            self.profiler.stop('Scanner:generate_controls')
    
        return controls
