from abc import ABC, abstractmethod
import numpy as np

class RadarControlManagerBase(ABC):
    def __init__(self, radar, t0, manager_period, logger=None, profiler=None):
        self.profiler = profiler
        self.logger = logger

        self.radar = radar
        self._t0 = t0
        self._manager_period = manager_period

        if self.logger is not None:
            self.logger.info(f"RadarControlManagerBase:init -> setting scheduling start time t0={t0}")

        if self._manager_period is not None:
            self.logger.info(f"RadarControlManagerBase:init -> setting scheduling period : manager_period={manager_period}")        
        else:
            self.logger.info("RadarControlManagerBase:init -> ignoring scheduling period...")   

    @abstractmethod
    def run(self, controls):
        '''Runs the control manager algorithm to obtain the final RADAR control sequence sent to the RADAR.

        Parameters
        ----------

        controls : np.ndarray/list
        Array of RADAR controls to be managed. The algorithm will arrange the time slices from those controls to create a control sequence compatible with the RADAR system.

        Return value
        ------------

        final_control_sequence : dict
        Final RADAR control sequence compatible with the RADAR system

        '''
        pass

    @property
    def manager_period(self):
        return self._manager_period

    @manager_period.setter
    def manager_period(self, val):
        try:
            val = float(val)
        except:
            raise ValueError("The manager period has to be a number (int/float)")

        self._manager_period = val

        if self.logger is not None:
            self.logger.info(f"RadarControlManagerBase:manager_period:setter -> setting scheduling period : manager_period={val}")        

    @property
    def t0(self):
        return self._t0
    
    @t0.setter
    def t0(self, val):
        try:
            val = float(val)
        except:
            raise ValueError("The manager start time has to be a number (int/float)")
        
        self._t0 = val
    
        if self.logger is not None:
            self.logger.info(f"RadarControlManagerBase:t0:setter -> setting scheduling start time : t0={val}")     