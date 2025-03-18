import logging, typing as t, abc
from datetime import datetime
import pandas as pd
from .. import controller_v2 as ctrlr

logger = logging.getLogger(__name__)


class SchedulerProtocol(t.Protocol):
    controllers: tuple[ctrlr.ControllerProtocol, ...] = ()
    res_ns = int(1e6)
    "time resolution in nanoseconds. defaults to `1e6` (1ms)"

    @abc.abstractmethod
    def generate_schedule(self, stt_tstmp: datetime, end_tstmp: datetime) -> pd.DataFrame:
        """
        Parameters
        ---

        stt_tstmp
            start timestamp
        end_time
            end timestamp

        Returns
        ---
        a DataFrame with these columns:

        |index     |coordinate_system|coh_int_bandwidth|pointing                 |ipp    |pulse_length|
        |:-        |:-               |:-               |:-                       |:-     |:-          |
        |datetime64|int8             |float64          |(float64,float64,float64)|float64|float64     |
        """
        ...
