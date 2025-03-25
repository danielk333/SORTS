import logging, typing as t, abc
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class ControllerProtocol(t.Protocol):
    @abc.abstractmethod
    def generate(self, stt_tstmp: datetime, end_tstmp: datetime, res_ns: int) -> pd.DataFrame:
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

        |index     |coh_int_bandwidth|pointing         |ipp    |pulse_length|
        |:-        |:-               |:-               |:-     |:-          |
        |datetime64|float64          |(float64,float64)|float64|float64     |
        """

        ...
