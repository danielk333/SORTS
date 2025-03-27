import logging, typing as t, abc
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class ControllerProtocol(t.Protocol):
    @abc.abstractmethod
    def generate(self, stt_tstmp: datetime, end_tstmp: datetime, res_us: int) -> pd.DataFrame:
        """
        Parameters
        ---

        stt_tstmp
            start timestamp
        end_time
            end timestamp
        res_us
            resolution in microseconds
        """
        ...
