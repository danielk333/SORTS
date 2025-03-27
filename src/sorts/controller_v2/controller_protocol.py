import logging, typing as t, abc
from datetime import datetime
from .. import scheduler_v2 as schr

logger = logging.getLogger(__name__)


class ControllerProtocol(t.Protocol):
    @abc.abstractmethod
    def generate(self, stt_tstmp: datetime, end_tstmp: datetime, res_us: int) -> schr.Schedule:
        """
        Parameters
        ---

        stt_tstmp
            start timestamp
        end_tstmp
            end timestamp
        res_us
            resolution in microseconds
        """
        ...
