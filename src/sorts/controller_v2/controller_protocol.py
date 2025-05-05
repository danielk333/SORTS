import logging, typing as t, abc
from datetime import datetime
from .. import scheduler_v2 as schr
from ..radar.radars.composite_key import RadarStationCompositeKey

logger = logging.getLogger(__name__)


class ControllerProtocol(t.Protocol):
    @abc.abstractmethod
    def generate(
        self, stt_tstmp: datetime, end_tstmp: datetime, res_us: int
    ) -> dict[RadarStationCompositeKey, schr.Schedule]:
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


__all__ = ["ControllerProtocol"]
