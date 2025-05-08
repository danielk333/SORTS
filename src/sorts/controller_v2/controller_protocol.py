import logging, typing as t, abc
from datetime import datetime
from sorts.schedule_v2 import Schedule
from sorts.radar.radars.composite_key import RadarStationCompositeKey

logger = logging.getLogger(__name__)


class ControllerProtocol(t.Protocol):
    @abc.abstractmethod
    def generate(
        self, stt_tstmp: datetime, end_tstmp: datetime, res_us: int
    ) -> dict[RadarStationCompositeKey, Schedule]:
        """
        Parameters
        ---

        stt_tstmp
            start timestamp
        end_tstmp
            end timestamp
        res_us
            resolution in microseconds

        TODO: do we still need `res_us`? should we renamed it to `alignment_us`?
        """
        ...


__all__ = ["ControllerProtocol"]
