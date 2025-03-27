from __future__ import annotations
import logging, typing as t, abc
from datetime import datetime
from .. import scheduler_v2 as schr, controller_v2 as ctrlr

logger = logging.getLogger(__name__)


class SchedulerProtocol(t.Protocol):
    controllers: tuple[ctrlr.ControllerProtocol, ...] = ()
    res_us = 1000
    "time resolution in microseconds. defaults to `1000` (1ms)"

    @abc.abstractmethod
    def generate_schedule(self, stt_tstmp: datetime, end_tstmp: datetime) -> schr.Schedule:
        """
        Parameters
        ---

        stt_tstmp
            start timestamp
        end_tstmp
            end timestamp
        """
        ...
