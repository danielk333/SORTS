import logging, typing as t, abc
from datetime import datetime
from sorts.scheduler_v2 import Schedule

logger = logging.getLogger(__name__)


class ControllerProtocol(t.Protocol):
    # TODO: re-eval what the return type should be
    def generate(
        self, stt_tstmp: datetime, end_tstmp: datetime, res_us: int
    ) -> tuple[Schedule, Schedule]:
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
