import logging, typing as t
from datetime import datetime

# from sorts.scheduler_v2 import Schedule

logger = logging.getLogger(__name__)


class ControllerProtocol(t.Protocol):
    # TODO: re-eval what the return type should be
    def generate(self, stt_tstmp: datetime, end_tstmp: datetime, res_us: int) -> t.Any:
        """Generate the schedule of a particular controller type"""
        ...
