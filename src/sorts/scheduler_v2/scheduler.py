import logging, typing as t, enum
import numpy as np
from . import schedule_metadata as schmeta

logger = logging.getLogger(__name__)


class Scheduler:
    """
    A Scheduler for executing time-slices of different radar controllers.

    v2, wip
    """

    def __init__(self):
        pass

    def generate_schedule(self):
        """Takes times and a corresponding generator that returns radar instances to generate a radar schedule."""

        logger.info(f"ok")
        print("ahh")
