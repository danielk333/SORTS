from __future__ import annotations
import logging, typing as t, abc
from dataclasses import dataclass, field, fields
from datetime import datetime
import numpy as np
import numpy.typing as npt
from .. import controller_v2 as ctrlr

logger = logging.getLogger(__name__)


class SchedulerProtocol(t.Protocol):
    controllers: tuple[ctrlr.ControllerProtocol, ...] = ()
    res_us = 1000
    "time resolution in microseconds. defaults to `1000` (1ms)"

    @abc.abstractmethod
    def generate_schedule(self, stt_tstmp: datetime, end_tstmp: datetime) -> Schedule:
        """
        Parameters
        ---

        stt_tstmp
            start timestamp
        end_tstmp
            end timestamp
        """
        ...


@dataclass(kw_only=True)
class Schedule:
    """
    a dataclass that holds fields of ndarrays which forms a schedule
    """

    stt_tstmp_ms: npt.NDArray[np.datetime64]

    # TODO: seems useful to add end_time_ms ?
    # end_tstmp_ms: npt.NDArray[np.datetime64]

    # TODO: add `exp_num` field

    pointing_az: npt.NDArray[np.float64]
    pointing_el: npt.NDArray[np.float64]

    coh_int_bandwidth: npt.NDArray[np.float64]
    ipp: npt.NDArray[np.float64]
    pulse_length: npt.NDArray[np.float64]

    def __post_init__(self):
        f_0, *f_rests = fields(self)  # Field objects
        fv_0, *fv_rests = t.cast(
            tuple[npt.NDArray, ...], [getattr(self, f.name) for f in fields(self)]
        )  # actual value of the fields

        for idx, f in enumerate(fv_rests):
            if f.shape != fv_0.shape:
                raise RuntimeError(
                    "fields of a `Schedule` must have equal lengths.\n"
                    + f"but shape of {f_rests[idx].name} is {fv_rests[idx].shape}, "
                    f"while shape of {f_0.name} is {fv_0.shape} "
                )
