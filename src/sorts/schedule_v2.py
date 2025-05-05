from __future__ import annotations
import logging, typing as t
from dataclasses import dataclass, fields
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Schedule:
    """
    a dataclass that holds fields of ndarrays which forms a schedule
    """

    # TODO: if `end_tstmp_ms` is not needed, this can be renamed to just `tstmp_ms`?
    stt_tstmp_us: npt.NDArray[np.datetime64]

    # TODO: seems useful to add end_tstmp_ms ?
    # end_tstmp_ms: npt.NDArray[np.datetime64]

    # TODO: re-eval the size of `exp_num`
    exp_num: npt.NDArray[np.int64]

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
                    "fields of a `Schedule` must have equal lengths. "
                    + f"but shape of {f_rests[idx].name} is {fv_rests[idx].shape}, "
                    f"while shape of {f_0.name} is {fv_0.shape} "
                )


__all__ = ["Schedule"]
