import logging, typing as t
import numpy as np
import numpy.typing as npt
from sorts.types import Timedelta64_us, Datetime64_us, Float64_as_deg, AzelrCoordinates_DegM

logger = logging.getLogger(__name__)


class ExperimentDetail(t.TypedDict):
    """A TypedDict of params"""

    id: int

    coh_int_bandwidth: float  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    ipp: float  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    pulse_length: float  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    power: float
    bandwidth: float
    duty_cycle: float  # TODO: invtg: not used in `sorts.signals.hard_target_snr`?
    noise_temp: float

    slice_duration: Timedelta64_us
    "Duration of a control slice, in micro-second"

    # TODO: this is a temp workaround to get multiple simutaneous rx pointings working
    num_simutaneous_pointings: t.NotRequired[int]


class ScheduleNdarrayDict(t.TypedDict):
    """
    A TypedDict, stores a collection of "control slices" (or "slices" in short).

    - Slice data are stored as columns of fields, each of which is a `ndarray`.
    - Metadata (`ExperimentDetail`s) are stored as a dict inside the `exp_detail_map` field.
    """

    # TODO: add `Station` into this class?
    exp_detail_map: dict[int, ExperimentDetail]

    start_time: npt.NDArray[Datetime64_us]
    end_time: npt.NDArray[Datetime64_us]

    # TODO: re-eval the size of `exp_num`
    exp_num: npt.NDArray[np.int64]

    pointing: AzelrCoordinates_DegM


# TODO: try to remove/dissolve this class?
class ScheduleNdarrayDict2(t.TypedDict):
    """
    A TypedDict, stores a collection of "control slices" (or "slices" in short).

    - Slice data are stored as columns of fields, each of which is a `ndarray`.
    - Metadata (`ExperimentDetail`s) are stored as a dict inside the `exp_detail_map` field.
    """

    # TODO: add `Station` into this class?
    exp_detail_map: dict[int, ExperimentDetail]

    start_time: npt.NDArray[Datetime64_us]

    # TODO: re-eval the size of `exp_num`
    exp_num: npt.NDArray[np.int64]

    pointing_az: npt.NDArray[Float64_as_deg]
    pointing_el: npt.NDArray[Float64_as_deg]
